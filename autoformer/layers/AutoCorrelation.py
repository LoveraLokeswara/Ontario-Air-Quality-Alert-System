import math
import torch
import torch.nn as nn


class AutoCorrelation(nn.Module):
    """
    Auto-Correlation Mechanism (NeurIPS 2021).

    Instead of dot-product attention (O(L^2)), this uses FFT-based cross-correlation
    to discover period-based dependencies and aggregate information at the series level,
    achieving O(L log L) complexity.

    Two phases:
      1. Period-based dependency discovery via FFT cross-correlation
      2. Time-delay aggregation: roll the value sequence by the top-k lag offsets
         and take a weighted sum, where weights are the softmax of the correlation scores.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        Batch-normalisation-style aggregation used during training.
        Picks a shared set of top-k lag indices (averaged over the batch),
        which is faster and more memory-friendly.

        values: [B, H, E, L]  (heads, embed_dim, length)
        corr:   [B, H, E, L]
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        top_k = int(self.factor * math.log(length))
        # Average correlation across heads and channels -> [B, L]
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        # Shared top-k indices across batch -> [top_k]
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        # Per-sample weights at those indices -> [B, top_k]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)

        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            # Circularly shift values by the lag, then weight-sum
            pattern = torch.roll(values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1).unsqueeze(1).unsqueeze(1)
                .expand(values.shape[0], head, channel, length)
            )
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        Per-sample top-k aggregation used during inference.
        Picks the best k lags individually for each sample in the batch.

        values: [B, H, E, L]
        corr:   [B, H, E, L]
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # Base indices for gather: [B, H, E, L]
        init_index = (
            torch.arange(length)
            .unsqueeze(0).unsqueeze(0).unsqueeze(0)
            .expand(batch, head, channel, length)
            .to(values.device)
        )
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [B, L]
        weights, delay = torch.topk(mean_value, top_k, dim=-1)   # [B, top_k]
        tmp_corr = torch.softmax(weights, dim=-1)

        # Double the sequence so circular shifts become simple gathers
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            # Shift index for lag i: [B, H, E, L]
            tmp_delay = init_index + (
                delay[:, i]
                .unsqueeze(1).unsqueeze(1).unsqueeze(1)
                .expand(batch, head, channel, length)
            )
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1).unsqueeze(1).unsqueeze(1)
                .expand(batch, head, channel, length)
            )
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # Align sequence lengths
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :, :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # FFT cross-correlation in frequency domain: O(L log L)
        # Shape after permute: [B, H, E, L]
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), dim=-1)

        # Time-delay aggregation
        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    """
    Multi-head wrapper around AutoCorrelation, mirroring the MultiheadAttention API.
    Projects Q, K, V to n_heads × d_keys/d_values, runs AutoCorrelation, then
    projects back to d_model.
    """

    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
