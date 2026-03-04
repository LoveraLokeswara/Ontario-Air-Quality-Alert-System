import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import (
    Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    my_Layernorm, series_decomp,
)


class Model(nn.Module):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term
    Series Forecasting (NeurIPS 2021).  https://arxiv.org/abs/2106.13008

    Architecture overview
    ─────────────────────
    1. Series decomposition at every sub-layer separates trend and seasonal
       components progressively.

    2. The encoder processes the look-back window and produces a context
       representation of the seasonal component only (trend is discarded from
       the encoder stream because it can be predicted more reliably by the
       decoder's explicit trend accumulator).

    3. The decoder is initialised with:
         • seasonal_init = last `label_len` seasonal values  +  zeros for pred_len
         • trend_init    = last `label_len` trend   values  +  mean of enc input

       At each decoder layer the Auto-Correlation (self + cross) extracts the
       seasonal signal, while the three decomp blocks accumulate a residual
       trend contribution that is added to trend_init iteratively.

    4. Final output = seasonal_part + trend_part  (both in output feature space)

    Config attributes used
    ──────────────────────
    seq_len, label_len, pred_len
    enc_in, dec_in, c_out
    d_model, n_heads, e_layers, d_layers, d_ff
    moving_avg   – kernel size for the moving-average decomposition (must be odd)
    factor       – controls top-k lags in Auto-Correlation  (k = factor * log L)
    dropout, embed, freq, activation
    output_attention
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Series decomposition (shared; used only at model init for the
        # encoder input — each layer has its own decomp blocks)
        self.decomp = series_decomp(configs.moving_avg)

        # ── Embeddings (no positional encoding, per the paper) ──────────────
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        # ── Encoder ─────────────────────────────────────────────────────────
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
        )

        # ── Decoder ─────────────────────────────────────────────────────────
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # Masked self-attention (causal) on the decoder sequence
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            mask_flag=True,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    # Cross-attention between decoder and encoder output
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        x_enc      : [B, seq_len,              enc_in]   encoder input
        x_mark_enc : [B, seq_len,              4    ]    encoder time features
        x_dec      : [B, label_len + pred_len, dec_in]   (only shape is used)
        x_mark_dec : [B, label_len + pred_len, 4    ]    decoder time features

        Returns    : [B, pred_len, c_out]
        """
        # ── Decompose encoder input to seed the decoder ──────────────────────
        # trend_init for the pred_len future steps: use the mean of x_enc
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).expand(-1, self.pred_len, -1)
        # zeros placeholder for the pred_len seasonal future
        zeros = torch.zeros(
            x_dec.shape[0], self.pred_len, x_dec.shape[2], device=x_enc.device
        )

        seasonal_enc, trend_enc = self.decomp(x_enc)

        # Decoder init: [label_len real | pred_len initialised]
        trend_init = torch.cat([trend_enc[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_enc[:, -self.label_len:, :], zeros], dim=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # ── Decoder ──────────────────────────────────────────────────────────
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out,
            x_mask=dec_self_mask, cross_mask=dec_enc_mask,
            trend=trend_init,
        )

        # Final prediction: seasonal + trend, take only the pred_len horizon
        dec_out = trend_part + seasonal_part
        dec_out = dec_out[:, -self.pred_len:, :]

        if self.output_attention:
            return dec_out, attns
        return dec_out
