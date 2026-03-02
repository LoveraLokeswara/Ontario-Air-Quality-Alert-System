"""
Temporal Convolutional Network (TCN) for multi-horizon PM2.5 forecasting.

Architecture overview
─────────────────────
Input (batch, seq_len, input_size)
  → Linear input projection  →  (batch, seq_len, num_channels[0])
  → Transpose                →  (batch, channels, seq_len)
  → N residual TCN blocks, dilation doubles each block (1, 2, 4, 8 …)
  → Final time-step          →  (batch, channels)
  → One FC head per forecast horizon

Each residual block
───────────────────
  CausalConv1d → BatchNorm → ReLU → Dropout
  CausalConv1d → BatchNorm → ReLU → Dropout
  + skip connection (1×1 conv if channel widths differ)

Causal convolution
──────────────────
  Left-padding only: pad = (kernel_size - 1) × dilation
  The padded tail is trimmed so output length == input length and
  no future time-steps are visible to the model.

Reference
─────────
  Bai et al. (2018) "An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling"
  https://arxiv.org/abs/1803.01271
"""

import torch
import torch.nn as nn


class _CausalConv1d(nn.Module):
    """Left-padded causal 1-D convolution with no future leakage."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.pad, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class _TCNBlock(nn.Module):
    """
    Residual TCN block: two stacked causal convolutions + skip connection.

    Input / Output shape: (batch, channels, seq_len)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_channels,  out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Match channel dimensions for the residual if needed
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.skip(x))


class PM25TCNForecaster(nn.Module):
    """
    Multi-horizon Temporal Convolutional Network for PM2.5 forecasting.

    Parameters
    ──────────
    input_size   : number of input features per time step
    num_channels : list of output channel sizes for each TCN block;
                   dilation doubles with each block (2^0, 2^1, …)
                   default: [64, 64, 128, 128]
    kernel_size  : convolutional kernel width          (default 3)
    dropout      : dropout probability                 (default 0.2)
    n_horizons   : forecast horizons to predict simultaneously
                   (default 3, corresponding to +4 h, +6 h, +24 h)

    Receptive field
    ───────────────
    RF = (kernel_size - 1) × sum(2^i for i in 0..N-1) × 2
    With kernel=3, 4 blocks: RF = 2 × (1+2+4+8) × 2 = 60 steps
    — comfortably covers a 24-hour lookback window.
    """

    def __init__(
        self,
        input_size:   int,
        num_channels: list[int] | None = None,
        kernel_size:  int               = 3,
        dropout:      float             = 0.2,
        n_horizons:   int               = 3,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 128, 128]

        # Project raw features into the TCN channel space
        self.input_proj = nn.Linear(input_size, num_channels[0])

        # Build residual TCN blocks; dilation = 2^i
        blocks: list[nn.Module] = []
        in_ch = num_channels[0]
        for i, out_ch in enumerate(num_channels):
            blocks.append(
                _TCNBlock(in_ch, out_ch, kernel_size,
                          dilation=2 ** i, dropout=dropout)
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)

        # Independent FC head for each forecast horizon
        self.heads = nn.ModuleList([
            nn.Linear(num_channels[-1], 1) for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        x = self.input_proj(x)   # (batch, seq_len, channels)
        x = x.transpose(1, 2)    # (batch, channels, seq_len)
        x = self.tcn(x)          # (batch, channels, seq_len)
        last = x[:, :, -1]       # (batch, channels) — final causal timestep
        preds = torch.cat(
            [head(last) for head in self.heads], dim=1
        )                        # (batch, n_horizons)
        return preds
