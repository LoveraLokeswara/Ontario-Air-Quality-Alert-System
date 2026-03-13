import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NHITSBlock(nn.Module):
    def __init__(self, input_dim, seq_len, max_horizon, pool_size, n_theta, hidden_dim, dropout_rate):
        super().__init__()
        self.pool_size = pool_size
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.max_horizon = max_horizon

        self.pooled_seq_len = int(np.ceil(seq_len / pool_size))
        flat_dim = input_dim * self.pooled_seq_len

        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )

        self.theta_b = nn.Linear(hidden_dim, seq_len * input_dim)
        self.theta_f = nn.Linear(hidden_dim, n_theta)

    def forward(self, x):
        x_pool = x.permute(0, 2, 1) 
        if self.pool_size > 1:
            x_pool = F.max_pool1d(x_pool, kernel_size=self.pool_size, stride=self.pool_size, ceil_mode=True)

        h = self.mlp(x_pool.reshape(x_pool.size(0), -1))
        backcast = self.theta_b(h).reshape(-1, self.seq_len, self.input_dim)
        theta_f = self.theta_f(h).unsqueeze(1) 
        forecast = F.interpolate(theta_f, size=self.max_horizon, mode='linear', align_corners=True).squeeze(1)

        return backcast, forecast

class RealNHITS(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.max_horizon = 24
        self.block1 = NHITSBlock(input_dim, seq_len, self.max_horizon, 4, 6, hidden_dim, dropout_rate)
        self.block2 = NHITSBlock(input_dim, seq_len, self.max_horizon, 2, 12, hidden_dim, dropout_rate)
        self.block3 = NHITSBlock(input_dim, seq_len, self.max_horizon, 1, 24, hidden_dim, dropout_rate)

    def forward(self, x):
        exog = x[:, :, 1:] 
        target = x[:, :, 0:1] 

        b1, f1 = self.block1(x)
        res_target = target - b1[:, :, 0:1] 

        input_2 = torch.cat([res_target, exog], dim=-1)
        b2, f2 = self.block2(input_2)
        res_target = res_target - b2[:, :, 0:1]

        input_3 = torch.cat([res_target, exog], dim=-1)
        _, f3 = self.block3(input_3)

        return (f1 + f2 + f3)[:, 3]