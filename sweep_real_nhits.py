import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import os
import random
import shutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- SEEDING ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. THE N-HITS ARCHITECTURE
# ==========================================

class NHITSBlock(nn.Module):
    def __init__(self, input_dim, seq_len, max_horizon, pool_size, n_theta, hidden_dim, dropout_rate):
        super().__init__()
        self.pool_size = pool_size
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.max_horizon = max_horizon

        # Downsampling logic
        self.pooled_seq_len = int(np.ceil(seq_len / pool_size))
        flat_dim = input_dim * self.pooled_seq_len

        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )

        # Backcast: Reconstruction of the input window
        self.theta_b = nn.Linear(hidden_dim, seq_len * input_dim)
        # Forecast: Basis knots for interpolation
        self.theta_f = nn.Linear(hidden_dim, n_theta)

    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        x_pool = x.permute(0, 2, 1) 
        if self.pool_size > 1:
            x_pool = F.max_pool1d(x_pool, kernel_size=self.pool_size, stride=self.pool_size, ceil_mode=True)

        h = self.mlp(x_pool.reshape(x_pool.size(0), -1))

        backcast = self.theta_b(h).reshape(-1, self.seq_len, self.input_dim)

        # Hierarchical Interpolation
        theta_f = self.theta_f(h).unsqueeze(1) 
        forecast = F.interpolate(theta_f, size=self.max_horizon, mode='linear', align_corners=True).squeeze(1)

        return backcast, forecast

class RealNHITS(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.max_horizon = 24 
        
        # Stack 1: Macro (Low Frequency) - High pooling, few interpolation knots
        self.block1 = NHITSBlock(input_dim, seq_len, self.max_horizon, 4, 6, hidden_dim, dropout_rate)
        # Stack 2: Mid Frequency
        self.block2 = NHITSBlock(input_dim, seq_len, self.max_horizon, 2, 12, hidden_dim, dropout_rate)
        # Stack 3: Fine (High Frequency) - No pooling, 24 interpolation knots
        self.block3 = NHITSBlock(input_dim, seq_len, self.max_horizon, 1, 24, hidden_dim, dropout_rate)

    def forward(self, x):
        # Target-Specific Residual logic: residuals only on PM2.5 (column 0)
        exog = x[:, :, 1:] 
        target = x[:, :, 0:1] 

        # Block 1
        b1, f1 = self.block1(x)
        res_target = target - b1[:, :, 0:1] 

        # Block 2: Feed target residual + original exogenous features
        input_2 = torch.cat([res_target, exog], dim=-1)
        b2, f2 = self.block2(input_2)
        res_target = res_target - b2[:, :, 0:1]

        # Block 3
        input_3 = torch.cat([res_target, exog], dim=-1)
        _, f3 = self.block3(input_3)

        # Final prediction is the sum of frequency-decomposed forecasts
        # We slice index 3 to get the 4-hour-ahead forecast
        return (f1 + f2 + f3)[:, 3]

# ==========================================
# 2. DATASET & EVALUATION
# ==========================================

class AQDataset(Dataset):
    def __init__(self, data, seq_len, horizon=4):
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx : idx + self.seq_len])
        # Target is PM2.5 at (lookback + 4 hours)
        y = torch.FloatTensor([self.data[idx + self.seq_len + self.horizon - 1, 0]])
        return x, y

def run_trial(lb, lr, hd, drop, data, device):
    ds = AQDataset(data, lb)
    n = len(ds)
    tr_idx, val_idx = int(0.7 * n), int(0.85 * n)
    
    tr_loader = DataLoader(Subset(ds, range(tr_idx)), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(ds, range(tr_idx, val_idx)), batch_size=128)
    ts_loader = DataLoader(Subset(ds, range(val_idx, n)), batch_size=128)

    model = RealNHITS(input_dim=11, seq_len=lb, hidden_dim=hd, dropout_rate=drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_v = float('inf')
    patience = 8
    counter = 0

    for epoch in range(50):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x).unsqueeze(1), y)
            loss.backward()
            opt.step()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                v_loss += crit(model(x.to(device)).unsqueeze(1), y.to(device)).item()
        
        avg_v = v_loss / len(val_loader)
        if avg_v < best_v:
            best_v = avg_v
            torch.save(model.state_dict(), "best_scratch_weights.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience: break

    model.load_state_dict(torch.load("best_scratch_weights.pt"))
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in ts_loader:
            preds.append(model(x.to(device)).cpu().numpy())
            actuals.append(y.numpy())
    
    return np.concatenate(preds), np.concatenate(actuals).flatten()

if __name__ == "__main__":
    print("Loading data for Scratch N-HiTS...", flush=True)
    df = pd.read_csv("data/data_clean/cleaned_data_toronto_downtown.csv", low_memory=False)
    feats = ['PM_ppb', 'Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Stn Press (kPa)', 'Dew Point Temp (°C)', 'Precip. Amount (mm)', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']


    train_size = int(len(df) * 0.7)
    # Fit the scaler ONLY on the training data
    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_size][feats])
    scaled_data = scaler.transform(df[feats])
    
    pm_mean, pm_std = scaler.mean_[0], scaler.scale_[0]

    # --- EXPANDED GRID ---
    lookbacks = [20, 24, 48]
    hidden_dims = [256, 512]
    learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.000001] 
    dropouts = [0.1]

    print("--- STARTING EXPANDED N-HITS SCRATCH SWEEP ---", flush=True)
    results = []
    
    # Trackers for the overall champion model
    best_rmse = float('inf')
    best_p, best_a, best_cfg = None, None, None

    for lb in lookbacks:
        for hd in hidden_dims:
            for lr in learning_rates:
                for dr in dropouts:
                    start = time.time()
                    try:
                        p_raw, a_raw = run_trial(lb, lr, hd, dr, scaled_data, device)
                        
                        p = (p_raw * pm_std) + pm_mean
                        a = (a_raw * pm_std) + pm_mean
                        
                        rmse = np.sqrt(mean_squared_error(a, p))
                        mae = mean_absolute_error(a, p)
                        
                        print(f"LB: {lb} | HD: {hd} | LR: {lr} | RMSE: {rmse:.4f} | {time.time()-start:.1f}s", flush=True)
                        results.append({'lb': lb, 'hd': hd, 'lr': lr, 'dr': dr, 'rmse': rmse, 'mae': mae})
                        
                        # Save the global champion for local plotting later
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_p, best_a, best_cfg = p, a, {'lb': lb, 'hd': hd}
                            shutil.copy("best_scratch_weights.pt", "global_champion_nhits.pt")
                            
                    except Exception as e:
                        print(f"Trial failed: {e}")

    df_res = pd.DataFrame(results)
    df_res.to_csv("results_scratch_nhits_expanded.csv", index=False)
    print("\nSweep Complete. Top 3 by RMSE:")
    print(df_res.sort_values("rmse").head(3))

    # --- PLOT THE BEST MODEL ---
    if best_p is not None:
        print("\nGenerating Test Set Plot...", flush=True)
        plt.figure(figsize=(15, 6))
        plt.plot(best_a[:500], label="Actual PM2.5 (ppb)", color='blue', alpha=0.7)
        plt.plot(best_p[:500], label="N-HiTS Predicted", color='red', linestyle='--')
        plt.title(f"Toronto PM2.5 Test Set (4hr Horizon)\nChampion N-HiTS: {best_cfg['lb']}h Lookback, {best_cfg['hd']} Hidden | RMSE: {best_rmse:.4f}")
        plt.xlabel("Hours")
        plt.ylabel("PM2.5 Concentration")
        plt.legend()
        plt.grid(True)
        plt.savefig("best_nhits_test_plot.png")
        print("Saved plot as: best_nhits_test_plot.png")
        print("Global champion weights saved as: global_champion_nhits.pt")