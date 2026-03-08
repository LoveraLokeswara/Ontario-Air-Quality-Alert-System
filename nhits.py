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
        y = torch.FloatTensor([self.data[idx + self.seq_len + self.horizon - 1, 0]])
        return x, y

def run_trial(lb, lr, hd, drop, data, device, pm_mean, pm_std):
    """Returns ONLY validation results to prevent grid search leakage."""
    ds = AQDataset(data, lb)
    n = len(ds)
    tr_idx, val_idx = int(0.7 * n), int(0.85 * n)
    
    tr_loader = DataLoader(Subset(ds, range(tr_idx)), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(ds, range(tr_idx, val_idx)), batch_size=128)

    model = RealNHITS(input_dim=11, seq_len=lb, hidden_dim=hd, dropout_rate=drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_v_rmse = float('inf')
    best_v_residuals = None
    patience, counter = 8, 0

    for epoch in range(50):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x).unsqueeze(1), y)
            loss.backward()
            opt.step()

        model.eval()
        v_preds, v_acts = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.to(device))
                v_preds.append(out.cpu().numpy())
                v_acts.append(y.numpy())
        
        # Calculate unscaled validation performance
        p_val = (np.concatenate(v_preds) * pm_std) + pm_mean
        a_val = (np.concatenate(v_acts).flatten() * pm_std) + pm_mean
        v_rmse = np.sqrt(mean_squared_error(a_val, p_val))

        if v_rmse < best_v_rmse:
            best_v_rmse = v_rmse
            best_v_residuals = a_val - p_val # Harvest Estimated residuals
            torch.save(model.state_dict(), "temp_best.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience: break
    
    return best_v_rmse, best_v_residuals

if __name__ == "__main__":
    print("Loading data for Scratch N-HiTS...", flush=True)
    df = pd.read_csv("data/data_clean/cleaned_data_toronto_downtown.csv", low_memory=False)
    feats = ['PM_ppb', 'Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Stn Press (kPa)', 'Dew Point Temp (°C)', 'Precip. Amount (mm)', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']

    # Scale strictly on 70% to prevent leakage
    train_size = int(len(df) * 0.7)
    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_size][feats])
    scaled_data = scaler.transform(df[feats])
    
    pm_mean, pm_std = scaler.mean_[0], scaler.scale_[0]

    # --- THE CLEAN SWEEP ---
    lookbacks = [24, 48]
    hidden_dims = [128, 256, 512]
    learning_rates = [0.001, 0.0005, 0.0001, 0.00005] 
    dropouts = [0.1]

    print("--- STARTING N-HITS SWEEP (VAL-OPTIMIZED) ---", flush=True)
    results = []
    best_v_global = float('inf')
    winning_residuals = None
    best_cfg = None

    for lb in lookbacks:
        for hd in hidden_dims:
            for lr in learning_rates:
                for dr in dropouts:
                    start = time.time()
                    v_rmse, v_res = run_trial(lb, lr, hd, dr, scaled_data, device, pm_mean, pm_std)
                    
                    print(f"LB: {lb} | HD: {hd} | LR: {lr} | VAL RMSE: {v_rmse:.4f} | {time.time()-start:.1f}s", flush=True)
                    results.append({'lb': lb, 'hd': hd, 'lr': lr, 'dr': dr, 'val_rmse': v_rmse})
                    
                    if v_rmse < best_v_global:
                        best_v_global = v_rmse
                        winning_residuals = v_res
                        best_cfg = {'lb': lb, 'hd': hd, 'lr': lr, 'dr': dr}
                        shutil.copy("temp_best.pt", "global_champion_nhits.pt")

    # Save search results
    pd.DataFrame(results).to_csv("nhits_sweep_pure_val.csv", index=False)

    # Calculate Estimated Standard Error from the winning validation residuals
    estimated_se = np.std(winning_residuals)
    print(f"\nWINNER CHOSEN: {best_cfg} | Estimated SE: {estimated_se:.4f}")

    # ==========================================
    # FINAL EVALUATION (THE BLIND TEST SET)
    # ==========================================
    print("\n--- PERFORMING FINAL BLIND TEST ---", flush=True)
    
    # Load winning model
    champ_model = RealNHITS(input_dim=11, seq_len=best_cfg['lb'], 
                            hidden_dim=best_cfg['hd'], dropout_rate=best_cfg['dr']).to(device)
    champ_model.load_state_dict(torch.load("global_champion_nhits.pt"))
    champ_model.eval()

    # Setup Test Loader
    final_ds = AQDataset(scaled_data, best_cfg['lb'])
    n_final = len(final_ds)
    test_idx = int(0.85 * n_final)
    test_loader = DataLoader(Subset(final_ds, range(test_idx, n_final)), batch_size=128)

    test_preds, test_acts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            test_preds.append(champ_model(x.to(device)).cpu().numpy())
            test_acts.append(y.numpy())

    p_final = (np.concatenate(test_preds) * pm_std) + pm_mean
    a_final = (np.concatenate(test_acts).flatten() * pm_std) + pm_mean
    
    final_rmse = np.sqrt(mean_squared_error(a_final, p_final))
    print(f"FINAL TEST RMSE: {final_rmse:.4f} | Estimated SE: {estimated_se:.4f}")
    
    # Plot with Estimated Error Bands
    plt.figure(figsize=(15, 6))
    plt.plot(a_final[:400], label="Actual PM2.5", color='black', alpha=0.5)
    plt.plot(p_final[:400], label="N-HiTS Forecast", color='red')
    plt.fill_between(range(400), 
                     p_final[:400] - 1.96 * estimated_se, 
                     p_final[:400] + 1.96 * estimated_se, 
                     color='red', alpha=0.15, label="95% CI (Estimated)")
    
    plt.title(f"Real N-HiTS Pure Test Set Showdown\nTest RMSE: {final_rmse:.2f} | Estimated SE: {estimated_se:.2f}")
    plt.legend()
    plt.savefig("final_pure_nhits_test.png")
    
    # Save the SE and residuals for later comparison
    np.save("Estimated_residuals.npy", winning_residuals)
    with open("Estimated_se.txt", "w") as f:
        f.write(str(estimated_se))