import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import os
import random 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}", flush=True)

# ISOLATED RESULTS FOLDER
RESULTS_DIR = "results_11feat_mse"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. THE DATASET CLASS
class NHITSAQDataset(Dataset):
    def __init__(self, scaled_data, scaler, seq_length, forecast_horizon=4):
        self.data = scaled_data
        self.scaler = scaler
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.seq_length - self.forecast_horizon

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx : idx + self.seq_length])
        y = torch.FloatTensor(self.data[idx + self.seq_length : idx + self.seq_length + self.forecast_horizon, 0])
        return x, y

# 2. THE DYNAMIC N-HITS ARCHITECTURE
class ModelNHITS(nn.Module):
    def __init__(self, input_dim, seq_len=24, output_dim=4, hidden_dim=128, dropout=0.1):
        super().__init__()
        flat_dim = input_dim * seq_len
        
        self.branch_high = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.1), nn.Linear(hidden_dim // 2, 32)
        )
        
        self.pool = nn.MaxPool1d(kernel_size=4)
        pool_len = seq_len // 4
        self.branch_low = nn.Sequential(
            nn.Linear(input_dim * pool_len, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.1), nn.Linear(hidden_dim // 2, 32)
        )
        
        self.fc_out = nn.Linear(32 + 32, output_dim)
        
    def forward(self, x):
        x_high = x.reshape(x.size(0), -1)
        out_high = self.branch_high(x_high)
        
        x_low = x.permute(0, 2, 1) 
        x_low = self.pool(x_low).reshape(x.size(0), -1)
        out_low = self.branch_low(x_low)
        
        return self.fc_out(torch.cat((out_high, out_low), dim=1))

def train_and_evaluate(lookback, lr, hidden_dim, dropout, dataset, device):
    n = len(dataset)
    train_idx, val_idx = int(0.7 * n), int(0.85 * n)
    
    train_loader = DataLoader(Subset(dataset, range(train_idx)), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(train_idx, val_idx)), batch_size=128)
    test_loader = DataLoader(Subset(dataset, range(val_idx, n)), batch_size=128)

    # 11 FEATURES
    model = ModelNHITS(input_dim=11, seq_len=lookback, output_dim=4, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # MSE LOSS FOR RMSE TARGETING
    criterion = nn.MSELoss()

    best_val_loss, patience, counter = float('inf'), 8, 0
    for epoch in range(40):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.to(device))
                val_loss += criterion(out, y.to(device)).item()
        
        avg_val = val_loss / len(val_loader)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # UNIQUE TEMP NAME
            torch.save(model.state_dict(), "temp_best_nhits_11f.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience: break

    model.load_state_dict(torch.load("temp_best_nhits_11f.pt"))
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device))
            preds.append(out.cpu().numpy())
            actuals.append(y.numpy())

    all_p, all_a = np.vstack(preds)[:, 3], np.vstack(actuals)[:, 3]
    pm_std = dataset.scaler.scale_[0]
    mae = mean_absolute_error(all_a, all_p) * pm_std
    rmse = np.sqrt(mean_squared_error(all_a, all_p)) * pm_std
    return mae, rmse, all_p, all_a

if __name__ == "__main__":
    data_path = "data/data_clean/cleaned_data_toronto_downtown.csv"
    
    df = pd.read_csv(data_path, low_memory=False)
    # 11 FEATURES LIST
    features = ['PM_ppb', 'Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Stn Press (kPa)', 'Dew Point Temp (°C)', 'Precip. Amount (mm)', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    
    train_size = int(len(df) * 0.7)
    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_size][features])
    scaled_data_array = scaler.transform(df[features].values)

    results = []
    
    # TRACKING RMSE INSTEAD OF MAE
    best_rmse = float('inf')
    best_p, best_a, best_config_str, best_ds = None, None, "", None

    # SHORTER LOOKBACK GRID
    lookbacks = [8, 12, 16, 20, 24] 
    hidden_dims = [128, 256, 512]
    dropouts = [0.1, 0.2, 0.3]
    learning_rates = [0.001, 0.0005, 0.0001, 0.00005]

    print("--- STARTING 11-FEATURE MSE N-HITS SWEEP ---", flush=True)

    for lb in lookbacks:
        ds = NHITSAQDataset(scaled_data_array, scaler, seq_length=lb)
        for hd in hidden_dims:
            for drop in dropouts:
                for lr in learning_rates:
                    print(f"Testing N-HiTS: Lookback={lb}, Hidden={hd}, Drop={drop}, LR={lr}", flush=True)
                    start_time = time.time()
                    try:
                        mae, rmse, p, a = train_and_evaluate(lb, lr, hd, drop, ds, device)
                        
                        # SAVING BEST MODEL BY RMSE
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_p, best_a, best_ds = p, a, ds
                            best_config_str = f"Lookback={lb}, Hidden={hd}, Drop={drop}, LR={lr}"
                            # ISOLATED CHAMPION FILE
                            os.replace("temp_best_nhits_11f.pt", os.path.join(RESULTS_DIR, "global_champion_nhits_11f.pt"))

                        results.append({
                            "lookback": lb, "hidden_dim": hd, "dropout": drop, "lr": lr, 
                            "mae": mae, "rmse": rmse
                        })
                        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | {time.time()-start_time:.1f}s", flush=True)
                    except Exception as e:
                        print(f"Trial failed: {e}", flush=True)
                        import traceback
                        traceback.print_exc()

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RESULTS_DIR, "nhits_11feat_mse_sweep_results.csv"), index=False)
    
    # SORTING FINAL LEADERBOARD BY RMSE
    print("\nSweep Complete. Top 5 by RMSE:")
    print(df_results.sort_values("rmse").head(5))

    if best_p is not None:
        pm_mean = best_ds.scaler.mean_[0]
        pm_std = best_ds.scaler.scale_[0]
        
        unscaled_preds = (best_p * pm_std) + pm_mean
        unscaled_actuals = (best_a * pm_std) + pm_mean
        
        plt.figure(figsize=(16, 6))
        plt.plot(unscaled_actuals[:500], label="Actual PM2.5 (ppb)", color='blue', alpha=0.7)
        plt.plot(unscaled_preds[:500], label="Predicted PM2.5 (ppb)", color='red', linestyle='--')
        
        # UPDATED PLOT TITLE
        plt.title(f"Toronto Downtown PM2.5 (4-Hour) - N-HiTS\nBest Config: {best_config_str}")
        plt.xlabel("Time Steps (Hours)")
        plt.ylabel("Concentration (ppb)")
        plt.legend()
        plt.grid(True)
        
        # UPDATED PLOT FILENAME
        plt.savefig(os.path.join(RESULTS_DIR, "best_pm25_4hr_nhits_11f.png"))