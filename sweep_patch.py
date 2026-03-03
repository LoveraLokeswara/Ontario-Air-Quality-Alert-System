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
from transformers import PatchTSTConfig, PatchTSTForPrediction

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

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. UPDATED DATASET CLASS: Accepts pre-scaled data to prevent leakage
class PatchAQDataset(Dataset):
    def __init__(self, scaled_data, scaler, seq_length, forecast_horizon=6):
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

def train_and_evaluate(d_model, dropout, patch_len, lookback, heads, lr, stride, dataset, device):
    n = len(dataset)
    # With 7 years of data, 70/15/15 is standard
    train_idx, val_idx = int(0.7 * n), int(0.85 * n)
    
    train_loader = DataLoader(Subset(dataset, range(train_idx)), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(train_idx, val_idx)), batch_size=128)
    test_loader = DataLoader(Subset(dataset, range(val_idx, n)), batch_size=128)

    config = PatchTSTConfig(
        num_input_channels=6,
        context_length=lookback, 
        prediction_length=6,
        patch_length=patch_len, 
        stride=stride, 
        d_model=d_model, 
        num_layers=3, 
        num_attention_heads=heads, 
        dropout=dropout, 
        prediction_channel_indices=[0]
    )
    
    model = PatchTSTForPrediction(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 2. UPDATED LOSS FUNCTION: Changed to L1Loss (MAE) to fix lag
    criterion = nn.L1Loss()

    best_val_loss, patience, counter = float('inf'), 8, 0
    for epoch in range(40):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(past_values=x).prediction_outputs[:, :, 0], y)
            loss.backward(); optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(past_values=x.to(device)).prediction_outputs
                val_loss += criterion(out[:, :, 0], y.to(device)).item()
        
        avg_val = val_loss / len(val_loader)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "temp_best_v4.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience: break

    model.load_state_dict(torch.load("temp_best_v4.pt"))
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(past_values=x.to(device)).prediction_outputs
            preds.append(out[:, :, 0].cpu().numpy())
            actuals.append(y.numpy())

    all_p, all_a = np.vstack(preds)[:, 5], np.vstack(actuals)[:, 5]
    pm_std = dataset.scaler.scale_[0]
    mae = mean_absolute_error(all_a, all_p) * pm_std
    rmse = np.sqrt(mean_squared_error(all_a, all_p)) * pm_std
    return mae, rmse, all_p, all_a

if __name__ == "__main__":
    data_path = "data/data_clean/cleaned_data_toronto_downtown.csv"
    
    # 3. SCALING FIX: Fit only on the first 70% of the raw dataframe
    df = pd.read_csv(data_path, low_memory=False)
    features = ['PM_ppb', 'Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Stn Press (kPa)', 'Dew Point Temp (°C)']
    
    train_size = int(len(df) * 0.7)
    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_size][features])
    
    # Transform the entire array using only the training distribution
    scaled_data_array = scaler.transform(df[features].values)

    results = []
    
    best_mae = float('inf')
    best_p = None
    best_a = None
    best_config_str = ""
    best_ds = None

    # --- UPDATED SEARCH GRID ---
    lookbacks = [24,48,72]
    d_models = [128, 256] 
    heads_list = [8]
    learning_rates = [0.001, 0.0005, 0.002]
    patch_lengths = [4, 6, 8] 

    print("--- STARTING HIGH-RESOLUTION SWEEP ---", flush=True)

    for lb in lookbacks:
        # Pass the pre-scaled array and the scaler to the dataset
        ds = PatchAQDataset(scaled_data_array, scaler, seq_length=lb)
        
        for dm in d_models:
            for nh in heads_list:
                for lr in learning_rates:
                    for pl in patch_lengths:
                        
                        # Dynamic 50% overlap stride
                        stride = pl // 2 
                        
                        print(f"Testing: LB={lb}, DM={dm}, Heads={nh}, LR={lr}, Patch={pl}, Stride={stride}", flush=True)
                        start_time = time.time()
                        try:
                            mae, rmse, p, a = train_and_evaluate(dm, 0.1, pl, lb, nh, lr, stride, ds, device)
                            
                            if mae < best_mae:
                                best_mae = mae
                                best_p = p
                                best_a = a
                                best_ds = ds
                                best_config_str = f"LB={lb}, DM={dm}, Heads={nh}, LR={lr}, Patch={pl}, Stride={stride}"
                                os.replace("temp_best_v4.pt", os.path.join(RESULTS_DIR, "global_champion_v5.pt"))

                            results.append({
                                "lookback": lb, "d_model": dm, "heads": nh, 
                                "lr": lr, "patch_len": pl, "stride": stride, 
                                "mae": mae, "rmse": rmse
                            })
                            print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | {time.time()-start_time:.1f}s", flush=True)
                        except Exception as e:
                            print(f"Trial failed: {e}", flush=True)

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RESULTS_DIR, "v5_antilag_sweep_results.csv"), index=False)
    print("\nSweep Complete. Top 5 by MAE:")
    print(df_results.sort_values("mae").head(5))

    # 4. PLOT UPDATE: Expanded to 500 hours
    if best_p is not None:
        pm_mean = best_ds.scaler.mean_[0]
        pm_std = best_ds.scaler.scale_[0]
        
        unscaled_preds = (best_p * pm_std) + pm_mean
        unscaled_actuals = (best_a * pm_std) + pm_mean
        
        plt.figure(figsize=(16, 6))
        # Sliced to 500 instead of 100
        plt.plot(unscaled_actuals[:500], label="Actual PM2.5 (ppb)", color='blue', alpha=0.7)
        plt.plot(unscaled_preds[:500], label="Predicted PM2.5 (ppb)", color='red', linestyle='--')
        plt.title(f"Toronto Downtown PM2.5 Forecast (6-Hour Lead Time)\nBest Config: {best_config_str}")
        plt.xlabel("Time Steps (Hours)")
        plt.ylabel("Concentration (ppb)")
        plt.legend()
        plt.grid(True)
        
        # Saved under a new name to avoid overwriting your old one
        plot_path = os.path.join(RESULTS_DIR, "best_pm25_500hr_forecast.png")
        plt.savefig(plot_path)