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

class PatchAQDataset(Dataset):
    def __init__(self, csv_path, seq_length, forecast_horizon=6):
        df = pd.read_csv(csv_path, low_memory=False)
        # Updated feature set (6 variables)
        self.features = ['PM_ppb', 'Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Stn Press (kPa)', 'Dew Point Temp (°C)']
        data = df[self.features].values
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data)
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
        num_input_channels=6, # Updated to 6
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
    criterion = nn.MSELoss()

    best_val_loss, patience, counter = float('inf'), 8, 0 # Lowered patience for efficiency
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
    
    results = []
    
    # Variables for tracking the best model silently
    best_mae = float('inf')
    best_p = None
    best_a = None
    best_config_str = ""
    best_ds = None

    # --- EXPANDED SEARCH GRID ---
    lookbacks = [24, 72, 156, 254]
    d_models = [128, 256] # Increased capacity for 6 features
    heads_list = [8]
    learning_rates = [0.001, 0.0005]
    patch_lengths = [8, 16, 24] 

    print("--- STARTING HEAVY-FEATURE MULTI-YEAR LONG SWEEP ---", flush=True)

    for lb in lookbacks:
        ds = PatchAQDataset(data_path, seq_length=lb)
        for dm in d_models:
            for nh in heads_list:
                for lr in learning_rates:
                    for pl in patch_lengths:
                        if pl >= lb: continue
                        
                        # Test a smaller, higher-overlap stride
                        stride = 4 if pl == 8 else 8
                        
                        print(f"Testing: LB={lb}, DM={dm}, Heads={nh}, LR={lr}, Patch={pl}", flush=True)
                        start_time = time.time()
                        try:
                            mae, rmse, p, a = train_and_evaluate(dm, 0.1, pl, lb, nh, lr, stride, ds, device)
                            
                            # Silently track the best model
                            if mae < best_mae:
                                best_mae = mae
                                best_p = p
                                best_a = a
                                best_ds = ds
                                best_config_str = f"LB={lb}, DM={dm}, Heads={nh}, LR={lr}, Patch={pl}"
                                os.replace("temp_best_v4.pt", os.path.join(RESULTS_DIR, "global_champion_v4.pt"))

                            results.append({
                                "lookback": lb, "d_model": dm, "heads": nh, 
                                "lr": lr, "patch_len": pl, "mae": mae, "rmse": rmse
                            })
                            print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | {time.time()-start_time:.1f}s", flush=True)
                        except Exception as e:
                            print(f"Trial failed: {e}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "v4_heavy_sweep_results.csv"), index=False)
    print("\nSweep Complete. Top 5 by MAE:")
    print(df.sort_values("mae").head(5))

    # Plot the first 100 hours of the best model
    if best_p is not None:
        pm_mean = best_ds.scaler.mean_[0]
        pm_std = best_ds.scaler.scale_[0]
        
        unscaled_preds = (best_p * pm_std) + pm_mean
        unscaled_actuals = (best_a * pm_std) + pm_mean
        
        plt.figure(figsize=(12, 6))
        plt.plot(unscaled_actuals[:100], label="Actual PM2.5 (ppb)", color='blue', alpha=0.7)
        plt.plot(unscaled_preds[:100], label="Predicted PM2.5 (ppb)", color='red', linestyle='--')
        plt.title(f"Toronto Downtown PM2.5 Forecast (6-Hour Lead Time)\nBest Config: {best_config_str}")
        plt.xlabel("Time Steps (Hours)")
        plt.ylabel("Concentration (ppb)")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(RESULTS_DIR, "best_pm25_100hr_forecast.png")
        plt.savefig(plot_path)