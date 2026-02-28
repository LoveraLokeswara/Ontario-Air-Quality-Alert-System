import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# --- Settings ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
FORECAST_HORIZON = 6  # Predicting 6 steps ahead

class AirQualityDataset(Dataset):
    def __init__(self, csv_path, seq_length=24, forecast_horizon=6):
        df = pd.read_csv(csv_path, low_memory=False)
        self.features = ['PM_ppb', 'Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)']
        data = df[self.features].values
        
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data)
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        # Ensure room for sequence + the 6-hour future target
        return len(self.data) - self.seq_length - self.forecast_horizon

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        # Target is the next 6 hours of PM_ppb (column 0)
        y = self.data[idx + self.seq_length : idx + self.seq_length + self.forecast_horizon, 0]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class MultiStepForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super(MultiStepForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Final layer outputs the full 6-hour horizon
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

def run_research():
    global_start_time = time.time()
    
    # 1. Load Data
    data_path = "data/data_clean/cleaned_data_toronto_downtown.csv"
    full_dataset = AirQualityDataset(data_path, forecast_horizon=FORECAST_HORIZON)
    
    # 2. Time-Series Split
    train_size = int(0.8 * len(full_dataset))
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(full_dataset)))
    
    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=256, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=256, shuffle=False)

    # 3. Model Setup (out_size=6)
    model = MultiStepForecaster(input_size=4, hidden_size=128, num_layers=3, 
                                out_size=FORECAST_HORIZON).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"--- RESEARCH RUN STARTED (6-HOUR FORECAST) ---")
    print(f"Device: {device}")
    print(f"Data Source: {data_path}")
    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")

    # 4. Training Loop
    epochs = 10 
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                val_loss += criterion(model(x_v), y_v).item()
        
        epoch_end = time.time()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Time: {epoch_end-epoch_start:.2f}s")

    # 5. Evaluation and Inverse Scaling
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for x_v, y_v in val_loader:
            preds = model(x_v.to(device)).cpu().numpy()
            all_preds.append(preds)
            all_actuals.append(y_v.numpy())
            
    all_preds = np.vstack(all_preds)
    all_actuals = np.vstack(all_actuals)

    # Inverse transform logic (using dummy columns for the 4 features)
    dummy = np.zeros((len(all_preds), 4))
    
    # We evaluate metrics specifically for the 6th hour prediction
    pred_6hr = all_preds[:, 5]
    actual_6hr = all_actuals[:, 5]

    dummy_preds = dummy.copy()
    dummy_preds[:, 0] = pred_6hr
    unscaled_preds_6hr = full_dataset.scaler.inverse_transform(dummy_preds)[:, 0]
    
    dummy_actuals = dummy.copy()
    dummy_actuals[:, 0] = actual_6hr
    unscaled_actuals_6hr = full_dataset.scaler.inverse_transform(dummy_actuals)[:, 0]

    # Calculate Metrics for the 6-hour lead time
    rmse = np.sqrt(mean_squared_error(unscaled_actuals_6hr, unscaled_preds_6hr))
    mae = mean_absolute_error(unscaled_actuals_6hr, unscaled_preds_6hr)
    
    print(f"\n--- EVALUATION RESULTS (HOUR +6) ---")
    print(f"Root Mean Squared Error: {rmse:.2f} ppb")
    print(f"Mean Absolute Error: {mae:.2f} ppb")

    # 6. Plotting (Original Look: 100 Hour Continuous Snippet)
    plt.figure(figsize=(12, 6))
    plt.plot(unscaled_actuals_6hr[:100], label="Actual PM2.5 (ppb)", color='blue', alpha=0.7)
    plt.plot(unscaled_preds_6hr[:100], label="Predicted PM2.5 (ppb)", color='red', linestyle='--')
    plt.title("Toronto Downtown PM2.5 Forecast (6-Hour Lead Time)")
    plt.xlabel("Time Steps (Hours)")
    plt.ylabel("Concentration (ppb)")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_DIR, "pm25_6hr_forecast.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    total_run_time = time.time() - global_start_time
    print(f"\nTOTAL EXECUTION TIME: {total_run_time:.2f} seconds")

if __name__ == "__main__":
    run_research()
