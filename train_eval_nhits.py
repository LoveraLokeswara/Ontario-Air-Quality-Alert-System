"""
N-HiTS Model Training and Evaluation Pipeline.

This script executes a grid search hyperparameter sweep for the Neural Hierarchical
Interpolation for Time Series (N-HiTS) model. It trains on historical air quality
and meteorological data, evaluates on a chronological validation set with early stopping, 
and finally evaluates the globally best-performing model on an isolated test set.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import os
import random
import shutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Import from refactored modules
from modules.nhits_model import RealNHITS
from modules.dataset import AQDataset

# --- SEEDING ---
def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure complete
    reproducibility of weight initializations and batch shuffling.
    
    Args:
        seed (int): The integer seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # Forces cuDNN to use deterministic algorithms
        torch.backends.cudnn.deterministic = True

def run_trial(lb, lr, hd, drop, data, device, pm_mean, pm_std):
    """
    Trains and validates an N-HiTS model for a specific set of hyperparameters.
    
    This function isolates the training loop, utilizing a chronological split
    (70% train, 15% validation). It includes an early stopping mechanism that halts
    training if the validation RMSE does not improve for a set number of epochs.
    
    Args:
        lb (int): Lookback window size (sequence length).
        lr (float): Learning rate for the optimizer.
        hd (int): Hidden dimension size for the N-HiTS MLP blocks.
        drop (float): Dropout rate for regularization.
        data (np.ndarray): The full array of scaled features.
        device (torch.device): Compute device to run on (CPU or CUDA).
        pm_mean (float): The mean of the target variable (used to unscale predictions).
        pm_std (float): The standard dev of the target variable (used to unscale predictions).

    Returns:
        tuple: (best validation RMSE, best validation residuals array)
    """
    # Initialize the custom PyTorch dataset
    ds = AQDataset(data, lb)
    n = len(ds)
    
    # Chronological splits: First 70% for training, next 15% for validation
    tr_idx, val_idx = int(0.7 * n), int(0.85 * n)
    
    # Dataloaders: Train data can be shuffled, but validation remains strictly ordered
    tr_loader = DataLoader(Subset(ds, range(tr_idx)), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(ds, range(tr_idx, val_idx)), batch_size=128)

    # Initialize model, optimizer, and loss function
    model = RealNHITS(input_dim=11, seq_len=lb, hidden_dim=hd, dropout_rate=drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_v_rmse = float('inf')
    best_v_residuals = None
    patience, counter = 8, 0  # Early stopping constraints

    for epoch in range(50):
        # --- TRAINING STAGE ---
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x).unsqueeze(1), y)
            loss.backward()
            opt.step()

        # --- VALIDATION STAGE ---
        model.eval()
        v_preds, v_acts = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.to(device))
                v_preds.append(out.cpu().numpy())
                v_acts.append(y.numpy())
        
        # Unscale predictions and actuals back to real-world units (ppb/ug/m3)
        p_val = (np.concatenate(v_preds) * pm_std) + pm_mean
        a_val = (np.concatenate(v_acts).flatten() * pm_std) + pm_mean
        v_rmse = np.sqrt(mean_squared_error(a_val, p_val))

        # Check early stopping criteria
        if v_rmse < best_v_rmse:
            best_v_rmse = v_rmse
            best_v_residuals = a_val - p_val # Harvest Estimated residuals
            # Save a temporary checkpoint for the best epoch of this specific trial
            torch.save(model.state_dict(), "temp_best.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience: 
                break # Halt training if patience is exceeded
    
    return best_v_rmse, best_v_residuals

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    print("Loading data for N-HiTS...", flush=True)
    # Ensure this path points correctly to your cleaned data
    df = pd.read_csv("data/data_clean/cleaned_data_toronto_downtown.csv", low_memory=False)
    
    # 1 target variable + 6 weather variables + 4 cyclical time variables = 11 features
    feats = ['PM_ppb', 'Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Stn Press (kPa)', 
             'Dew Point Temp (°C)', 'Precip. Amount (mm)', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']

    # --- STRICT SCALING ---
    # We fit the scaler strictly on the first 70% of chronological data to ensure 
    # no future information leaks into the training stats (mean/std).
    train_size = int(len(df) * 0.7)
    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_size][feats])
    scaled_data = scaler.transform(df[feats])
    
    # Extract PM2.5 mean and std to unscale predictions later
    pm_mean, pm_std = scaler.mean_[0], scaler.scale_[0]

    # --- THE CLEAN SWEEP ---
    # Define hyperparameter grid
    lookbacks = [24, 48, 72, 96]
    hidden_dims = [128, 256, 512]
    learning_rates = [0.001, 0.0005, 0.0001, 0.00005] 
    dropouts = [0.1]

    print("--- STARTING N-HITS SWEEP ---", flush=True)
    results = []
    best_v_global = float('inf')
    winning_residuals = None
    best_cfg = None

    # Nested loops to exhaustively search the hyperparameter space
    for lb in lookbacks:
        for hd in hidden_dims:
            for lr in learning_rates:
                for dr in dropouts:
                    start = time.time()
                    
                    # Run the trial for the current configuration
                    v_rmse, v_res = run_trial(lb, lr, hd, dr, scaled_data, device, pm_mean, pm_std)
                    
                    print(f"LB: {lb} | HD: {hd} | LR: {lr} | VAL RMSE: {v_rmse:.4f} | {time.time()-start:.1f}s", flush=True)
                    results.append({'lb': lb, 'hd': hd, 'lr': lr, 'dr': dr, 'val_rmse': v_rmse})
                    
                    # If this trial is the best we've seen overall, crown it champion
                    if v_rmse < best_v_global:
                        best_v_global = v_rmse
                        winning_residuals = v_res
                        best_cfg = {'lb': lb, 'hd': hd, 'lr': lr, 'dr': dr}
                        # Promote the temporary best to the global champion model file
                        shutil.copy("temp_best.pt", "global_champion_nhits.pt")

    # Save hyperparameter search results for later review
    pd.DataFrame(results).to_csv("nhits_sweep_pure_val.csv", index=False)

    # Calculate the Estimated Standard Error based on the best validation residuals
    estimated_se = np.std(winning_residuals)
    print(f"\nWINNER CHOSEN: {best_cfg} | Estimated SE: {estimated_se:.4f}")

    # ==========================================
    # FINAL EVALUATION (THE BLIND TEST SET)
    # ==========================================
    print("\n--- PERFORMING FINAL TEST ---", flush=True)
    
    # Instantiate the model with the winning hyperparameters and load the saved weights
    champ_model = RealNHITS(input_dim=11, seq_len=best_cfg['lb'], 
                            hidden_dim=best_cfg['hd'], dropout_rate=best_cfg['dr']).to(device)
    champ_model.load_state_dict(torch.load("global_champion_nhits.pt"))
    champ_model.eval()

    # Setup the Test Loader (The final completely isolated 15% block of data)
    final_ds = AQDataset(scaled_data, best_cfg['lb'])
    n_final = len(final_ds)
    test_idx = int(0.85 * n_final)
    test_loader = DataLoader(Subset(final_ds, range(test_idx, n_final)), batch_size=128)

    test_preds, test_acts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            test_preds.append(champ_model(x.to(device)).cpu().numpy())
            test_acts.append(y.numpy())

    # Unscale final test predictions and actuals
    p_final = (np.concatenate(test_preds) * pm_std) + pm_mean
    a_final = (np.concatenate(test_acts).flatten() * pm_std) + pm_mean
    
    # Calculate final RMSE on the isolated test set
    final_rmse = np.sqrt(mean_squared_error(a_final, p_final))
    print(f"FINAL TEST RMSE: {final_rmse:.4f} | Estimated SE: {estimated_se:.4f}")
    
    # --- PLOTTING ---
    plt.figure(figsize=(15, 6))
    
    # Plot actuals vs. predictions for the first 400 points of the test set
    plt.plot(a_final[:400], label="Actual PM2.5", color='black', alpha=0.5)
    plt.plot(p_final[:400], label="N-HiTS Forecast", color='red')
    
    # Add a 95% Confidence Interval band using the Standard Error (1.96 * SE)
    plt.fill_between(range(400), 
                     p_final[:400] - 1.96 * estimated_se, 
                     p_final[:400] + 1.96 * estimated_se, 
                     color='red', alpha=0.15, label="95% CI (Estimated)")
    
    plt.title(f"N-HiTS Test Set Evaluation\nTest RMSE: {final_rmse:.2f} | Estimated SE: {estimated_se:.2f}")
    plt.legend()
    plt.savefig("final_nhits_test.png")
    
    # Save the SE and residuals to disk for future metric comparisons
    np.save("Estimated_residuals.npy", winning_residuals)
    with open("Estimated_se.txt", "w") as f:
        f.write(str(estimated_se))