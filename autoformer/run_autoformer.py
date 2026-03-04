"""
Autoformer PM2.5 Forecasting – Toronto Downtown
================================================
Trains the Autoformer model on the cleaned air-quality dataset and evaluates
it on a held-out validation set.

Run from the project root:
    python autoformer/run_autoformer.py

Output
------
  autoformer/saved_models/autoformer_pm25_best.pth   – best checkpoint
  results/autoformer_pm25_forecast.png               – 200-step forecast plot
  results/autoformer_error_by_horizon.png            – RMSE/MAE per horizon
  results/autoformer_training_loss.png               – loss curves
"""

import sys
import os

# Make sure 'autoformer/' is on the path so relative layer/model imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from models.Autoformer import Model


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class Configs:
    # Sequence lengths
    seq_len   = 96   # look-back window  (4 days × 24 h)
    label_len = 48   # decoder start token  (2 days × 24 h)
    pred_len  = 24   # forecast horizon  (next 24 hours)

    # Feature dimensions
    enc_in = 4   # encoder input features
    dec_in = 4   # decoder input features (same feature set)
    c_out  = 4   # output features (all 4; PM2.5 extracted at eval time)

    # Model size
    d_model = 64    # embedding / hidden dimension
    n_heads = 4     # Auto-Correlation heads
    e_layers = 2    # encoder layers
    d_layers = 1    # decoder layers
    d_ff    = 256   # point-wise FFN inner dimension

    # Decomposition & Auto-Correlation
    moving_avg = 25  # moving-average kernel (must be odd)
    factor     = 3   # top-k = factor × log(L) lags kept per head

    # Training
    dropout    = 0.05
    embed      = 'timeF'
    freq       = 'h'
    activation = 'gelu'

    output_attention = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_time_features(datetime_series: pd.Series) -> np.ndarray:
    """
    Returns normalised time features with shape [N, 4] in [-0.5, 0.5]:
      col-0  HourOfDay   hour / 23  - 0.5
      col-1  DayOfWeek   dayofweek / 6 - 0.5
      col-2  DayOfMonth  (day - 1) / 30 - 0.5
      col-3  DayOfYear   (dayofyear - 1) / 365 - 0.5
    Matches the 4-dimensional input expected by TimeFeatureEmbedding(freq='h').
    """
    dt = pd.DatetimeIndex(datetime_series)
    return np.column_stack([
        dt.hour / 23.0 - 0.5,
        dt.dayofweek / 6.0 - 0.5,
        (dt.day - 1) / 30.0 - 0.5,
        (dt.dayofyear - 1) / 365.0 - 0.5,
    ]).astype(np.float32)


class AutoformerDataset(Dataset):
    """
    Sliding-window dataset for the Autoformer encoder-decoder architecture.

    One sample:
      x_enc      [seq_len,              n_feat]   – encoder input
      x_mark_enc [seq_len,              4     ]   – encoder time features
      x_dec      [label_len + pred_len, n_feat]   – decoder input (shape only used by model)
      x_mark_dec [label_len + pred_len, 4     ]   – decoder time features
      y          [pred_len,             n_feat]   – ground truth
    """

    def __init__(self, data: np.ndarray, time_marks: np.ndarray,
                 seq_len: int, label_len: int, pred_len: int):
        self.data = data
        self.time_marks = time_marks
        self.seq_len   = seq_len
        self.label_len = label_len
        self.pred_len  = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end   = s_begin + self.seq_len
        r_begin = s_end   - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        x_enc      = self.data[s_begin:s_end]
        x_mark_enc = self.time_marks[s_begin:s_end]
        x_dec      = self.data[r_begin:r_end]
        x_mark_dec = self.time_marks[r_begin:r_end]
        y          = self.data[s_end:s_end + self.pred_len]

        return (
            torch.FloatTensor(x_enc),
            torch.FloatTensor(x_mark_enc),
            torch.FloatTensor(x_dec),
            torch.FloatTensor(x_mark_dec),
            torch.FloatTensor(y),
        )


def inverse_scale_pm25(arr_1d: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse-scale a 1-D array of PM2.5 (column 0) values."""
    dummy = np.zeros((len(arr_1d), 4), dtype=np.float32)
    dummy[:, 0] = arr_1d
    return scaler.inverse_transform(dummy)[:, 0]


# ─────────────────────────────────────────────────────────────────────────────
# Main training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_autoformer():
    global_start = time.time()
    configs = Configs()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    data_path   = os.path.join(base_dir, '..', 'data', 'data_clean',
                               'cleaned_data_toronto_downtown.csv')
    save_dir    = os.path.join(base_dir, 'saved_models')
    results_dir = os.path.join(base_dir, '..', 'results')
    os.makedirs(save_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ── Load & preprocess ────────────────────────────────────────────────────
    print("Loading data …", flush=True)
    df = pd.read_csv(data_path, low_memory=False)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

    features = ['PM_ppb', 'Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)']
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df[features] = df[features].ffill().bfill()
    df = df.dropna(subset=features).reset_index(drop=True)

    print(f"  Rows after cleaning: {len(df):,}", flush=True)
    print(f"  Date range: {df['Datetime'].min()} → {df['Datetime'].max()}", flush=True)

    scaler     = StandardScaler()
    data       = scaler.fit_transform(df[features].values).astype(np.float32)
    time_marks = get_time_features(df['Datetime'])

    # ── Build datasets ───────────────────────────────────────────────────────
    full_dataset = AutoformerDataset(
        data, time_marks,
        configs.seq_len, configs.label_len, configs.pred_len
    )
    n_total  = len(full_dataset)
    n_train  = int(0.8 * n_total)
    n_val    = n_total - n_train

    train_ds = Subset(full_dataset, list(range(n_train)))
    val_ds   = Subset(full_dataset, list(range(n_train, n_total)))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, drop_last=False, num_workers=0)

    print(f"  Train samples: {n_train:,} | Val samples: {n_val:,}", flush=True)

    # ── Build model ──────────────────────────────────────────────────────────
    model       = Model(configs).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nAutoformer parameters: {param_count:,}", flush=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )

    # ── Training loop ────────────────────────────────────────────────────────
    epochs        = 10
    best_val_loss = math.inf
    train_losses, val_losses = [], []

    print(f"\n{'─'*60}", flush=True)
    print(f"  AUTOFORMER TRAINING  |  device={device}", flush=True)
    print(f"  seq={configs.seq_len}  label={configs.label_len}  pred={configs.pred_len}", flush=True)
    print(f"  d_model={configs.d_model}  heads={configs.n_heads}  "
          f"enc_layers={configs.e_layers}  dec_layers={configs.d_layers}", flush=True)
    print(f"{'─'*60}", flush=True)

    for epoch in range(epochs):
        t0 = time.time()

        # Train
        model.train()
        running_loss = 0.0
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in train_loader:
            x_enc      = x_enc.to(device)
            x_mark_enc = x_mark_enc.to(device)
            x_dec      = x_dec.to(device)
            x_mark_dec = x_mark_dec.to(device)
            y          = y.to(device)

            optimizer.zero_grad()
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)

        # Validate
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for x_enc, x_mark_enc, x_dec, x_mark_dec, y in val_loader:
                out = model(
                    x_enc.to(device), x_mark_enc.to(device),
                    x_dec.to(device), x_mark_dec.to(device)
                )
                running_val += criterion(out, y.to(device)).item()

        avg_val = running_val / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        scheduler.step(avg_val)
        elapsed = time.time() - t0

        saved_tag = ''
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model_path = os.path.join(save_dir, 'autoformer_pm25_best.pth')
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             best_val_loss,
                'scaler_mean':          scaler.mean_,
                'scaler_scale':         scaler.scale_,
                'configs': {
                    'seq_len':   configs.seq_len,
                    'label_len': configs.label_len,
                    'pred_len':  configs.pred_len,
                    'enc_in':    configs.enc_in,
                    'dec_in':    configs.dec_in,
                    'c_out':     configs.c_out,
                    'd_model':   configs.d_model,
                    'n_heads':   configs.n_heads,
                    'e_layers':  configs.e_layers,
                    'd_layers':  configs.d_layers,
                    'd_ff':      configs.d_ff,
                    'moving_avg': configs.moving_avg,
                    'factor':    configs.factor,
                    'embed':     configs.embed,
                    'freq':      configs.freq,
                    'activation': configs.activation,
                },
            }, model_path)
            saved_tag = '  ✓ saved'

        print(
            f"  Epoch {epoch+1:02d}/{epochs}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}  "
            f"time={elapsed:.1f}s{saved_tag}"
        )

    # ── Load best checkpoint and evaluate ────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("  EVALUATION  (best checkpoint)", flush=True)
    print(f"{'─'*60}", flush=True)

    checkpoint = torch.load(
        os.path.join(save_dir, 'autoformer_pm25_best.pth'),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds, all_actuals = [], []
    with torch.no_grad():
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in val_loader:
            out = model(
                x_enc.to(device), x_mark_enc.to(device),
                x_dec.to(device), x_mark_dec.to(device)
            )
            all_preds.append(out.cpu().numpy())
            all_actuals.append(y.numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # [N, pred_len, 4]
    actuals = np.concatenate(all_actuals, axis=0)   # [N, pred_len, 4]

    # Evaluate PM2.5 (channel 0) at every forecast horizon
    rmse_per_h = []
    mae_per_h  = []
    for h in range(configs.pred_len):
        p = inverse_scale_pm25(preds[:, h, 0],   scaler)
        a = inverse_scale_pm25(actuals[:, h, 0], scaler)
        rmse_per_h.append(math.sqrt(mean_squared_error(a, p)))
        mae_per_h.append(mean_absolute_error(a, p))

    print(f"\n  {'Horizon':>9} | {'RMSE (ppb)':>10} | {'MAE (ppb)':>10}", flush=True)
    print(f"  {'-'*36}", flush=True)
    for h in [0, 5, 11, 23]:
        print(f"  H+{h+1:02d}      | {rmse_per_h[h]:>10.3f} | {mae_per_h[h]:>10.3f}", flush=True)
    print(f"  {'-'*36}", flush=True)
    print(f"  Average    | {np.mean(rmse_per_h):>10.3f} | {np.mean(mae_per_h):>10.3f}", flush=True)

    # ── Plots ─────────────────────────────────────────────────────────────────

    # 1. Forecast vs Actual – first 200 validation samples at H+1
    h_plot = 0
    p_h = inverse_scale_pm25(preds[:200,   h_plot, 0], scaler)
    a_h = inverse_scale_pm25(actuals[:200, h_plot, 0], scaler)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(a_h, label='Actual PM2.5',        color='steelblue', alpha=0.85)
    ax.plot(p_h, label='Predicted PM2.5 (H+1)', color='tomato', linestyle='--', alpha=0.9)
    ax.set_title('Autoformer – PM2.5 Next-Hour Forecast  |  Toronto Downtown',
                 fontsize=13)
    ax.set_xlabel('Validation Time Steps (Hours)', fontsize=11)
    ax.set_ylabel('PM2.5 Concentration (ppb)',     fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'autoformer_pm25_forecast.png'), dpi=150)
    plt.close(fig)

    # 2. Error vs Forecast Horizon
    fig, ax = plt.subplots(figsize=(10, 5))
    horizons = list(range(1, configs.pred_len + 1))
    ax.plot(horizons, rmse_per_h, 'o-', color='steelblue', label='RMSE')
    ax.plot(horizons, mae_per_h,  's-', color='tomato',    label='MAE')
    ax.set_title('Autoformer – Error vs Forecast Horizon', fontsize=13)
    ax.set_xlabel('Hours Ahead', fontsize=11)
    ax.set_ylabel('Error (ppb)', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'autoformer_error_by_horizon.png'), dpi=150)
    plt.close(fig)

    # 3. Training loss curves
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, epochs + 1), train_losses, 'o-', color='steelblue', label='Train')
    ax.plot(range(1, epochs + 1), val_losses,   's-', color='tomato',    label='Val')
    ax.set_title('Autoformer – Training Loss', fontsize=13)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE Loss (scaled)', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'autoformer_training_loss.png'), dpi=150)
    plt.close(fig)

    total_time = time.time() - global_start
    print(f"\n  Total runtime : {total_time:.1f}s", flush=True)
    print(f"  Model saved   : {os.path.join(save_dir, 'autoformer_pm25_best.pth')}", flush=True)
    print(f"  Plots saved   : {results_dir}/", flush=True)


if __name__ == '__main__':
    run_autoformer()
