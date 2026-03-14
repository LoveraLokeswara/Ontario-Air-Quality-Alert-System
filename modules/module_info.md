# Air Quality Forecasting Module (N-HiTS)

This module contains the PyTorch implementation of the Neural Hierarchical Interpolation for Time Series (N-HiTS) model and the custom data loader designed for the Ontario Air Quality Alert System project. 

It is specifically tailored for multi-variate time-series forecasting, predicting future PM2.5 concentrations based on historical meteorological and air quality data.

## Module Structure

- `__init__.py`: Initializes the module and exposes the core classes.
- `dataset.py`: Contains the `AQDataset` class for transforming tabular time-series data into PyTorch tensors.
- `nhits_model.py`: Contains the deep learning architecture, including the base `NHITSBlock` and the primary `RealNHITS` model.

---

## Components Overview

### 1. `AQDataset` (`dataset.py`)
A custom PyTorch `Dataset` designed to handle sliding window time-series generation.

**Key Features:**
- **Lookback Window (`seq_len`):** Extracts a contiguous sequence of historical data to serve as the input tensor `x`.
- **Target Horizon (`horizon`):** Automatically extracts the specific future target value.
- **Target Assumption:** Assumes that the variable to be predicted (PM2.5) is located at index `0` of the feature array.

### 2. `RealNHITS` & `NHITSBlock` (`nhits_model.py`)
A custom implementation of the N-HiTS architecture, designed to capture both slow-moving climate trends and sudden, non-linear weather shocks using multi-rate signal decomposition.

**Architecture Highlights:**
- **Hierarchical Pooling:** The model utilizes a stack of 3 `NHITSBlock` layers with decreasing MaxPool kernel sizes (`4`, `2`, `1`). 
  - *Block 1 (Pool 4):* Captures long-term, low-frequency atmospheric trends.
  - *Block 2 (Pool 2):* Captures mid-range temporal patterns.
  - *Block 3 (Pool 1):* Processes raw, unpooled data to capture sudden high-frequency weather shocks.
- **Residual Connections:** After each block, the model generates a "backcast" which is subtracted from the target signal. This forces subsequent blocks to focus strictly on the residual errors (the patterns the previous blocks missed).
- **Interpolated Forecast:** Uses `F.interpolate` to smoothly project the learned hidden representations into the future horizon.
- **Specific Output:** The `forward` pass aggregates the forecasts from all blocks and specifically returns the prediction for the 4th hour (`[:, 3]`).
