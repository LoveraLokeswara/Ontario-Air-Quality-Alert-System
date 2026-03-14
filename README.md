# PM₂.₅ Short-Term Forecasting for TTC Air Quality Management

This repository contains the code and data used to forecast short-term PM₂.₅ (fine particulate matter) concentrations in Ontario using meteorological and environmental data. The project evaluates whether modern deep learning time-series models improve forecasting accuracy relative to simpler statistical baselines.

The goal is to support operational decision-making for the Toronto Transit Commission (TTC) by predicting pollution spikes that may impact underground station ventilation systems.

## Project Overview

This analysis investigates how accurately short-term PM₂.₅ concentrations can be predicted by:
- Training a deep learning time-series model (**N-HiTS**) to capture nonlinear temporal dynamics.
- Comparing predictive performance against a baseline (**LASSO**) to determine whether more complex models provide meaningful improvements.
- The forecasting task focuses on short-term horizons (specifically the **4th-hour prediction**) using recent historical observations.

---

## Project Structure

```text
Ontario-Air-Quality-Alert-System/
├── pyproject.toml                     # Project configuration and dependencies
├── uv.lock                            # Lock file for reproducible environments
│
├── src/
│   ├── collect_air_quality.py         # Script to fetch raw PM2.5 data from Air Quality Ontario
│   ├── collect_weather.py             # Script to fetch meteorological data from Environment Canada
│   ├── data_clean.py                  # Script for cleaning, merging, and feature engineering
│
├── modules/                           # Core model and data handling components
│   ├── dataset.py                     # Custom PyTorch Dataset (AQDataset) for sliding windows
│   ├── nhits_model.py                 # N-HiTS architecture implementation (RealNHITS)
│   └── module_info.md                 # Detailed documentation for the modules
│
├── data/
│   ├── air_quality/                   # Raw PM2.5 monitoring station data and station info
│   ├── weather_data/                  # Raw meteorological data from Environment Canada
│   └── data_clean/                    # Final processed datasets for modeling
│
├── model/                             # Serialized trained models (e.g., global_champion_nhits.pt)
├── result/                            # Forecast plots, presentation files, and evaluation metrics
│   ├── images/                        # Architecture diagrams and forecast plots
│   ├── sections/                      # Quarto (.qmd) source files for the presentation
│   └── presentation.html              # Rendered technical presentation
│
├── train_eval_nhits.py                # Main pipeline for N-HiTS training and hyperparameter sweep
└── model_comparison.ipynb             # Interactive notebook for comparing LASSO vs. N-HiTS performance
```

---

## Model Comparison & Evaluation (`model_comparison.ipynb`)

The `model_comparison.ipynb` notebook provides a comprehensive evaluation of the forecasting models. It serves as the primary tool for comparing the deep learning approach against traditional statistical baselines.

### Key Features of the Comparison:
1. **LASSO Baseline Training**:
   - Implements a **LASSO** regression model using `scikit-learn`.
   - Uses `LassoCV` with `TimeSeriesSplit` (5-fold) to automatically tune the regularization strength.
   - The model is trained on flattened lookback windows of all 11 features to capture linear relationships.

2. **N-HiTS Inference**:
   - Loads the pre-trained "global champion" **N-HiTS** model (`model/global_champion_nhits.pt`).
   - Performs inference on the same test set used for the LASSO baseline to ensure a fair comparison.

3. **Performance Metrics**:
   - Calculates **RMSE** (Root Mean Squared Error) and **MAE** (Mean Absolute Error) for both models.
   - Provides a direct comparison of how much the non-linear deep learning model improves upon the linear baseline.

4. **Uncertainty Estimation**:
   - Calculates validation residuals to estimate a **95% Confidence Interval** (Margin of Error) for the N-HiTS predictions.
   - This provides operational context for the reliability of the forecasts.

5. **Visualization**:
   - Generates high-resolution plots comparing **Actual vs. LASSO vs. N-HiTS** forecasts over a 300-hour window.
   - Includes the 95% CI band to visualize prediction uncertainty.

---

## How to Run the Analysis

This project uses **[uv](https://github.com/astral-sh/uv)** for efficient Python package management.

### 1. Setup & Installation
Clone the repository and set up the environment:
```bash
# Clone the repository
git clone https://github.com/LoveraLokeswara/Traffic-and-Air-Quality-Analysis
cd Traffic-and-Air-Quality-Analysis

# Install dependencies and create virtual environment
uv sync
```

### 2. Data Collection (Optional)
To fetch the latest data from official sources:
```bash
# Collect Air Quality data
uv run python src/collect_air_quality.py

# Collect Weather data
uv run python src/collect_weather.py
```

### 3. Data Processing
To generate the cleaned datasets from raw sources, run:
```bash
uv run python src/data_clean.py
```
This script merges air quality and weather data, handles missing values (forward fill up to 2 hours), and generates cyclical time features (sine/cosine of hour and month).

### 4. Model Training and Evaluation
To execute the N-HiTS training pipeline, including hyperparameter sweep and final test evaluation:
```bash
uv run python train_eval_nhits.py
```
This will:
- Perform a grid search over lookback windows, hidden dimensions, and learning rates.
- Save the best model as `global_champion_nhits.pt`.
- Generate evaluation plots (e.g., `final_nhits_test.png`).

---

## Datasets & Features

### Air Quality Data
Hourly PM₂.₅ concentrations collected from Ontario environmental monitoring stations (e.g., Toronto Downtown).
- **Target:** `PM_ppb` (PM₂.₅ concentration).

### Meteorological Data
Hourly weather measurements from Environment Canada (Toronto City Centre):
- Temperature, Relative Humidity, Wind Speed, Station Pressure, Dew Point, and Precipitation.

### Feature Engineering
- **Cyclical Time Features:** `hour_sin`, `hour_cos`, `month_sin`, `month_cos` to capture diurnal and seasonal patterns.
- **Lagged Observations:** Handled via the sliding window in `AQDataset`.

---

## Modeling Approach: N-HiTS

The project implements the **Neural Hierarchical Interpolation for Time Series (N-HiTS)** architecture, which is designed to capture multi-scale temporal patterns.

### Key Features:
- **Hierarchical Pooling:** Uses multiple blocks with different pooling rates to capture both long-term trends and short-term shocks.
- **Residual Connections:** Each block learns to predict the residual error of the previous blocks (backcasting).
- **Multi-rate Signal Decomposition:** Effectively handles the high-frequency nature of air quality data alongside smoother meteorological trends.

---

## Evaluation Results
Models are evaluated on a strictly chronological test set (final 15% of data) using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Visual Inspection:** Forecast vs. Actual plots with 95% confidence intervals.

Detailed results and architecture diagrams can be found in the `result/` folder and the `presentation.html`.
