# PM₂.₅ Short-Term Forecasting for TTC Air Quality Management

This repository contains the code and data used to forecast short-term PM₂.₅ (fine particulate matter) concentrations in Ontario using meteorological and environmental data. The project evaluates whether modern deep learning time-series models improve forecasting accuracy relative to simpler statistical baselines.
The goal is to support operational decision-making for the Toronto Transit Commission (TTC) by predicting pollution spikes that may impact underground station ventilation systems.
## Project Overview


This analysis investigates how accurately short-term PM₂.₅ concentrations can be predicted by:
- Training a linear baseline model (LASSO) using meteorological predictors and lagged PM₂.₅ values
- Training a deep learning time-series model (N-HiTS) to capture nonlinear temporal dynamics
- Comparing predictive performance to determine whether more complex models provide meaningful improvements
- The forecasting task focuses on short-term horizons (4-hour predictions) using recent historical observations.

---

## Project Structure

```
Ontario-Air-Quality-Alert-System/
├── README.md                          # Project documentation
├── pyproject.toml                     # Project configuration and dependencies
├── uv.lock                            # Lock file for reproducible environments
│
├── src/
│   ├── analysis.py                    # Main analysis script: trains models and generates results
│   ├── collect_air_quality.py         # Script to fetch raw PM2.5 data
│   ├── collect_weather.py             # Script to fetch meteorological data
│   ├── data_clean.py                  # Script for cleaning and merging datasets
│   └── forecasting_models.py          # Model definitions (LASSO, N-HiTS)
│
├── data/
│   ├── air_quality/                   # Raw PM2.5 monitoring station data
│   ├── weather/                       # Raw meteorological data
│   └── data_clean/                    # Final processed datasets for modeling
│
├── models/                            # Serialized trained models
├── results/                           # Forecast plots, RMSE tables, and evaluation metrics
└── presentation/                      # Source files for the technical presentation
```

## How to Run the Analysis

This project uses **[uv](https://github.com/astral-sh/uv)** for efficient Python package management.

### 1. Setup & Installation
Clone the repository and set up the environment:
```bash
git clone https://github.com/LoveraLokeswara/Traffic-and-Air-Quality-Analysis
cd Traffic-and-Air-Quality-Analysis
uv sync
```

### 2. Data Processing
To regenerate the cleaned datasets from raw sources, run:
```bash
uv run python src/data_clean.py
```

### 3. Model Training and Analysis
To train the models and generate all plots in the `results/` folder, execute:
```bash
uv run python src/analysis.py
```
## Datasets Overview

The project integrates two main datasets:

**Air Quality Data**
Hourly PM₂.₅ concentrations collected from Ontario environmental monitoring stations.
Variables include:
* Timestamp
* PM₂.₅ concentration (µg/m³)
* Station identifier
  
**Meteorological Data**
Hourly weather measurements used as predictors, including:
* Temperature
* Wind speed
* Precipitation
* Humidity
* Air pressure
* Dew point temperature
These variables influence pollutant dispersion and accumulation.

---
## Models
Two forecasting approaches are implemented.

1. LASSO Regression (Baseline)
- A linear regression model with L1 regularization used as a strong interpretable baseline.

Key characteristics:
- Handles multicollinearity
- Performs automatic feature selection
- Captures linear relationships between weather variables and PM₂.₅


2. N-HiTS (Neural Hierarchical Interpolation for Time Series)
- A deep learning architecture designed for long-horizon time-series forecasting.

Advantages include:
- Capturing nonlinear temporal dynamics
- Modelling multi-scale temporal patterns
- Handling complex interactions between meteorology and pollution levels

 
# Evaluation Metrics
Models are evaluated using:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)





