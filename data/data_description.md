# Data Description

This document summarizes the source datasets used in the project and describes the cleaned data format used for each station.

## Overview

The project integrates three original data sources:
- Weather
- Air quality

These were cleaned, aligned by date and station, and merged into per-station CSV files in `data/data_clean/` (one CSV per station). Each cleaned CSV contains daily observations for the station.

---

## Original datasets

### Weather
- Contents: meteorological observations such as temperature, precipitation, wind speed/gust and wind direction, etc.
- Variables: PM_ppb, Temp (°C), Rel Hum (%), Wind Spd (km/h), Stn Press (kPa), Dew Point Temp (°C), Precip. Amount (mm), hour_sin, hour_cos, month_sin, and month_cos.
- Link to dataset: https://climate.weather.gc.ca

### Air quality
- Contents: pollutant measurements (e.g., NO₂) recorded at monitoring stations.
- Variables: `Date`,`Hour`, `Station`, pollutant concentrations (e.g., `NO2_Mean`).
- Link to dataset: https://www150.statcan.gc.ca/n1/pub/71-607-x/71-607-x2022018-eng.htm


---

## Cleaned data (per station)

The cleaned CSVs are stored in `data/data_clean/`. Each file contains daily rows for a single station. Below is the column reference (Data name | Type | Description) describing the fields present in any cleaned station file.

| Data name             | Type  | Description                                                                                                        |
| --------------------- | ----- | ------------------------------------------------------------------------------------------------------------------ |
| `PM_ppb`              | float | Hourly particulate matter concentration measured at the monitoring station (ppb: parts per billion).               |
| `Temp (°C)`           | float | Hourly air temperature in degrees Celsius.                                                                         |
| `Rel Hum (%)`         | float | Relative humidity expressed as a percentage (0–100%), indicating moisture in the air.                              |
| `Wind Spd (km/h)`     | float | Wind speed measured in kilometers per hour.                                                                        |
| `Stn Press (kPa)`     | float | Atmospheric pressure measured at the station in kilopascals (kPa).                                                 |
| `Dew Point Temp (°C)` | float | Dew point temperature in degrees Celsius — the temperature at which air becomes saturated and condensation begins. |
| `Precip. Amount (mm)` | float | Precipitation amount during the observation period measured in millimeters.                                        |
| `hour_sin`            | float | Sine transformation of the hour of day used to capture cyclical daily patterns in time-series models.              |
| `hour_cos`            | float | Cosine transformation of the hour of day used with `hour_sin` to represent the cyclical nature of time.            |
| `month_sin`           | float | Sine transformation of the month used to capture cyclical seasonal patterns.                                       |
| `month_cos`           | float | Cosine transformation of the month used with `month_sin` to represent annual seasonality.                          |
