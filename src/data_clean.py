import os
import pandas as pd
import numpy as np

def prep_and_merge(aq_filepath, weather_filepath):
    print(f"Loading Air Quality Data from {aq_filepath}...")
    
    # Load data (skipping the top 10 metadata rows)
    aq_df = pd.read_csv(aq_filepath, skiprows=10, index_col=False)
    
    hour_cols = [f'H{str(i).zfill(2)}' for i in range(1, 25)]
    
    # Melt the wide data into long format
    aq_long = pd.melt(
        aq_df, 
        id_vars=['Station ID', 'Pollutant', 'Date'], 
        value_vars=hour_cols,
        var_name='Hour_Col', 
        value_name='PM_raw' 
    )
    
    # --- UPDATED PM CLEANING LOGIC ---
    # Convert to numeric first (errors='coerce' handles stray text/flags)
    aq_long['PM_ppb'] = pd.to_numeric(aq_long['PM_raw'], errors='coerce')
    # Filter non-physical placeholders and outliers
    aq_long.loc[(aq_long['PM_ppb'] > 500) | (aq_long['PM_ppb'] < 0), 'PM_ppb'] = np.nan
    # ---------------------------------

    aq_long['Hour_Int'] = aq_long['Hour_Col'].str.replace('H', '').astype(int) - 1
    
    # 1. Convert Date column to Datetime
    aq_long['Clean_Date'] = pd.to_datetime(aq_long['Date'], errors='coerce')
    
    # 2. Drop the footer rows
    aq_long = aq_long.dropna(subset=['Clean_Date'])
    
    # 3. Time math
    aq_long['Datetime'] = aq_long['Clean_Date'] + pd.to_timedelta(aq_long['Hour_Int'], unit='h')
    
    aq_clean = aq_long[['Datetime', 'Station ID', 'PM_ppb']].rename(columns={'Station ID': 'AQ_Station_ID'})

    print(f"Loading Weather Data from {weather_filepath}...")
    weather_df = pd.read_csv(weather_filepath, low_memory=False)
    weather_df['Datetime'] = pd.to_datetime(weather_df['Date/Time (LST)'], errors='coerce')

    print("Merging datasets on Datetime...")
    merged_df = pd.merge(aq_clean, weather_df, on='Datetime', how='inner')
    merged_df = merged_df.sort_values('Datetime').reset_index(drop=True)
    
    # Forward fill PM2.5 
    merged_df['PM_ppb'] = merged_df['PM_ppb'].ffill().bfill()

    # Forward fill weather conditions
    merged_df['Weather'] = merged_df['Weather'].ffill().bfill()

    # Forward fill and sanitize weather numerical columns
    numerical_weather_cols = ['Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Stn Press (kPa)', 'Dew Point Temp (°C)']
    for col in numerical_weather_cols:
        # Force numeric and remove placeholders (9999, etc.) for weather too
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        if 'Press' in col:
            merged_df.loc[(merged_df[col] > 110) | (merged_df[col] < 85), col] = np.nan
        else:
            merged_df.loc[(merged_df[col] > 1000) | (merged_df[col] < -1000), col] = np.nan
            
        merged_df[col] = merged_df[col].ffill().bfill()
    
    print(f"Successfully merged! Output contains {len(merged_df)} rows.")
    return merged_df

# Paths
aq_file = "data/air_quality/aq_data/station_31129_data.csv"
weather_file = "data/weather_data/Toronto_City_Centre_Downtown_hourly_2018_2025.csv" 
output_path = "data/data_clean/cleaned_data_toronto_downtown.csv"

# Execution
final_merged_data = prep_and_merge(aq_file, weather_file)
os.makedirs(os.path.dirname(output_path), exist_ok=True) 
final_merged_data.to_csv(output_path, index=False)

# Verification prints for your peace of mind
print(f"Final Stats - Mean: {final_merged_data.PM_ppb.mean():.2f}, Std: {final_merged_data.PM_ppb.std():.2f}")
print(f"Data successfully saved to: {output_path}")