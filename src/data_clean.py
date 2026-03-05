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
        value_name='PM_ppb' # Changed to PM_ppb
    )
    
    # Clean the PM values (Fixed KeyReference here)
    aq_long['PM_ppb'] = pd.to_numeric(aq_long['PM_ppb'].replace([9999, 999, -999], np.nan), errors='coerce')
    aq_long['Hour_Int'] = aq_long['Hour_Col'].str.replace('H', '').astype(int) - 1
    
    # 1. Convert Date column to Datetime, forcing footer text to become 'NaT'
    aq_long['Clean_Date'] = pd.to_datetime(aq_long['Date'], errors='coerce')
    
    # 2. Drop the footer rows (where the date is NaT)
    aq_long = aq_long.dropna(subset=['Clean_Date'])
    
    # 3. Safely perform the time math now that the garbage text is gone
    aq_long['Datetime'] = aq_long['Clean_Date'] + pd.to_timedelta(aq_long['Hour_Int'], unit='h')
    # -----------------------
    
    # Keep only what we need (Fixed KeyReference here too)
    aq_clean = aq_long[['Datetime', 'Station ID', 'PM_ppb']].rename(columns={'Station ID': 'AQ_Station_ID'})

    print(f"Loading Weather Data from {weather_filepath}...")
    weather_df = pd.read_csv(weather_filepath, low_memory=False)
    weather_df['Datetime'] = pd.to_datetime(weather_df['Date/Time (LST)'])

    print("Merging datasets on Datetime...")
    merged_df = pd.merge(aq_clean, weather_df, on='Datetime', how='inner')
    merged_df = merged_df.sort_values('Datetime').reset_index(drop=True)
    
    # --- EDITED SECTION: PM2.5 Forward Fill (up to 3 hours), Drop, and Print ---
    initial_rows = len(merged_df)
    
    # Forward fill PM2.5 for gaps up to 3 hours, then drop the remaining empty rows
    merged_df['PM_ppb'] = merged_df['PM_ppb'].ffill(limit=4)
    merged_df = merged_df.dropna(subset=['PM_ppb'])
    
    # --- NEW: Cyclical Time Features ---
    # This gives the models a "clock" to understand diurnal patterns, preventing 24h flatlining
    merged_df['hour'] = merged_df['Datetime'].dt.hour
    merged_df['month'] = merged_df['Datetime'].dt.month

    merged_df['hour_sin'] = np.sin(2 * np.pi * merged_df['hour'] / 24.0)
    merged_df['hour_cos'] = np.cos(2 * np.pi * merged_df['hour'] / 24.0)
    merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12.0)
    merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12.0)
    
    # Calculate and print the dropped rows
    dropped_rows = initial_rows - len(merged_df)
    print(f"\n--- GAP FILTERING DIAGNOSTICS ---")
    print(f"Total merged rows before dropping: {initial_rows}")
    print(f"Rows dropped due to missing PM2.5 (>4h gap): {dropped_rows} ({(dropped_rows/initial_rows)*100:.2f}%)")
    print(f"Remaining clean rows: {len(merged_df)}")
    print(f"---------------------------------\n")
    # ---------------------------------------------------------------------------

    # Forward fill weather conditions (categorical)
    merged_df['Weather'] = merged_df['Weather'].ffill()

    # Forward fill remaining weather numerical columns
    numerical_weather_cols = ['Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Stn Press (kPa)', 'Dew Point Temp (°C)', 'Precip. Amount (mm)']
    merged_df[numerical_weather_cols] = merged_df[numerical_weather_cols].ffill()
    
    print(f"Successfully merged! Output contains {len(merged_df)} rows.")
    return merged_df

# Specify data paths 
aq_file = "data/air_quality/aq_data/station_31129_data.csv"
weather_file = "data/weather_data/Toronto_City_Centre_Downtown_hourly_2018_2025.csv" 

final_merged_data = prep_and_merge(aq_file, weather_file)

# --- NEW CODE TO SAVE THE FILE AND CREATE FOLDER ---
output_path = "data/data_clean/cleaned_data_toronto_downtown.csv"

# 1. Get just the directory part of the path ("data/data_clean")
output_dir = os.path.dirname(output_path)

# 2. Safely create the directory if it does not exist
# exist_ok=True prevents an error if you run the script a second time
os.makedirs(output_dir, exist_ok=True) 

# 3. Save to CSV
final_merged_data.to_csv(output_path, index=False)

print(f"Data successfully saved to: {output_path}")