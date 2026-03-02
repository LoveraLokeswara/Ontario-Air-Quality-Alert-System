import pandas as pd
import requests
from io import StringIO
import os
import time

# --- CONFIGURATION ---
# Format: { StationID : "Custom_Name" }
# Station 48549 is "Toronto City Centre" which reliably provides hourly data.
STATIONS = {
    48549: "Toronto_City_Centre_Downtown", 
    # Add other stations here as needed, ensuring they support hourly recording
}

# Set your "couple of years" range here
START_YEAR = 2018
END_YEAR = 2025

def get_hourly_station_data(station_id, station_name, start_year, end_year):
    base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    frames = []

    print(f"Fetching hourly data for {station_name} (ID: {station_id}) from {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        # Environment Canada restricts hourly bulk downloads to one month at a time
        for month in range(1, 13):
            params = {
                "format": "csv",
                "stationID": station_id,
                "Year": year,
                "Month": month,
                "Day": 1, 
                "timeframe": 1,  # 1 = Hourly Data
                "submit": "Download Data" 
            }

            try:
                # Send the request
                response = requests.get(base_url, params=params)
                response.raise_for_status()

                # Validate that we actually got a CSV back and not an HTML error page
                if "Date/Time" not in response.text[:300] and "LOCAL_DATE" not in response.text[:300]:
                    print(f"  - No hourly data found for {year}-{month:02d}")
                    continue

                # Parse the CSV string into a pandas DataFrame
                csv_data = StringIO(response.content.decode('utf-8'))
                df = pd.read_csv(csv_data)
                
                # Tag the data with the station info for easier merging later
                df['Station ID'] = station_id
                df['Station Name'] = station_name
                
                frames.append(df)
                print(f"  - Successfully downloaded {year}-{month:02d}")
                
                # Pause for 0.5 seconds to avoid hitting rate limits
                time.sleep(0.5)

            except Exception as e:
                print(f"  - Error on {year}-{month:02d}: {e}")

    # Combine all the monthly dataframes into one large dataframe
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()

def collect_weather():
    # --- MAIN EXECUTION ---
    output_folder = "data/weather_data"
    os.makedirs(output_folder, exist_ok=True)

    for stn_id, stn_name in STATIONS.items():
        df = get_hourly_station_data(stn_id, stn_name, START_YEAR, END_YEAR)
        
        if not df.empty:
            # Clean up column names (strip quotes and trailing spaces)
            df.columns = [c.replace('"', '').strip() for c in df.columns]
            
            # Save the compiled multi-year dataset
            filename = f"{output_folder}/{stn_name}_hourly_{START_YEAR}_{END_YEAR}.csv"
            df.to_csv(filename, index=False)
            print(f"\n-> SAVED: {filename}\n")
        else:
            print(f"\n-> SKIPPED: {stn_name} (No hourly data found in this range)\n")

    print("All downloads complete.")

if __name__ == "__main__":
    collect_weather()