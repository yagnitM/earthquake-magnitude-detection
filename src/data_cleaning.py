import numpy as np
import pandas as pd
import os

RAW_DATA_PATH = '../data/raw/usa_earthquake_data.csv'
PROCESSED_DATA_PATH = '../data/processed/usa_earthquake_cleaned.csv'

def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"Loaded data with shape: {df.shape}")
        df.replace("", np.nan, inplace=True)

        columns_to_fill = ['gap', 'horizontalError', 'magError']
        for col in columns_to_fill:
            if col in df.columns:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled missing value in {col} with median: {median_val:.4f}")
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df['date'] = df['time'].dt.date
        df['time_only'] = df['time'].dt.time
        print(f"Extracted 'date' and 'time_only' (UTC).")

        columns_to_drop = ['nst', 'dmin', 'magNst', 'time', 'id']
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")

        return df
    except FileNotFoundError:
        print(f"File not found at {path}")
        return None


if __name__ == "__main__":
    print("Starting earthquake data processing...")
    df = load_data(RAW_DATA_PATH)
    
    if df is not None:
        print("Data loaded and cleaned successfully!")
        print("\nFirst few rows of processed data:")
        print(df.head())

        try:
            df.to_csv(PROCESSED_DATA_PATH, index=False)
            print(f"\nCleaned data saved at: {PROCESSED_DATA_PATH}")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        print("Some error occurred during loading.")    