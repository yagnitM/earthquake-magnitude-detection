import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

PROCESSED_DATA_PATH = '../data/processed/usa_earthquake_cleaned.csv'
FEATURES_DATA_PATH = '../data/processed/usa_earthquake_features.csv'

def load_data():
    print("Loading cleaned data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Data loaded: {df.shape}")
    return df

def create_simple_features(df):
    print("Creating simple features...")
    
    df = df.copy()
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = pd.to_numeric(df['time_only'].str.split(':').str[0], errors='coerce').fillna(0)
    
    print("Created temporal features")
    
    usa_center_lat = 39.8283
    usa_center_lon = -98.5795
    df['distance_from_center'] = np.sqrt(
        (df['latitude'] - usa_center_lat)**2 + 
        (df['longitude'] - usa_center_lon)**2
    )
    
    df['region'] = 'east'
    df.loc[df['longitude'] < -90, 'region'] = 'central'
    df.loc[df['longitude'] < -120, 'region'] = 'west'
    
    print("Created geographic features")
    
    df['mag_error_ratio'] = df['magError'] / (df['mag'] + 0.01)
    df['depth_error_ratio'] = df['depthError'] / (df['depth'] + 0.01)
    
    df['is_shallow'] = (df['depth'] < 10).astype(int)
    
    print("Created measurement features")
    
    label_encoders = {}
    categorical_cols = ['magType', 'net', 'type', 'status', 'region']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded {col}")
    
    print(f"Created {len(categorical_cols)} encoded features")
    
    return df

def select_final_features(df):
    print("\nSelecting final features...")
    
    feature_columns = [
        'latitude', 'longitude', 'depth', 'mag', 'gap', 'rms',
        'horizontalError', 'depthError', 'magError',
        'year', 'month', 'day', 'hour',
        'distance_from_center', 'is_shallow',
        'mag_error_ratio', 'depth_error_ratio',
        'magtype_encoded', 'net_encoded', 'type_encoded', 
        'status_encoded', 'region_encoded'
    ]
    
    available_features = [col for col in feature_columns if col in df.columns]
    
    for col in df.columns:
        if '_encoded' in col and col not in available_features:
            available_features.append(col)
    
    final_df = df[available_features].copy()
    
    final_df = final_df.fillna(0)
    
    print(f"Final features selected: {len(final_df.columns)}")
    print(f"Final data shape: {final_df.shape}")
    
    return final_df

def save_features(df):
    print(f"\nSaving features to {FEATURES_DATA_PATH}...")
    
    os.makedirs(os.path.dirname(FEATURES_DATA_PATH), exist_ok=True)
    
    df.to_csv(FEATURES_DATA_PATH, index=False)
    
    print("Features saved successfully!")
    
    print(f"\n📊 FEATURE SUMMARY:")
    print(f"   Total features: {len(df.columns)}")
    print(f"   Total samples: {len(df)}")
    print(f"   Target column: mag")
    
    print(f"\n📋 Feature columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")

def main():
    print("EARTHQUAKE MAGNITUDE PREDICTION - SIMPLE FEATURE ENGINEERING")
    print("=" * 60)
    
    df = load_data()
    
    df_features = create_simple_features(df)
    
    final_df = select_final_features(df_features)
    
    save_features(final_df)
    
    print("\n" + "="*60)
    print("SIMPLE FEATURE ENGINEERING COMPLETED!")
    print("Ready to run baseline_models.py")

if __name__ == "__main__":
    main()
