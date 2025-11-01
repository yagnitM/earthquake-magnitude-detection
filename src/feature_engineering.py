import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')


PROCESSED_DATA_PATH = '../data/processed/usa_earthquake_cleaned.csv'
FEATURES_DATA_PATH = '../data/processed/usa_earthquake_features.csv'


# Major seismic zones in the USA (approximate boundaries)
SEISMIC_ZONES = {
    'san_andreas': {'lat': (32, 40), 'lon': (-125, -114), 'risk': 'very_high'},
    'cascadia': {'lat': (40, 49), 'lon': (-125, -121), 'risk': 'very_high'},
    'wasatch': {'lat': (38, 42), 'lon': (-113, -111), 'risk': 'high'},
    'new_madrid': {'lat': (35, 38), 'lon': (-91, -88), 'risk': 'moderate'},
    'alaska': {'lat': (55, 72), 'lon': (-170, -130), 'risk': 'very_high'},
    'hawaii': {'lat': (18, 23), 'lon': (-161, -154), 'risk': 'high'},
    'yellowstone': {'lat': (44, 45), 'lon': (-111, -110), 'risk': 'moderate'}
}


def load_data():
    """Load cleaned earthquake data"""
    print("Loading cleaned data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"✓ Data loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    return df


def validate_data(df):
    """Validate data quality before feature engineering"""
    print("\nValidating data quality...")
    
    issues = []
    
    # Check for required columns
    required_cols = ['latitude', 'longitude', 'depth', 'mag', 'date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for invalid ranges
    if 'latitude' in df.columns:
        invalid_lat = df[(df['latitude'] < -90) | (df['latitude'] > 90)]
        if len(invalid_lat) > 0:
            issues.append(f"Invalid latitude values: {len(invalid_lat)} rows")
    
    if 'longitude' in df.columns:
        invalid_lon = df[(df['longitude'] < -180) | (df['longitude'] > 180)]
        if len(invalid_lon) > 0:
            issues.append(f"Invalid longitude values: {len(invalid_lon)} rows")
    
    if 'depth' in df.columns:
        invalid_depth = df[df['depth'] < 0]
        if len(invalid_depth) > 0:
            issues.append(f"Negative depth values: {len(invalid_depth)} rows")
    
    if 'mag' in df.columns:
        invalid_mag = df[(df['mag'] < 0) | (df['mag'] > 10)]
        if len(invalid_mag) > 0:
            issues.append(f"Invalid magnitude values: {len(invalid_mag)} rows")
    
    if issues:
        print("⚠ Data quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Data validation passed")
    
    return issues


def create_temporal_features(df):
    """Create time-based features"""
    print("\nCreating temporal features...")
    
    df = df.copy()
    
    # Parse datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract temporal components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    
    # Extract hour from time_only if available
    if 'time_only' in df.columns:
        df['hour'] = pd.to_numeric(
            df['time_only'].str.split(':').str[0], 
            errors='coerce'
        ).fillna(0)
    else:
        df['hour'] = 0
    
    # Cyclical encoding for temporal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    print("✓ Created temporal features (year, month, day, hour, cyclical encodings)")
    
    return df


def identify_seismic_zone(lat, lon):
    """Identify which major seismic zone an earthquake belongs to"""
    for zone_name, bounds in SEISMIC_ZONES.items():
        lat_min, lat_max = bounds['lat']
        lon_min, lon_max = bounds['lon']
        
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return zone_name, bounds['risk']
    
    return 'other', 'low'


def create_geographic_features(df):
    """Create location-based features"""
    print("\nCreating geographic features...")
    
    df = df.copy()
    
    # Distance from USA geographic center
    usa_center_lat = 39.8283
    usa_center_lon = -98.5795
    df['distance_from_center'] = np.sqrt(
        (df['latitude'] - usa_center_lat)**2 + 
        (df['longitude'] - usa_center_lon)**2
    )
    
    # Identify seismic zones
    zones_and_risks = df.apply(
        lambda row: identify_seismic_zone(row['latitude'], row['longitude']),
        axis=1
    )
    df['seismic_zone'] = zones_and_risks.apply(lambda x: x[0])
    df['seismic_risk'] = zones_and_risks.apply(lambda x: x[1])
    
    # Simple regional classification (kept for compatibility)
    df['region'] = 'east'
    df.loc[df['longitude'] < -90, 'region'] = 'central'
    df.loc[df['longitude'] < -120, 'region'] = 'west'
    
    # Distance to major fault lines (approximations)
    san_andreas_lat, san_andreas_lon = 36.0, -120.5
    df['distance_to_san_andreas'] = np.sqrt(
        (df['latitude'] - san_andreas_lat)**2 + 
        (df['longitude'] - san_andreas_lon)**2
    )
    
    # Coastal proximity (distance to nearest coast)
    # Simplified: distance to Pacific coast (longitude -125) or Atlantic coast (longitude -75)
    df['distance_to_pacific'] = np.abs(df['longitude'] - (-125))
    df['distance_to_atlantic'] = np.abs(df['longitude'] - (-75))
    df['distance_to_coast'] = np.minimum(df['distance_to_pacific'], df['distance_to_atlantic'])
    
    print("✓ Created geographic features (distances, zones, regions)")
    
    return df


def create_measurement_features(df):
    """Create features from measurement quality indicators"""
    print("\nCreating measurement quality features...")
    
    df = df.copy()
    
    # Depth-related features
    df['is_shallow'] = (df['depth'] < 10).astype(int)
    df['is_intermediate'] = ((df['depth'] >= 10) & (df['depth'] < 70)).astype(int)
    df['is_deep'] = (df['depth'] >= 70).astype(int)
    
    # Depth categories
    df['depth_category'] = pd.cut(
        df['depth'],
        bins=[0, 10, 35, 70, 300],
        labels=['shallow', 'mid_shallow', 'intermediate', 'deep']
    ).astype(str)
    
    # Error ratios (but NOT magError!)
    df['depth_error_ratio'] = df['depthError'] / (df['depth'] + 0.01)
    df['horizontal_error_ratio'] = df['horizontalError'] / (df['depth'] + 0.01)
    
    # Measurement quality score (lower is better)
    # CRITICAL: Does NOT include magError
    df['measurement_quality'] = (
        df['gap'] / 360.0 +  # Gap normalized
        df['rms'] +  # RMS value
        df['depth_error_ratio'] +  # Depth uncertainty
        df['horizontal_error_ratio']  # Position uncertainty
    ) / 4.0
    
    # Station coverage quality
    df['is_good_coverage'] = (df['gap'] < 90).astype(int)
    df['is_poor_coverage'] = (df['gap'] > 180).astype(int)
    
    print("✓ Created measurement quality features (NO magError included)")
    
    return df


def create_interaction_features(df):
    """Create interaction features between existing variables"""
    print("\nCreating interaction features...")
    
    df = df.copy()
    
    # Depth × Location interactions
    df['depth_x_distance'] = df['depth'] * df['distance_from_center']
    df['depth_x_lat'] = df['depth'] * df['latitude']
    
    # Quality × Depth interactions
    df['gap_x_depth'] = df['gap'] * df['depth']
    df['rms_x_depth'] = df['rms'] * df['depth']
    
    # Location squared terms (for non-linear relationships)
    df['latitude_squared'] = df['latitude'] ** 2
    df['longitude_squared'] = df['longitude'] ** 2
    df['depth_squared'] = df['depth'] ** 2
    
    print("✓ Created interaction features")
    
    return df


def encode_categorical_features(df):
    """Encode categorical variables"""
    print("\nEncoding categorical features...")
    
    df = df.copy()
    label_encoders = {}
    
    categorical_cols = [
        'magType', 'net', 'type', 'status', 
        'region', 'seismic_zone', 'seismic_risk', 'depth_category'
    ]
    
    encoded_count = 0
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            encoded_count += 1
    
    print(f"✓ Encoded {encoded_count} categorical variables")
    
    return df, label_encoders


def select_final_features(df):
    """Select and organize final feature set for modeling"""
    print("\nSelecting final features...")
    
    # CRITICAL: Features that should NEVER be included (data leakage)
    forbidden_features = [
        'mag',  # Target variable
        'magError',  # Measurement error of magnitude (data leakage!)
        'mag_error_ratio',  # Derived from magError
        'date',  # Raw date
        'time_only',  # Raw time
    ]
    
    # Core features to include
    core_features = [
        # Location
        'latitude', 'longitude', 'depth',
        
        # Measurement quality (NO magError!)
        'gap', 'rms', 'horizontalError', 'depthError',
        
        # Temporal
        'year', 'month', 'day', 'hour', 'dayofweek', 'quarter',
        'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
        
        # Geographic
        'distance_from_center', 'distance_to_san_andreas',
        'distance_to_coast', 'distance_to_pacific', 'distance_to_atlantic',
        
        # Depth-related
        'is_shallow', 'is_intermediate', 'is_deep',
        'depth_error_ratio', 'horizontal_error_ratio',
        
        # Measurement quality
        'measurement_quality', 'is_good_coverage', 'is_poor_coverage',
        
        # Interactions
        'depth_x_distance', 'depth_x_lat', 'gap_x_depth', 'rms_x_depth',
        'latitude_squared', 'longitude_squared', 'depth_squared',
    ]
    
    # Add all encoded categorical features
    encoded_features = [col for col in df.columns if col.endswith('_encoded')]
    core_features.extend(encoded_features)
    
    # Filter to only available features
    available_features = [col for col in core_features if col in df.columns]
    
    # CRITICAL: Remove any forbidden features that might have slipped through
    available_features = [
        col for col in available_features 
        if col not in forbidden_features and not any(f in col.lower() for f in ['magerror'])
    ]
    
    # Create final dataframe with features + target
    final_df = df[available_features].copy()
    
    # Add magnitude as target (separate column)
    if 'mag' in df.columns:
        final_df['mag'] = df['mag']
    
    # Fill any remaining NaN values
    final_df = final_df.fillna(0)
    
    # Replace infinite values
    final_df = final_df.replace([np.inf, -np.inf], 0)
    
    print(f"✓ Selected {len(available_features)} features (excluding target 'mag')")
    print(f"✓ Final dataset shape: {final_df.shape}")
    
    # Display feature categories
    print("\n📊 Feature Categories:")
    location_features = [f for f in available_features if any(x in f for x in ['lat', 'lon', 'distance', 'coast'])]
    temporal_features = [f for f in available_features if any(x in f for x in ['year', 'month', 'day', 'hour', 'quarter'])]
    depth_features = [f for f in available_features if 'depth' in f]
    quality_features = [f for f in available_features if any(x in f for x in ['gap', 'rms', 'error', 'quality', 'coverage'])]
    
    print(f"  Location features:    {len(location_features)}")
    print(f"  Temporal features:    {len(temporal_features)}")
    print(f"  Depth features:       {len(depth_features)}")
    print(f"  Quality features:     {len(quality_features)}")
    print(f"  Encoded features:     {len(encoded_features)}")
    print(f"  Interaction features: {len([f for f in available_features if '_x_' in f or 'squared' in f])}")
    
    return final_df, available_features


def save_features(df, feature_list):
    """Save engineered features and metadata"""
    print(f"\nSaving features to {FEATURES_DATA_PATH}...")
    
    os.makedirs(os.path.dirname(FEATURES_DATA_PATH), exist_ok=True)
    
    # Save feature data
    df.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"✓ Features saved: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Save feature list for reference
    feature_list_path = FEATURES_DATA_PATH.replace('.csv', '_list.txt')
    with open(feature_list_path, 'w') as f:
        f.write("Feature Engineering Summary\n")
        f.write("="*60 + "\n\n")
        f.write("CRITICAL: magError is NOT included (prevents data leakage)\n\n")
        f.write("Feature List:\n")
        for i, feat in enumerate(feature_list, 1):
            f.write(f"{i}. {feat}\n")
    print(f"✓ Feature list saved to {feature_list_path}")
    
    # Save basic statistics
    stats_path = FEATURES_DATA_PATH.replace('.csv', '_stats.csv')
    stats_df = df.describe().T
    stats_df.to_csv(stats_path)
    print(f"✓ Feature statistics saved to {stats_path}")


def main():
    print("="*60)
    print("🌍 EARTHQUAKE MAGNITUDE PREDICTION - FEATURE ENGINEERING")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Validate data quality
    validate_data(df)
    
    # Create features step by step
    df = create_temporal_features(df)
    df = create_geographic_features(df)
    df = create_measurement_features(df)
    df = create_interaction_features(df)
    df, label_encoders = encode_categorical_features(df)
    
    # Select final features
    final_df, feature_list = select_final_features(df)
    
    # Save everything
    save_features(final_df, feature_list)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ FEATURE ENGINEERING COMPLETED!")
    print("="*60)
    print("\n🔒 CRITICAL DATA LEAKAGE PREVENTION:")
    print("  ✓ magError excluded from features")
    print("  ✓ mag_error_ratio excluded from features")
    print("  ✓ Only pre-measurement features included")
    
    print("\n💡 Next steps:")
    print("  1. Review feature_list.txt for all features")
    print("  2. Check feature_stats.csv for data distribution")
    print("  3. Run baseline_models.py to train models")
    print("="*60)


if __name__ == "__main__":
    main()