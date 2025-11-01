import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = '../models/'


def load_trained_models():
    """Load all trained regression models and preprocessing objects"""
    print("Loading trained models...")
    models = {}
    
    model_files = {
        'Linear_Regression': 'linear_regression.pkl',
        'Ridge_Regression': 'ridge_regression.pkl',
        'Lasso_Regression': 'lasso_regression.pkl',
        'Random_Forest': 'random_forest.pkl',
        'Gradient_Boosting': 'gradient_boosting.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for name, filename in model_files.items():
        try:
            model_path = os.path.join(MODELS_DIR, filename)
            models[name] = joblib.load(model_path)
            print(f"✓ Loaded {name}")
        except FileNotFoundError:
            print(f"✗ {name} not found")
    
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        print("✓ Loaded Scaler")
    except FileNotFoundError:
        print("✗ Scaler not found")
        return None, None, None
    
    try:
        test_data = joblib.load(os.path.join(MODELS_DIR, 'test_data.pkl'))
        feature_names = test_data.get('feature_names', [])
        print(f"✓ Loaded feature names ({len(feature_names)} features)")
    except FileNotFoundError:
        print("✗ test_data.pkl not found")
        feature_names = []
    
    if not models:
        print("\n❌ No models found! Please run baseline_models.py first.")
        return None, None, None
    
    return models, scaler, feature_names


def get_magnitude_category(magnitude):
    """Convert magnitude to risk category"""
    if magnitude < 2.0:
        return "Very Low", "🟢"
    elif magnitude < 3.0:
        return "Low", "🟡"
    elif magnitude < 4.0:
        return "Moderate", "🟠"
    elif magnitude < 5.0:
        return "High", "🔴"
    else:
        return "Very High", "🔴🔴"


def get_quick_input():
    """Get essential features from user with validation"""
    print("\n⚡ QUICK PREDICTION MODE")
    print("="*60)
    print("Enter 5 essential features (rest will be auto-filled)")
    print("="*60)
    
    earthquake_data = {}
    
    # Get inputs with validation
    print("\n🌍 Essential Features:")
    
    while True:
        try:
            lat = float(input("  1. Latitude (-90 to 90, e.g., 35.5): "))
            if -90 <= lat <= 90:
                earthquake_data['latitude'] = lat
                break
            print("     ⚠ Latitude must be between -90 and 90")
        except ValueError:
            print("     ⚠ Please enter a valid number")
    
    while True:
        try:
            lon = float(input("  2. Longitude (-180 to 180, e.g., -118.2): "))
            if -180 <= lon <= 180:
                earthquake_data['longitude'] = lon
                break
            print("     ⚠ Longitude must be between -180 and 180")
        except ValueError:
            print("     ⚠ Please enter a valid number")
    
    while True:
        try:
            depth = float(input("  3. Depth in km (0-700, e.g., 8.5): "))
            if 0 <= depth <= 700:
                earthquake_data['depth'] = depth
                break
            print("     ⚠ Depth must be between 0 and 700 km")
        except ValueError:
            print("     ⚠ Please enter a valid number")
    
    while True:
        try:
            gap = float(input("  4. Gap in degrees (0-360, e.g., 45.0): "))
            if 0 <= gap <= 360:
                earthquake_data['gap'] = gap
                break
            print("     ⚠ Gap must be between 0 and 360 degrees")
        except ValueError:
            print("     ⚠ Please enter a valid number")
    
    while True:
        try:
            rms = float(input("  5. RMS (e.g., 0.3): "))
            if rms >= 0:
                earthquake_data['rms'] = rms
                break
            print("     ⚠ RMS must be positive")
        except ValueError:
            print("     ⚠ Please enter a valid number")
    
    print("\n🔧 Auto-filling remaining features...")
    
    # Auto-fill temporal features
    now = datetime.now()
    earthquake_data['year'] = now.year
    earthquake_data['month'] = now.month
    earthquake_data['day'] = now.day
    earthquake_data['hour'] = now.hour
    earthquake_data['dayofweek'] = now.weekday()
    earthquake_data['quarter'] = (now.month - 1) // 3 + 1
    
    # Cyclical temporal encoding
    earthquake_data['month_sin'] = np.sin(2 * np.pi * now.month / 12)
    earthquake_data['month_cos'] = np.cos(2 * np.pi * now.month / 12)
    earthquake_data['hour_sin'] = np.sin(2 * np.pi * now.hour / 24)
    earthquake_data['hour_cos'] = np.cos(2 * np.pi * now.hour / 24)
    
    print(f"  ✓ Temporal: {now.year}-{now.month:02d}-{now.day:02d} {now.hour:02d}:00")
    
    # Geographic features
    usa_center_lat = 39.8283
    usa_center_lon = -98.5795
    earthquake_data['distance_from_center'] = np.sqrt(
        (earthquake_data['latitude'] - usa_center_lat)**2 + 
        (earthquake_data['longitude'] - usa_center_lon)**2
    )
    
    # Distance to San Andreas Fault
    san_andreas_lat, san_andreas_lon = 36.0, -120.5
    earthquake_data['distance_to_san_andreas'] = np.sqrt(
        (earthquake_data['latitude'] - san_andreas_lat)**2 + 
        (earthquake_data['longitude'] - san_andreas_lon)**2
    )
    
    # Coastal distances
    earthquake_data['distance_to_pacific'] = abs(earthquake_data['longitude'] - (-125))
    earthquake_data['distance_to_atlantic'] = abs(earthquake_data['longitude'] - (-75))
    earthquake_data['distance_to_coast'] = min(
        earthquake_data['distance_to_pacific'],
        earthquake_data['distance_to_atlantic']
    )
    
    print(f"  ✓ Distance from center: {earthquake_data['distance_from_center']:.2f}°")
    
    # Depth-related features
    earthquake_data['is_shallow'] = 1 if earthquake_data['depth'] < 10 else 0
    earthquake_data['is_intermediate'] = 1 if 10 <= earthquake_data['depth'] < 70 else 0
    earthquake_data['is_deep'] = 1 if earthquake_data['depth'] >= 70 else 0
    print(f"  ✓ Depth category: {'Shallow' if earthquake_data['is_shallow'] else 'Intermediate' if earthquake_data['is_intermediate'] else 'Deep'}")
    
    # Measurement quality features (typical values)
    earthquake_data['horizontalError'] = 0.8
    earthquake_data['depthError'] = 1.2
    earthquake_data['depth_error_ratio'] = earthquake_data['depthError'] / (earthquake_data['depth'] + 0.01)
    earthquake_data['horizontal_error_ratio'] = earthquake_data['horizontalError'] / (earthquake_data['depth'] + 0.01)
    
    earthquake_data['measurement_quality'] = (
        earthquake_data['gap'] / 360.0 +
        earthquake_data['rms'] +
        earthquake_data['depth_error_ratio'] +
        earthquake_data['horizontal_error_ratio']
    ) / 4.0
    
    earthquake_data['is_good_coverage'] = 1 if earthquake_data['gap'] < 90 else 0
    earthquake_data['is_poor_coverage'] = 1 if earthquake_data['gap'] > 180 else 0
    
    print(f"  ✓ Measurement quality: {'Good' if earthquake_data['is_good_coverage'] else 'Poor' if earthquake_data['is_poor_coverage'] else 'Fair'}")
    
    # Interaction features
    earthquake_data['depth_x_distance'] = earthquake_data['depth'] * earthquake_data['distance_from_center']
    earthquake_data['depth_x_lat'] = earthquake_data['depth'] * earthquake_data['latitude']
    earthquake_data['gap_x_depth'] = earthquake_data['gap'] * earthquake_data['depth']
    earthquake_data['rms_x_depth'] = earthquake_data['rms'] * earthquake_data['depth']
    
    # Squared terms
    earthquake_data['latitude_squared'] = earthquake_data['latitude'] ** 2
    earthquake_data['longitude_squared'] = earthquake_data['longitude'] ** 2
    earthquake_data['depth_squared'] = earthquake_data['depth'] ** 2
    
    # Encoded categorical features (using most common values)
    earthquake_data['magType_encoded'] = 4
    earthquake_data['net_encoded'] = 5
    earthquake_data['type_encoded'] = 0
    earthquake_data['status_encoded'] = 0
    earthquake_data['depth_category_encoded'] = 0 if earthquake_data['is_shallow'] else 2 if earthquake_data['is_deep'] else 1
    
    # Region detection
    if earthquake_data['longitude'] < -120:
        earthquake_data['region_encoded'] = 2
        earthquake_data['seismic_zone_encoded'] = 0  # West/San Andreas
        earthquake_data['seismic_risk_encoded'] = 2  # Very high
        region = "West (High seismic activity)"
    elif earthquake_data['longitude'] < -90:
        earthquake_data['region_encoded'] = 1
        earthquake_data['seismic_zone_encoded'] = 7  # Central
        earthquake_data['seismic_risk_encoded'] = 1  # Moderate
        region = "Central"
    else:
        earthquake_data['region_encoded'] = 0
        earthquake_data['seismic_zone_encoded'] = 7  # East
        earthquake_data['seismic_risk_encoded'] = 0  # Low
        region = "East"
    
    print(f"  ✓ Region: {region}")
    
    return earthquake_data


def get_example_earthquakes():
    """Pre-loaded real earthquake examples"""
    examples = {
        '1': {
            'name': '🌋 California Shallow Earthquake (San Andreas)',
            'data': {
                'latitude': 35.5, 'longitude': -118.2, 'depth': 8.5,
                'gap': 45.0, 'rms': 0.3, 'horizontalError': 0.8,
                'depthError': 1.2, 'year': 2024, 'month': 10,
                'day': 30, 'hour': 14, 'dayofweek': 2, 'quarter': 4,
                'month_sin': -0.5, 'month_cos': -0.866,
                'hour_sin': 0.809, 'hour_cos': -0.588,
                'distance_from_center': 30.5,
                'distance_to_san_andreas': 5.2,
                'distance_to_pacific': 7.2, 'distance_to_atlantic': 43.2,
                'distance_to_coast': 7.2,
                'is_shallow': 1, 'is_intermediate': 0, 'is_deep': 0,
                'depth_error_ratio': 0.14, 'horizontal_error_ratio': 0.094,
                'measurement_quality': 0.15,
                'is_good_coverage': 1, 'is_poor_coverage': 0,
                'depth_x_distance': 259.25, 'depth_x_lat': 301.75,
                'gap_x_depth': 382.5, 'rms_x_depth': 2.55,
                'latitude_squared': 1260.25, 'longitude_squared': 13971.24,
                'depth_squared': 72.25,
                'magType_encoded': 4, 'net_encoded': 5,
                'type_encoded': 0, 'status_encoded': 0,
                'region_encoded': 2, 'seismic_zone_encoded': 0,
                'seismic_risk_encoded': 2, 'depth_category_encoded': 0
            }
        },
        '2': {
            'name': '❄️ Alaska Deep Earthquake',
            'data': {
                'latitude': 64.5, 'longitude': -148.9, 'depth': 52.0,
                'gap': 89.9, 'rms': 0.8, 'horizontalError': 0.5,
                'depthError': 1.6, 'year': 2024, 'month': 10,
                'day': 30, 'hour': 12, 'dayofweek': 2, 'quarter': 4,
                'month_sin': -0.5, 'month_cos': -0.866,
                'hour_sin': 0.0, 'hour_cos': 1.0,
                'distance_from_center': 55.0,
                'distance_to_san_andreas': 45.2,
                'distance_to_pacific': 23.9, 'distance_to_atlantic': 73.9,
                'distance_to_coast': 23.9,
                'is_shallow': 0, 'is_intermediate': 1, 'is_deep': 0,
                'depth_error_ratio': 0.03, 'horizontal_error_ratio': 0.0096,
                'measurement_quality': 0.22,
                'is_good_coverage': 1, 'is_poor_coverage': 0,
                'depth_x_distance': 2860.0, 'depth_x_lat': 3354.0,
                'gap_x_depth': 4674.8, 'rms_x_depth': 41.6,
                'latitude_squared': 4160.25, 'longitude_squared': 22171.21,
                'depth_squared': 2704.0,
                'magType_encoded': 4, 'net_encoded': 0,
                'type_encoded': 0, 'status_encoded': 0,
                'region_encoded': 2, 'seismic_zone_encoded': 4,
                'seismic_risk_encoded': 2, 'depth_category_encoded': 1
            }
        },
        '3': {
            'name': '🌊 West Coast Moderate Earthquake',
            'data': {
                'latitude': 38.8, 'longitude': -122.8, 'depth': 12.0,
                'gap': 65.0, 'rms': 0.15, 'horizontalError': 0.3,
                'depthError': 0.97, 'year': 2024, 'month': 10,
                'day': 30, 'hour': 9, 'dayofweek': 2, 'quarter': 4,
                'month_sin': -0.5, 'month_cos': -0.866,
                'hour_sin': -0.866, 'hour_cos': 0.5,
                'distance_from_center': 24.5,
                'distance_to_san_andreas': 4.8,
                'distance_to_pacific': 2.2, 'distance_to_atlantic': 47.8,
                'distance_to_coast': 2.2,
                'is_shallow': 0, 'is_intermediate': 1, 'is_deep': 0,
                'depth_error_ratio': 0.081, 'horizontal_error_ratio': 0.025,
                'measurement_quality': 0.13,
                'is_good_coverage': 1, 'is_poor_coverage': 0,
                'depth_x_distance': 294.0, 'depth_x_lat': 465.6,
                'gap_x_depth': 780.0, 'rms_x_depth': 1.8,
                'latitude_squared': 1505.44, 'longitude_squared': 15079.84,
                'depth_squared': 144.0,
                'magType_encoded': 2, 'net_encoded': 5,
                'type_encoded': 0, 'status_encoded': 0,
                'region_encoded': 2, 'seismic_zone_encoded': 0,
                'seismic_risk_encoded': 2, 'depth_category_encoded': 1
            }
        },
        '4': {
            'name': '🌴 Hawaii Volcanic Earthquake',
            'data': {
                'latitude': 19.3, 'longitude': -155.5, 'depth': 30.0,
                'gap': 165.0, 'rms': 0.11, 'horizontalError': 0.6,
                'depthError': 0.76, 'year': 2024, 'month': 10,
                'day': 30, 'hour': 18, 'dayofweek': 2, 'quarter': 4,
                'month_sin': -0.5, 'month_cos': -0.866,
                'hour_sin': 0.951, 'hour_cos': -0.309,
                'distance_from_center': 60.5,
                'distance_to_san_andreas': 38.2,
                'distance_to_pacific': 30.5, 'distance_to_atlantic': 80.5,
                'distance_to_coast': 30.5,
                'is_shallow': 0, 'is_intermediate': 1, 'is_deep': 0,
                'depth_error_ratio': 0.025, 'horizontal_error_ratio': 0.02,
                'measurement_quality': 0.26,
                'is_good_coverage': 0, 'is_poor_coverage': 1,
                'depth_x_distance': 1815.0, 'depth_x_lat': 579.0,
                'gap_x_depth': 4950.0, 'rms_x_depth': 3.3,
                'latitude_squared': 372.49, 'longitude_squared': 24180.25,
                'depth_squared': 900.0,
                'magType_encoded': 2, 'net_encoded': 3,
                'type_encoded': 0, 'status_encoded': 0,
                'region_encoded': 2, 'seismic_zone_encoded': 5,
                'seismic_risk_encoded': 1, 'depth_category_encoded': 1
            }
        },
        '5': {
            'name': '🛢️ Texas Induced Earthquake',
            'data': {
                'latitude': 31.7, 'longitude': -104.1, 'depth': 4.5,
                'gap': 63.0, 'rms': 0.3, 'horizontalError': 0.4,
                'depthError': 1.2, 'year': 2024, 'month': 10,
                'day': 30, 'hour': 16, 'dayofweek': 2, 'quarter': 4,
                'month_sin': -0.5, 'month_cos': -0.866,
                'hour_sin': 1.0, 'hour_cos': 0.0,
                'distance_from_center': 9.8,
                'distance_to_san_andreas': 18.5,
                'distance_to_pacific': 20.9, 'distance_to_atlantic': 29.1,
                'distance_to_coast': 20.9,
                'is_shallow': 1, 'is_intermediate': 0, 'is_deep': 0,
                'depth_error_ratio': 0.27, 'horizontal_error_ratio': 0.089,
                'measurement_quality': 0.24,
                'is_good_coverage': 1, 'is_poor_coverage': 0,
                'depth_x_distance': 44.1, 'depth_x_lat': 142.65,
                'gap_x_depth': 283.5, 'rms_x_depth': 1.35,
                'latitude_squared': 1004.89, 'longitude_squared': 10836.81,
                'depth_squared': 20.25,
                'magType_encoded': 4, 'net_encoded': 11,
                'type_encoded': 0, 'status_encoded': 0,
                'region_encoded': 1, 'seismic_zone_encoded': 7,
                'seismic_risk_encoded': 1, 'depth_category_encoded': 0
            }
        }
    }
    
    print("\n📋 SELECT AN EXAMPLE EARTHQUAKE:")
    print("="*60)
    for key, example in examples.items():
        print(f"{key}. {example['name']}")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice in examples:
        selected = examples[choice]
        print(f"\n✓ Selected: {selected['name']}")
        return selected['data']
    else:
        print("\n⚠ Invalid choice. Using example 1.")
        return examples['1']['data']


def preprocess_input(input_data, feature_names):
    """Preprocess input data to match training format"""
    if isinstance(input_data, dict):
        if not isinstance(list(input_data.values())[0], list):
            input_data = {k: [v] for k, v in input_data.items()}
        df = pd.DataFrame(input_data)
    else:
        df = input_data.copy()
    
    # Add missing features with default values
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        for feature in missing_features:
            df[feature] = 0
    
    # Ensure correct column order
    df = df[feature_names]
    
    # Fill any NaN or infinite values
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


def predict_magnitude(input_data, models, scaler, feature_names):
    """Predict earthquake magnitude using regression models"""
    X = preprocess_input(input_data, feature_names)
    X_scaled = scaler.transform(X)
    
    results = []
    
    for i in range(len(X)):
        result = {'Input': i + 1, 'predictions': {}}
        
        for name, model in models.items():
            # Make prediction
            pred_magnitude = model.predict(X_scaled[i:i+1])[0]
            
            # Ensure magnitude is within reasonable bounds
            pred_magnitude = max(0.0, min(10.0, pred_magnitude))
            
            # Get risk category
            risk_category, emoji = get_magnitude_category(pred_magnitude)
            
            result['predictions'][name] = {
                'magnitude': pred_magnitude,
                'risk_category': risk_category,
                'emoji': emoji
            }
        
        results.append(result)
    
    return results[0] if len(results) == 1 else results


def display_results(result):
    """Display prediction results in a user-friendly format"""
    print("\n" + "="*60)
    print("🔮 EARTHQUAKE MAGNITUDE PREDICTIONS")
    print("="*60)
    
    predictions = result['predictions']
    
    # Calculate statistics across models
    magnitudes = [p['magnitude'] for p in predictions.values()]
    mean_mag = np.mean(magnitudes)
    std_mag = np.std(magnitudes)
    min_mag = np.min(magnitudes)
    max_mag = np.max(magnitudes)
    
    # Display ensemble prediction
    print(f"\n📊 ENSEMBLE PREDICTION (Average of all models):")
    print("-" * 60)
    risk_cat, emoji = get_magnitude_category(mean_mag)
    print(f"  {emoji} Predicted Magnitude:  {mean_mag:.2f}")
    print(f"     Risk Category:       {risk_cat}")
    print(f"     Confidence Range:    {min_mag:.2f} - {max_mag:.2f}")
    print(f"     Std Deviation:       ±{std_mag:.2f}")
    
    # Display individual model predictions
    print(f"\n📈 INDIVIDUAL MODEL PREDICTIONS:")
    print("-" * 60)
    
    for model_name, pred in sorted(predictions.items(), key=lambda x: x[1]['magnitude'], reverse=True):
        mag = pred['magnitude']
        risk = pred['risk_category']
        emoji = pred['emoji']
        
        print(f"\n{model_name.replace('_', ' ')}:")
        print(f"  {emoji} Magnitude: {mag:.2f}  |  Risk: {risk}")
    
    # Interpretation guide
    print("\n" + "="*60)
    print("📖 MAGNITUDE INTERPRETATION:")
    print("-" * 60)
    print("  < 2.0  🟢 Very Low:    Rarely felt, no damage")
    print("  2.0-3.0  🟡 Low:       Often felt, rarely causes damage")
    print("  3.0-4.0  🟠 Moderate:  Noticeable shaking, minor damage")
    print("  4.0-5.0  🔴 High:      Can cause damage to buildings")
    print("  5.0+   🔴🔴 Very High: Serious damage, dangerous")
    print("="*60)


def save_prediction(input_data, result):
    """Save prediction to file"""
    try:
        predictions_file = '../results/predictions_log.csv'
        
        # Prepare data for saving
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'latitude': input_data.get('latitude'),
            'longitude': input_data.get('longitude'),
            'depth': input_data.get('depth'),
            'gap': input_data.get('gap'),
            'rms': input_data.get('rms'),
        }
        
        # Add predictions from all models
        for model_name, pred in result['predictions'].items():
            record[f'{model_name}_magnitude'] = pred['magnitude']
            record[f'{model_name}_risk'] = pred['risk_category']
        
        # Calculate ensemble
        magnitudes = [p['magnitude'] for p in result['predictions'].values()]
        record['ensemble_magnitude'] = np.mean(magnitudes)
        record['ensemble_std'] = np.std(magnitudes)
        
        # Save to CSV
        df = pd.DataFrame([record])
        
        if os.path.exists(predictions_file):
            df.to_csv(predictions_file, mode='a', header=False, index=False)
        else:
            df.to_csv(predictions_file, mode='w', header=True, index=False)
        
        print(f"\n💾 Prediction saved to: {predictions_file}")
        
    except Exception as e:
        print(f"\n⚠ Could not save prediction: {str(e)}")


def main():
    print("="*60)
    print("🌍 EARTHQUAKE MAGNITUDE PREDICTION SYSTEM")
    print("="*60)
    
    # Load models
    models, scaler, feature_names = load_trained_models()
    
    if models is None:
        return
    
    print(f"\n✓ Loaded {len(models)} regression models")
    print(f"✓ Ready to predict exact magnitudes (0.0 - 10.0)")
    
    while True:
        print("\n" + "="*60)
        print("CHOOSE PREDICTION MODE:")
        print("="*60)
        print("1. ⚡ Quick Prediction (5 inputs only)")
        print("2. 🎯 Example Earthquakes (pre-loaded)")
        print("3. 💾 View Prediction History")
        print("4. 🚪 Exit")
        
        choice = input("\nEnter choice (1/2/3/4): ").strip()
        
        if choice == '1':
            earthquake_data = get_quick_input()
        elif choice == '2':
            earthquake_data = get_example_earthquakes()
        elif choice == '3':
            try:
                history = pd.read_csv('../results/predictions_log.csv')
                print("\n" + "="*60)
                print("📜 RECENT PREDICTIONS:")
                print("="*60)
                print(history.tail(10).to_string(index=False))
                print("="*60)
                continue
            except FileNotFoundError:
                print("\n⚠ No prediction history found yet.")
                continue
        elif choice == '4':
            print("\n👋 Thank you for using the Earthquake Prediction System!")
            print("   Stay safe! 🌍")
            break
        else:
            print("\n❌ Invalid choice. Please try again.")
            continue
        
        # Make prediction
        print("\n🔮 Making predictions...")
        print("-" * 60)
        
        result = predict_magnitude(earthquake_data, models, scaler, feature_names)
        
        # Display results
        display_results(result)
        
        # Ask to save
        save_choice = input("\n💾 Save this prediction? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_prediction(earthquake_data, result)
        
        # Ask for another prediction
        again = input("\n🔄 Make another prediction? (y/n): ").strip().lower()
        if again != 'y':
            print("\n👋 Thank you for using the Earthquake Prediction System!")
            print("   Stay safe! 🌍")
            break


if __name__ == "__main__":
    main()