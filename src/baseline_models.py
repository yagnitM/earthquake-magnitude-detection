import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


FEATURES_DATA_PATH = '../data/processed/usa_earthquake_features.csv'
MODELS_DIR = '../models/'


os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    """Load earthquake features dataset"""
    print("Loading data...")
    df = pd.read_csv(FEATURES_DATA_PATH)
    print(f"✓ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df


def prepare_data(df):
    """Prepare features and target for regression"""
    print("\nPreparing data for REGRESSION...")
    
    # Target is continuous magnitude (NO BINNING!)
    if 'mag' not in df.columns:
        raise ValueError("ERROR: 'mag' column not found in dataset!")
    
    y = df['mag'].copy()
    
    # Remove any samples with invalid magnitudes
    valid_mask = (y >= 0) & (y <= 10) & (~y.isna())
    df = df[valid_mask].copy()
    y = y[valid_mask]
    
    print(f"✓ Valid samples: {len(y)}")
    print(f"✓ Magnitude range: {y.min():.2f} to {y.max():.2f}")
    print(f"✓ Mean magnitude: {y.mean():.2f}")
    print(f"✓ Median magnitude: {y.median():.2f}")
    
    # Select numerical features (exclude target)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numerical_cols if col not in ['mag']]
    
    # CRITICAL: Remove any features that leak information about magnitude
    leakage_features = ['mag_class', 'mag_error_ratio', 'magError']
    feature_cols = [col for col in feature_cols if col not in leakage_features]
    
    if any(leak in feature_cols for leak in leakage_features):
        print("⚠ WARNING: Data leakage detected! Removing problematic features...")
    
    X = df[feature_cols].copy()
    
    print(f"\n✓ Features selected: {len(feature_cols)}")
    print(f"  Feature list: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
    
    # Clean data
    print("\nCleaning data...")
    
    # Replace infinities with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median
    X = X.fillna(X.median())
    X = X.fillna(0)
    
    # Clip outliers at 0.1% and 99.9% percentiles
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            upper_limit = X[col].quantile(0.999)
            lower_limit = X[col].quantile(0.001)
            X[col] = X[col].clip(lower=lower_limit, upper=upper_limit)
    
    # Final check for invalid values
    if np.any(np.isinf(X.values)) or np.any(np.isnan(X.values)):
        print("⚠ Warning: Still have infinite/NaN values. Applying final cleanup...")
        X = X.replace([np.inf, -np.inf, np.nan], 0)
    
    print("✓ Data cleaning completed")
    
    # Display magnitude distribution
    print(f"\n📊 Magnitude Distribution:")
    print(f"  0-1:   {sum((y >= 0) & (y < 1))} earthquakes")
    print(f"  1-2:   {sum((y >= 1) & (y < 2))} earthquakes")
    print(f"  2-3:   {sum((y >= 2) & (y < 3))} earthquakes")
    print(f"  3-4:   {sum((y >= 3) & (y < 4))} earthquakes")
    print(f"  4-5:   {sum((y >= 4) & (y < 5))} earthquakes")
    print(f"  5+:    {sum(y >= 5)} earthquakes")
    
    return X, y, feature_cols


def train_models(X_train, y_train):
    """Train multiple regression models"""
    print("\n" + "="*60)
    print("TRAINING REGRESSION MODELS...")
    print("="*60)
    
    models = {
        'Linear_Regression': LinearRegression(),
        'Ridge_Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso_Regression': Lasso(alpha=0.1, random_state=42),
        'Random_Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient_Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name.replace('_', ' ')}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"{name.lower()}.pkl")
            joblib.dump(model, model_path)
            print(f"   ✓ Saved to {model_path}")
            
        except Exception as e:
            print(f"   ✗ Failed: {str(e)}")
    
    return trained_models


def evaluate_models(models, X_test, y_test):
    """Evaluate regression models on test set"""
    print("\n" + "="*60)
    print("EVALUATING MODELS ON TEST SET")
    print("="*60)
    
    results = []
    
    for name, model in models.items():
        print(f"\n📊 {name.replace('_', ' ')}:")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate percentage within tolerance
            tolerance_01 = np.mean(np.abs(y_test - y_pred) <= 0.1) * 100
            tolerance_05 = np.mean(np.abs(y_test - y_pred) <= 0.5) * 100
            tolerance_10 = np.mean(np.abs(y_test - y_pred) <= 1.0) * 100
            
            print(f"  RMSE:                {rmse:.4f}")
            print(f"  MAE:                 {mae:.4f}")
            print(f"  R² Score:            {r2:.4f}")
            print(f"  Within ±0.1 mag:     {tolerance_01:.1f}%")
            print(f"  Within ±0.5 mag:     {tolerance_05:.1f}%")
            print(f"  Within ±1.0 mag:     {tolerance_10:.1f}%")
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Within_0.1': tolerance_01,
                'Within_0.5': tolerance_05,
                'Within_1.0': tolerance_10
            })
            
        except Exception as e:
            print(f"  ✗ Evaluation failed: {str(e)}")
    
    # Display summary
    if results:
        print("\n" + "="*60)
        print("📈 MODELS SUMMARY (Sorted by RMSE)")
        print("="*60)
        results_df = pd.DataFrame(results).sort_values('RMSE')
        print(results_df.to_string(index=False))
        
        # Save results
        results_path = os.path.join(MODELS_DIR, 'model_comparison.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to {results_path}")
    
    return results


def main():
    print("="*60)
    print("🌍 EARTHQUAKE MAGNITUDE PREDICTION - REGRESSION MODELS")
    print("="*60)
    
    # Load and prepare data
    df = load_data()
    X, y, feature_cols = prepare_data(df)
    
    # Split data
    print("\n" + "="*60)
    print("SPLITTING DATA...")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set:  {X_test.shape[0]} samples")
    
    # Scale features
    print("\n" + "="*60)
    print("SCALING FEATURES...")
    print("="*60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Scaling completed using StandardScaler")
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Save test data and feature names
    test_data = {
        'X_test': X_test_scaled,
        'y_test': y_test,
        'feature_names': feature_cols
    }
    test_path = os.path.join(MODELS_DIR, 'test_data.pkl')
    joblib.dump(test_data, test_path)
    print(f"✓ Test data saved to {test_path}")
    
    # Train models
    trained_models = train_models(X_train_scaled, y_train)
    
    # Evaluate models
    if trained_models:
        evaluate_models(trained_models, X_test_scaled, y_test)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)
    print(f"📁 Models saved in: {MODELS_DIR}")
    print(f"📊 {len(trained_models)} models trained successfully")
    print("\n💡 Next steps:")
    print("   1. Run predict.py to make predictions")
    print("   2. Check model_comparison.csv for detailed metrics")
    print("="*60)


if __name__ == "__main__":
    main()