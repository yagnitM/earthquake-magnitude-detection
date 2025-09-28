import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURES_DATA_PATH = '../data/processed/usa_earthquake_features.csv'
MODELS_DIR = '../models/'

os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    print("Loading data...")
    df = pd.read_csv(FEATURES_DATA_PATH)
    print(f"Data loaded: {df.shape}")
    return df

def prepare_data(df):
    print("Preparing data...")
    df['mag_class'] = pd.cut(df['mag'], bins=[0, 2, 4, 10], labels=[0, 1, 2])
    df['mag_class'] = pd.to_numeric(df['mag_class'], errors='coerce')
    df['mag_class'] = df['mag_class'].fillna(0).astype(int)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numerical_cols if col not in ['mag', 'mag_class']]
    X = df[feature_cols].copy()
    y = df['mag_class']
    print("Cleaning data...")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            upper_limit = X[col].quantile(0.999)
            lower_limit = X[col].quantile(0.001)
            X[col] = X[col].clip(lower=lower_limit, upper=upper_limit)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Classes distribution:")
    print(f"  Low (0-2):     {sum(y == 0)} samples")
    print(f"  Medium (2-4):  {sum(y == 1)} samples") 
    print(f"  High (4+):     {sum(y == 2)} samples")
    
    if np.any(np.isinf(X.values)) or np.any(np.isnan(X.values)):
        print("Warning: Still have infinite/NaN values. Applying final cleanup...")
        X = X.replace([np.inf, -np.inf, np.nan], 0)
    
    print("Data cleaning completed")
    
    return X, y

def train_models(X_train, y_train):
    print("\nTraining models...")
    
    models = {
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        model_path = os.path.join(MODELS_DIR, f"{name.lower()}.pkl")
        joblib.dump(model, model_path)
        print(f"{name} saved to {model_path}")
    
    return trained_models

def main():
    print("EARTHQUAKE MAGNITUDE CLASSIFICATION - SIMPLE BASELINE MODELS")
    print("=" * 60)
    
    df = load_data()
    X, y = prepare_data(df)
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("Scaling features...")
    scaler = StandardScaler()
    print(f"Data range check - Min: {X_train.min().min():.2f}, Max: {X_train.max().max():.2f}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Scaling completed")
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    test_data = {
        'X_test': X_test_scaled,
        'y_test': y_test
    }
    test_path = os.path.join(MODELS_DIR, 'test_data.pkl')
    joblib.dump(test_data, test_path)
    print(f"Test data saved to {test_path}")
    
    trained_models = train_models(X_train_scaled, y_train)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("Run model_evaluation.py to see results")

if __name__ == "__main__":
    main()
