import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

MODELS_DIR = '../models/'

def load_models_and_data():
    print("Loading models and test data...")
    models = {}
    
    try:
        models['Logistic_Regression'] = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
        print(" Loaded Logistic Regression")
    except FileNotFoundError:
        print("✗ Logistic Regression model not found!")
    
    try:
        models['Random_Forest'] = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
        print(" Loaded Random Forest")
    except FileNotFoundError:
        print("✗ Random Forest model not found!")  
    
    if not models:
        print("No models found! Please run baseline_models.py first.")
        return None, None, None
    try:
        test_data = joblib.load(os.path.join(MODELS_DIR, 'test_data.pkl'))
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        print(f" Test data loaded: {X_test.shape}")
    except FileNotFoundError:
        print("Test data not found! Please run baseline_models.py first.")
        return None, None, None
    
    return models, X_test, y_test

def evaluate_models(models, X_test, y_test):
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    results = []
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            result = {
                'Model': name.replace('_', ' '),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            }
            results.append(result)
            
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
        except Exception as e:
            print(f"✗ Error evaluating {name}: {e}")
    
    return results

def print_comparison_table(results):
    if not results:
        print("No results to display!")
        return
        
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    display_df = df_results.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1_Score']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    if len(results) > 1:
        best_idx = df_results['F1_Score'].idxmax()
        best_model = results[best_idx]['Model']
        best_f1 = results[best_idx]['F1_Score']
        print(f"\nBEST MODEL: {best_model} (F1-Score: {best_f1:.4f})")
    else:
        print(f"\n📊 Single model result: {results[0]['Model']}")

def detailed_classification_report(models, X_test, y_test):
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*60)
    
    class_names = ['Low (0-2)', 'Medium (2-4)', 'High (4+)']
    
    for name, model in models.items():
        print(f"\n{name.replace('_', ' ')} Classification Report:")
        print("-" * 50)
        
        try:
            y_pred = model.predict(X_test)
            report = classification_report(
                y_test, y_pred, 
                target_names=class_names, 
                digits=4, 
                zero_division=0
            )
            print(report)
        except Exception as e:
            print(f"✗ Error generating report for {name}: {e}")

def save_results(results):
    if not results:
        print("No results to save!")
        return
        
    print("\nSaving results...")
    
    try:
        df_results = pd.DataFrame(results)
        results_path = os.path.join(MODELS_DIR, 'evaluation_results.csv')
        df_results.to_csv(results_path, index=False)
        print(f" Results saved to {results_path}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")

def main():
    print("EARTHQUAKE MAGNITUDE CLASSIFICATION - MODEL EVALUATION")
    print("=" * 60)
    models, X_test, y_test = load_models_and_data()
    
    if models is None:
        print("Cannot proceed without models and test data!")
        return
    results = evaluate_models(models, X_test, y_test)
    
    if not results:
        print("No evaluation results generated!")
        return
    print_comparison_table(results)
    detailed_classification_report(models, X_test, y_test)

    save_results(results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED!")

if __name__ == "__main__":
    main()
