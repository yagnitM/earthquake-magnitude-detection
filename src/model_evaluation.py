import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


MODELS_DIR = '../models/'
RESULTS_DIR = '../results/'


os.makedirs(RESULTS_DIR, exist_ok=True)


def load_models_and_data():
    """Load trained regression models and test data"""
    print("Loading models and test data...")
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
            print(f"✗ {name} not found (skipping)")
    
    if not models:
        print("❌ No models found! Please run baseline_models.py first.")
        return None, None, None
    
    try:
        test_data = joblib.load(os.path.join(MODELS_DIR, 'test_data.pkl'))
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        print(f"✓ Test data loaded: {X_test.shape[0]} samples")
        print(f"✓ Magnitude range: {y_test.min():.2f} to {y_test.max():.2f}")
    except FileNotFoundError:
        print("❌ Test data not found! Please run baseline_models.py first.")
        return None, None, None
    
    return models, X_test, y_test


def calculate_tolerance_metrics(y_true, y_pred):
    """Calculate percentage of predictions within tolerance thresholds"""
    errors = np.abs(y_true - y_pred)
    
    tolerances = {
        '±0.1': np.mean(errors <= 0.1) * 100,
        '±0.2': np.mean(errors <= 0.2) * 100,
        '±0.3': np.mean(errors <= 0.3) * 100,
        '±0.5': np.mean(errors <= 0.5) * 100,
        '±1.0': np.mean(errors <= 1.0) * 100,
        '±1.5': np.mean(errors <= 1.5) * 100,
    }
    
    return tolerances


def calculate_magnitude_bin_performance(y_true, y_pred):
    """Evaluate performance across different magnitude ranges"""
    bins = [
        (0, 1, 'Very Low (0-1)'),
        (1, 2, 'Low (1-2)'),
        (2, 3, 'Moderate (2-3)'),
        (3, 4, 'Medium (3-4)'),
        (4, 5, 'High (4-5)'),
        (5, 10, 'Very High (5+)')
    ]
    
    bin_performance = []
    
    for min_mag, max_mag, label in bins:
        mask = (y_true >= min_mag) & (y_true < max_mag)
        if mask.sum() > 0:
            bin_true = y_true[mask]
            bin_pred = y_pred[mask]
            
            bin_performance.append({
                'Range': label,
                'Count': mask.sum(),
                'MAE': mean_absolute_error(bin_true, bin_pred),
                'RMSE': np.sqrt(mean_squared_error(bin_true, bin_pred)),
                'Mean_Error': np.mean(bin_pred - bin_true)
            })
    
    return pd.DataFrame(bin_performance)


def evaluate_models(models, X_test, y_test):
    """Evaluate all regression models with comprehensive metrics"""
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION RESULTS (REGRESSION)")
    print("="*60)
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        print(f"\n🔄 Evaluating {name.replace('_', ' ')}...")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            
            # Calculate core regression metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            median_ae = median_absolute_error(y_test, y_pred)
            
            # Calculate tolerance metrics
            tolerances = calculate_tolerance_metrics(y_test, y_pred)
            
            # Calculate error statistics
            errors = y_pred - y_test
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            max_error = np.max(np.abs(errors))
            
            # Store results
            result = {
                'Model': name.replace('_', ' '),
                'RMSE': rmse,
                'MAE': mae,
                'Median_AE': median_ae,
                'R2_Score': r2,
                'Mean_Error': mean_error,
                'Std_Error': std_error,
                'Max_Error': max_error,
                **{f'Within_{k}': v for k, v in tolerances.items()}
            }
            results.append(result)
            
            # Print key metrics
            print(f"  RMSE:            {rmse:.4f} magnitudes")
            print(f"  MAE:             {mae:.4f} magnitudes")
            print(f"  Median AE:       {median_ae:.4f} magnitudes")
            print(f"  R² Score:        {r2:.4f}")
            print(f"  Within ±0.5:     {tolerances['±0.5']:.1f}%")
            print(f"  Within ±1.0:     {tolerances['±1.0']:.1f}%")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    return results, predictions


def print_comparison_table(results):
    """Display model comparison table"""
    if not results:
        print("❌ No results to display!")
        return
    
    print("\n" + "="*60)
    print("📈 MODEL COMPARISON TABLE")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    # Format key metrics for display
    display_cols = ['Model', 'RMSE', 'MAE', 'R2_Score', 'Within_±0.5', 'Within_±1.0']
    display_df = df_results[display_cols].copy()
    
    for col in ['RMSE', 'MAE', 'R2_Score']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    for col in ['Within_±0.5', 'Within_±1.0']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")
    
    print(display_df.to_string(index=False))
    
    # Identify best models
    if len(results) > 1:
        print("\n🏆 BEST MODELS:")
        best_rmse_idx = df_results['RMSE'].idxmin()
        best_r2_idx = df_results['R2_Score'].idxmax()
        best_mae_idx = df_results['MAE'].idxmin()
        
        print(f"  Lowest RMSE:     {results[best_rmse_idx]['Model']} ({results[best_rmse_idx]['RMSE']:.4f})")
        print(f"  Highest R²:      {results[best_r2_idx]['Model']} ({results[best_r2_idx]['R2_Score']:.4f})")
        print(f"  Lowest MAE:      {results[best_mae_idx]['Model']} ({results[best_mae_idx]['MAE']:.4f})")


def print_detailed_tolerance_analysis(results):
    """Display detailed tolerance analysis"""
    print("\n" + "="*60)
    print("🎯 TOLERANCE ANALYSIS")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    tolerance_cols = [col for col in df_results.columns if col.startswith('Within_')]
    
    print("\nPercentage of predictions within tolerance:")
    print("-" * 60)
    
    for idx, row in df_results.iterrows():
        print(f"\n{row['Model']}:")
        for col in tolerance_cols:
            tolerance = col.replace('Within_', '')
            print(f"  {tolerance:8s}: {row[col]:5.1f}%")


def analyze_by_magnitude_bins(models, X_test, y_test):
    """Analyze model performance across magnitude ranges"""
    print("\n" + "="*60)
    print("📊 PERFORMANCE BY MAGNITUDE RANGE")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{name.replace('_', ' ')}:")
        print("-" * 60)
        
        try:
            y_pred = model.predict(X_test)
            bin_performance = calculate_magnitude_bin_performance(y_test, y_pred)
            
            display_df = bin_performance.copy()
            display_df['MAE'] = display_df['MAE'].apply(lambda x: f"{x:.4f}")
            display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"{x:.4f}")
            display_df['Mean_Error'] = display_df['Mean_Error'].apply(lambda x: f"{x:+.4f}")
            
            print(display_df.to_string(index=False))
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")


def create_visualizations(models, X_test, y_test, predictions):
    """Create visualization plots for model evaluation"""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS...")
    print("="*60)
    
    try:
        # 1. Predictions vs Actual scatter plot
        plt.figure(figsize=(15, 10))
        
        n_models = len(models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        for idx, (name, y_pred) in enumerate(predictions.items(), 1):
            plt.subplot(n_rows, n_cols, idx)
            
            plt.scatter(y_test, y_pred, alpha=0.5, s=20)
            plt.plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
            
            plt.xlabel('Actual Magnitude')
            plt.ylabel('Predicted Magnitude')
            plt.title(f'{name.replace("_", " ")}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        scatter_path = os.path.join(RESULTS_DIR, 'predictions_vs_actual.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {scatter_path}")
        plt.close()
        
        # 2. Residual plots
        plt.figure(figsize=(15, 10))
        
        for idx, (name, y_pred) in enumerate(predictions.items(), 1):
            plt.subplot(n_rows, n_cols, idx)
            
            residuals = y_pred - y_test
            plt.scatter(y_pred, residuals, alpha=0.5, s=20)
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            
            plt.xlabel('Predicted Magnitude')
            plt.ylabel('Residual (Predicted - Actual)')
            plt.title(f'{name.replace("_", " ")} Residuals')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        residual_path = os.path.join(RESULTS_DIR, 'residual_plots.png')
        plt.savefig(residual_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {residual_path}")
        plt.close()
        
        # 3. Error distribution
        plt.figure(figsize=(15, 10))
        
        for idx, (name, y_pred) in enumerate(predictions.items(), 1):
            plt.subplot(n_rows, n_cols, idx)
            
            errors = y_pred - y_test
            plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
            
            plt.xlabel('Prediction Error (Magnitude)')
            plt.ylabel('Frequency')
            plt.title(f'{name.replace("_", " ")} Error Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        error_dist_path = os.path.join(RESULTS_DIR, 'error_distribution.png')
        plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {error_dist_path}")
        plt.close()
        
        # 4. Model comparison bar chart
        df_results = pd.DataFrame([{
            'Model': name,
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        } for name, y_pred in predictions.items()])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['RMSE', 'MAE', 'R2']
        titles = ['Root Mean Squared Error (Lower is Better)',
                 'Mean Absolute Error (Lower is Better)',
                 'R² Score (Higher is Better)']
        
        for ax, metric, title in zip(axes, metrics, titles):
            df_sorted = df_results.sort_values(metric, ascending=(metric != 'R2'))
            colors = ['green' if i == 0 else 'skyblue' for i in range(len(df_sorted))]
            
            ax.barh(df_sorted['Model'], df_sorted[metric], color=colors)
            ax.set_xlabel(metric)
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        comparison_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {comparison_path}")
        plt.close()
        
    except Exception as e:
        print(f"⚠ Warning: Could not create visualizations: {str(e)}")
        print("  (This is optional - results are still saved)")


def save_results(results):
    """Save evaluation results to files"""
    if not results:
        print("❌ No results to save!")
        return
    
    print("\n" + "="*60)
    print("💾 SAVING RESULTS...")
    print("="*60)
    
    try:
        df_results = pd.DataFrame(results)
        
        # Save main results
        results_path = os.path.join(MODELS_DIR, 'evaluation_results.csv')
        df_results.to_csv(results_path, index=False)
        print(f"✓ Results saved: {results_path}")
        
        # Save summary report
        summary_path = os.path.join(RESULTS_DIR, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EARTHQUAKE MAGNITUDE PREDICTION - EVALUATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write("MODEL COMPARISON:\n")
            f.write("-"*60 + "\n")
            f.write(df_results.to_string(index=False))
            f.write("\n\n")
            
            f.write("BEST MODELS:\n")
            f.write("-"*60 + "\n")
            best_rmse_idx = df_results['RMSE'].idxmin()
            best_r2_idx = df_results['R2_Score'].idxmax()
            best_mae_idx = df_results['MAE'].idxmin()
            
            f.write(f"Lowest RMSE:  {results[best_rmse_idx]['Model']} ({results[best_rmse_idx]['RMSE']:.4f})\n")
            f.write(f"Highest R²:   {results[best_r2_idx]['Model']} ({results[best_r2_idx]['R2_Score']:.4f})\n")
            f.write(f"Lowest MAE:   {results[best_mae_idx]['Model']} ({results[best_mae_idx]['MAE']:.4f})\n")
        
        print(f"✓ Summary saved: {summary_path}")
        
    except Exception as e:
        print(f"✗ Error saving results: {str(e)}")


def main():
    print("="*60)
    print("🌍 EARTHQUAKE MAGNITUDE PREDICTION - MODEL EVALUATION")
    print("="*60)
    
    # Load models and data
    models, X_test, y_test = load_models_and_data()
    
    if models is None:
        print("\n❌ Cannot proceed without models and test data!")
        return
    
    # Evaluate models
    results, predictions = evaluate_models(models, X_test, y_test)
    
    if not results:
        print("\n❌ No evaluation results generated!")
        return
    
    # Display results
    print_comparison_table(results)
    print_detailed_tolerance_analysis(results)
    analyze_by_magnitude_bins(models, X_test, y_test)
    
    # Create visualizations
    create_visualizations(models, X_test, y_test, predictions)
    
    # Save results
    save_results(results)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETED!")
    print("="*60)
    print(f"\n📁 Results saved in:")
    print(f"   {MODELS_DIR}evaluation_results.csv")
    print(f"   {RESULTS_DIR}evaluation_summary.txt")
    print(f"   {RESULTS_DIR}predictions_vs_actual.png")
    print(f"   {RESULTS_DIR}residual_plots.png")
    print(f"   {RESULTS_DIR}error_distribution.png")
    print(f"   {RESULTS_DIR}model_comparison.png")
    print("="*60)


if __name__ == "__main__":
    main()