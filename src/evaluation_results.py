import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Paths
RESULTS_PATH = '../models/evaluation_results.csv'
RESULTS_DIR = '../results/'

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_results():
    """Load evaluation results"""
    print("Loading evaluation results...")
    try:
        df = pd.read_csv(RESULTS_PATH)
        print(f"✓ Loaded results for {len(df)} models")
        print(f"  Columns: {', '.join(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {RESULTS_PATH} not found!")
        print("   Please run model_evaluation.py first.")
        return None


def plot_core_metrics_comparison(df):
    """Plot RMSE, MAE, and R² comparison"""
    print("\n📊 Creating core metrics comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define metrics and their properties
    metrics_info = [
        ('RMSE', 'Root Mean Squared Error', 'Lower is Better', True),
        ('MAE', 'Mean Absolute Error', 'Lower is Better', True),
        ('R2_Score', 'R² Score', 'Higher is Better', False)
    ]
    
    for ax, (metric, title, subtitle, lower_better) in zip(axes, metrics_info):
        # Sort data
        df_sorted = df.sort_values(metric, ascending=lower_better)
        
        # Color: best model in green, others in blue
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(df_sorted))]
        
        # Create horizontal bar chart
        bars = ax.barh(df_sorted['Model'], df_sorted[metric], color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df_sorted[metric])):
            ax.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\n({subtitle})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Highlight best value
        best_val = df_sorted[metric].iloc[0]
        ax.axvline(best_val, color='#27ae60', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'core_metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_tolerance_analysis(df):
    """Plot tolerance/accuracy analysis"""
    print("\n🎯 Creating tolerance analysis plot...")
    
    # Extract tolerance columns
    tolerance_cols = [col for col in df.columns if col.startswith('Within_')]
    
    if not tolerance_cols:
        print("⚠ No tolerance columns found, skipping...")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for grouped bar chart
    x = np.arange(len(df))
    width = 0.15
    
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
    
    for i, col in enumerate(tolerance_cols):
        tolerance_label = col.replace('Within_', '').replace('±', '±')
        offset = width * (i - len(tolerance_cols)/2 + 0.5)
        bars = ax.bar(x + offset, df[col], width, label=tolerance_label, 
                     color=colors[i % len(colors)], alpha=0.8, edgecolor='black')
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Only show if bar is tall enough
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Accuracy Within Tolerance Thresholds', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend(title='Tolerance', loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'tolerance_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_error_statistics(df):
    """Plot error statistics (Mean Error, Std Error, Max Error)"""
    print("\n📉 Creating error statistics plot...")
    
    error_cols = ['Mean_Error', 'Std_Error', 'Max_Error']
    
    # Check if columns exist
    available_cols = [col for col in error_cols if col in df.columns]
    if not available_cols:
        print("⚠ No error statistic columns found, skipping...")
        return
    
    fig, axes = plt.subplots(1, len(available_cols), figsize=(6*len(available_cols), 6))
    
    if len(available_cols) == 1:
        axes = [axes]
    
    titles = {
        'Mean_Error': 'Mean Prediction Error\n(Bias)',
        'Std_Error': 'Standard Deviation of Errors\n(Variance)',
        'Max_Error': 'Maximum Absolute Error\n(Worst Case)'
    }
    
    for ax, col in zip(axes, available_cols):
        df_sorted = df.sort_values(col, key=lambda x: abs(x) if col == 'Mean_Error' else x)
        
        # Color coding: green for better, red for worse
        if col == 'Mean_Error':
            colors = ['#2ecc71' if abs(v) < 0.1 else '#e74c3c' if abs(v) > 0.3 else '#f39c12' 
                     for v in df_sorted[col]]
        else:
            max_val = df_sorted[col].max()
            colors = ['#2ecc71' if v < max_val*0.5 else '#e74c3c' if v > max_val*0.8 else '#f39c12' 
                     for v in df_sorted[col]]
        
        bars = ax.barh(df_sorted['Model'], df_sorted[col], color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, df_sorted[col]):
            label_x = bar.get_width() + (bar.get_width()*0.02 if value > 0 else bar.get_width()*-0.02)
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                   f'{value:+.3f}' if col == 'Mean_Error' else f'{value:.3f}',
                   va='center', fontsize=9, fontweight='bold')
        
        if col == 'Mean_Error':
            ax.axvline(0, color='black', linestyle='-', linewidth=1.5)
        
        ax.set_xlabel(col.replace('_', ' ') + ' (magnitudes)', fontsize=11, fontweight='bold')
        ax.set_title(titles[col], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'error_statistics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_overall_ranking(df):
    """Create overall model ranking visualization"""
    print("\n🏆 Creating overall ranking plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize metrics for ranking (0-1 scale)
    df_norm = df.copy()
    
    # Lower is better: RMSE, MAE
    for col in ['RMSE', 'MAE']:
        if col in df.columns:
            df_norm[f'{col}_norm'] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # Higher is better: R2_Score
    if 'R2_Score' in df.columns:
        df_norm['R2_Score_norm'] = (df['R2_Score'] - df['R2_Score'].min()) / (df['R2_Score'].max() - df['R2_Score'].min())
    
    # Calculate overall score (average of normalized metrics)
    norm_cols = [col for col in df_norm.columns if col.endswith('_norm')]
    df_norm['Overall_Score'] = df_norm[norm_cols].mean(axis=1)
    
    # Sort by overall score
    df_norm = df_norm.sort_values('Overall_Score', ascending=False)
    
    # Color gradient: best (green) to worst (red)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_norm)))
    
    bars = ax.barh(df_norm['Model'], df_norm['Overall_Score'] * 100, 
                   color=colors, alpha=0.8, edgecolor='black')
    
    # Add score labels
    for bar, score in zip(bars, df_norm['Overall_Score'] * 100):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f'{score:.1f}', va='center', fontsize=10, fontweight='bold')
    
    # Add rank numbers
    for i, (bar, model) in enumerate(zip(bars, df_norm['Model']), 1):
        ax.text(2, bar.get_y() + bar.get_height()/2,
               f'#{i}', va='center', ha='left', fontsize=11, 
               fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.set_xlabel('Overall Score (0-100)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Model Ranking\n(Based on RMSE, MAE, and R²)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'overall_ranking.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_radar_chart(df):
    """Create radar chart comparing models across multiple metrics"""
    print("\n🎯 Creating radar chart...")
    
    # Select key metrics
    metrics = ['RMSE', 'MAE', 'R2_Score']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) < 3:
        print("⚠ Not enough metrics for radar chart, skipping...")
        return
    
    # Normalize metrics (0-1, where 1 is best)
    df_norm = df.copy()
    for col in ['RMSE', 'MAE']:
        if col in df.columns:
            df_norm[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
    
    if 'R2_Score' in df.columns:
        df_norm['R2_Score'] = (df['R2_Score'] - df['R2_Score'].min()) / (df['R2_Score'].max() - df['R2_Score'].min() + 1e-10)
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (_, row) in enumerate(df_norm.iterrows()):
        values = [row[m] for m in available_metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], 
               color=colors[idx % len(colors)], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax.set_title('Model Performance Radar Chart\n(Normalized Metrics)', 
                fontsize=14, fontweight='bold', pad=30)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'radar_chart.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_summary_dashboard(df):
    """Create a comprehensive summary dashboard"""
    print("\n📊 Creating summary dashboard...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. RMSE comparison
    ax1 = fig.add_subplot(gs[0, 0])
    df_sorted = df.sort_values('RMSE')
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(df_sorted))]
    ax1.barh(df_sorted['Model'], df_sorted['RMSE'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('RMSE', fontweight='bold')
    ax1.set_title('Root Mean Squared Error', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. MAE comparison
    ax2 = fig.add_subplot(gs[0, 1])
    df_sorted = df.sort_values('MAE')
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(df_sorted))]
    ax2.barh(df_sorted['Model'], df_sorted['MAE'], color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('MAE', fontweight='bold')
    ax2.set_title('Mean Absolute Error', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. R² comparison
    ax3 = fig.add_subplot(gs[0, 2])
    df_sorted = df.sort_values('R2_Score', ascending=False)
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(df_sorted))]
    ax3.barh(df_sorted['Model'], df_sorted['R2_Score'], color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('R² Score', fontweight='bold')
    ax3.set_title('R² Score (Higher is Better)', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Tolerance comparison (if available)
    tolerance_cols = [col for col in df.columns if 'Within_±0.5' in col or 'Within_±1.0' in col]
    if tolerance_cols:
        ax4 = fig.add_subplot(gs[1, :])
        x = np.arange(len(df))
        width = 0.35
        
        if 'Within_±0.5' in df.columns:
            ax4.bar(x - width/2, df['Within_±0.5'], width, label='Within ±0.5', 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
        if 'Within_±1.0' in df.columns:
            ax4.bar(x + width/2, df['Within_±1.0'], width, label='Within ±1.0', 
                   color='#3498db', alpha=0.8, edgecolor='black')
        
        ax4.set_xlabel('Model', fontweight='bold')
        ax4.set_ylabel('Percentage (%)', fontweight='bold')
        ax4.set_title('Prediction Accuracy Within Tolerance', fontweight='bold', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Best model summary
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    best_rmse_idx = df['RMSE'].idxmin()
    best_mae_idx = df['MAE'].idxmin()
    best_r2_idx = df['R2_Score'].idxmax()
    
    summary_text = f"""
    🏆 BEST MODELS SUMMARY
    
    ✓ Lowest RMSE:     {df.loc[best_rmse_idx, 'Model']}  ({df.loc[best_rmse_idx, 'RMSE']:.4f} magnitudes)
    ✓ Lowest MAE:      {df.loc[best_mae_idx, 'Model']}  ({df.loc[best_mae_idx, 'MAE']:.4f} magnitudes)
    ✓ Highest R²:      {df.loc[best_r2_idx, 'Model']}  ({df.loc[best_r2_idx, 'R2_Score']:.4f})
    
    💡 Interpretation:
       • RMSE < 0.5: Excellent prediction accuracy
       • MAE < 0.3: Very good average error
       • R² > 0.7: Strong explanatory power
    """
    
    ax5.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    fig.suptitle('Earthquake Magnitude Prediction - Model Evaluation Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'summary_dashboard.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    print("="*60)
    print("📊 EARTHQUAKE PREDICTION - RESULTS VISUALIZATION")
    print("="*60)
    
    # Load results
    df = load_results()
    
    if df is None:
        return
    
    print(f"\n📈 Creating visualizations for {len(df)} models...")
    
    # Create all visualizations
    try:
        plot_core_metrics_comparison(df)
        plot_tolerance_analysis(df)
        plot_error_statistics(df)
        plot_overall_ranking(df)
        plot_radar_chart(df)
        create_summary_dashboard(df)
        
        print("\n" + "="*60)
        print("✅ ALL VISUALIZATIONS CREATED!")
        print("="*60)
        print(f"\n📁 Saved in: {RESULTS_DIR}")
        print("   • core_metrics_comparison.png")
        print("   • tolerance_analysis.png")
        print("   • error_statistics.png")
        print("   • overall_ranking.png")
        print("   • radar_chart.png")
        print("   • summary_dashboard.png")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()