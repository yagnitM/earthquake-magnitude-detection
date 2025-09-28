import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_df = pd.read_csv('../models/evaluation_results.csv')

plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
models = results_df['Model']

x = range(len(models))
width = 0.2

for i, metric in enumerate(metrics):
    plt.bar([xi + width*i for xi in x], results_df[metric], 
            width=width, label=metric, alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks([xi + width*1.5 for xi in x], models)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/model_comparison.png', dpi=300)
plt.show()
