import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = {}

with open('results/baseline_results.txt', 'r') as f:
    lines = f.readlines()
    results['Linear Regression'] = {
        'RMSE': float(lines[0].split(': ')[1]),
        'MAE': float(lines[1].split(': ')[1]),
        'R2': float(lines[2].split(': ')[1])
    }

with open('results/rf_results.txt', 'r') as f:
    lines = f.readlines()
    results['Random Forest'] = {
        'RMSE': float(lines[3].split(': ')[1]),
        'MAE': float(lines[4].split(': ')[1]),
        'R2': float(lines[5].split(': ')[1])
    }

with open('results/xgb_results.txt', 'r') as f:
    lines = f.readlines()
    results['XGBoost'] = {
        'RMSE': float(lines[3].split(': ')[1]),
        'MAE': float(lines[4].split(': ')[1]),
        'R2': float(lines[5].split(': ')[1])
    }

df = pd.DataFrame(results).T
print("=== MODEL COMPARISON SUMMARY ===")
print(df.round(4))
print()

# Find best model
best_rmse = df['RMSE'].idxmin()
best_r2 = df['R2'].idxmax()

print(f"Best RMSE: {best_rmse} ({df.loc[best_rmse, 'RMSE']:.4f})")
print(f"Best R2: {best_r2} ({df.loc[best_r2, 'R2']:.4f})")

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

colors = ['#3498db', '#2ecc71', '#e74c3c']

df['RMSE'].plot(kind='bar', ax=axes[0], color=colors)
axes[0].set_title('RMSE (lower is better)', fontsize=14)
axes[0].set_ylabel('RMSE')
axes[0].tick_params(axis='x', rotation=45)

df['MAE'].plot(kind='bar', ax=axes[1], color=colors)
axes[1].set_title('MAE (lower is better)', fontsize=14)
axes[1].set_ylabel('MAE')
axes[1].tick_params(axis='x', rotation=45)

df['R2'].plot(kind='bar', ax=axes[2], color=colors)
axes[2].set_title('R-squared (higher is better)', fontsize=14)
axes[2].set_ylabel('R-squared')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/final_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n Final comparison chart saved to results/final_comparison.png")

# Write summary report
with open('results/summary_report.txt', 'w') as f:
    f.write("=== MODEL COMPARISON SUMMARY REPORT ===\n\n")
    f.write("Model Performance:\n")
    f.write(df.round(4).to_string())
    f.write("\n\n")
    f.write(f"Best performing model (RMSE): {best_rmse}\n")
    f.write(f"Best performing model (R2): {best_r2}\n")
    f.write("\nRecommendation: Use XGBoost for best predictive performance.\n")
    f.write("Random Forest is a good alternative if interpretability is important.\n")

print(" Summary report saved to results/summary_report.txt")
