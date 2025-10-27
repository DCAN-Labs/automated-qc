import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Analyze QU motion scores and generate markdown report')
parser.add_argument('--csv', required=True, help='Path to CSV file with QU_motion and predicted_qu_motion_score columns')
parser.add_argument('--png', required=True, help='Path to PNG file to embed in markdown')
parser.add_argument('--output', default='analysis_results.md', help='Output markdown filename (default: analysis_results.md)')

args = parser.parse_args()

# Configuration from arguments
csv_file = args.csv
png_file = args.png
output_md = args.output

# Read the CSV
df = pd.read_csv(csv_file)

# Extract the columns
actual = df['QU_motion']
predicted = df['predicted_qu_motion_score']

# Remove any NaN values
mask = ~(actual.isna() | predicted.isna())
actual = actual[mask]
predicted = predicted[mask]

# Calculate statistics
# 1. RMSE
rmse = np.sqrt(mean_squared_error(actual, predicted))

# 2. Standardized RMSE (RMSE / std of actual values)
std_actual = np.std(actual, ddof=1)
standardized_rmse = rmse / std_actual

# 3. Correlation and p-value
correlation, p_value = stats.pearsonr(actual, predicted)

# 4. Standard Error of the estimate
residuals = actual - predicted
n = len(actual)
standard_error = np.sqrt(np.sum(residuals**2) / (n - 2))

# Print results to console
print(f"Statistics Summary:")
print(f"==================")
print(f"Sample size: {n}")
print(f"RMSE: {rmse:.4f}")
print(f"Standardized RMSE: {standardized_rmse:.4f}")
print(f"Correlation: {correlation:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Standard Error: {standard_error:.4f}")

# Create markdown content
markdown_content = f"""# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Sample Size | {n} |
| RMSE | {rmse:.4f} |
| Standardized RMSE | {standardized_rmse:.4f} |
| Correlation (r) | {correlation:.4f} |
| P-value | {p_value:.4e} |
| Standard Error | {standard_error:.4f} |

## Visualization

![QU Motion Score Analysis]({os.path.basename(png_file)})

## Interpretation

- **Correlation**: {correlation:.4f} indicates a {"strong positive" if correlation > 0.7 else "moderate positive" if correlation > 0.4 else "weak positive" if correlation > 0 else "negative"} relationship between actual and predicted scores.
- **P-value**: {p_value:.4e} {"is statistically significant (p < 0.05)" if p_value < 0.05 else "is not statistically significant (p ≥ 0.05)"}.
- **Standardized RMSE**: {standardized_rmse:.4f} represents the RMSE as a proportion of the standard deviation of the actual values.
"""

# Write to markdown file
with open(output_md, 'w') as f:
    f.write(markdown_content)

print(f"\nMarkdown report saved to: {output_md}")