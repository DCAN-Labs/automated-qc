import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
import os
import argparse
import re

# Set up argument parser
parser = argparse.ArgumentParser(description='Analyze QU motion scores and generate markdown report')
parser.add_argument('--csv', required=False, help='Path to CSV file with QU_motion and predicted_qu_motion_score columns (optional if executable is provided)')
parser.add_argument('--png', required=False, help='Path to PNG file to embed in markdown (optional if executable is provided)')
parser.add_argument('--executable', required=True, help='Path to the executable bash script that specifies the model hyperparameters used during training')
parser.add_argument('--output', default='analysis_results.md', help='Output markdown filename (default: analysis_results.md)')
parser.add_argument('--note', default='', help='Additional notes to include in the markdown report (optional)')

args = parser.parse_args()

# Configuration from arguments
csv_file = args.csv
png_file = args.png
output_md = args.output
executable_script = args.executable

def extract_hyperparameters(script_content):
    """
    Extract hyperparameters from the executable script content.
    Supports:
      --flag
      --flag value
      --flag "value with spaces"
      --flag='value with spaces'
      --flag=value
    Flags without values are set to True.
    """
    hyperparameters = {}

    pattern = re.compile(
        r'''--(?P<name>[a-zA-Z0-9-]+)
            (?:
                (?:=|\s+)
                (?:
                    "(?P<dq>[^"]*)"           # double-quoted value
                    |
                    '(?P<sq>[^']*)'           # single-quoted value
                    |
                    (?P<uvalue>(?!-{2})\S+)   # unquoted value not starting with --
                )
            )?
        ''',
        re.VERBOSE
    )

    for m in pattern.finditer(script_content):
        name = m.group('name').replace('-', '_')
        value = m.group('dq') or m.group('sq') or m.group('uvalue')

        if value is None:
            param_value = True
        else:
            s = value.strip()
            low = s.lower()
            if low in ('true', 'yes', 'on'):
                param_value = True
            elif low in ('false', 'no', 'off'):
                param_value = False
            else:
                # Try numeric conversion
                try:
                    if re.fullmatch(r'[-+]?\d+', s):
                        param_value = int(s)
                    else:
                        param_value = float(s)
                except ValueError:
                    param_value = s

        hyperparameters[name] = param_value

    return hyperparameters

def format_hyperparameters_markdown(hyperparameters):
    """
    Format hyperparameters as a markdown table.
    """
    if not hyperparameters:
        return "No hyperparameters found in the script."
    
    # Separate into training-related and other parameters
    training_params = {}
    other_params = {}
    
    training_keywords = ['model', 'lr', 'scheduler', 'batch_size', 'epochs', 
                         'optimizer', 'split_strategy', 'train_split', 'num_workers', 
                         'use_amp']
    
    for key, value in hyperparameters.items():
        if key == 'model_save_location':
            other_params[key] = value
        elif any(keyword in key for keyword in training_keywords):
            training_params[key] = value
        else:
            other_params[key] = value
    
    markdown = "## Hyperparameters\n\n"
    
    if training_params:
        markdown += "### Training Parameters\n\n"
        markdown += "| Parameter | Value |\n"
        markdown += "|-----------|-------|\n"
        for key, value in sorted(training_params.items()):
            markdown += f"| {key} | {value} |\n"
        markdown += "\n"
    
    if other_params:
        markdown += "### Configuration\n\n"
        markdown += "| Parameter | Value |\n"
        markdown += "|-----------|-------|\n"
        for key, value in sorted(other_params.items()):
            markdown += f"| {key} | {value} |\n"
        markdown += "\n"
    
    return markdown

# Extract and display hyperparameters if executable is provided
hyperparameters = {}
if executable_script:
    if os.path.exists(executable_script):
        with open(executable_script, 'r') as f:
            script_content = f.read()
        
        hyperparameters = extract_hyperparameters(script_content)
        
        if hyperparameters:
            print(f"Using hyperparameters from: {executable_script}")
            print(f"\nExtracted {len(hyperparameters)} hyperparameters:")
            for key, value in sorted(hyperparameters.items()):
                print(f"  {key}: {value}")
        else:
            print(f"Warning: No hyperparameters found in {executable_script}")
    else:
        print(f"Warning: Executable script not found at {executable_script}")

if csv_file is not None and png_file is not None:
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
else:
    print("CSV file or PNG file not provided; skipping statistical analysis.")
    n = rmse = standardized_rmse = correlation = p_value = standard_error = 0
    png_file = ""

# Create markdown content
markdown_content = f"""# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | {n} |
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
    if args.note:
        f.write(f"## Notes\n\n{args.note}\n")
    # Append hyperparameters section if available
    if hyperparameters:
        hyperparams_md = format_hyperparameters_markdown(hyperparameters)
        f.write(hyperparams_md)

print(f"\nMarkdown report saved to: {output_md}")