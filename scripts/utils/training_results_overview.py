import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
import os
import argparse
import re
from pathlib import Path

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
"""
        
if csv_file is not None and png_file is not None:
    # sensitivity and specificity analysis at different thresholds
    thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    markdown_content += "\n## Sensitivity, Specificity, PPV, and NPV Analysis\n\n"
    markdown_content += "| Threshold | Sensitivity | Specificity | PPV | NPV |\n"
    markdown_content += "|-----------|-------------|-------------|-----|-----|\n"
    
    # Store metrics for interpretation
    metrics = []
    
    for threshold in thresholds:
        tp = np.sum((predicted >= threshold) & (actual >= threshold))
        fn = np.sum((predicted < threshold) & (actual >= threshold))
        tn = np.sum((predicted < threshold) & (actual < threshold))
        fp = np.sum((predicted >= threshold) & (actual < threshold))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        markdown_content += f"| {threshold} | {sensitivity:.4f} | {specificity:.4f} | {ppv:.4f} | {npv:.4f} |\n"
        
        metrics.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv
        })
    
    # Add automated interpretation
    markdown_content += "\n### Interpretation\n\n"
    markdown_content += "Higher motion scores indicate more motion artifact (worse image quality). "
    markdown_content += "Thresholds represent the maximum acceptable motion score before flagging an image.\n\n"
    
    # Find balanced threshold (best trade-off between sensitivity and specificity)
    balanced_idx = None
    best_balance = 0
    for i, m in enumerate(metrics[1:], 1):  # Skip threshold 0
        # Use harmonic mean (F1-like) of sensitivity and specificity
        if m['sensitivity'] > 0 and m['specificity'] > 0:
            balance = 2 * (m['sensitivity'] * m['specificity']) / (m['sensitivity'] + m['specificity'])
            if balance > best_balance:
                best_balance = balance
                balanced_idx = i
    
    # Find high sensitivity threshold (catches most motion)
    high_sens_idx = None
    for i, m in enumerate(metrics[1:], 1):
        if m['sensitivity'] >= 0.75 and m['specificity'] >= 0.50:
            high_sens_idx = i
            break
    
    # Find high specificity threshold (minimizes false positives)
    high_spec_idx = None
    for i in range(len(metrics)-1, 0, -1):
        m = metrics[i]
        if m['specificity'] >= 0.90 and m['sensitivity'] >= 0.45:
            high_spec_idx = i
            break
    
    # Generate recommendations
    markdown_content += "**Threshold Recommendations**:\n\n"
    
    if balanced_idx:
        m = metrics[balanced_idx]
        markdown_content += f"- **Balanced (Threshold {m['threshold']})**: "
        markdown_content += f"Detects {m['sensitivity']*100:.0f}% of motion-corrupted images while maintaining "
        markdown_content += f"{m['specificity']*100:.0f}% specificity. "
        markdown_content += f"When flagged, {m['ppv']*100:.0f}% are true positives.\n"
    
    if high_sens_idx and high_sens_idx != balanced_idx:
        m = metrics[high_sens_idx]
        markdown_content += f"- **High Sensitivity (Threshold {m['threshold']})**: "
        markdown_content += f"Catches {m['sensitivity']*100:.0f}% of motion artifacts but rejects "
        markdown_content += f"{(1-m['specificity'])*100:.0f}% of acceptable images.\n"
    
    if high_spec_idx and high_spec_idx != balanced_idx:
        m = metrics[high_spec_idx]
        markdown_content += f"- **High Specificity (Threshold {m['threshold']})**: "
        markdown_content += f"Minimizes false positives ({m['specificity']*100:.0f}% specificity) but only catches "
        markdown_content += f"{m['sensitivity']*100:.0f}% of motion artifacts.\n"
    
    # Add performance summary
    markdown_content += "**Key Observations**:\n\n"
    
    # Check for good PPV at recommended thresholds
    if balanced_idx and metrics[balanced_idx]['ppv'] >= 0.80:
        markdown_content += f"- High precision: When images are flagged at threshold {metrics[balanced_idx]['threshold']}, "
        markdown_content += f"they're truly motion-corrupted {metrics[balanced_idx]['ppv']*100:.0f}% of the time.\n"
    
    # Check specificity range
    # low_spec = min(m['specificity'] for m in metrics[1:3])
    # high_spec = max(m['specificity'] for m in metrics[-2:])
    # markdown_content += f"- Specificity ranges from {low_spec*100:.0f}% (strict quality control) to "
    # markdown_content += f"{high_spec*100:.0f}% (lenient), showing clear threshold-dependent behavior.\n"
    
    # Note about missed cases
    if balanced_idx:
        missed_pct = (1 - metrics[balanced_idx]['sensitivity']) * 100
        if missed_pct > 20:
            markdown_content += f"- At the balanced threshold, {missed_pct:.0f}% of motion artifacts go undetected. "
            markdown_content += "These mild cases may still be diagnostic and will be caught during radiologist review if problematic.\n"

if png_file:
    # Resolve PNG path relative to the output markdown file
    png_path = Path(png_file).expanduser()
    output_path = Path(output_md).expanduser()
    
    # Make PNG path absolute first
    if not png_path.is_absolute():
        png_path = (Path.cwd() / png_path).resolve()
    
    # Check if the PNG file exists
    if not png_path.is_file():
        print(f"Warning: PNG file not found at {png_path}")
        png_file = None  # Set png_file to None to skip visualization
    else:
        # Make output path absolute
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        
        # Calculate relative path from markdown to PNG
        try:
            relative_png_path = os.path.relpath(png_path, output_path.parent)
        except ValueError:
            # Fallback to absolute if on different drives (Windows)
            relative_png_path = str(png_path)
        
        markdown_content += f"""
## Visualization

![QU Motion Score Analysis]({relative_png_path})
"""

if png_file:  # Only add interpretation if PNG was successfully processed
    markdown_content += f"""

## Interpretation

- **Correlation**: {correlation:.4f} indicates a {"strong positive" if correlation > 0.7 else "moderate positive" if correlation > 0.4 else "weak positive" if correlation > 0 else "negative"} relationship between actual and predicted scores.
- **P-value**: {p_value:.4e} {"is statistically significant (p < 0.05)" if p_value < 0.05 else "is not statistically significant (p ≥ 0.05)"}.
- **Standardized RMSE**: {standardized_rmse:.4f} represents the RMSE as a proportion of the standard deviation of the actual values.
- **Standard Error**: {standard_error:.4f} provides an estimate of the average distance that the observed values fall from the regression line.
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