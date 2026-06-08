# training_results_overview.py

## Overview

This script has been updated to support analyzing 5-fold cross-validation (CV) results in addition to single model validation results. It generates a comprehensive markdown report with:

- Combined statistics across all 5 folds
- Per-fold performance metrics
- Hyperparameter extraction from training scripts
- Per-fold visualizations
- Statistical interpretation

## Usage

### For 5-Fold Cross-Validation Results

```bash
cd /path/to/model/directory

python /path/to/training_results_overview.py \
  --csv-pattern "model_name_fold_*.csv" \
  --png-pattern "model_name_fold_*.png" \
  --executable "/path/to/training/script.sh" \
  --output "model_cv_summary.md" \
  --note "Optional notes about this model"
```

**Example for model_02r7:**

```bash
cd /users/1/lundq163/projects/automated-qc/doc/models/model_02r7

/users/1/lundq163/projects/automated-qc/.venv/bin/python \
  /users/1/lundq163/projects/automated-qc/scripts/utils/training_results_overview.py \
  --csv-pattern "model_02r7_fold_*.csv" \
  --png-pattern "model_02r7_fold_*.png" \
  --executable "/users/1/lundq163/projects/automated-qc/scripts/config/auto-qc-training_model_agate_subset_1024_model_02r_kfold2.sh" \
  --output "model_02r7_cv_summary.md"
```

### For Single Model Results (Original Behavior)

```bash
python training_results_overview.py \
  --csv "/path/to/results.csv" \
  --png "/path/to/results.png" \
  --executable "/path/to/training/script.sh" \
  --output "analysis_results.md"
```

## Arguments

### Required Arguments
- `--executable`: Path to the bash script containing model hyperparameters

### For 5-Fold CV (use both):
- `--csv-pattern`: Glob pattern for fold CSV files (e.g., `"model_02r7_fold_*.csv"`)
- `--png-pattern`: Glob pattern for fold PNG files (e.g., `"model_02r7_fold_*.png"`)

### For Single Model (use both):
- `--csv`: Path to single CSV file with results
- `--png`: Path to single PNG file with visualization

### Optional Arguments
- `--output`: Output markdown filename (default: `analysis_results.md`)
- `--note`: Additional notes to include in the report

## Notes

- The script expects fold CSV files to have a `validation` column (1 = validation, 0 = training)
- Only validation samples are included in the final statistics
- Fold indices in the output are determined by the alphabetical order of CSV files
- PNG files are embedded with relative paths to the markdown output location
- Hyperparameters are extracted using regex pattern matching from the bash script