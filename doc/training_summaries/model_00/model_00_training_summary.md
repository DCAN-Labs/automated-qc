# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 48 |
| RMSE | 1.5558 |
| Standardized RMSE | 1.4331 |
| Correlation (r) | 0.1751 |
| P-value | 2.3389e-01 |
| Standard Error | 1.5892 |

## Visualization

![QU Motion Score Analysis](model_00.png)

## Interpretation

- **Correlation**: 0.1751 indicates a weak positive relationship between actual and predicted scores.
- **P-value**: 2.3389e-01 is not statistically significant (p ≥ 0.05).
- **Standardized RMSE**: 1.4331 represents the RMSE as a proportion of the standard deviation of the actual values.
## Additional Notes

preliminary test case for storing cache on scratch
## Hyperparameters

### Training Parameters

| Parameter | Value |
|-----------|-------|
| batch_size | 4 |
| epochs | 50 |
| lr | 0.001 |
| model | Regressor |
| num_workers | 12 |
| optimizer | Adam |
| scheduler | plateau |
| split_strategy | stratified |
| train_split | 0.8 |

### Configuration

| Parameter | Value |
|-----------|-------|
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/anat_qc_t1w_t2w_subset_256.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_00/model_00.csv |
| folder | /scratch.global/lundq163/auto_qc_subset_256/ |
| model_save_location | /scratch.global/lundq163/auto_qc_model_00/model_00.pt |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_00/model_00.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |

