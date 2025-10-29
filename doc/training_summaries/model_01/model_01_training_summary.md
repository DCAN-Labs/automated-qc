# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 48 |
| RMSE | 1.1410 |
| Standardized RMSE | 1.0511 |
| Correlation (r) | 0.5134 |
| P-value | 1.9031e-04 |
| Standard Error | 1.1655 |

## Visualization

![QU Motion Score Analysis](model_01.png)

## Interpretation

- **Correlation**: 0.5134 indicates a moderate positive relationship between actual and predicted scores.
- **P-value**: 1.9031e-04 is statistically significant (p < 0.05).
- **Standardized RMSE**: 1.0511 represents the RMSE as a proportion of the standard deviation of the actual values.
## Notes

includes PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, double epochs and batch-size from model_00, same prelim set
## Hyperparameters

### Training Parameters

| Parameter | Value |
|-----------|-------|
| batch_size | 8 |
| epochs | 100 |
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
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_01/model_01.csv |
| folder | /scratch.global/lundq163/auto_qc_subset_256/ |
| model_save_location | /scratch.global/lundq163/auto_qc_model_01/model_01.pt |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_01/model_01.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |

