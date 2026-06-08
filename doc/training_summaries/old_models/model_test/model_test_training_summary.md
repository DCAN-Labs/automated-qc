# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 3 |
| RMSE | 92.6136 |
| Standardized RMSE | 160.4114 |
| Correlation (r) | 0.4041 |
| P-value | 7.3514e-01 |
| Standard Error | 160.4114 |

## Visualization

![QU Motion Score Analysis](../../models/model_test/model_test.png)


## Interpretation

- **Correlation**: 0.4041 indicates a moderate positive relationship between actual and predicted scores.
- **P-value**: 7.3514e-01 is not statistically significant (p ≥ 0.05).
- **Standardized RMSE**: 160.4114 represents the RMSE as a proportion of the standard deviation of the actual values.
## Hyperparameters

### Training Parameters

| Parameter | Value |
|-----------|-------|
| batch_size | 1 |
| epochs | 1 |
| model | Regressor |
| num_workers | 1 |
| split_strategy | stratified |
| train_split | 0.8 |

### Configuration

| Parameter | Value |
|-----------|-------|
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/anat_qc_t1w_t2w_test_subset.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_test/model_test.csv |
| folder | /scratch.global/lundq163/auto_qc_test/ |
| gres | gpu:a100:2 |
| job_name | automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 180g |
| model_save_location | /users/1/lundq163/projects/automated-qc/models/model_test.pt |
| ntasks | 6 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_test/model_test.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 1:00:00 |

