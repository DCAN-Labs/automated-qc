# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 199 |
| RMSE | 0.9273 |
| Standardized RMSE | 0.9339 |
| Correlation (r) | 0.5554 |
| P-value | 1.6970e-17 |
| Standard Error | 0.9320 |

## Visualization

![QU Motion Score Analysis](../../models/model_02/model_02_fixed_scores.png)


## Interpretation

- **Correlation**: 0.5554 indicates a moderate positive relationship between actual and predicted scores.
- **P-value**: 1.6970e-17 is statistically significant (p < 0.05).
- **Standardized RMSE**: 0.9339 represents the RMSE as a proportion of the standard deviation of the actual values.
## Notes

same as model_02 but includes the updated scores
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
| use_amp | True |

### Configuration

| Parameter | Value |
|-----------|-------|
| cpus_per_task | 16 |
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/raw/auto_qc_t1w_t2w_subset_1024_curated.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_02/model_02_fixed_scores.csv |
| folder | /scratch.global/lundq163/auto_qc/auto_qc_subset_1024_fixed_scores/ |
| gres | gpu:a100:1 |
| job_name | automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 128g |
| model_save_location | /scratch.global/lundq163/auto_qc/auto_qc_model_02_fixed_scores/model_02fs.pt |
| ntasks | 1 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_02/model_02_fixed_scores.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 24:00:00 |

