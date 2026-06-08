# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 807 |
| RMSE | 0.8524 |
| Standardized RMSE | 0.9707 |
| Correlation (r) | 0.4406 |
| P-value | 1.1643e-39 |
| Standard Error | 0.8534 |

## Visualization

![QU Motion Score Analysis](../../models/model_03rs/model_03rs.png)


## Interpretation

- **Correlation**: 0.4406 indicates a moderate positive relationship between actual and predicted scores.
- **P-value**: 1.1643e-39 is statistically significant (p < 0.05).
- **Standardized RMSE**: 0.9707 represents the RMSE as a proportion of the standard deviation of the actual values.
## Notes

Utilizes larger subset, preprocessing involves registration then skullstripping, and removes majority of scans where preproc was poor. Run time was 17h 39m 59s
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
| use_amp | True |

### Configuration

| Parameter | Value |
|-----------|-------|
| cpus_per_task | 16 |
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/raw/auto_qc_t1w_t2w_subset_4096rs.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_03rs/model_03rs.csv |
| folder | /scratch.global/lundq163/auto_qc_subset_4096rs/ |
| gres | gpu:a100:1 |
| job_name | automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 128g |
| model_save_location | /scratch.global/lundq163/auto_qc_model_03rs/model_03rs.pt |
| ntasks | 1 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_03rs/model_03rs.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 24:00:00 |

