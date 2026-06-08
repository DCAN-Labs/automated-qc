# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 206 |
| RMSE | 0.6962 |
| Standardized RMSE | 0.6604 |
| Correlation (r) | 0.7612 |
| P-value | 3.1768e-40 |
| Standard Error | 0.6996 |

## Visualization

![QU Motion Score Analysis](../../models/model_02/model_02.png)


## Interpretation

- **Correlation**: 0.7612 indicates a strong positive relationship between actual and predicted scores.
- **P-value**: 3.1768e-40 is statistically significant (p < 0.05).
- **Standardized RMSE**: 0.6604 represents the RMSE as a proportion of the standard deviation of the actual values.
## Notes

fixed PYTORCH_ALLOC_CONF, uses larger subset: ~10% of total dataset, same hyperparameters as model_01, also introduces automatic mixed precision flag for more efficient processing, took about 3-4 hours on the node
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
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/anat_qc_t1w_t2w_subset_1024.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_02/model_02.csv |
| folder | /scratch.global/lundq163/auto_qc_subset_1024/ |
| gres | gpu:a100:1 |
| job_name | automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 128g |
| model_save_location | /scratch.global/lundq163/auto_qc_model_02/model_02.pt |
| ntasks | 1 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_02/model_02.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 24:00:00 |

