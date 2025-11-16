# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 212 |
| RMSE | 0.9231 |
| Standardized RMSE | 0.9348 |
| Correlation (r) | 0.3961 |
| P-value | 2.2426e-09 |
| Standard Error | 0.9275 |

## Visualization

![QU Motion Score Analysis](../../models/model_02s/model_02s2.png)


## Interpretation

- **Correlation**: 0.3961 indicates a weak positive relationship between actual and predicted scores.
- **P-value**: 2.2426e-09 is statistically significant (p < 0.05).
- **Standardized RMSE**: 0.9348 represents the RMSE as a proportion of the standard deviation of the actual values.
## Notes

Same paramters as model_02s1, but removes majority of scans where skull-stripping was poor. Run time was 12h 41m 12s
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
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/raw/auto_qc_t1w_t2w_subset_1024s.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_02s/model_02s.csv |
| folder | /scratch.global/lundq163/auto_qc_subset_1024s/ |
| gres | gpu:a100:1 |
| job_name | automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 128g |
| model_save_location | /scratch.global/lundq163/auto_qc_model_02s/model_02s.pt |
| ntasks | 1 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_02s/model_02s.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 24:00:00 |

