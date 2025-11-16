# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 0 |
| RMSE | 0.0000 |
| Standardized RMSE | 0.0000 |
| Correlation (r) | 0.0000 |
| P-value | 0.0000e+00 |
| Standard Error | 0.0000 |
## Notes

ran into time out error. same as model_03r instead using skull stripping and no registration
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
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/raw/auto_qc_t1w_t2w_subset_4096s.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_03s/model_03s.csv |
| folder | /scratch.global/lundq163/auto_qc_subset_4096s/ |
| gres | gpu:a100:1 |
| job_name | automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 128g |
| model_save_location | /scratch.global/lundq163/auto_qc_model_03s/model_03s.pt |
| ntasks | 1 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_03s/model_03s.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 24:00:00 |

