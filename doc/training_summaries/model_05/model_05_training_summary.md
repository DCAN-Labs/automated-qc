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

ran out of time on the node
## Hyperparameters

### Training Parameters

| Parameter | Value |
|-----------|-------|
| batch_size | 12 |
| epochs | 30 |
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
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/raw/anat_qc_t1w_t2w_subset_4096.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_05/model_05.csv |
| folder | /scratch.global/lundq163/auto_qc_subset_4096/ |
| gres | gpu:a100:1 |
| job_name | automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 128g |
| model_save_location | /scratch.global/lundq163/auto_qc_model_05/model_05.pt |
| ntasks | 1 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_05/model_05.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 24:00:00 |

