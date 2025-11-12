# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 210 |
| RMSE | 0.8797 |
| Standardized RMSE | 0.8301 |
| Correlation (r) | 0.6228 |
| P-value | 5.9191e-24 |
| Standard Error | 0.8839 |

## Visualization

![QU Motion Score Analysis](../../models/model_02r/model_02r.png)


## Interpretation

- **Correlation**: 0.6228 indicates a moderate positive relationship between actual and predicted scores.
- **P-value**: 5.9191e-24 is statistically significant (p < 0.05).
- **Standardized RMSE**: 0.8301 represents the RMSE as a proportion of the standard deviation of the actual values.
## Notes

Same paramters and input data size as model_02, but removes three scans with poor QU_motion gt scores. Run time was 3h 11m 52s
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
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/raw/anat_qc_t1w_t2w_subset_1024r.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_02r/model_02r.csv |
| folder | /scratch.global/lundq163/auto_qc_subset_1024r/ |
| gres | gpu:a100:1 |
| job_name | automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 128g |
| model_save_location | /scratch.global/lundq163/auto_qc_model_02r/model_02r.pt |
| ntasks | 1 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_02r/model_02r.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 24:00:00 |

