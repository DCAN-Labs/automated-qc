# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 840 |
| RMSE | 0.7977 |
| Standardized RMSE | 0.7798 |
| Correlation (r) | 0.6624 |
| P-value | 3.1018e-107 |
| Standard Error | 0.7987 |

## Visualization

![QU Motion Score Analysis](../../models/model_03r/model_03r.png)


## Interpretation

- **Correlation**: 0.6624 indicates a moderate positive relationship between actual and predicted scores.
- **P-value**: 3.1018e-107 is statistically significant (p < 0.05).
- **Standardized RMSE**: 0.7798 represents the RMSE as a proportion of the standard deviation of the actual values.
## Notes

Same paramters and input data size as model_03, but uses data registered to an MNIInfant template with ridge + affine transform using ANTs and removes three scans with poor QU_motion gt scores. Run time was 12h 26m 44s
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
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/raw/anat_qc_t1w_t2w_subset_4096r.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_03r/model_03r.csv |
| folder | /scratch.global/lundq163/auto_qc_subset_4096r/ |
| gres | gpu:a100:1 |
| job_name | automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 128g |
| model_save_location | /scratch.global/lundq163/auto_qc_model_03r/model_03r.pt |
| ntasks | 1 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_03r/model_03r.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 24:00:00 |

