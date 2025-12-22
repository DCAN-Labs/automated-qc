# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 199 |
| RMSE | 1.5613 |
| Standardized RMSE | 1.5724 |
| Correlation (r) | 0.3508 |
| P-value | 3.7710e-07 |
| Standard Error | 1.5692 |

## Visualization

![QU Motion Score Analysis](../../models/model_07/model_07.png)


## Interpretation

- **Correlation**: 0.3508 indicates a weak positive relationship between actual and predicted scores.
- **P-value**: 3.7710e-07 is statistically significant (p < 0.05).
- **Standardized RMSE**: 1.5724 represents the RMSE as a proportion of the standard deviation of the actual values.
## Notes

same as rescored model_02r but uses a smaller sized image: 200, 240, 160. Does not perform as well
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
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/raw/auto_qc_t1w_t2w_subset_1024r_curated.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_07/model_07.csv |
| folder | /scratch.global/lundq163/auto_qc/auto_qc_subset_1024r_fixed_scores/ |
| gres | gpu:a100:1 |
| job_name | model-07-automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 128g |
| model_save_location | /scratch.global/lundq163/auto_qc/auto_qc_model_07/model_07.pt |
| ntasks | 1 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_07/model_07.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 24:00:00 |

