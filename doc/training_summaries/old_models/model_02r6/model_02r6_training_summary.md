# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 219 |
| RMSE | 0.6284 |
| Standardized RMSE | 0.6529 |
| Correlation (r) | 0.7710 |
| P-value | 2.0998e-44 |
| Standard Error | 0.6313 |

## Sensitivity, Specificity, PPV, and NPV Analysis

| Threshold | Sensitivity | Specificity | PPV | NPV |
|-----------|-------------|-------------|-----|-----|
| 0.0 | 0.9909 | 0.0000 | 1.0000 | 0.0000 |
| 0.5 | 0.9340 | 0.5000 | 0.9436 | 0.4583 |
| 1.0 | 0.7515 | 0.7407 | 0.8986 | 0.4938 |
| 1.5 | 0.6455 | 0.8807 | 0.8452 | 0.7111 |
| 2.0 | 0.5921 | 0.9650 | 0.9000 | 0.8166 |
| 2.5 | 0.5000 | 0.9939 | 0.9643 | 0.8586 |
| 3.0 | 0.3684 | 0.9890 | 0.8750 | 0.8818 |

### Interpretation

Higher motion scores indicate more motion artifact (worse image quality). Thresholds represent the maximum acceptable motion score before flagging an image.

**Threshold Recommendations**:

- **Balanced (Threshold 1.0)**: Detects 75% of motion-corrupted images while maintaining 74% specificity. When flagged, 90% are true positives.
- **High Sensitivity (Threshold 0.5)**: Catches 93% of motion artifacts but rejects 50% of acceptable images.
- **High Specificity (Threshold 2.5)**: Minimizes false positives (99% specificity) but only catches 50% of motion artifacts.
**Key Observations**:

- High precision: When images are flagged at threshold 1.0, they're truly motion-corrupted 90% of the time.
- At the balanced threshold, 25% of motion artifacts go undetected. These mild cases may still be diagnostic and will be caught during radiologist review if problematic.

## Visualization

![QU Motion Score Analysis](../../models/model_02r/model_02r6.png)


## Interpretation

- **Correlation**: 0.7710 indicates a strong positive relationship between actual and predicted scores.
- **P-value**: 2.0998e-44 is statistically significant (p < 0.05).
- **Standardized RMSE**: 0.6529 represents the RMSE as a proportion of the standard deviation of the actual values.
- **Standard Error**: 0.6313 provides an estimate of the average distance that the observed values fall from the regression line.
## Notes

same as model 02r5 but with some edited scores
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
| csv_input_file | /users/1/lundq163/projects/automated-qc/data/raw/auto_qc_t1w_t2w_subset_1024r_curated_for_02r6.csv |
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_02r/model_02r6.csv |
| folder | /scratch.global/lundq163/auto_qc/auto_qc_subset_1024r_fixed_scores/ |
| gres | gpu:a100:1 |
| job_name | automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 128g |
| model_save_location | /scratch.global/lundq163/auto_qc/auto_qc_model_02r6/model_02r6.pt |
| ntasks | 1 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_02r/model_02r6.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 24:00:00 |

