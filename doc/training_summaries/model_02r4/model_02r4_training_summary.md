# QU Motion Score Analysis Results

## Statistical Metrics

| Metric | Value |
|--------|-------|
| Validation Sample Size | 207 |
| RMSE | 0.6128 |
| Standardized RMSE | 0.6204 |
| Correlation (r) | 0.7843 |
| P-value | 2.2226e-44 |
| Standard Error | 0.6158 |

## Sensitivity, Specificity, PPV, and NPV Analysis

| Threshold | Sensitivity | Specificity | PPV | NPV |
|-----------|-------------|-------------|-----|-----|
| 0.0 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.5 | 0.9844 | 0.1333 | 0.9356 | 0.4000 |
| 1.0 | 0.8182 | 0.6226 | 0.8630 | 0.5410 |
| 1.5 | 0.7238 | 0.8922 | 0.8736 | 0.7583 |
| 2.0 | 0.6575 | 0.9552 | 0.8889 | 0.8366 |
| 2.5 | 0.4727 | 0.9803 | 0.8966 | 0.8371 |
| 3.0 | 0.2609 | 0.9938 | 0.9231 | 0.8247 |

### Interpretation

Higher motion scores indicate more motion artifact (worse image quality). Thresholds represent the maximum acceptable motion score before flagging an image.

**Threshold Recommendations**:

- **Balanced (Threshold 1.5)**: Detects 72% of motion-corrupted images while maintaining 89% specificity. When flagged, 87% are true positives.
- **High Sensitivity (Threshold 1.0)**: Catches 82% of motion artifacts but rejects 38% of acceptable images.
- **High Specificity (Threshold 2.5)**: Minimizes false positives (98% specificity) but only catches 47% of motion artifacts.
**Key Observations**:

- High precision: When images are flagged at threshold 1.5, they're truly motion-corrupted 87% of the time.
- At the balanced threshold, 28% of motion artifacts go undetected. These mild cases may still be diagnostic and will be caught during radiologist review if problematic.

## Visualization

![QU Motion Score Analysis](../../models/model_02r/model_02r4.png)


## Interpretation

- **Correlation**: 0.7843 indicates a strong positive relationship between actual and predicted scores.
- **P-value**: 2.2226e-44 is statistically significant (p < 0.05).
- **Standardized RMSE**: 0.6204 represents the RMSE as a proportion of the standard deviation of the actual values.
- **Standard Error**: 0.6158 provides an estimate of the average distance that the observed values fall from the regression line.
## Notes

same set up as the other 02r models but corrected scores from previous runs
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
| csv_output_file | /users/1/lundq163/projects/automated-qc/doc/models/model_02r/model_02r5.csv |
| folder | /scratch.global/lundq163/auto_qc/auto_qc_subset_1024r_fixed_scores/ |
| gres | gpu:a100:1 |
| job_name | automated-qc-Regressor |
| mail_type | end |
| mail_user | lundq163@umn.edu |
| mem | 128g |
| model_save_location | /scratch.global/lundq163/auto_qc/auto_qc_model_02r5/model_02r5.pt |
| ntasks | 1 |
| plot_location | /users/1/lundq163/projects/automated-qc/doc/models/model_02r/model_02r5.png |
| tb_run_dir | /users/1/lundq163/projects/automated-qc/src/training/runs/ |
| time | 24:00:00 |

