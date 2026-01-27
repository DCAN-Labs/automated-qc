
## Usage

### Quick Start

Run the master script to submit all 5 fold jobs at once:

```bash
bash /users/1/lundq163/projects/automated-qc/scripts/config/submit_kfold_parallel.sh
```

This will:
- Submit 5 separate SLURM jobs (folds 0-4)
- Print out all job IDs
- Provide commands for monitoring progress

### Manual Submission (Individual Fold)

To train a specific fold manually:

```bash
# Train fold 0
sbatch --export=FOLD_IDX=0 /users/1/lundq163/projects/automated-qc/scripts/config/auto-qc-training_model_agate_subset_1024_model_02r_kfold_single_fold.sh

# Train fold 2
sbatch --export=FOLD_IDX=2 /users/1/lundq163/projects/automated-qc/scripts/config/auto-qc-training_model_agate_subset_1024_model_02r_kfold_single_fold.sh
```

### For Single Fold Testing

If you want to run without SLURM (for testing):

```bash
export FOLD_IDX=0
bash /users/1/lundq163/projects/automated-qc/scripts/config/auto-qc-training_model_agate_subset_1024_model_02r_kfold_single_fold.sh
```


## Output Structure

After all 5 jobs complete, you'll have:

```
/scratch.global/lundq163/auto_qc/auto_qc_model_02r7/
├── model_02r7_fold_0.pt
├── model_02r7_fold_1.pt
├── model_02r7_fold_2.pt
├── model_02r7_fold_3.pt
└── model_02r7_fold_4.pt

/users/1/lundq163/projects/automated-qc/doc/models/model_02r/
├── model_02r7_fold_0.csv       (predictions for fold 0)
├── model_02r7_fold_1.csv       (predictions for fold 1)
├── model_02r7_fold_2.csv       (predictions for fold 2)
├── model_02r7_fold_3.csv       (predictions for fold 3)
├── model_02r7_fold_4.csv       (predictions for fold 4)
├── model_02r7_fold_0.png       (plot for fold 0)
├── model_02r7_fold_1.png       (plot for fold 1)
├── model_02r7_fold_2.png       (plot for fold 2)
├── model_02r7_fold_3.png       (plot for fold 3)
└── model_02r7_fold_4.png       (plot for fold 4)

/users/1/lundq163/projects/automated-qc/scripts/config/logs/
├── automated-qc-Regressor-fold-[JOBID1].out
├── automated-qc-Regressor-fold-[JOBID1].err
├── automated-qc-Regressor-fold-[JOBID2].out
├── automated-qc-Regressor-fold-[JOBID2].err
└── ... (one pair per fold)
```
