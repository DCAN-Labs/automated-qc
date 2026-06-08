# Stratified K-Fold Training Workflow

This document describes the two-step k-fold training process with stratified fold assignments.

## Workflow Overview

The process is now split into two steps:

1. **Step 1: Prepare Stratified K-Folds** (run once)
   - Create a copy of `prepare_stratified_kfolds.sh` for your specific training case
   - Run `prepare_stratified_kfolds.sh` to create stratified fold assignments
   - This creates fold split files that balance QU_motion score distribution
   - The split information is saved for reuse

2. **Step 2: Train Folds in Parallel** (run after step 1)
   - Create a copy of `submit_kfold_parallel.sh` for your specific training case
   - Run `submit_kfold_parallel.sh` with the fold assignment file
   - Each fold trains independently on the pre-stratified data
   - All 5 folds train in parallel

## Detailed Instructions

### Step 1: Prepare Stratified K-Folds

**Before running this step, make sure to edit the bash script for your specific case**

Create the stratified fold assignments with this script:

```bash
sbatch ~/projects/automated-qc/scripts/utils/prepare_stratified_kfolds.sh
```

This will:
- Read your input CSV file
- Create 5 stratified k-fold splits based on QU_motion distribution
- Save fold assignments to a path similar to this one, depending upon your specified output-dir:
  ```
  ~/projects/automated-qc/doc/models/model_02r/fold_assignments_02r7/
    ├── fold_assignments.json    (subject -> fold mapping)
    ├── fold_details.json        (fold statistics)
    ├── fold_assignments.csv     (human-readable table)
    └── fold_*_subset.csv        (train-val column info)
  ```

Wait for this job to complete before proceeding to Step 2.

### Step 2: Submit Parallel Fold Training

**Before running this step, make sure to edit the following bash scripts for your specific case:**

**`~/projects/automated-qc/scripts/utils/submit_kfold_parallel.sh`**

**`~/projects/automated-qc/scripts/utils/auto-qc-training_model_agate_subset_1024_model_02r_kfold.sh`**

After Step 1 completes, run the parallel fold training after editing the variable for your specific path:

```bash
FOLD_CSV_DIR=~/projects/automated-qc/doc/models/model_02r7/fold_assignments_02r7 ~/projects/automated-qc/scripts/utils/submit_kfold_parallel.sh
```

This will:
- Submit 5 independent SLURM jobs (one per fold)
- Each job reads the pre-computed fold assignments
- All jobs run in parallel with their own 24-hour time limits

Monitor progress:
```bash
squeue -u <x500> | grep fold
```

## Output Structure

After both steps complete:

```
/scratch.global/<x500>/auto_qc/auto_qc_model_02r7/
├── model_02r7_fold_0.pt
├── model_02r7_fold_1.pt
├── model_02r7_fold_2.pt
├── model_02r7_fold_3.pt
└── model_02r7_fold_4.pt

~/projects/automated-qc/doc/models/model_02r7/
├── model_02r7_fold_0.csv
├── model_02r7_fold_1.csv
├── model_02r7_fold_2.csv
├── model_02r7_fold_3.csv
├── model_02r7_fold_4.csv
├── model_02r7_fold_0.png
├── model_02r7_fold_1.png
├── model_02r7_fold_2.png
├── model_02r7_fold_3.png
└── model_02r7_fold_4.png

~/projects/automated-qc/doc/models/model_02r7/fold_assignments_02r7/
├── fold_assignments.json    (used by training script)
├── fold_details.json        (fold statistics)
├── fold_assignments.csv     (human-readable table)
└── fold_*_subset.csv        (train-val column info)
```

