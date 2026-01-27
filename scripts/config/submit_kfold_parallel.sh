#!/bin/bash

# Master script to submit 5 parallel k-fold jobs
# This script submits 5 separate SLURM jobs, each training a single fold
# Each job has its own 24-hour time limit, avoiding timeout issues

# TODO: find a way to clarify which fold is which in log file name and job names

# Configuration
MODEL_NAME="model_02r7"
NUM_FOLDS=5
CONFIG_TEMPLATE="/users/1/lundq163/projects/automated-qc/scripts/config/auto-qc-training_model_agate_subset_1024_model_02r_kfold_single_fold.sh"
LOG_DIR="/users/1/lundq163/projects/automated-qc/scripts/config/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "Submitting ${NUM_FOLDS} parallel fold training jobs for ${MODEL_NAME}"
echo "========================================================================"
echo ""

# Array to store job IDs
declare -a JOB_IDS

# Submit a job for each fold
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
    echo "Submitting fold $fold..."
    
    # Submit the job with FOLD_IDX environment variable
    JOB_ID=$(sbatch --export=FOLD_IDX=$fold "$CONFIG_TEMPLATE" | awk '{print $NF}')
    
    if [ -z "$JOB_ID" ]; then
        echo "ERROR: Failed to submit job for fold $fold"
        exit 1
    fi
    
    JOB_IDS+=($JOB_ID)
    echo "  Submitted fold $fold with job ID: $JOB_ID"
done

echo ""
echo "========================================================================"
echo "All fold jobs submitted successfully!"
echo "========================================================================"
echo ""
echo "Job IDs:"
for i in "${!JOB_IDS[@]}"; do
    echo "  Fold $i: ${JOB_IDS[$i]}"
done

echo ""
echo "Monitor job status with:"
echo "  squeue --user=lundq163"
echo ""
echo "View individual job output with:"
for i in "${!JOB_IDS[@]}"; do
    echo "  tail -f ${LOG_DIR}/automated-qc-Regressor-fold-${JOB_IDS[$i]}.out"
done
echo ""
echo "All outputs will be saved to: $LOG_DIR"
echo "========================================================================"
