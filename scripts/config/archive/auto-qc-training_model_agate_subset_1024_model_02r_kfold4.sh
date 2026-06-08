#!/bin/sh

#SBATCH --mem=128g        
#SBATCH --time=24:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

#SBATCH --job-name=fold_4_redo
#SBATCH -e /users/1/lundq163/projects/automated-qc/scripts/utils/logs/automated-qc-Regressor-fold_4_redo-%j.err
#SBATCH -o /users/1/lundq163/projects/automated-qc/scripts/utils/logs/automated-qc-Regressor-fold_4_redo-%j.out

#SBATCH --mail-type=BEGIN,END,FAIL        
#SBATCH --mail-user=lundq163@umn.edu
#SBATCH -A feczk001

# This script trains a single fold using a pre-stratified fold subset CSV.
# The fold index (FOLD_IDX) should be passed as an environment variable.
# The fold CSV file path should be passed as FOLD_CSV.
FOLD_IDX=4
FOLD_CSV="/users/1/lundq163/projects/automated-qc/doc/models/model_02r7/fold_assignments_02r7/fold_4_subset.csv"

if [ -z "$FOLD_CSV" ]; then
    echo "ERROR: FOLD_CSV environment variable not set"
    exit 1
fi

if [ ! -f "$FOLD_CSV" ]; then
    echo "ERROR: Fold CSV file not found: $FOLD_CSV"
    exit 1
else
    echo "Training fold index: $FOLD_IDX"
    echo "Using fold CSV: $FOLD_CSV"
fi

AUTO_QC_CACHE_DIR=/scratch.global/lundq163/auto_qc/auto_qc_model_02r7_cache/auto-qc-training_model_agate_subset_1024_model_02r_kfold_fold_${FOLD_IDX}/

if [ ! -d "$AUTO_QC_CACHE_DIR" ]; then
    mkdir -p "$AUTO_QC_CACHE_DIR"
else
    echo "Using existing cache directory: $AUTO_QC_CACHE_DIR"
fi


cd /users/1/lundq163/projects/automated-qc/src/training || exit

export PYTHONPATH=/users/1/lundq163/projects/automated-qc/src:$PYTHONPATH
export AUTO_QC_CACHE_DIR=$AUTO_QC_CACHE_DIR
export PYTORCH_ALLOC_CONF=expandable_segments:True

/users/1/lundq163/projects/automated-qc/.venv/bin/python \
/users/1/lundq163/projects/automated-qc/src/training/training.py \
--model-save-location "/scratch.global/lundq163/auto_qc/auto_qc_model_02r7/model_02r7_fold_${FOLD_IDX}.pt" \
--plot-location "/users/1/lundq163/projects/automated-qc/doc/models/model_02r7/model_02r7_fold_${FOLD_IDX}.png" \
--folder "/scratch.global/lundq163/auto_qc/auto_qc_subset_1024r_fixed_scores/" \
--csv-input-file "$FOLD_CSV" \
--csv-output-file "/users/1/lundq163/projects/automated-qc/doc/models/model_02r7/model_02r7_fold_${FOLD_IDX}.csv" \
--tb-run-dir "/users/1/lundq163/projects/automated-qc/src/training/runs/" \
--use-train-validation-cols \
--model "Regressor" \
--lr 0.001 \
--scheduler "plateau" \
--batch-size 8 \
--epochs 100 \
--optimizer "Adam" \
--num-workers 12 \
--use-amp
