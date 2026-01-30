#!/bin/sh

#SBATCH --job-name=prep-kfolds
#SBATCH --mem=8g        
#SBATCH --time=00:10:00          
#SBATCH -p msismall,msilarge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4    

#SBATCH --mail-type=BEGIN,END,FAIL        
#SBATCH --mail-user=lundq163@umn.edu
#SBATCH -e logs/prepare-kfolds-model_02r7-%j.err
#SBATCH -o logs/prepare-kfolds-model_02r7-%j.out
#SBATCH -A feczk001

# This script prepares stratified k-fold splits for cross-validation.
# Run this ONCE before submitting parallel fold training jobs.
# It generates fold assignment files that are used by the training script.

cd /users/1/lundq163/projects/automated-qc/src/data_sets || exit

export PYTHONPATH=/users/1/lundq163/projects/automated-qc/src:$PYTHONPATH

/users/1/lundq163/projects/automated-qc/.venv/bin/python \
/users/1/lundq163/projects/automated-qc/src/data_sets/prepare_stratified_kfolds.py \
--csv-input-file "/users/1/lundq163/projects/automated-qc/data/raw/auto_qc_t1w_t2w_subset_1024r_curated.csv" \
--output-dir "/users/1/lundq163/projects/automated-qc/doc/models/model_02r7/fold_assignments_02r7" \
--k-folds 5 \
--random-seed 42

echo "K-fold preparation complete!"
echo "Fold assignments saved to: /users/1/lundq163/projects/automated-qc/doc/models/model_02r7/fold_assignments_02r7/"
echo ""
echo "Now run the training script with the fold assignment file:"
echo "  sbatch --export=FOLD_ASSIGNMENTS=/users/1/lundq163/projects/automated-qc/doc/models/model_02r7/fold_assignments_02r7/fold_assignments.json /users/1/lundq163/projects/automated-qc/scripts/config/submit_kfold_parallel.sh"