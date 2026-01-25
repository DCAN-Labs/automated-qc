#!/bin/sh

#SBATCH --job-name=automated-qc-Regressor # job name

#SBATCH --mem=128g        
#SBATCH --time=24:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    

#SBATCH --mail-type=begin       
#SBATCH --mail-type=end          
#SBATCH --mail-user=lundq163@umn.edu
#SBATCH -e logs/automated-qc-Regressor-kfold-%j.err
#SBATCH -o logs/automated-qc-Regressor-kfold-%j.out
#SBATCH -A feczk001

cd /users/1/lundq163/projects/automated-qc/src/training || exit

export PYTHONPATH=/users/1/lundq163/projects/automated-qc/src:$PYTHONPATH
export AUTO_QC_CACHE_DIR=/scratch.global/lundq163/auto_qc/auto_qc_model_02r7_cache/
export PYTORCH_ALLOC_CONF=expandable_segments:True

/users/1/lundq163/projects/automated-qc/.venv/bin/python \
/users/1/lundq163/projects/automated-qc/src/training/training.py \
--model-save-location "/scratch.global/lundq163/auto_qc/auto_qc_model_02r7/model_02r7.pt" \
--plot-location "/users/1/lundq163/projects/automated-qc/doc/models/model_02r/model_02r7.png" \
--folder "/scratch.global/lundq163/auto_qc/auto_qc_subset_1024r_fixed_scores/" \
--csv-input-file "/users/1/lundq163/projects/automated-qc/data/raw/auto_qc_t1w_t2w_subset_1024r_curated.csv" \
--csv-output-file "/users/1/lundq163/projects/automated-qc/doc/models/model_02r/model_02r7.csv" \
--tb-run-dir "/users/1/lundq163/projects/automated-qc/src/training/runs/" \
--model "Regressor" \
--lr 0.001 \
--scheduler "plateau" \
--batch-size 8 \
--epochs 100 \
--optimizer "Adam" \
--num-workers 12 \
--use-kfold \
--k-folds 5