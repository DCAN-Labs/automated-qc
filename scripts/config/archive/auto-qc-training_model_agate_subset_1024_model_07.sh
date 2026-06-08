#!/bin/sh

#SBATCH --job-name=model-07-automated-qc-Regressor # job name

#SBATCH --mem=128g        
#SBATCH --time=24:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    

#SBATCH --mail-type=begin       
#SBATCH --mail-type=end          
#SBATCH --mail-user=lundq163@umn.edu
#SBATCH -e logs/model-07-automated-qc-Regressor-%j.err
#SBATCH -o logs/model-07-automated-qc-Regressor-%j.out
#SBATCH -A feczk001

cd /users/1/lundq163/projects/automated-qc/src/training || exit

export PYTHONPATH=/users/1/lundq163/projects/automated-qc/src:$PYTHONPATH
export AUTO_QC_CACHE_DIR=/scratch.global/lundq163/auto_qc/auto_qc_model_07_cache/
export PYTORCH_ALLOC_CONF=expandable_segments:True

/users/1/lundq163/projects/automated-qc/.venv/bin/python \
/users/1/lundq163/projects/automated-qc/src/training/training.py \
--model-save-location "/scratch.global/lundq163/auto_qc/auto_qc_model_07/model_07.pt" \
--plot-location "/users/1/lundq163/projects/automated-qc/doc/models/model_07/model_07.png" \
--folder "/scratch.global/lundq163/auto_qc/auto_qc_subset_1024r_fixed_scores/" \
--csv-input-file "/users/1/lundq163/projects/automated-qc/data/raw/auto_qc_t1w_t2w_subset_1024r_curated.csv" \
--csv-output-file "/users/1/lundq163/projects/automated-qc/doc/models/model_07/model_07.csv" \
--tb-run-dir "/users/1/lundq163/projects/automated-qc/src/training/runs/" \
--split-strategy "stratified" \
--train-split 0.8 \
--model "Regressor" \
--lr 0.001 \
--scheduler "plateau" \
--batch-size 8 \
--epochs 100 \
--optimizer "Adam" \
--num-workers 12 \
--use-amp
