#!/bin/bash -l
#SBATCH -J reg-ss
#SBATCH -c 2
#SBATCH --mem=8gb
#SBATCH --tmp=5gb
#SBATCH -t 0:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lundq163@umn.edu
#SBATCH -p msismall
#SBATCH -o output_logs/hbcd-reg-ss_%A_%a.out
#SBATCH -e output_logs/hbcd-reg-ss_%A_%a.err
#SBATCH -A cdni-nih-bdc

cd run_files.preproc_reg+skullstrip

file=run${SLURM_ARRAY_TASK_ID}

bash ${file}