#!/bin/bash -l
#SBATCH -J hbcd-registration
#SBATCH -c 2
#SBATCH --mem=8gb
#SBATCH --tmp=5gb
#SBATCH -t 0:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lundq163@umn.edu
#SBATCH -p msismall
#SBATCH -o output_logs/hbcd-registration_%A_%a.out
#SBATCH -e output_logs/hbcd-registration_%A_%a.err
#SBATCH -A faird

cd run_files.registration

module load singularity

file=run${SLURM_ARRAY_TASK_ID}

bash ${file}