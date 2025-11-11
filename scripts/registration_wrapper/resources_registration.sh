#!/bin/bash -l
#SBATCH -J hbcd-registration
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH --tmp=10gb
#SBATCH -t 4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lundq163@umn.edu
#SBATCH -p msismall
#SBATCH -o output_logs/hbcd-registration_%A_%a.out
#SBATCH -e output_logs/hbcd-registration_%A_%a.err
#SBATCH -A rando149

cd run_files.registration

module load singularity

file=run${SLURM_ARRAY_TASK_ID}

bash ${file}