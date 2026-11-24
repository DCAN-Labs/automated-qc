#!/bin/bash
# Submit all cross-validation folds for a 4D field map training run.
#
# Reads every path and SLURM setting from the arguments file, then passes the
# SLURM options as sbatch command line flags -- those override the #SBATCH
# directives inside the job script, which is the only way to configure them from
# a config file (sbatch parses the directives before the script ever runs).
#
# Usage:
#   ./submit_4d_kfold.sh
#   ARGUMENTS_FILE=/path/to/arguments.txt ./submit_4d_kfold.sh

set -u

AUTO_QC_HOME=${AUTO_QC_HOME:-$HOME/projects/automated-qc}
ARGUMENTS_FILE=${ARGUMENTS_FILE:-$AUTO_QC_HOME/config/arguments.txt}

if [ ! -f "$ARGUMENTS_FILE" ]; then
    echo "ERROR: arguments file not found: $ARGUMENTS_FILE" >&2
    echo "Create one from the tracked template:" >&2
    echo "  cp $AUTO_QC_HOME/config/arguments.example.txt $AUTO_QC_HOME/config/arguments.txt" >&2
    exit 1
fi

. "$AUTO_QC_HOME/scripts/utils/load_arguments.sh" "$ARGUMENTS_FILE"

JOB_SCRIPT="$PROJECT_DIR/scripts/config/auto-qc-training_model_agate_fmap_4d_kfold.sh"
NUM_FOLDS=${K_FOLDS:-5}

echo "Submitting $NUM_FOLDS fold training jobs for $MODEL_NAME"
echo "Arguments file: $ARGUMENTS_FILE"
echo "Job script:     $JOB_SCRIPT"

mkdir -p "$LOG_DIR"

declare -a JOB_IDS

for fold_idx in $(seq 0 $((NUM_FOLDS - 1))); do
    # Resolve the fold CSV the same way the job will, to fail fast on a typo
    # rather than after the job has queued.
    fold_csv=$(FOLD_IDX=$fold_idx sh -c ". \"$AUTO_QC_HOME/scripts/utils/load_arguments.sh\" \"$ARGUMENTS_FILE\" > /dev/null; printf '%s' \"\$CSV_INPUT_FILE\"")

    if [ ! -f "$fold_csv" ]; then
        echo "ERROR: Fold CSV not found: $fold_csv" >&2
        exit 1
    fi

    echo "Submitting fold $fold_idx using CSV: $fold_csv"

    job_output=$(sbatch \
        --export=ALL,FOLD_IDX=$fold_idx,ARGUMENTS_FILE="$ARGUMENTS_FILE",AUTO_QC_HOME="$AUTO_QC_HOME" \
        --job-name="${MODEL_NAME}_fold_${fold_idx}" \
        --account="$SLURM_ACCOUNT" \
        --mail-user="$SLURM_EMAIL" \
        --partition="$SLURM_GPU_PARTITION" \
        --gres="$SLURM_GPU_GRES" \
        --mem="$SLURM_MEM" \
        --time="$SLURM_TIME" \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        -e "${LOG_DIR}/${MODEL_NAME}-fold_${fold_idx}-%j.err" \
        -o "${LOG_DIR}/${MODEL_NAME}-fold_${fold_idx}-%j.out" \
        "$JOB_SCRIPT")

    job_id=$(echo "$job_output" | awk '{print $NF}')
    JOB_IDS[$fold_idx]=$job_id
    echo "  Submitted fold $fold_idx with job ID: $job_id"
done

echo ""
echo "All $NUM_FOLDS fold training jobs submitted!"
for fold_idx in $(seq 0 $((NUM_FOLDS - 1))); do
    echo "  Fold $fold_idx: ${JOB_IDS[$fold_idx]}"
done
echo ""
echo "Watch all folds: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
