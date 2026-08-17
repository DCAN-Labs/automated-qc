#!/bin/sh

#SBATCH --mem=128g
#SBATCH --time=24:00:00
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=BEGIN,END,FAIL

# NOTE: sbatch reads #SBATCH directives before this script body executes, so the
# values above cannot come from the arguments file. scripts/utils/submit_4d_kfold.sh
# passes SLURM_MEM / SLURM_TIME / SLURM_ACCOUNT / SLURM_EMAIL etc. as sbatch
# command line flags, which override these. The directives here are only the
# fallback for a bare `sbatch` of this file.

# 4D field map training for a single cross-validation fold.
#
# Every path comes from the arguments file. The only thing this script needs to
# be told is where that file lives:
#
#   sbatch --export=ALL,FOLD_IDX=0,ARGUMENTS_FILE=/path/to/arguments.txt \
#          scripts/config/auto-qc-training_model_agate_fmap_4d_kfold.sh
#
# or use scripts/utils/submit_4d_kfold.sh, which submits all folds.

FOLD_IDX=${FOLD_IDX:-0}
export FOLD_IDX

# The single bootstrap path: where to find the arguments file. Everything else
# is read from it. Override with --export=ARGUMENTS_FILE=...
AUTO_QC_HOME=${AUTO_QC_HOME:-$HOME/projects/automated-qc}
ARGUMENTS_FILE=${ARGUMENTS_FILE:-$AUTO_QC_HOME/config/arguments.txt}

if [ ! -f "$ARGUMENTS_FILE" ]; then
    echo "ERROR: arguments file not found: $ARGUMENTS_FILE" >&2
    echo "Create one from the tracked template:" >&2
    echo "  cp $AUTO_QC_HOME/config/arguments.example.txt $AUTO_QC_HOME/config/arguments.txt" >&2
    echo "Pass one with --export=ALL,ARGUMENTS_FILE=/path/to/arguments.txt" >&2
    exit 1
fi

# load_arguments.sh lives next to the arguments file's project, so resolve it
# from AUTO_QC_HOME first, then re-resolve from PROJECT_DIR once loaded.
. "$AUTO_QC_HOME/scripts/utils/load_arguments.sh" "$ARGUMENTS_FILE"

echo "Training fold index: $FOLD_IDX"
echo "Project dir:         $PROJECT_DIR"
echo "Fold CSV:            $CSV_INPUT_FILE"
echo "Model output:        $MODEL_SAVE_LOCATION"

if [ ! -f "$CSV_INPUT_FILE" ]; then
    echo "ERROR: Fold CSV file not found: $CSV_INPUT_FILE" >&2
    exit 1
fi

# The preprocessing config is part of the disk cache key, so a changed
# TARGET_SHAPE or NUM_FRAMES will not serve stale tensors. A per-model cache dir
# still keeps scratch usage bounded.
AUTO_QC_CACHE_DIR="$CACHE_DIR"
mkdir -p "$AUTO_QC_CACHE_DIR"
mkdir -p "$(dirname "$MODEL_SAVE_LOCATION")"
mkdir -p "$MODEL_DIR"

cd "$PROJECT_DIR/src/training" || exit 1

export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export AUTO_QC_CACHE_DIR
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Only --arguments-file is passed: the Python entry point reads the same file
# and applies every other value itself. Add flags here to override for a one-off
# run without editing shared config.
"$VENV_PYTHON" \
"$PROJECT_DIR/src/training/training.py" \
--arguments-file "$ARGUMENTS_FILE"
