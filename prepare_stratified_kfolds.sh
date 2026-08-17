#!/bin/sh

#SBATCH --job-name=prep-kfolds
#SBATCH --mem=8g
#SBATCH --time=00:10:00
#SBATCH -p msismall,msilarge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL

# Prepares stratified k-fold splits for cross-validation.
# Run this ONCE before submitting parallel fold training jobs.
#
# All paths come from the arguments file; see config/arguments.txt.

AUTO_QC_HOME=${AUTO_QC_HOME:-$HOME/projects/automated-qc}
ARGUMENTS_FILE=${ARGUMENTS_FILE:-$AUTO_QC_HOME/config/arguments.txt}

if [ ! -f "$ARGUMENTS_FILE" ]; then
    echo "ERROR: arguments file not found: $ARGUMENTS_FILE" >&2
    echo "Create one from the tracked template:" >&2
    echo "  cp $AUTO_QC_HOME/config/arguments.example.txt $AUTO_QC_HOME/config/arguments.txt" >&2
    exit 1
fi

. "$AUTO_QC_HOME/scripts/utils/load_arguments.sh" "$ARGUMENTS_FILE"

if [ ! -f "$RAW_CSV" ]; then
    echo "ERROR: input CSV not found: $RAW_CSV" >&2
    exit 1
fi

cd "$PROJECT_DIR/src/data_sets" || exit 1
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

mkdir -p "$FOLD_ASSIGNMENTS_DIR"

"$VENV_PYTHON" "$PROJECT_DIR/src/data_sets/prepare_stratified_kfolds.py" \
--csv-input-file "$RAW_CSV" \
--output-dir "$FOLD_ASSIGNMENTS_DIR" \
--k-folds "$K_FOLDS" \
--random-seed "$RANDOM_SEED"

echo "K-fold preparation complete!"
echo "Fold assignments saved to: $FOLD_ASSIGNMENTS_DIR"
echo ""
echo "Now submit the training jobs:"
echo "  $PROJECT_DIR/scripts/utils/submit_4d_kfold.sh"
