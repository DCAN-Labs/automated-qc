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

# A shell script exits with the status of its LAST command. Without this check
# the trailing echoes below would mask a Python crash, and the job would be
# recorded COMPLETED 0:0 having written nothing -- which is exactly what
# happened when a broken pandas install went unnoticed for hours.
status=$?
if [ "$status" -ne 0 ]; then
    echo "ERROR: prepare_stratified_kfolds.py exited $status" >&2
    exit "$status"
fi

# Exit 0 is necessary but not sufficient: verify the folds are actually there.
produced=$(ls "$FOLD_ASSIGNMENTS_DIR"/fold_*_subset.csv 2>/dev/null | wc -l)
if [ "$produced" -ne "$K_FOLDS" ]; then
    echo "ERROR: expected $K_FOLDS fold files in $FOLD_ASSIGNMENTS_DIR," \
         "found $produced" >&2
    exit 1
fi

echo "K-fold preparation complete! ($produced fold files)"
echo "Fold assignments saved to: $FOLD_ASSIGNMENTS_DIR"
echo ""
echo "Now submit the training jobs:"
echo "  $PROJECT_DIR/scripts/utils/submit_4d_kfold.sh"
