#!/bin/sh

#SBATCH --job-name=inspect-4d-dims
#SBATCH --mem=8g
#SBATCH --time=00:20:00
#SBATCH -p msismall,msilarge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL

# Surveys NIfTI header dimensions across the cohort using nibabel.
# Run ONCE before configuring a 4D training run: its output supplies the
# TARGET_SHAPE and NUM_FRAMES values in config/arguments.txt
#
# Header-only reads, so this is cheap enough to run interactively:
#   . scripts/utils/load_arguments.sh config/arguments.txt
#   "$VENV_PYTHON" scripts/utils/inspect_4d_dims.py "$FOLDER" --pattern "$FILE_PATTERN"

AUTO_QC_HOME=${AUTO_QC_HOME:-$HOME/projects/automated-qc}
ARGUMENTS_FILE=${ARGUMENTS_FILE:-$AUTO_QC_HOME/config/arguments.txt}

if [ ! -f "$ARGUMENTS_FILE" ]; then
    echo "ERROR: arguments file not found: $ARGUMENTS_FILE" >&2
    echo "Create one from the tracked template:" >&2
    echo "  cp $AUTO_QC_HOME/config/arguments.example.txt $AUTO_QC_HOME/config/arguments.txt" >&2
    exit 1
fi

. "$AUTO_QC_HOME/scripts/utils/load_arguments.sh" "$ARGUMENTS_FILE"

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# With SURVEY_CSV set, restrict the survey to scans a fold CSV references, which
# also surfaces rows whose file is missing from disk.
SURVEY_CSV=${SURVEY_CSV:-}

if [ -n "$SURVEY_CSV" ]; then
    "$VENV_PYTHON" "$PROJECT_DIR/scripts/utils/inspect_4d_dims.py" \
        "$FOLDER" --csv "$SURVEY_CSV"
else
    "$VENV_PYTHON" "$PROJECT_DIR/scripts/utils/inspect_4d_dims.py" \
        "$FOLDER" --pattern "$FILE_PATTERN"
fi

echo ""
echo "Copy the suggested flags above into TARGET_SHAPE / NUM_FRAMES in:"
echo "  $ARGUMENTS_FILE"
