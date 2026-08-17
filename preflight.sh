#!/bin/bash
# Reports on every stage of the smoke-test setup. Read-only; changes nothing.
# Find the repo root: this dir, or automated-qc/ underneath it.
if [ -d src/models ] && [ -d scripts/utils ]; then :
elif [ -d automated-qc/src/models ]; then cd automated-qc || exit 1
else
  echo "ERROR: not in the automated-qc repo (no src/models here)."
  echo "  You are in: $PWD"
  echo "  Try:  cd automated-qc  &&  bash preflight.sh"
  exit 1
fi
ok(){ printf '  \033[32mOK\033[0m   %s\n' "$1"; }
no(){ printf '  \033[31mNO\033[0m   %s\n' "$1"; }
hm(){ printf '  --   %s\n' "$1"; }

echo "REPO: $PWD"

echo; echo "[1] code landed"
n=$(git status --short 2>/dev/null | wc -l)
[ "$n" -ge 17 ] && ok "$n changed paths" || no "$n changed paths (expected ~18) - did you extract the zip?"
[ -f src/models/temporal.py ] && ok "src/models/temporal.py" || no "src/models/temporal.py MISSING"
[ -f scripts/utils/build_fmap_csv.py ] && ok "scripts/utils/build_fmap_csv.py" || no "build_fmap_csv.py MISSING"
[ -d _local ] && no "_local/ still present - rm -rf _local" || ok "_local removed"

echo; echo "[2] config"
if [ -f config/arguments.txt ]; then
  ok "config/arguments.txt exists"
  git check-ignore -q config/arguments.txt && ok "correctly gitignored" || no "NOT gitignored"
else no "config/arguments.txt MISSING - cp from _local or the template"; fi

echo; echo "[3] venv"
if [ -f config/arguments.txt ] && [ -f scripts/utils/load_arguments.sh ]; then
  # Set the env var too: dash ignores arguments passed to a sourced script.
  ARGUMENTS_FILE=config/arguments.txt; export ARGUMENTS_FILE
  . scripts/utils/load_arguments.sh config/arguments.txt >/dev/null 2>&1
else
  no "skipping - config/arguments.txt or load_arguments.sh not found"
fi
if [ -n "$VENV_PYTHON" ] && [ -x "$VENV_PYTHON" ]; then ok "VENV_PYTHON runs: $VENV_PYTHON"
else no "VENV_PYTHON not executable: $VENV_PYTHON"
     hm "active interpreter is: $(command -v python)"; fi

: "${FOLDER:=}" ; : "${RAW_CSV:=}" ; : "${FOLD_ASSIGNMENTS_DIR:=}"
: "${MODEL_NAME:=}" ; : "${TARGET_SHAPE:=}" ; : "${NUM_FRAMES:=}"
: "${EPOCHS:=}" ; : "${FILE_PATTERN:=*.nii.gz}" ; : "${LOG_DIR:=scripts/utils/logs}"
: "${VENV_PYTHON:=}"

echo; echo "[4] resolved paths"
for k in FOLDER RAW_CSV FOLD_ASSIGNMENTS_DIR MODEL_NAME TARGET_SHAPE NUM_FRAMES EPOCHS; do
  eval "printf '  %-20s = %s\n' \"$k\" \"\$$k\""
done

echo; echo "[5] data"
if [ -n "$FOLDER" ] && [ -d "$FOLDER" ]; then
  ok "FOLDER readable"
  c=$(find "$FOLDER" -name "$FILE_PATTERN" 2>/dev/null | head -5000 | wc -l)
  hm "$c files matching $FILE_PATTERN (capped at 5000)"
else no "FOLDER not readable: $FOLDER"; fi

echo; echo "[6] dummy CSV"
if [ -n "$RAW_CSV" ] && [ -f "$RAW_CSV" ]; then
  ok "$RAW_CSV"
  hm "rows: $(($(wc -l < "$RAW_CSV") - 1))"
  head -1 "$RAW_CSV" | grep -q QU_motion_source && ok "tagged SYNTHETIC" || no "no QU_motion_source column"
else no "not built yet - run build_fmap_csv.py"; fi

echo; echo "[7] folds"
if [ -n "$FOLD_ASSIGNMENTS_DIR" ] && [ -d "$FOLD_ASSIGNMENTS_DIR" ]; then
  f=$(ls "$FOLD_ASSIGNMENTS_DIR"/fold_*_subset.csv 2>/dev/null | wc -l)
  [ "$f" -gt 0 ] && ok "$f fold CSVs" || no "directory exists but no fold_*_subset.csv"
else no "not built yet - run prepare_stratified_kfolds.sh"; fi

echo; echo "[8] jobs"
squeue -u "$USER" 2>/dev/null | tail -n +2 | grep -q . \
  && squeue -u "$USER" || hm "nothing queued or running"
sacct -u "$USER" --starttime today --format=JobID%14,JobName%24,State,Elapsed,ExitCode 2>/dev/null | head -12

echo; echo "[9] latest log"
L=$(ls -t "$LOG_DIR"/*.out 2>/dev/null | head -1)
if [ -n "$L" ]; then
  hm "$L"
  grep -E "Input configuration|Applied from arguments|standardized_rmse|correlation_coefficient|Error|Traceback" "$L" | tail -8
else hm "no logs in $LOG_DIR yet"; fi
