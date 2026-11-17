# Automated MRI QC training for HBCD data - Quick Start Guide

## .venv set up on MSI

### 1. Clone the repository in a user specific `projects` directory (i.e. `~/projects/`):
Do not clone in a shared shared directory, as you will be setting up a virtual environment and do not want your changes conflicting with others. 

`git clone git@github.com:DCAN-Labs/automated-qc.git`

### 2. If you have already set up your virtual environment, activate it and skip step 3:

example command if you have already done step 3: `source ~/projects/automated-qc/.venv/bin/activate`

### 3. Set up your virtual environment:

Note that this is specific for MSI users.

`cd ~/projects/automated-qc/`

`module load python3/3.12.4_anaconda2024.06-1_libmamba`

`python3.12 -m venv .venv`

`source .venv/bin/activate`

`pip install -r requirements.txt`

From here, create a new branch for your development work separate from main

## Run configuration: `config/arguments.txt`

Paths are no longer hardcoded in the job scripts. A single `KEY=VALUE` file is
read by **both** the shell scripts (`scripts/utils/load_arguments.sh`) and the
Python entry points (`src/util/arguments.py`), so the two cannot drift apart.

```
PROJECT_DIR=/users/1/<x500>/projects/automated-qc
SCRATCH_DIR=/scratch.global/<x500>/auto_qc
MODEL_NAME=model_04d0

# ${...} interpolates earlier keys, then the environment
FOLDER=${SCRATCH_DIR}/fmaps/
MODEL_SAVE_LOCATION=${SCRATCH_DIR}/${MODEL_NAME}/${MODEL_NAME}_fold_${FOLD_IDX}.pt
```

Keys map to command line options by lowercasing: `FOLDER` sets `--folder`,
`CSV_INPUT_FILE` sets `--csv-input-file`. Keys with no matching option
(`PROJECT_DIR`, `SLURM_ACCOUNT`, `VENV_PYTHON`) are used only by the shell scripts.

**Precedence: command line > arguments file > built-in default.** A one-off run
can override any single value without editing shared config:

```bash
python src/training/training.py --arguments-file config/arguments.txt --epochs 5
```

Values are type-checked and validated against each option's `choices` when the
file loads, so a typo fails immediately rather than an hour into a job.
`${FOLD_IDX}` and friends resolve from the environment, which is how the
submission script fans one config out across folds.

Two things the arguments file deliberately cannot do:

- **`#SBATCH` directives.** `sbatch` parses those before the script body runs.
  `scripts/utils/submit_4d_kfold.sh` reads `SLURM_ACCOUNT` / `SLURM_EMAIL` /
  `SLURM_MEM` etc. and passes them as `sbatch` command line flags, which override
  the in-file directives.
- **Bootstrap its own location.** Each script keeps one overridable default,
  `AUTO_QC_HOME` (defaults to `$HOME/projects/automated-qc`). Override with
  `--export=ALL,ARGUMENTS_FILE=/path/to/arguments.txt`.

`config/arguments.example.txt` is the tracked template; your working copy at
`config/arguments.txt` is gitignored, so personal paths stay out of version
control and `git status` stays clean between runs. First thing after cloning:

```bash
cp config/arguments.example.txt config/arguments.txt
# then replace every <x500> with your username
```

## Input data: dimensionality and frames

`INPUT_MODE` decides how image files are read.

| value | meaning |
|---|---|
| `auto` | inspect the NIfTI headers and decide |
| `2d` | single-slice images |
| `3d` | one volume per file |
| `4d` | several volumes per file |

`auto` counts axes with extent greater than one rather than trusting `dim0`, so
a file declaring 4D while carrying a single volume is read as 3D, and a volume
shaped `(X, Y, 1)` is read as 2D. In the HBCD field map cohort 77 of 4412 files
are in the first category.

Detection is a property of the **cohort**, not of individual files: the
network's spatial rank is fixed when the model is built. A mixed cohort is
therefore an error, with the distribution printed, unless `ALLOW_MIXED_DIMS=true`
tells it to take the majority. `DIM_SURVEY_LIMIT` caps how many headers are
read; `0` reads all of them.

An explicit mode is honoured even when the data disagrees -- you get a warning,
not a failure, because forcing a mode is sometimes the point.

`2d` builds a 2D network, and `TARGET_SHAPE` then takes two values rather than
three. `MODEL=AlexNet` is 3D-only, so use `MODEL=Regressor`.

The survey itself lives in `src/data_sets/dims.py`; `scripts/utils/inspect_4d_dims.py`
is the standalone version to run before training, and prints the same
distribution.

### Handling the frame axis

There is no `nn.Conv4d`, and MONAI's convolution factory registers only 1/2/3
spatial dimensions, so a 4D file reaches the network one of three ways.

- `FRAME_MODE=pool` (default) encodes each frame with a shared 3D backbone and
  pools the per-frame predictions. Handles a variable frame count, and at one
  frame it is identical to the original 3D model, so `model_02r7` checkpoints
  load directly via `PRETRAINED_WEIGHTS`. The effective forward batch is
  `BATCH_SIZE x NUM_FRAMES`.
- `FRAME_MODE=channels` feeds frames as conv input channels. Cheaper, but needs
  a fixed frame count across every scan and cannot reuse a 1-channel checkpoint.
- `SPLIT_FRAMES=true` makes each frame an independent 3D sample. The forward
  pass matches `pool` -- that already encodes frames separately -- but the loss
  does not:

| | `FRAME_MODE=pool` | `SPLIT_FRAMES=true` |
|---|---|---|
| loss | `(mean(p1, p2) - y)^2` | `(p1 - y)^2 + (p2 - y)^2` |
| samples | one per file | one per frame (~2x) |
| network | `FrameWiseRegressor` wrapper | plain 3D regressor |

Frame counts come from the NIfTI header rather than `NUM_FRAMES`, so a file
carrying a single volume yields one sample instead of a duplicated pair. Both
frames of a scan share a subject and splits are made on `subject_id`, so no scan
lands with one frame in training and the other in validation.

Splitting interacts with normalization: a sliced frame is a 3D array, so
`NORMALIZE=global` becomes per-frame and erases intensity differences between
frames. `NORMALIZE=shared` keeps whole-acquisition statistics, which makes split
frames numerically identical to the planes of a pooled run -- so a comparison
between the two isolates the loss change alone.

## What the QC scores describe

Two settings, deliberately separate from how frames reach the network: how
volumes get into the model and what the ratings describe are different
questions.

**`LABEL_SCOPE`** says what one manual rating covers.

- `scan` -- one rating for the whole file. When frames are split they each
  inherit it. For `QU_line` that duplication was expensive: half the files hold
  two volumes a rater scores differently, and training on the duplicate took
  pooled `r` from 0.751 down to 0.470.
- `frame` -- one rating per volume. Requires a `frame_index` column in
  `RAW_CSV`; build it with
  `merge_qc_labels.py --frame-column <column>`, naming the column in the
  manual-rating export that holds the volume index.

**`PREDICTION_SCOPE`** says what the output CSV reports.

- `scan` -- one score per file, averaging its volumes. Comparable with a single
  manual rating.
- `frame` -- each volume's own prediction, one row per volume.
- `both` -- per-volume rows plus the scan-level mean, which carries an empty
  `frame_index`.

Averaging makes per-volume prediction differences identically zero, so any
measure of whether the model separates the two volumes needs `frame` or `both`.
Summary statistics and the scatter plot are always computed per scan, so they
stay comparable across settings.

## Network architecture

The MONAI Regressor is configurable from the arguments file: `NET_CHANNELS`,
`NET_STRIDES`, `NET_KERNEL_SIZE`, `NET_NUM_RES_UNITS`, `NET_DROPOUT`,
`NET_ACT`, `NET_NORM`, `NET_BIAS`. Leaving a value empty keeps MONAI's own
default rather than pinning it here. `MODEL=AlexNet` has a fixed architecture
and warns that it is ignoring these.

Channel depth has to suit the input: the `260x320x320` anatomical default was
seven stages, six stride-2 downsamples. At `96x96x64` the Z axis supports about
five before collapsing to a single voxel.

## Loss weighting

`USE_WEIGHTED_LOSS=true` weights a label value with count `n` by
`n ** -WEIGHT_POWER`.

`WEIGHT_POWER=1.0` is plain inverse frequency, and it is worth being explicit
about what that implies: every **distinct** label value receives the same total
weight, so the loss-optimal constant prediction becomes the mean of the value
*set* rather than of the sample distribution. For `QU_line`, whose values are
`{0, 0.3, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.7, 3.0}`, that optimum is 1.42
against a data mean of 0.31 -- which is why predictions from the weighted runs
sat near 1.0 and why `standardized_rmse`, scored against a mean predictor,
could never look good. A value rated once also outweighs a value rated 3459
times by 3459x.

`WEIGHT_POWER=0.5` keeps minority emphasis while cutting that ratio to
`sqrt(n)`. `0.0` is uniform, equivalent to plain MSE. `WEIGHT_CLIP` caps any
weight at a multiple of the median; `WEIGHT_NORMALIZE` puts the loss back on
the MSE scale without moving the optimum (it does change gradient magnitude, so
revisit `LR`).

## Augmentation

`AUGMENT=true` applies random left-right flips and small affine jitter to
training samples only, **after** the disk cache, so each epoch sees a fresh
perturbation. Before this was fixed, augmentation ran before memoization: one
random perturbation per sample was frozen into the cache on the first epoch and
replayed for every epoch and every fold thereafter.

## Workflow

```bash
# 1. Measure the cohort -- suggests TARGET_SHAPE and NUM_FRAMES
python scripts/utils/inspect_4d_dims.py "$FOLDER" --pattern '*_epi.nii.gz'
#    (scripts/utils/inspect_4d_dims_fsl.sh is an fslhd equivalent needing no venv)

# 2. Edit config/arguments.txt with the suggested values

# 3. Build fold assignments
sbatch scripts/utils/prepare_stratified_kfolds.sh

# 4. Submit all folds
./scripts/utils/submit_4d_kfold.sh
```

Step 1 still matters with `INPUT_MODE=auto`: auto settles dimensionality and
frame counts, but not `TARGET_SHAPE`, and the defaults in `dsets.py` are
placeholders.

Fold preparation verifies its own output -- it checks the Python exit status and
counts the fold files, rather than reporting success because the last `echo`
succeeded.

## Reporting results

`doc/analysis/fmap_qc_results.qmd` renders out-of-fold metrics, the scanner
confound decomposition, screening thresholds with ROC, per-frame agreement and a
model comparison:

```bash
quarto render doc/analysis/fmap_qc_results.qmd -P models:"model_a,model_b"
```

Models are grouped by the set of scans they were scored on, so a model evaluated
on an artifact-enriched subset is never tabled against one evaluated on the full
cohort. `scripts/utils/fmap_results_analyze.py` is the Python equivalent for the
metrics and figures, without the screening tables.

## Notes on model naming schema

Current schema is `model_##{r/s}#`

- First set of numbers (`##`) signifies a unique combination of the hyperparameters and size of the dataset. As of June 2026, the stable iteration is `02`.

- The {r/s} signification describes the preprocessing done on the dataset prior to training. `r` stands for "registration," `s` stands for "skull-stripping," `rs` notes that both were completed, and if these are missing from the model name then no preprocessing was completed. As of June 2026, testing as revealed that a rigid+affine registration without skull-stripping performs the best, so `r` is the current stable configuration.

- The last number (`#`) signifies the job testing iteration after ground truth score correcting or codebase changes. As of June 2026, the most recent iteration count is at `7`, the first with cross validation fully built in to the training workflow. 

## Tensorboard set-up test (after executing a training job)

### 1. Find your log directory

The logs are being written to:

```bash
~/projects/automated-qc/src/training/runs/<tb_prefix>/<time_str>-trn_cls-<comment>/
~/projects/automated-qc/src/training/runs/<tb_prefix>/<time_str>-val_cls-<comment>/
```

### 2. Launch TensorBoard on the cluster

SSH into your cluster and run:

```bash
cd ~/projects/automated-qc/src/training
~/projects/automated-qc/.venv/bin/tensorboard --logdir=runs --port=6006 --bind_all
```

### 3. Create an SSH tunnel from your local machine

On your local machine, open a terminal and run:

```bash
ssh -L 6006:localhost:6006 <x500>@<your-cluster-hostname>
```

Replace `<your-cluster-hostname>` with your actual cluster address (e.g., `login.msi.umn.edu` or similar for UMN systems).

### 4. Open TensorBoard in your browser

Navigate to:

<http://localhost:6006>

You should now see your training metrics updating in real-time!

Tips:

- Keep the SSH tunnel open while you want to monitor training

- If port 6006 is already in use, try a different port (e.g., --port=6007)

- The --bind_all flag allows TensorBoard to be accessible from any network interface

Alternative if you're on MSI (UMN): MSI may have a web portal or specific instructions for port forwarding. Check their documentation or use their OnDemand portal if available.
