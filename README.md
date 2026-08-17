# Automated MRI QC training for HBCD data - Quick Start Guide

## .venv set up on MSI

### 1. Clone the repository in a user specific `projects` directory (i.e. `~/projects/`):
Do not clone in a shared shared directory, as you will be setting up a virtual environment and do not want your changes conflicting with others. 

`git clone git@github.com:DCAN-Labs/automated-qc.git`

### 2. If you have already set up your virtual environment, activate it and skip step 3:

example command if you have already done step 3: `source ~/projects/automated-qc/.venv/bin/activate`

### 3. Set up your virtual environment:

Note that this is specific for MSI users.

`cd ~/projects/dcan-nnunet-v2/`

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

## 4D input (field maps)

The pipeline accepts 4D scans as well as 3D anatomicals. Since there is no
`nn.Conv4d` and MONAI's convolution factory only registers 1/2/3 spatial
dimensions, the frame axis is handled one of two ways:

- `FRAME_MODE=pool` (default) encodes each frame with a shared 3D backbone and
  pools the per-frame predictions. Handles a variable frame count, and at one
  frame it is identical to the original 3D model, so `model_02r7` checkpoints
  load directly via `PRETRAINED_WEIGHTS`.
- `FRAME_MODE=channels` feeds frames as conv input channels. Cheaper, but needs
  a fixed frame count across every scan and cannot reuse a 1-channel checkpoint.

Note that with `pool` the effective forward batch is `BATCH_SIZE x NUM_FRAMES`.

### Workflow

```bash
# 1. Measure the cohort -- sets TARGET_SHAPE and NUM_FRAMES
python scripts/utils/inspect_4d_dims.py "$FOLDER" --pattern '*_epi.nii.gz'
#    (scripts/utils/inspect_4d_dims_fsl.sh is an fslhd equivalent needing no venv)

# 2. Edit config/arguments.txt with the suggested values

# 3. Build fold assignments
sbatch scripts/utils/prepare_stratified_kfolds.sh

# 4. Submit all folds
./scripts/utils/submit_4d_kfold.sh
```

Step 1 matters: the defaults in `dsets.py` are placeholders. It also reports any
files that declare `dim0=4` while carrying `dim4=1` -- a single volume wearing a
4D header, which would otherwise be silently edge-padded up to `NUM_FRAMES` by
duplicating that one frame.

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
