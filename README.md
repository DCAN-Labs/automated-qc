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
