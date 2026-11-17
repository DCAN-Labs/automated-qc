# Dataset preparation using HBCD data

This documents how to prepare the csv-input-file and the curated HBCD dataset after any score corrections. 

## Workflow Overview

This process is split into X steps. 

1. **Step 1: Update scores as needed in the csv-input-file**
     - Load in corrected scores and previous scores
     - Can utilize `score_fix_CV.ipynb` to update any previous scores with corrected scores
     - Save out new csv-input-file

2. **Step 2: Prepare scans training folder**
     - Copy all registered files down from s3 to scratch
     - Use `copy_from_csv.py` to move necessary scans to the new training dataset folder

## Detailed Instructions

### Step 1: Update scores as needed in the csv-input-file

Refer to any recent traking sheets as needed. It is suggested to create a new tracking sheet then copy it over as a csv format after updating scores for new model training iterations. QU_motion scores of anatomical images for model_02r8 are up to date. Any changes to scores requires creating a new csv-input-file of the following format:

```
subject_id,session_id,run_id,suffix,scan,QU_motion
sub-######,ses-V0#,#,T#w,sub-######_ses-V0#_run-#_T#w.nii.gz,#.#
...
...
```

Follow the notebook `score_fix_CV.ipynb` within `automated-qc/scripts/utils/data_notebooks/` and replace csv variables with full paths to any new csvs. After creating a newly updated csv, proceed to setting up the dataset in step 2.

### Step 2: Prepare scans training folder

First you will need to copy over the relevant files from the registered data directory: `/projects/standard/feczk001/shared/data/automated-qc/hbcd-T1w-T2w-scans-RigidAffine-registered/`

This process is streamlined with the `copy_from_csv.py` script available in `automated-qc/scripts/utils/`

Please refer to the following example command when setting arguments for running the script:

```
python automated-qc/scripts/utils/copy_from_csv.py --csv automated-qc/data/model_##r#/csv_name.csv --src /projects/standard/feczk001/shared/data/automated-qc/hbcd-T1w-T2w-scans-RigidAffine-registered/ --dst /somewhere/on/scratch.global/
```

Note that `copy_from_csv.py` also has the ability to include an additional csv of the same format to make sure the copy command skips these specified files, a limit if you only want the first set of rows copied, or an output csv if you would like information on files that were copied.