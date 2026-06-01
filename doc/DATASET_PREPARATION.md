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
     - Remove the source directory of registered scans

## Detailed Instructions

### Step 1: Update scores as needed in the csv-input-file

### Step 2: Prepare scans training folder