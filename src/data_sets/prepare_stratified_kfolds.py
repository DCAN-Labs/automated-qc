"""
Prepare stratified k-fold splits for cross-validation.

This script creates stratified k-fold splits based on QU_motion score distribution
and saves the fold assignments. These fold assignments can then be used by the
training script to train models on stratified data.

Usage:
    python prepare_stratified_kfolds.py \
        --csv-input-file data.csv \
        --output-dir fold_assignments \
        --k-folds 5 \
        --random-seed 42
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def create_stratified_kfolds(csv_file, output_dir, k_folds=5, random_seed=42):
    """
    Create stratified k-fold splits based on QU_motion scores.
    
    Args:
        csv_file (str): Path to input CSV file
        output_dir (str): Directory to save fold assignments
        k_folds (int): Number of folds (default: 5)
        random_seed (int): Random seed for reproducibility (default: 42)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the CSV file
    log.info(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Get unique subjects
    all_subjects = sorted(df['subject_id'].unique())
    log.info(f"Found {len(all_subjects)} unique subjects")
    
    # Calculate average QU_motion per subject for stratification
    subject_motion_map = {}
    for subject in all_subjects:
        subject_rows = df[df['subject_id'] == subject]
        avg_motion = subject_rows['QU_motion'].mean()
        subject_motion_map[subject] = avg_motion
    
    # Create stratification labels by binning QU_motion scores
    motion_values = np.array([subject_motion_map[s] for s in all_subjects])
    
    try:
        # Create k strata based on percentiles
        strata = pd.qcut(motion_values, q=k_folds, labels=False, duplicates='drop')
        log.info(f"Created {len(np.unique(strata))} strata based on QU_motion percentiles")
    except Exception as e:
        log.warning(f"Percentile-based stratification failed ({e}), falling back to random k-fold")
        strata = np.random.RandomState(random_seed).randint(0, k_folds, len(all_subjects))
    
    # Create stratified k-fold splits
    skfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    # Store fold assignments
    fold_assignments = {}
    fold_details = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(skfold.split(all_subjects, strata)):
        train_subjects = [all_subjects[i] for i in train_indices]
        val_subjects = [all_subjects[i] for i in val_indices]
        
        # Log statistics for this fold
        train_motion_scores = [subject_motion_map[s] for s in train_subjects]
        val_motion_scores = [subject_motion_map[s] for s in val_subjects]
        
        log.info(f"\nFold {fold_idx}:")
        log.info(f"  Training: {len(train_subjects)} subjects, "
                f"QU_motion mean={np.mean(train_motion_scores):.3f}, "
                f"std={np.std(train_motion_scores):.3f}")
        log.info(f"  Validation: {len(val_subjects)} subjects, "
                f"QU_motion mean={np.mean(val_motion_scores):.3f}, "
                f"std={np.std(val_motion_scores):.3f}")
        
        # Store assignments - each subject gets ONE validation fold (and appears in other training folds)
        # Validation fold assignment (one per subject) - ALWAYS set when subject is in validation
        for subject in val_subjects:
            if subject not in fold_assignments:
                fold_assignments[subject] = {
                    'validation_fold': fold_idx,
                    'avg_qu_motion': float(subject_motion_map[subject]),
                    'training_folds': []
                }
            else:
                # Subject already exists (from training in previous fold), now set their validation fold
                fold_assignments[subject]['validation_fold'] = fold_idx
        
        # Training fold assignments (many per subject)
        for subject in train_subjects:
            if subject not in fold_assignments:
                fold_assignments[subject] = {
                    'validation_fold': None,
                    'avg_qu_motion': float(subject_motion_map[subject]),
                    'training_folds': []
                }
            fold_assignments[subject]['training_folds'].append(fold_idx)
        
        fold_details.append({
            'fold': fold_idx,
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'train_count': len(train_subjects),
            'val_count': len(val_subjects),
            'train_qu_motion_mean': float(np.mean(train_motion_scores)),
            'train_qu_motion_std': float(np.std(train_motion_scores)),
            'val_qu_motion_mean': float(np.mean(val_motion_scores)),
            'val_qu_motion_std': float(np.std(val_motion_scores)),
        })
        
        # CREATE SUBSET CSV for this fold
        # Include rows where subject is in either train or val subjects
        fold_subjects = set(train_subjects + val_subjects)
        fold_df = df[df['subject_id'].isin(fold_subjects)].copy()
        
        # Add training/validation columns
        fold_df['training'] = fold_df['subject_id'].apply(lambda x: 1 if x in train_subjects else 0)
        fold_df['validation'] = fold_df['subject_id'].apply(lambda x: 1 if x in val_subjects else 0)
        
        # Save fold-specific CSV
        fold_csv_path = os.path.join(output_dir, f'fold_{fold_idx}_subset.csv')
        fold_df.to_csv(fold_csv_path, index=False)
        log.info(f"  Saved fold subset CSV to: {fold_csv_path}")
    
    # Save fold assignments as JSON
    assignments_file = os.path.join(output_dir, 'fold_assignments.json')
    with open(assignments_file, 'w') as f:
        json.dump(fold_assignments, f, indent=2)
    log.info(f"\nSaved fold assignments to: {assignments_file}")
    
    # Save fold details as JSON
    details_file = os.path.join(output_dir, 'fold_details.json')
    with open(details_file, 'w') as f:
        json.dump(fold_details, f, indent=2)
    log.info(f"Saved fold details to: {details_file}")
    
    # Also save as CSV for easy viewing
    assignments_csv = os.path.join(output_dir, 'fold_assignments.csv')
    assignments_df = pd.DataFrame([
        {
            'subject_id': subject,
            'validation_fold': data['validation_fold'],
            'training_folds': ','.join(map(str, data['training_folds'])),
            'avg_qu_motion': data['avg_qu_motion']
        }
        for subject, data in fold_assignments.items()
    ])
    # Convert validation_fold to nullable integer type (preserves ints and nulls, not floats)
    assignments_df['validation_fold'] = assignments_df['validation_fold'].astype('Int64')
    # Sort by validation fold for easier viewing
    assignments_df = assignments_df.sort_values('validation_fold').reset_index(drop=True)
    assignments_df.to_csv(assignments_csv, index=False)
    log.info(f"Saved fold assignments CSV to: {assignments_csv}")
    
    log.info("\n" + "="*60)
    log.info("K-FOLD PREPARATION COMPLETE")
    log.info("="*60)
    log.info(f"Total subjects: {len(all_subjects)}")
    log.info(f"Number of folds: {k_folds}")
    log.info(f"Random seed: {random_seed}")
    log.info(f"\nOutput files:")
    log.info(f"  - {assignments_file}")
    log.info(f"  - {details_file}")
    log.info(f"  - {assignments_csv}")
    for fold_idx in range(k_folds):
        fold_csv_path = os.path.join(output_dir, f'fold_{fold_idx}_subset.csv')
        log.info(f"  - {fold_csv_path}")
    
    return fold_assignments, fold_details


def main():
    parser = argparse.ArgumentParser(
        description="Prepare stratified k-fold splits for cross-validation"
    )
    parser.add_argument(
        "--csv-input-file",
        required=True,
        help="Path to input CSV file with subject_id, session_id, QU_motion columns"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save fold assignment files"
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of folds (default: 5)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.csv_input_file):
        log.error(f"CSV file not found: {args.csv_input_file}")
        sys.exit(1)
    
    # Create stratified k-folds
    create_stratified_kfolds(
        csv_file=args.csv_input_file,
        output_dir=args.output_dir,
        k_folds=args.k_folds,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
