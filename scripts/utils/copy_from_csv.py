#!/usr/bin/env python3
"""
Copy files from source to destination based on CSV inclusion/exclusion criteria.
"""

import argparse
import csv
import shutil
from pathlib import Path


def load_scan_list(csv_path, scan_column='scan'):
    """Load scan identifiers from a CSV file, preserving order and row data."""
    scans = []
    rows_data = {}  # Map from scan to full row data
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames  # Preserve original field names
        for row in reader:
            scan = row[scan_column]
            scans.append(scan)
            rows_data[scan] = row
    return scans, rows_data, fieldnames


def write_output_csv(output_path, copied_rows):
    """Write CSV with metadata for copied files."""
    if not copied_rows:
        print(f"No files were copied, skipping output CSV")
        return
    
    # Extract columns if they exist in the data
    fieldnames = ['subject_id', 'session_id', 'run_id', 'suffix', 'scan', 'QU_motion']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for row in copied_rows:
            # Only write fields that exist in the row
            output_row = {field: row.get(field, '') for field in fieldnames}
            writer.writerow(output_row)
    
    print(f"Output CSV written to {output_path} with {len(copied_rows)} rows")


def main():
    parser = argparse.ArgumentParser(
        description='Copy files based on CSV inclusion/exclusion criteria'
    )
    parser.add_argument('--csv', required=True, help='Main CSV file with scans to include')
    parser.add_argument('--src', required=True, help='Source directory')
    parser.add_argument('--dst', required=True, help='Destination directory')
    parser.add_argument('--exclude-csv', help='CSV file with scans to exclude')
    parser.add_argument('--limit', type=int, help='Maximum number of files to copy')
    parser.add_argument('--output-csv', help='Output CSV with metadata of copied files')
    
    args = parser.parse_args()
    
    # Load scan lists
    print(f"Loading scans from {args.csv}...")
    include_scans, include_rows, _ = load_scan_list(args.csv)
    print(f"  Found {len(include_scans)} scans to include")
    
    exclude_scans = set()
    exclude_rows = {}
    if args.exclude_csv:
        print(f"Loading scans to exclude from {args.exclude_csv}...")
        exclude_scans_list, exclude_rows, _ = load_scan_list(args.exclude_csv)
        exclude_scans = set(exclude_scans_list)
        print(f"  Found {len(exclude_scans)} scans to exclude")
    
    # Determine final scan list (preserve order from main CSV)
    final_scans = [scan for scan in include_scans if scan not in exclude_scans]
    ignored_count = len(include_scans) - len(final_scans)
    print(f"\nAfter exclusions: {len(final_scans)} scans to copy ({ignored_count} ignored)")
    
    # Note: limit applies to successfully copied files, not the candidate list
    if args.limit:
        print(f"Target: copying {args.limit} scans (will continue through candidate list if some files are not found)")
    
    # Setup directories
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files and track copied rows
    print(f"\nCopying files from {src_dir} to {dst_dir}...")
    copied = 0
    not_found = 0
    copied_rows = []
    
    for scan in final_scans:
        # Stop if we've reached the limit
        if args.limit and copied >= args.limit:
            break
        
        src_file = src_dir / scan
        
        if not src_file.exists():
            not_found += 1
            print(f"  Warning: {scan} not found in source directory")
            continue
        
        dst_file = dst_dir / scan
        shutil.copy2(src_file, dst_file)
        copied += 1
        copied_rows.append(include_rows[scan])
        
        if copied % 100 == 0:
            print(f"  Copied {copied} files...")
    
    print(f"\nComplete!")
    print(f"  Successfully copied: {copied}")
    print(f"  Not found: {not_found}")
    
    # Write output CSV if requested
    if args.output_csv:
        write_output_csv(args.output_csv, copied_rows)


if __name__ == '__main__':
    main()