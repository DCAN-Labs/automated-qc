#!/usr/bin/env python3
"""
Copy files from source to destination based on CSV inclusion/exclusion criteria.
"""

import argparse
import csv
import shutil
from pathlib import Path


def load_scan_list(csv_path, scan_column='scan'):
    """Load scan identifiers from a CSV file, preserving order."""
    scans = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scans.append(row[scan_column])
    return scans


def main():
    parser = argparse.ArgumentParser(
        description='Copy files based on CSV inclusion/exclusion criteria'
    )
    parser.add_argument('--csv', required=True, help='Main CSV file with scans to include')
    parser.add_argument('--src', required=True, help='Source directory')
    parser.add_argument('--dst', required=True, help='Destination directory')
    parser.add_argument('--exclude-csv', help='CSV file with scans to exclude')
    parser.add_argument('--limit', type=int, help='Maximum number of files to copy')
    
    args = parser.parse_args()
    
    # Load scan lists
    print(f"Loading scans from {args.csv}...")
    include_scans = load_scan_list(args.csv)
    print(f"  Found {len(include_scans)} scans to include")
    
    exclude_scans = set()
    if args.exclude_csv:
        print(f"Loading scans to exclude from {args.exclude_csv}...")
        exclude_scans = set(load_scan_list(args.exclude_csv))
        print(f"  Found {len(exclude_scans)} scans to exclude")
    
    # Determine final scan list (preserve order from main CSV)
    final_scans = [scan for scan in include_scans if scan not in exclude_scans]
    print(f"\nAfter exclusions: {len(final_scans)} scans to copy")
    
    # Apply limit if specified
    if args.limit and len(final_scans) > args.limit:
        final_scans = final_scans[:args.limit]
        print(f"Applied limit: copying {args.limit} scans")
    
    # Setup directories
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print(f"\nCopying files from {src_dir} to {dst_dir}...")
    copied = 0
    not_found = 0
    
    for scan in final_scans:
        src_file = src_dir / scan
        
        if not src_file.exists():
            not_found += 1
            print(f"  Warning: {scan} not found in source directory")
            continue
        
        dst_file = dst_dir / scan
        shutil.copy2(src_file, dst_file)
        copied += 1
        
        if copied % 100 == 0:
            print(f"  Copied {copied} files...")
    
    print(f"\nComplete!")
    print(f"  Successfully copied: {copied}")
    print(f"  Not found: {not_found}")


if __name__ == '__main__':
    main()