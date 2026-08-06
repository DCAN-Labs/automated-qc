#!/usr/bin/env python3
"""Survey NIfTI dimensions with nibabel before choosing --target-shape / --num-frames.

Reads headers only (no voxel data), so it is fast even over thousands of field
maps. Run this first: the defaults in dsets.py are placeholders and should be
replaced with numbers measured from the actual cohort.

This reports the raw NIfTI header fields rather than nibabel's interpreted
`img.shape`, so the output matches `fslhd` field for field:

    dim0 = how many dimensions the file declares (3 or 4)
    dim1, dim2, dim3 = X, Y, Z
    dim4 = how many volumes/frames

Those last two disagree more often than expected. A file can declare dim0=4
while carrying dim4=1, which is a single volume wearing a 4D header. It will
load fine and then get edge-padded up to --num-frames by duplicating that one
volume, producing a valid tensor with fabricated temporal content. Those files
are listed separately so they can be excluded or investigated.

Usage:
    python inspect_4d_dims.py /path/to/fmaps --pattern '*_epi.nii.gz'
    python inspect_4d_dims.py /path/to/fmaps --csv folds.csv
"""

import argparse
import os
import glob
from collections import Counter

import nibabel as nib
import numpy as np


def collect_paths(directory, pattern, csv_path):
    """Resolve the list of files to inspect, either by glob or from a CSV."""
    if csv_path:
        import pandas as pd

        df = pd.read_csv(csv_path)
        if "scan" in df.columns:
            names = df["scan"].dropna().astype(str)
        else:
            names = (
                df["subject_id"].astype(str)
                + "_"
                + df["session_id"].astype(str)
                + "_run-"
                + df["run_id"].astype(str)
                + "_"
                + df["suffix"].astype(str)
                + ".nii.gz"
            )
        return [os.path.join(directory, n) for n in names]

    return sorted(glob.glob(os.path.join(directory, pattern)))


def read_dims(path):
    """Return (dim0, dim1, dim2, dim3, dim4) straight from the NIfTI header.

    Uses the raw `dim` field rather than img.shape so the numbers line up
    exactly with `fslhd`. nibabel maps the file lazily, so no voxel data is
    touched here.
    """
    header = nib.load(path).header
    dim = header["dim"]
    return tuple(int(dim[i]) for i in range(5))


def round_up_8(value):
    """Round up to a multiple of 8 so repeated stride-2 downsampling stays clean."""
    return int(np.ceil(value / 8) * 8)


def stride_2_stages(smallest_dim):
    """How many stride-2 stages a dimension can absorb before hitting 1 voxel."""
    stages, current = 0, smallest_dim
    while current // 2 >= 2:
        current //= 2
        stages += 1
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory containing NIfTI files")
    parser.add_argument("--pattern", default="*.nii.gz", help="Glob pattern")
    parser.add_argument("--csv", default=None, help="Read file names from this CSV")
    parser.add_argument(
        "--percentile",
        type=float,
        default=100.0,
        help="Percentile used for the suggested spatial target (default 100, "
        "i.e. cover every scan without cropping).",
    )
    args = parser.parse_args()

    paths = collect_paths(args.directory, args.pattern, args.csv)
    if not paths:
        print(f"No files matched {os.path.join(args.directory, args.pattern)}")
        return

    records, missing, unreadable = [], [], []

    for path in paths:
        if not os.path.exists(path):
            missing.append(path)
            continue
        try:
            records.append((read_dims(path), path))
        except Exception as exc:
            unreadable.append((path, exc))

    print(
        f"Scanned {len(paths)} paths "
        f"({len(missing)} missing, {len(unreadable)} unreadable)"
    )

    if missing:
        print("\nFirst few missing files:")
        for path in missing[:5]:
            print(f"  {path}")

    if unreadable:
        print("\nUnreadable files:")
        for path, exc in unreadable[:5]:
            print(f"  {path}: {exc}")

    if not records:
        return

    print("\n=== shape histogram (count | dim0 dim1 dim2 dim3 dim4) ===")
    for dims, count in Counter(d for d, _ in records).most_common():
        print(f"  {count:>6}  {' '.join(str(v) for v in dims)}")

    print("\n=== frame counts (dim4) ===")
    for frames, count in sorted(Counter(d[4] for d, _ in records).items()):
        print(f"  {count:>6}  {frames} frames")

    print("\n=== files that are not genuinely 4D ===")
    not_4d = [(d, p) for d, p in records if d[0] < 4 or d[4] < 2]
    if not not_4d:
        print("  (none)")
    else:
        for dims, path in not_4d[:20]:
            print(f"  {path}  dim0={dims[0]} dim4={dims[4]}")
        if len(not_4d) > 20:
            print(f"  ... and {len(not_4d) - 20} more ({len(not_4d)} total)")

    spatial = np.array([d[1:4] for d, _ in records])
    frames = np.array([d[4] for d, _ in records])

    target = tuple(
        round_up_8(v) for v in np.percentile(spatial, args.percentile, axis=0)
    )
    median_frames = int(np.median(frames))

    print("\n=== suggested flags ===")
    print(f"  --target-shape {','.join(str(v) for v in target)}")
    print(f"  --num-frames {median_frames}")

    cropped = int((spatial > np.array(target)).any(axis=1).sum())
    if cropped:
        print(
            f"\n  {cropped} scans exceed this target on at least one axis and "
            "would be center-cropped."
        )

    smallest = min(target)
    stages = stride_2_stages(smallest)
    print(
        f"\n  Smallest target dim is {smallest}, supporting about {stages} "
        "stride-2 stages.\n  Default --net-strides is 2,2,2,2,2 (5 stages) "
        "against a 260x320x320\n  anatomical; trim it if the number above is smaller."
    )

    distinct = len(set(frames.tolist()))
    if distinct > 1:
        print(
            f"\n  NOTE: frame count varies across the cohort "
            f"({distinct} distinct values),\n"
            "  so --frame-mode pool is the safer choice than channels."
        )


if __name__ == "__main__":
    main()
