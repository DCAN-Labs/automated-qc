#!/usr/bin/env python3
"""Build a training CSV from a BIDS-nested field map tree.

No staging or copying required. The `scan` column holds each file's path
relative to the tree root, so `os.path.join(FOLDER, scan)` resolves straight
into the nested layout:

    sub-105370/ses-V03/fmap/sub-105370_ses-V03_dir-AP_run-1_epi.nii.gz

Emits the columns the pipeline expects -- subject_id, session_id, run_id,
suffix, scan, QU_motion -- plus `dir` and `acq` when those entities are present.

QU_motion is left empty unless --labels supplies it. That column is the training
signal; the pipeline cannot run without it.

For pipeline smoke tests before real ratings exist, --dummy-labels fills it with
random values and tags every row QU_motion_source=SYNTHETIC. A model trained on
those labels should score r ~ 0; anything higher means something is leaking.

Usage:
    python build_fmap_csv.py /scratch.global/fayzu001/hbcd_fmap_qc_sorted \\
        --output data/model_04d0/auto_qc_fmap_curated.csv

    # Merge ratings from an existing inventory
    python build_fmap_csv.py <root> --output out.csv \\
        --labels fmap_inventory.tsv --label-column qu_motion

    # Synthetic labels for a fast smoke test
    python build_fmap_csv.py <root> --output data/DUMMY_fmap.csv \\
        --dummy-labels --limit-subjects 20
"""

import argparse
import glob
import os
import re

import pandas as pd

# BIDS entity extraction from a filename
ENTITY_RE = re.compile(r"(?:^|_)(sub|ses|acq|dir|run)-([A-Za-z0-9]+)")
SUFFIX_RE = re.compile(r"_([A-Za-z0-9]+)\.nii(?:\.gz)?$")


def parse_entities(filename):
    """Pull BIDS entities and the suffix out of a filename."""
    entities = dict(ENTITY_RE.findall(filename))
    suffix_match = SUFFIX_RE.search(filename)
    entities["suffix"] = suffix_match.group(1) if suffix_match else None
    return entities


def build_rows(root, pattern):
    """Walk the tree and produce one row per matching file."""
    paths = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))

    rows, skipped = [], []

    for path in paths:
        rel = os.path.relpath(path, root)
        entities = parse_entities(os.path.basename(path))

        if not entities.get("sub") or not entities.get("ses"):
            skipped.append(rel)
            continue

        rows.append(
            {
                "subject_id": f"sub-{entities['sub']}",
                "session_id": f"ses-{entities['ses']}",
                "run_id": int(entities["run"]) if entities.get("run") else 1,
                "suffix": entities.get("suffix"),
                "dir": entities.get("dir"),
                "acq": entities.get("acq"),
                "scan": rel,
                "QU_motion": pd.NA,
            }
        )

    return pd.DataFrame(rows), skipped


def merge_labels(df, labels_path, label_column):
    """Join QU_motion ratings from a separate inventory file.

    Joins on `scan` if the labels file has it, otherwise on the
    subject/session/run/dir combination -- which for field maps needs `dir`,
    since AP and PA share every other identifier.
    """
    sep = "\t" if labels_path.endswith((".tsv", ".txt")) else ","
    labels = pd.read_csv(labels_path, sep=sep)

    if label_column not in labels.columns:
        raise SystemExit(
            f"--label-column {label_column!r} not in {labels_path}. "
            f"Available: {', '.join(labels.columns)}"
        )

    if "scan" in labels.columns:
        keys = ["scan"]
    else:
        keys = [k for k in ("subject_id", "session_id", "run_id", "dir") if k in labels.columns]
        if not keys:
            raise SystemExit(
                f"{labels_path} has no `scan` column and no subject/session keys "
                "to join on."
            )
        print(f"  joining on {keys} (no `scan` column in the labels file)")

    merged = df.drop(columns=["QU_motion"]).merge(
        labels[keys + [label_column]], on=keys, how="left"
    )
    merged = merged.rename(columns={label_column: "QU_motion"})

    if len(merged) != len(df):
        print(
            f"  WARNING: row count changed {len(df)} -> {len(merged)}. The join "
            "keys are probably not unique in the labels file."
        )

    return merged


def add_dummy_labels(df, low, high, step, seed):
    """Fill QU_motion with random ratings for pipeline testing.

    Every row is tagged QU_motion_source=SYNTHETIC so a dummy CSV can never be
    mistaken for real ratings later. Seeded, so re-running on the same tree
    reproduces the same labels and fold assignments.

    These labels carry no signal by construction. That is the point: a model
    trained on them should land near r = 0. A meaningfully positive correlation
    would mean information is leaking from somewhere it should not.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n_steps = int(round((high - low) / step)) + 1
    choices = [round(low + i * step, 4) for i in range(n_steps)]

    df = df.copy()
    df["QU_motion"] = rng.choice(choices, size=len(df))
    df["QU_motion_source"] = "SYNTHETIC"

    print("\n" + "!" * 74)
    print("!! SYNTHETIC LABELS -- FOR PIPELINE TESTING ONLY")
    print(f"!! {len(df)} rows filled with random values from {choices}")
    print(f"!! seed={seed}, every row tagged QU_motion_source=SYNTHETIC")
    print("!! A model trained on these should score r ~ 0. Higher means a leak.")
    print("!" * 74)

    return df


def warn_if_not_obviously_dummy(output_path):
    """Nudge toward a filename that cannot be confused with real ratings."""
    name = os.path.basename(output_path).lower()
    if "dummy" not in name and "synthetic" not in name and "test" not in name:
        print(
            f"\n  SUGGESTION: rename {os.path.basename(output_path)} to include "
            "'DUMMY'.\n  When the real ratings arrive, filename is the fastest way "
            "to tell them apart."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Root of the BIDS-nested tree")
    parser.add_argument("--output", required=True, help="CSV to write")
    parser.add_argument("--pattern", default="*_epi.nii.gz",
                        help="Filename pattern (default excludes TB1AFI etc.)")
    parser.add_argument("--labels", default=None,
                        help="CSV/TSV holding QU_motion ratings to merge in")
    parser.add_argument("--label-column", default="QU_motion",
                        help="Column in --labels holding the rating")
    parser.add_argument("--limit-subjects", type=int, default=None,
                        help="Keep only the first N subjects. Use for smoke tests "
                             "so a run finishes in minutes rather than days.")
    parser.add_argument("--dummy-labels", action="store_true",
                        help="Fill QU_motion with random values for pipeline "
                             "testing. Tags every row QU_motion_source=SYNTHETIC.")
    parser.add_argument("--dummy-min", type=float, default=1.0,
                        help="Lowest synthetic rating (default 1.0)")
    parser.add_argument("--dummy-max", type=float, default=4.0,
                        help="Highest synthetic rating (default 4.0)")
    parser.add_argument("--dummy-step", type=float, default=0.5,
                        help="Synthetic rating granularity (default 0.5)")
    parser.add_argument("--dummy-seed", type=int, default=42,
                        help="Seed, so the same tree gives the same dummy labels")
    args = parser.parse_args()

    if args.dummy_labels and args.labels:
        raise SystemExit("--dummy-labels and --labels are mutually exclusive.")

    df, skipped = build_rows(args.root, args.pattern)

    if df.empty:
        raise SystemExit(f"No files matched {args.pattern} under {args.root}")

    print(f"Found {len(df)} files matching {args.pattern}")
    if skipped:
        print(f"  skipped {len(skipped)} without sub-/ses- entities, e.g. {skipped[0]}")

    print(f"  subjects: {df['subject_id'].nunique()}")
    print(f"  sessions: {df.groupby(['subject_id', 'session_id']).ngroups}")

    for col in ("dir", "run_id", "suffix"):
        if df[col].notna().any():
            counts = df[col].value_counts().to_dict()
            print(f"  {col}: {counts}")

    # Uniqueness matters: predictions are matched back by `scan`, and the
    # subject/session/run/suffix tuple alone collides for AP vs PA.
    dupes = df["scan"].duplicated().sum()
    if dupes:
        print(f"  ERROR: {dupes} duplicate scan paths")
    legacy_key = ["subject_id", "session_id", "run_id", "suffix"]
    legacy_dupes = df.duplicated(subset=legacy_key).sum()
    if legacy_dupes:
        print(
            f"\n  NOTE: {legacy_dupes} rows share a subject/session/run/suffix key "
            "(expected -- AP and PA differ only by `dir`).\n"
            "  Predictions are matched back by `scan`, which stays unique."
        )

    if args.limit_subjects:
        keep = sorted(df["subject_id"].unique())[: args.limit_subjects]
        df = df[df["subject_id"].isin(keep)].reset_index(drop=True)
        print(f"\nLimited to {len(keep)} subjects -> {len(df)} files")

    if args.labels:
        print(f"\nMerging ratings from {args.labels}")
        df = merge_labels(df, args.labels, args.label_column)

    if args.dummy_labels:
        df = add_dummy_labels(df, args.dummy_min, args.dummy_max,
                              args.dummy_step, args.dummy_seed)
        warn_if_not_obviously_dummy(args.output)

    rated = int(df["QU_motion"].notna().sum())
    print(f"\nRows with a QU_motion rating: {rated} / {len(df)}")
    if rated == 0:
        print("  The pipeline cannot train until this column is populated.")
    elif rated < len(df):
        print(f"  {len(df) - rated} rows unrated; drop or rate them before training.")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    column_order = ["subject_id", "session_id", "run_id", "suffix", "dir", "acq",
                    "scan", "QU_motion", "QU_motion_source"]
    df[[c for c in column_order if c in df.columns]].to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
