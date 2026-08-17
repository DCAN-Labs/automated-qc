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


# Column-name aliases seen in HBCD manual-QC exports.
LABEL_ALIASES = {
    "subject": "subject_id",
    "participant_id": "subject_id",
    "session": "session_id",
    "ses": "session_id",
}


def basename_key(value):
    """Reduce any path to its bare filename, for use as a join key.

    Label exports and generated manifests disagree about how much path to
    carry: an export may say `fmap/sub-X_ses-Y_dir-AP_run-1_epi.nii.gz` while
    the manifest holds `sub-X/ses-Y/fmap/sub-X_ses-Y_dir-AP_run-1_epi.nii.gz`.
    The basename is identical in both and already unique across the dataset,
    since it encodes subject, session, direction and run.
    """
    if not isinstance(value, str):
        return None
    return os.path.basename(value.strip()) or None


def merge_labels(df, labels_path, label_column, status_column="qc_status",
                 status_value="review complete", drop_unrated=True):
    """Join ratings from a separate QC export onto the manifest.

    Joins on filename basename, which survives the differing path conventions
    between exports and generated manifests. Falls back to `scan` or the
    subject/session/run/dir tuple when no filename-like column exists.

    Rows without a rating are dropped by default: the pipeline calls
    float(row["QU_motion"]) per sample and takes a per-subject mean for
    stratification, so a single NaN breaks both.
    """
    sep = "\t" if labels_path.endswith((".tsv", ".txt")) else ","
    labels = pd.read_csv(labels_path, sep=sep)

    print(f"  labels file: {len(labels)} rows, columns: {', '.join(labels.columns)}")

    if label_column not in labels.columns:
        raise SystemExit(
            f"--label-column {label_column!r} not in {labels_path}. "
            f"Available: {', '.join(labels.columns)}"
        )

    labels = labels.rename(columns={k: v for k, v in LABEL_ALIASES.items()
                                    if k in labels.columns})

    # Keep only completed reviews, when the export says which are complete.
    if status_column in labels.columns and status_value:
        before = len(labels)
        keep = labels[status_column].astype(str).str.strip().str.lower()
        labels = labels[keep == status_value.strip().lower()]
        if len(labels) != before:
            print(f"  {status_column}: kept {len(labels)} of {before} rows "
                  f"matching '{status_value}'")

    # Pick a join strategy.
    path_col = next((c for c in ("scan", "filename", "file", "path")
                     if c in labels.columns), None)

    if path_col:
        print(f"  joining on basename of '{path_col}'")
        labels = labels.copy()
        labels["_join_key"] = labels[path_col].map(basename_key)
        left = df.copy()
        left["_join_key"] = left["scan"].map(basename_key)
        keys = ["_join_key"]
    else:
        keys = [k for k in ("subject_id", "session_id", "run_id", "dir")
                if k in labels.columns]
        if not keys:
            raise SystemExit(
                f"{labels_path} has no filename-like column and no "
                "subject/session keys to join on."
            )
        print(f"  joining on {keys}")
        left = df.copy()

    dupes = labels.duplicated(subset=keys).sum()
    if dupes:
        print(f"  WARNING: {dupes} duplicate join keys in the labels file; "
              "keeping the first occurrence of each.")
        labels = labels.drop_duplicates(subset=keys, keep="first")

    extra = [c for c in ("scanner_manufacturer", "nrev") if c in labels.columns]

    merged = left.drop(columns=["QU_motion"]).merge(
        labels[keys + [label_column] + extra], on=keys, how="left"
    )
    merged = merged.drop(columns=["_join_key"], errors="ignore")
    merged = merged.rename(columns={label_column: "QU_motion"})
    merged["QU_motion_source"] = f"{os.path.basename(labels_path)}:{label_column}"

    if len(merged) != len(df):
        print(f"  WARNING: row count changed {len(df)} -> {len(merged)}")

    matched = int(merged["QU_motion"].notna().sum())
    print(f"  matched {matched} of {len(merged)} manifest rows")

    unmatched_labels = len(labels) - matched
    if unmatched_labels > 0:
        print(f"  {unmatched_labels} label rows had no matching file on disk")

    if drop_unrated and matched < len(merged):
        missing = merged[merged["QU_motion"].isna()]
        print(f"\n  Dropping {len(missing)} unrated rows "
              f"({merged['subject_id'].nunique() - missing['subject_id'].nunique()} "
              "subjects retain at least one rated scan).")
        print("  First few unrated:")
        for scan in missing["scan"].head(3):
            print(f"    {scan}")
        merged = merged[merged["QU_motion"].notna()].reset_index(drop=True)

    return merged


def report_label_distribution(df):
    """Summarize the rating column, since real ratings are rarely uniform."""
    vals = df["QU_motion"].dropna()
    if vals.empty:
        return

    print("\n=== label distribution ===")
    counts = vals.value_counts().sort_index()
    total = len(vals)
    for value, count in counts.items():
        bar = "#" * max(1, int(60 * count / total))
        print(f"  {value:>6}  {count:>6}  {100*count/total:>5.1f}%  {bar}")
    print(f"  mean {vals.mean():.4f}  sd {vals.std():.4f}  "
          f"min {vals.min()}  max {vals.max()}")

    if vals.std() == 0:
        print("\n  ERROR: zero variance -- every scan has the same rating. "
              "Nothing to learn, and standardized_rmse will divide by zero.")
    elif vals.nunique() <= 3:
        top = counts.max() / total
        print(f"\n  NOTE: only {vals.nunique()} distinct values, "
              f"largest class {100*top:.0f}%. This is closer to a "
              "classification target than a continuous one.")
        print("  standardized_rmse is error / label sd, so it stays "
              "interpretable, but consider USE_WEIGHTED_LOSS=true if the "
              "classes are lopsided.")

    if "scanner_manufacturer" in df.columns:
        print("\n=== by scanner ===")
        by = df.groupby("scanner_manufacturer")["QU_motion"].agg(
            ["count", "mean", "std"]).round(3)
        print(by.to_string())

    if "nrev" in df.columns and df["nrev"].notna().any():
        print("\n=== reviewers per scan ===")
        print("  " + df["nrev"].value_counts().sort_index().to_string().replace("\n", "\n  "))


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
    parser.add_argument("--status-column", default="qc_status",
                        help="Column marking review completion (empty to skip)")
    parser.add_argument("--status-value", default="review complete",
                        help="Value in --status-column that counts as complete")
    parser.add_argument("--keep-unrated", action="store_true",
                        help="Keep rows with no rating. The pipeline cannot train "
                             "on NaN labels, so only use this for inspection.")
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
        df = merge_labels(df, args.labels, args.label_column,
                          status_column=args.status_column,
                          status_value=args.status_value,
                          drop_unrated=not args.keep_unrated)
        report_label_distribution(df)

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
                    "scan", "QU_motion", "QU_motion_source",
                    "scanner_manufacturer", "nrev"]
    df[[c for c in column_order if c in df.columns]].to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
