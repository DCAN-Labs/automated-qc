#!/usr/bin/env python3
"""Join an HBCD manual-QC export onto a manifest built by build_fmap_csv.py.

Kept separate from build_fmap_csv.py so the manifest step and the labelling
step can be rerun independently -- useful when ratings arrive in batches.

Joins on filename basename: the export carries 'fmap/<name>.nii.gz' while the
manifest carries 'sub-X/ses-Y/fmap/<name>.nii.gz'. The basename is identical in
both and already unique, since it encodes subject, session, direction and run.

Usage:
    python merge_qc_labels.py MANIFEST.csv LABELS.tsv OUTPUT.csv --label-column QU_sus
"""

import argparse
import os
import sys

import pandas as pd

ALIASES = {"subject": "subject_id", "participant_id": "subject_id",
           "session": "session_id", "ses": "session_id"}


def base(v):
    return os.path.basename(str(v).strip()) if isinstance(v, str) else None


def check_frame_counts(out, folder, per_frame):
    """Compare ratings per file against the volume count in the NIfTI header.

    Some files declare dim0=4 while carrying a single volume (65 of 3520 in the
    QU_line cohort). A rater working from a viewer that always shows two panes
    could produce two ratings for such a file, and the second would reference a
    volume that does not exist. Reads headers only, no voxel data.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("\n--check-frames: nibabel not importable, skipping")
        return

    counts = out.groupby("scan").size() if per_frame else None
    bad_count, bad_index, missing = [], [], 0

    for scan in out["scan"].unique():
        path = os.path.join(folder, str(scan))
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            dim = nib.load(path).header["dim"]
            n = int(dim[4]) if int(dim[0]) >= 4 and int(dim[4]) > 0 else 1
        except Exception as exc:
            print(f"  could not read {scan}: {exc}")
            continue

        if per_frame:
            rated = int(counts[scan])
            if rated != n:
                bad_count.append((scan, rated, n))
            hi = out.loc[out["scan"] == scan, "frame_index"].max()
            if hi is not None and hi == hi and int(hi) >= n:
                bad_index.append((scan, int(hi), n))

    print(f"\n=== frame check against headers in {folder} ===")
    if missing:
        print(f"  {missing} scans not found on disk")
    if not per_frame:
        print("  scan-level labels: nothing to reconcile")
        return
    if bad_count:
        print(f"  {len(bad_count)} files whose rating count != volume count:")
        for s, r, n in bad_count[:5]:
            print(f"    {os.path.basename(s)}: {r} ratings, {n} volume(s)")
    if bad_index:
        print(f"  {len(bad_index)} files with a frame_index past the last volume:")
        for s, hi, n in bad_index[:5]:
            print(f"    {os.path.basename(s)}: index {hi}, only {n} volume(s)")
    if not bad_count and not bad_index:
        print("  OK: every rated file has one rating per volume, indices in range")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", help="CSV from build_fmap_csv.py (QU_motion empty)")
    p.add_argument("labels", help="QC export .tsv or .csv")
    p.add_argument("output", help="CSV to write")
    p.add_argument("--label-column", default="QU_sus", help="Rating column in labels")
    p.add_argument("--status-column", default="qc_status")
    p.add_argument("--status-value", default="review complete")
    p.add_argument("--keep-unrated", action="store_true",
                   help="Keep rows with no rating. The pipeline cannot train on "
                        "NaN, so use only for inspection.")
    p.add_argument("--frame-column", default=None,
                   help="Column in the labels file naming which volume of a 4D "
                        "scan each rating refers to. Supply this once per-frame "
                        "ratings exist. Without it, two ratings for one file "
                        "collide on the filename and one is lost.")
    p.add_argument("--frame-base", type=int, default=0, choices=[0, 1],
                   help="Whether the rater's frame numbering starts at 0 or 1. "
                        "Output is always 0-based, matching nibabel indexing.")
    p.add_argument("--check-frames", metavar="FOLDER", default=None,
                   help="Validate rating counts against the NIfTI headers under "
                        "FOLDER. Catches files rated twice that hold only one "
                        "volume under a 4D header.")
    a = p.parse_args()

    man = pd.read_csv(a.manifest)
    sep = "\t" if a.labels.endswith((".tsv", ".txt")) else ","
    lab = pd.read_csv(a.labels, sep=sep)

    print(f"manifest: {len(man)} rows | labels: {len(lab)} rows")
    print(f"label columns: {', '.join(lab.columns)}")

    if a.label_column not in lab.columns:
        sys.exit(f"ERROR: --label-column {a.label_column!r} not found")

    lab = lab.rename(columns={k: v for k, v in ALIASES.items() if k in lab.columns})

    # Keep only completed reviews.
    if a.status_column in lab.columns and a.status_value:
        n0 = len(lab)
        m = lab[a.status_column].astype(str).str.strip().str.lower()
        lab = lab[m == a.status_value.strip().lower()]
        print(f"{a.status_column}: kept {len(lab)} of {n0}")

    path_col = next((c for c in ("scan", "filename", "file", "path")
                     if c in lab.columns), None)
    if not path_col:
        sys.exit("ERROR: labels file has no scan/filename/file/path column")
    print(f"joining on basename of '{path_col}'")

    lab = lab.copy()
    lab["_k"] = lab[path_col].map(base)
    man = man.copy()
    man["_k"] = man["scan"].map(base)

    per_frame = a.frame_column is not None
    if per_frame and a.frame_column not in lab.columns:
        sys.exit(f"ERROR: --frame-column {a.frame_column!r} not found. "
                 f"Columns present: {', '.join(lab.columns)}")

    if per_frame:
        # One row per (file, frame). Normalise the rater's numbering to 0-based
        # so it indexes nibabel's trailing axis directly.
        lab["frame_index"] = (
            pd.to_numeric(lab[a.frame_column], errors="coerce") - a.frame_base
        )
        bad = int(lab["frame_index"].isna().sum())
        if bad:
            sys.exit(f"ERROR: {bad} label rows have an unparseable "
                     f"{a.frame_column!r}; fix the export before merging")
        if (lab["frame_index"] < 0).any():
            sys.exit(f"ERROR: negative frame_index after subtracting "
                     f"--frame-base {a.frame_base}; check the numbering base")
        lab["frame_index"] = lab["frame_index"].astype(int)
        keys = ["_k", "frame_index"]
        print(f"per-frame ratings: joining on (basename, {a.frame_column}), "
              f"{a.frame_base}-based -> 0-based")
    else:
        keys = ["_k"]

    dup = int(lab.duplicated(subset=keys).sum())
    if dup:
        if per_frame:
            sys.exit(f"ERROR: {dup} duplicate (file, frame) keys in labels. "
                     f"Each volume should be rated once; resolve the "
                     f"duplicates rather than letting one win silently.")
        ex = lab[lab.duplicated(subset=keys, keep=False)]["_k"].unique()[:3]
        sys.exit(f"ERROR: {dup} duplicate filenames in labels with no "
                 f"--frame-column given, e.g. {', '.join(map(str, ex))}.\n"
                 f"If these are per-frame ratings, pass --frame-column so both "
                 f"are kept. Silently keeping the first would discard half the "
                 f"ratings and still train.")

    extra = [c for c in ("scanner_manufacturer", "nrev") if c in lab.columns]
    cols = keys + [a.label_column] + extra

    # Joining on "_k" alone is one-to-many when the labels are per-frame, so a
    # scan-level manifest row fans out into one row per rated frame. That is
    # exactly the shape the dataset wants.
    n_manifest = len(man)
    out = (man.drop(columns=["QU_motion"], errors="ignore")
              .merge(lab[cols], on="_k", how="left")
              .drop(columns=["_k"])
              .rename(columns={a.label_column: "QU_motion"}))
    out["QU_motion_source"] = f"{os.path.basename(a.labels)}:{a.label_column}"

    matched = int(out["QU_motion"].notna().sum())
    if per_frame:
        print(f"\n{n_manifest} manifest files -> {len(out)} frame rows, "
              f"{matched} rated")
    else:
        print(f"\nmatched {matched} of {len(out)} manifest rows")
    if len(lab) - matched > 0:
        print(f"{len(lab) - matched} label rows had no matching file on disk")

    if not a.keep_unrated and matched < len(out):
        drop = out[out["QU_motion"].isna()]
        print(f"dropping {len(drop)} unrated rows, e.g.:")
        for s in drop["scan"].head(3):
            print(f"  {s}")
        out = out[out["QU_motion"].notna()].reset_index(drop=True)

    if out.empty:
        sys.exit("ERROR: no rated rows remain -- check the join keys match")

    if a.check_frames:
        check_frame_counts(out, a.check_frames, per_frame)

    v = out["QU_motion"]
    print("\n=== label distribution ===")
    for val, n in v.value_counts().sort_index().items():
        print(f"  {val:>6}  {n:>6}  {100*n/len(v):>5.1f}%  {'#'*max(1,int(50*n/len(v)))}")
    print(f"  mean {v.mean():.4f}  sd {v.std():.4f}  min {v.min()}  max {v.max()}")
    print(f"  subjects: {out.subject_id.nunique()}")

    if v.std() == 0:
        print("\n  ERROR: zero variance. Nothing to learn; standardized_rmse "
              "would divide by zero.")
    elif v.nunique() <= 3:
        top = v.value_counts().max() / len(v)
        print(f"\n  NOTE: {v.nunique()} distinct values, largest class "
              f"{100*top:.0f}%. Closer to classification than regression; "
              "consider USE_WEIGHTED_LOSS=true.")

    if "scanner_manufacturer" in out.columns:
        print("\n=== by scanner ===")
        print(out.groupby("scanner_manufacturer")["QU_motion"]
                 .agg(["count", "mean", "std"]).round(3).to_string())

    d = os.path.dirname(a.output)
    if d:
        os.makedirs(d, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
