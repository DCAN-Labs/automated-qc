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

    dup = int(lab.duplicated(subset=["_k"]).sum())
    if dup:
        print(f"WARNING: {dup} duplicate keys in labels; keeping first of each")
        lab = lab.drop_duplicates(subset=["_k"], keep="first")

    extra = [c for c in ("scanner_manufacturer", "nrev") if c in lab.columns]
    cols = ["_k", a.label_column] + extra

    out = man.drop(columns=["QU_motion"], errors="ignore").merge(
        lab[cols], on="_k", how="left").drop(columns=["_k"])
    out = out.rename(columns={a.label_column: "QU_motion"})
    out["QU_motion_source"] = f"{os.path.basename(a.labels)}:{a.label_column}"

    matched = int(out["QU_motion"].notna().sum())
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
