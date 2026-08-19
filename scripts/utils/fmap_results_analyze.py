#!/usr/bin/env python3
"""Summarize a k-fold field map QC training run: metrics, confound checks, figures.

Complements scripts/utils/training_results_overview.py, which reports per-fold
accuracy. This script asks a different question: is the correlation real, or an
artifact of scanner differences and of the model regressing toward the mean?

Three things it computes that a plain correlation does not show:

  1. A mean-predictor baseline. standardized_rmse is error divided by label sd,
     so a model that always predicts the label mean scores exactly 1.0. Values
     at or above 1.0 mean the model has learned nothing useful, however
     significant its correlation p-value.

  2. Between- vs within-group decomposition. When a grouping variable such as
     scanner manufacturer shifts the label mean, a model that only learns the
     group offset earns a pooled correlation while carrying no within-group
     signal. This reports scanner-only eta^2 and the sample-weighted
     within-group r alongside the pooled figure.

  3. Convergence from the training logs. "New best validation loss" lines say
     when the saved checkpoint was actually taken. A run that stops improving
     early spent the remaining epochs producing nothing.

Usage:
    python analyze_fmap_results.py \\
        --csv-pattern 'doc/models/model_04d0/*fold_*.csv' \\
        --log-pattern 'scripts/utils/logs/real-fold*.err' \\
        --outdir doc/models/model_04d0/report
"""

import argparse
import glob
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

TRUTH = "QU_motion"
PRED = "predicted_qu_motion_score"

# Timestamp + metric lines as emitted by training.py's logger.
RE_BEST = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*New best validation loss: ([\d.eE+-]+)")
RE_EPOCH = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Epoch (\d+)/(\d+)")
RE_SRMSE = re.compile(r"standardized_rmse: ([\d.eE+-]+)")
RE_CORR = re.compile(r"correlation_coefficient: ([-\d.eE+]+)")
RE_CONFIG = re.compile(r"Input configuration: (.+)$")
RE_NSAMP = re.compile(r": (\d+) (training|validation) samples")

TS = "%Y-%m-%d %H:%M:%S"


def to_md(df, floatfmt=".4f"):
    """Markdown table with integer columns kept as integers.

    tabulate applies a single floatfmt to every numeric column, so an int64
    column renders as 0.0000 unless it is stringified first.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].astype("object").map(
                lambda v: "" if pd.isna(v) else str(int(v)))
    return out.to_markdown(index=False, floatfmt=floatfmt)


def fold_id(path):
    """Fold index from a filename. Handles fold_0, fold-0 and fold0."""
    m = re.search(r"fold[_-]?(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


# ----------------------------------------------------------------- loading


def load_predictions(pattern):
    """Read per-fold prediction CSVs into one frame tagged with fold index."""
    frames = []
    for path in sorted(glob.glob(pattern), key=fold_id):
        df = pd.read_csv(path)
        if TRUTH not in df.columns or PRED not in df.columns:
            print(f"  skipping {os.path.basename(path)}: missing {TRUTH}/{PRED}")
            continue
        df = df[df[PRED].notna() & df[TRUTH].notna()].copy()
        if len(df) < 3:
            continue
        df["fold"] = fold_id(path)
        df["_source"] = os.path.basename(path)
        frames.append(df)

    if not frames:
        raise SystemExit(f"No usable prediction CSVs matched {pattern!r}")

    return pd.concat(frames, ignore_index=True)


def parse_logs(pattern):
    """Pull convergence and final metrics out of the training logs.

    Logging goes to stderr, so these are normally the .err files.
    """
    runs = []
    for path in sorted(glob.glob(pattern), key=fold_id):
        text = open(path, errors="replace").read()
        lines = text.splitlines()

        bests = [(datetime.strptime(m.group(1), TS), float(m.group(2)))
                 for line in lines if (m := RE_BEST.match(line))]
        epochs = [(datetime.strptime(m.group(1), TS), int(m.group(2)), int(m.group(3)))
                  for line in lines if (m := RE_EPOCH.match(line))]

        if not epochs:
            continue

        srmse = RE_SRMSE.findall(text)
        corr = RE_CORR.findall(text)
        cfg = RE_CONFIG.findall(text)
        samples = {kind: int(n) for n, kind in RE_NSAMP.findall(text)}

        start = epochs[0][0]
        end = max([e[0] for e in epochs] + [b[0] for b in bests])
        total_epochs = epochs[0][2]

        # Which epoch was the saved checkpoint taken at? Match each improvement
        # to the most recent epoch banner before it.
        last_best_epoch = None
        if bests:
            t = bests[-1][0]
            prior = [e for e in epochs if e[0] <= t]
            last_best_epoch = prior[-1][1] if prior else None

        runs.append(dict(
            file=os.path.basename(path),
            fold=fold_id(path),
            config=cfg[0] if cfg else "",
            n_train=samples.get("training"),
            n_val=samples.get("validation"),
            epochs_run=epochs[-1][1],
            epochs_requested=total_epochs,
            n_improvements=len(bests),
            best_val_loss=bests[-1][1] if bests else np.nan,
            last_improvement_epoch=last_best_epoch,
            wall_minutes=(end - start).total_seconds() / 60,
            minutes_to_last_improvement=((bests[-1][0] - start).total_seconds() / 60)
                                        if bests else np.nan,
            standardized_rmse=float(srmse[-1]) if srmse else np.nan,
            correlation=float(corr[-1]) if corr else np.nan,
        ))

    return pd.DataFrame(runs)


# ----------------------------------------------------------------- analysis


def per_fold_metrics(df):
    rows = []
    for fold, g in df.groupby("fold"):
        r, p = pearsonr(g[TRUTH], g[PRED])
        rs, _ = spearmanr(g[TRUTH], g[PRED])
        rmse = float(np.sqrt(np.mean((g[TRUTH] - g[PRED]) ** 2)))
        sd = g[TRUTH].std()
        rows.append(dict(fold=fold, n=len(g), pearson_r=r, p_value=p,
                         spearman_r=rs, rmse=rmse, label_sd=sd,
                         standardized_rmse=rmse / sd if sd else np.nan))
    return pd.DataFrame(rows).sort_values("fold")


def group_decomposition(df, group_col):
    """Split pooled correlation into between-group and within-group parts.

    eta^2 is the share of label variance explained by group membership alone --
    the score a lookup table of group means would achieve. If the model's
    pooled r^2 is at or below that, it has not learned anything the grouping
    variable does not already encode.
    """
    if group_col not in df.columns or df[group_col].nunique() < 2:
        return None

    total_var = df[TRUTH].var(ddof=0)
    grand = df[TRUTH].mean()

    rows, between, weighted_r, n_used = [], 0.0, 0.0, 0
    for name, g in df.groupby(group_col):
        between += len(g) * (g[TRUTH].mean() - grand) ** 2
        if len(g) > 20 and g[TRUTH].std() > 0 and g[PRED].std() > 0:
            r, p = pearsonr(g[TRUTH], g[PRED])
            weighted_r += len(g) * r
            n_used += len(g)
        else:
            r, p = np.nan, np.nan
        rows.append(dict(group=name, n=len(g), label_mean=g[TRUTH].mean(),
                         pred_mean=g[PRED].mean(), within_r=r, within_p=p))

    eta2 = (between / len(df)) / total_var if total_var else np.nan
    pooled_r, _ = pearsonr(df[TRUTH], df[PRED])

    return dict(
        table=pd.DataFrame(rows).sort_values("n", ascending=False),
        eta2=eta2,
        group_only_r=np.sqrt(eta2) if eta2 == eta2 else np.nan,
        pooled_r=pooled_r,
        weighted_within_r=weighted_r / n_used if n_used else np.nan,
    )


# ------------------------------------------------------------------ figures


def fig_scatter(df, out, group_col=None):
    """Truth vs prediction, with the mean-predictor line for reference."""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    lo = min(df[TRUTH].min(), df[PRED].min()) - 0.2
    hi = max(df[TRUTH].max(), df[PRED].max()) + 0.2

    if group_col and group_col in df.columns:
        for name, g in df.groupby(group_col):
            ax.scatter(g[TRUTH], g[PRED], s=14, alpha=0.45, label=f"{name} (n={len(g)})")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    else:
        ax.scatter(df[TRUTH], df[PRED], s=14, alpha=0.4, color="#4C72B0")

    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="_perfect")
    ax.axhline(df[TRUTH].mean(), color="crimson", lw=1.4, ls=":",
               label="_mean")
    ax.text(hi, df[TRUTH].mean(), " predicting the mean", color="crimson",
            va="bottom", ha="right", fontsize=8)
    ax.text(hi, hi, "perfect ", va="top", ha="right", fontsize=8)

    r, p = pearsonr(df[TRUTH], df[PRED])
    ax.set_title(f"Observed vs predicted   (n={len(df)}, r={r:+.3f}, p={p:.2e})")
    ax.set_xlabel("Manual rating")
    ax.set_ylabel("Predicted")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_spread(df, out):
    """Prediction spread against label spread -- shows regression to the mean."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))

    bins = np.linspace(min(df[TRUTH].min(), df[PRED].min()),
                       max(df[TRUTH].max(), df[PRED].max()), 45)
    a1.hist(df[TRUTH], bins=bins, alpha=0.6, label=f"manual (sd={df[TRUTH].std():.3f})")
    a1.hist(df[PRED], bins=bins, alpha=0.6, label=f"predicted (sd={df[PRED].std():.3f})")
    a1.set_title("Distribution: predictions are far narrower")
    a1.set_xlabel("Rating")
    a1.set_ylabel("Count")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.25)

    # Mean prediction per observed rating level, with a 95% CI.
    lv = df.groupby(TRUTH)[PRED].agg(["mean", "std", "count"])
    lv = lv[lv["count"] >= 5]
    if not lv.empty:
        err = 1.96 * lv["std"] / np.sqrt(lv["count"])
        a2.errorbar(lv.index, lv["mean"], yerr=err, fmt="o-", capsize=4, color="#4C72B0")
        for x, y, n in zip(lv.index, lv["mean"], lv["count"]):
            a2.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7)
    a2.axhline(df[TRUTH].mean(), color="crimson", ls=":", lw=1.4)
    a2.text(lv.index.max() if not lv.empty else 0, df[TRUTH].mean(),
            " overall mean", color="crimson", va="bottom", ha="right", fontsize=8)
    lo = min(df[TRUTH].min(), 0)
    a2.plot([lo, df[TRUTH].max()], [lo, df[TRUTH].max()], "k--", lw=1)
    a2.set_title("Mean prediction per rating level")
    a2.set_xlabel("Manual rating")
    a2.set_ylabel("Mean prediction")
    a2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_folds(per_fold, out):
    """Per-fold correlation and standardized RMSE against their null baselines."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = per_fold["fold"].astype(str)

    a1.bar(x, per_fold["pearson_r"], color="#4C72B0")
    a1.axhline(0, color="k", lw=1)
    a1.set_title("Pearson r by fold")
    a1.set_xlabel("Fold")
    a1.set_ylabel("r")
    a1.grid(alpha=0.25, axis="y")

    colors = ["#55A868" if v < 1 else "#C44E52" for v in per_fold["standardized_rmse"]]
    a2.bar(x, per_fold["standardized_rmse"], color=colors)
    a2.axhline(1.0, color="crimson", ls="--", lw=1.5)
    a2.text(len(per_fold) - 0.5, 1.0, " predicting the mean", color="crimson",
            va="bottom", ha="right", fontsize=8)
    a2.set_title("Standardized RMSE by fold (lower is better)")
    a2.set_xlabel("Fold")
    a2.set_ylabel("RMSE / label sd")
    a2.set_ylim(0, max(1.25, per_fold["standardized_rmse"].max() * 1.1))
    a2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_groups(dec, out, group_col):
    """Within-group r, and the variance a group-mean lookup would explain."""
    t = dec["table"].dropna(subset=["within_r"])
    if t.empty:
        return False

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))

    labels = [f"{g}\n(n={n})" for g, n in zip(t["group"], t["n"])]
    colors = ["#55A868" if p < 0.05 else "#BBBBBB" for p in t["within_p"]]
    a1.barh(labels, t["within_r"], color=colors)
    a1.axvline(0, color="k", lw=1)
    a1.axvline(dec["pooled_r"], color="#4C72B0", ls="--", lw=1.5)
    a1.text(dec["pooled_r"], len(t) - 0.4, f" pooled r={dec['pooled_r']:+.3f}",
            color="#4C72B0", fontsize=8, va="center", ha="left")
    a1.set_title(f"Within-{group_col} correlation\n(grey = not significant)")
    a1.set_xlabel("Pearson r")
    a1.tick_params(labelsize=8)
    a1.grid(alpha=0.25, axis="x")

    vals = [100 * dec["eta2"], 100 * dec["pooled_r"] ** 2,
            100 * dec["weighted_within_r"] ** 2]
    names = [f"{group_col}\nalone", "model\n(pooled)", "model\n(within-group)"]
    a2.bar(names, vals, color=["#C44E52", "#4C72B0", "#55A868"])
    for i, v in enumerate(vals):
        a2.text(i, v, f" {v:.2f}%", ha="center", va="bottom", fontsize=9)
    a2.set_title("Share of label variance explained")
    a2.set_ylabel("% of variance")
    a2.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
    a2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def fig_convergence(runs, out):
    """When the saved checkpoint was taken, relative to the run length."""
    r = runs.dropna(subset=["minutes_to_last_improvement"])
    if r.empty:
        return False

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(r) + 2.2))
    y = np.arange(len(r))
    ax.barh(y, r["wall_minutes"], color="#DDDDDD", label="epochs that changed nothing")
    ax.barh(y, r["minutes_to_last_improvement"], color="#4C72B0",
            label="up to the last improvement")
    for i, (_, row) in enumerate(r.iterrows()):
        ep = row["last_improvement_epoch"]
        ax.text(row["wall_minutes"], i,
                f"  {int(row['n_improvements'])} improvement(s)"
                + (f", last at epoch ~{int(ep)}" if ep == ep else ""),
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"fold {int(f)}" for f in r["fold"]])
    ax.set_xlabel("Wall-clock minutes")
    ax.set_title("Where the saved checkpoint came from")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


# ------------------------------------------------------------------- report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-pattern", required=True,
                   help="Glob for per-fold prediction CSVs")
    p.add_argument("--log-pattern", default=None,
                   help="Glob for training .err logs (optional)")
    p.add_argument("--outdir", default="report", help="Directory for figures and report")
    p.add_argument("--group-col", default="scanner_manufacturer",
                   help="Column to test as a confound (default scanner_manufacturer)")
    p.add_argument("--title", default="Field map QC training results")
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    md, fig_paths = [], {}

    df = load_predictions(a.csv_pattern)
    per_fold = per_fold_metrics(df)
    pooled_r, pooled_p = pearsonr(df[TRUTH], df[PRED])
    pooled_rs, _ = spearmanr(df[TRUTH], df[PRED])
    label_sd = df[TRUTH].std()
    rmse = float(np.sqrt(np.mean((df[TRUTH] - df[PRED]) ** 2)))
    srmse = rmse / label_sd

    md.append(f"# {a.title}\n")
    md.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M}_\n")

    # --- headline
    md.append("## Summary\n")
    md.append(f"- **{len(df)} predictions** across {df['fold'].nunique()} folds, "
              f"{df.subject_id.nunique() if 'subject_id' in df.columns else '?'} subjects")
    md.append(f"- **Pooled r = {pooled_r:+.3f}** (p = {pooled_p:.2e}), "
              f"Spearman = {pooled_rs:+.3f}")
    md.append(f"- **r² = {100*pooled_r**2:.2f}%** of label variance explained")
    md.append(f"- **standardized RMSE = {srmse:.3f}**")

    verdict = ("at or above 1.0, so the model does **not** beat simply predicting "
               "the label mean" if srmse >= 0.995 else
               f"below 1.0, so the model beats the mean baseline by "
               f"{100*(1-srmse):.1f}%")
    md.append(f"  - a mean-predictor scores exactly 1.0; this is {verdict}")
    md.append(f"- prediction sd {df[PRED].std():.3f} vs label sd {label_sd:.3f} "
              f"— predictions span {100*df[PRED].std()/label_sd:.0f}% of the label spread\n")

    # --- per fold
    md.append("## Per-fold\n")
    show = per_fold.copy()
    show["fold"] = show["fold"].astype(int)
    show["n"] = show["n"].astype(int)
    show.columns = ["fold", "n", "r", "p", "spearman", "rmse", "label_sd", "std_rmse"]
    md.append(to_md(show) + "\n")

    fig_folds(per_fold, os.path.join(a.outdir, "per_fold.png"))
    fig_paths["per_fold.png"] = "Per-fold correlation and standardized RMSE"

    fig_scatter(df, os.path.join(a.outdir, "scatter.png"),
                a.group_col if a.group_col in df.columns else None)
    fig_paths["scatter.png"] = "Observed vs predicted"

    fig_spread(df, os.path.join(a.outdir, "spread.png"))
    fig_paths["spread.png"] = "Prediction spread and calibration"

    # --- confound
    dec = group_decomposition(df, a.group_col)
    if dec:
        md.append(f"## Confound check: {a.group_col}\n")
        md.append("If a grouping variable shifts the label mean, a model that "
                  "learns only the group offset earns a pooled correlation "
                  "while carrying no within-group signal.\n")
        t = dec["table"].copy()
        t["n"] = t["n"].astype(int)
        t.columns = ["group", "n", "label mean", "pred mean", "within r", "within p"]
        md.append(to_md(t) + "\n")

        share = (dec["weighted_within_r"] / dec["pooled_r"]
                 if dec["pooled_r"] else np.nan)
        md.append(f"- {a.group_col} alone explains **{100*dec['eta2']:.2f}%** of "
                  f"label variance (a group-mean lookup would score "
                  f"r = {dec['group_only_r']:+.3f})")
        md.append(f"- model pooled r = {dec['pooled_r']:+.3f} "
                  f"(r² = {100*dec['pooled_r']**2:.2f}%)")
        md.append(f"- sample-weighted **within-group r = "
                  f"{dec['weighted_within_r']:+.3f}** "
                  f"(r² = {100*dec['weighted_within_r']**2:.2f}%)")
        if share == share:
            md.append(f"- **~{100*(1-share):.0f}% of the pooled correlation is "
                      f"between-group**, not within\n")
        if dec["group_only_r"] > abs(dec["pooled_r"]):
            md.append(f"> The model scores **below** what {a.group_col} identity "
                      f"alone would achieve.\n")

        if fig_groups(dec, os.path.join(a.outdir, "groups.png"), a.group_col):
            fig_paths["groups.png"] = f"Within-{a.group_col} signal vs confound"

    # --- convergence
    if a.log_pattern:
        runs = parse_logs(a.log_pattern)
        if not runs.empty:
            md.append("## Convergence\n")
            r = runs[["fold", "n_train", "n_val", "epochs_requested",
                      "n_improvements", "last_improvement_epoch",
                      "minutes_to_last_improvement", "wall_minutes",
                      "best_val_loss"]].copy()
            for c in ("fold", "n_train", "n_val", "epochs_requested",
                      "n_improvements", "last_improvement_epoch"):
                r[c] = r[c].astype("Int64")
            r.columns = ["fold", "train", "val", "epochs", "improvements",
                         "last impr. epoch", "min to last impr.", "wall min",
                         "best val loss"]
            md.append(to_md(r, floatfmt=".2f") + "\n")

            wasted = (runs["wall_minutes"] - runs["minutes_to_last_improvement"]).sum()
            md.append(f"- total wall time {runs['wall_minutes'].sum()/60:.1f} h; "
                      f"**{wasted/60:.1f} h produced no improvement**")
            if runs["best_val_loss"].notna().any():
                bl = runs["best_val_loss"].mean()
                md.append(f"- mean best validation loss {bl:.4f} vs label variance "
                          f"{label_sd**2:.4f} "
                          f"({100*(bl/label_sd**2 - 1):+.1f}% vs the mean predictor)")
            md.append("- if improvement stops early, more epochs will not help; "
                      "add early stopping to make iteration cheap\n")

            if fig_convergence(runs, os.path.join(a.outdir, "convergence.png")):
                fig_paths["convergence.png"] = "Where the saved checkpoint came from"

            if runs["config"].notna().any() and runs["config"].iloc[0]:
                md.append(f"Input configuration: `{runs['config'].iloc[0]}`\n")

    # --- figures
    md.append("## Figures\n")
    for name, caption in fig_paths.items():
        md.append(f"### {caption}\n")
        md.append(f"![{caption}]({name})\n")

    out_md = os.path.join(a.outdir, "results.md")
    with open(out_md, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print("\n".join(md[:40]))
    print(f"\nWrote {out_md}")
    for name in fig_paths:
        print(f"Wrote {os.path.join(a.outdir, name)}")


if __name__ == "__main__":
    main()