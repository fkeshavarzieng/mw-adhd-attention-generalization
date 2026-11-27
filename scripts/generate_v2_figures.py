#!/usr/bin/env python3
"""Regenerate corrected-v2 research figures from aggregate results and private features.

From the project root:
    python3 scripts/generate_v2_figures.py
    python3 scripts/generate_v2_figures.py --features attlapse_behavior_features_v1.csv

Outputs SVG and PDF graphics under figures/v2_recomputed/ (ignored by Git).
The four summary figures use committed aggregate tables; ROC curves and
coefficient stability use private derived features and retrain LOSO models.
No individual observations or participant IDs are written to figures.
Never substitutes historical v1 figure data for corrected v2 metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["rt_median", "rt_cv", "rt_slope", "omission_rate", "commission_rate"]
BLUE = "#23759C"
RUST = "#C46D4C"
NAVY = "#18283A"


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": NAVY, "text.color": NAVY,
        "axes.titleweight": "bold", "figure.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none"})


def export(fig, stem: str, folder: Path):
    for fmt in ("svg", "pdf"):
        path = folder / f"{stem}.{fmt}"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        fig.savefig(path, bbox_inches="tight", dpi=250)
        print(f"Saved {path}")
    plt.close(fig)


def state_figure(folder: Path):
    old = [446, 120, 274]
    new = [429, 137, 274]
    names = ["On-task", "Deliberate MW", "Spontaneous MW"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    ax.bar(x - 0.19, old, 0.35, label="Archived v1", color="#C8D1DA")
    ax.bar(x + 0.19, new, 0.35, label="Corrected v2", color=BLUE)
    ax.set_xticks(x, names)
    ax.set_ylabel("Thought probes (n)")
    ax.set_title("Attention-state label correction")
    ax.legend(frameon=False)
    ax.set_ylim(0, 530)
    for i, v in enumerate(new):
        ax.text(i + 0.19, v + 8, str(v), ha="center", fontsize=9)
    fig.text(0.13, 0.01,
             "17 score-43 probes reassigned from on-task to deliberate mind-wandering.",
             fontsize=9)
    export(fig, "v2_state_distribution", folder)


def auc_figure(folder: Path, primary: pd.DataFrame):
    rows = primary[(primary.Model == "Primary Logistic")].copy()
    order = ["Pooled LOSO", "ADHD-only LOSO", "Control-only LOSO",
             "Control → ADHD", "ADHD → Control"]
    rows = rows.set_index("Analysis").loc[order].reset_index()
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = [NAVY, BLUE, RUST, BLUE, RUST]
    for i, r in rows.iterrows():
        a = float(r.AUC)
        ax.errorbar(a, i, xerr=[[a-float(r.AUC_CI_low)],
                                [float(r.AUC_CI_high)-a]],
                    fmt="o", capsize=4, color=colors[i], markersize=6)
        ax.text(0.81, i, f"{a:.3f}", va="center", fontsize=10)
    ax.axvline(0.5, color="#98A8B6", linewidth=1.5, linestyle="--")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(0.40, 0.86)
    ax.set_xlabel("ROC AUC (95% participant-bootstrap CI)")
    ax.set_title("Corrected v2 participant-independent prediction")
    export(fig, "v2_primary_auc_ci", folder)


def time_figure(folder: Path, robustness: pd.DataFrame):
    models = ["Behavior", "Reduced feature set", "Probe number only",
              "Behavior + probe number", "Detrended behavior"]
    renames = ["Behavior", "Behavior without RT slope", "Probe number only",
               "Behavior + probe number", "Time-detrended behavior"]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9, 4.7))
    w = 0.34
    for group, offset, color in [("ADHD", -w/2, BLUE),
                                  ("Control", w/2, RUST)]:
        sub = robustness[(robustness.section == "feature_time") &
                         (robustness.group == group)].set_index("analysis")
        vals = [float(sub.loc[m, "AUC"]) for m in models]
        ax.bar(x + offset, vals, w, color=color, label=group)
    ax.axhline(.5, linestyle="--", color="#8F9EAA", linewidth=1)
    ax.set_xticks(x, renames, rotation=16, ha="right")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("LOSO ROC AUC")
    ax.set_title("Time-on-task and feature-reduction sensitivity")
    ax.legend(frameon=False)
    fig.text(.10, 0.01, "Descriptive model comparisons; not pairwise significance tests.",
             fontsize=9)
    export(fig, "v2_time_on_task", folder)


def subtype_figure(folder: Path, robustness: pd.DataFrame):
    sub = robustness[robustness.section == "subtype"].copy()
    names = ["At least one each", "At least two each"]
    vals = [float(sub.loc[sub.analysis == q, "AUC"].iloc[0])
            for q in [">=1 of each subtype", ">=2 of each subtype"]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(names, vals, color=BLUE, width=0.48)
    ax.axhline(0.5, linestyle="--", color="#8F9EAA", linewidth=1)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("LOSO ROC AUC")
    ax.set_title("Exploratory deliberate vs spontaneous MW")
    for x, v in enumerate(vals):
        ax.text(x, v + 0.015, f"{v:.3f}", ha="center")
    export(fig, "v2_subtype_exploratory", folder)


def corrected_data(path: Path):
    df = pd.read_csv(path)
    required = set(FEATURES + ["participant", "group", "raw_att_code",
                               "probe_number"])
    if missing := sorted(required - set(df.columns)):
        raise ValueError(f"Missing columns: {missing}")
    if not df.group.isin(["ADHD", "Control"]).all():
        raise ValueError("Unexpected group")
    if df.duplicated(["participant", "probe_number"]).any():
        raise ValueError("Duplicate participant/probe")
    if (df.groupby("participant").group.nunique() != 1).any():
        raise ValueError("Participant ID shared across cohorts")
    score = pd.to_numeric(df.raw_att_code, errors="raise")
    if score.isna().any() or (score != score.astype(int)).any() or (score == 0).any():
        raise ValueError("Invalid attention-state code")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Missing or infinite features")
    df["y"] = (score >= 43).astype(int)  # corrected JSON: OT < 43
    return df


def clf():
    return Pipeline([("scaler", StandardScaler()),
                     ("classifier", LogisticRegression(
                         class_weight="balanced", max_iter=2000,
                         random_state=42))])


def loso_with_coefficients(frame: pd.DataFrame):
    X = frame[FEATURES]
    y = frame.y.to_numpy(dtype=int)
    g = frame.participant.to_numpy()
    p = np.full(len(frame), np.nan)
    coeff = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        fit = clf().fit(X.iloc[tr], y[tr])
        p[te] = fit.predict_proba(X.iloc[te])[:, 1]
        coeff.append(fit.named_steps["classifier"].coef_[0].copy())
    return y, p, np.stack(coeff)


def model_figures(folder: Path, df: pd.DataFrame, primary: pd.DataFrame):
    curves = {}
    coefs = {}
    for group in ["Pooled", "ADHD", "Control"]:
        sub = df if group == "Pooled" else df[df.group == group]
        y, p, coef = loso_with_coefficients(sub)
        curves[group] = (y, p)
        if group != "Pooled":
            coefs[group] = coef
        expected = float(primary.loc[primary.Analysis ==
                                     (group + "-only LOSO" if group != "Pooled"
                                      else "Pooled LOSO")].query(
                                      'Model == "Primary Logistic"').AUC.iloc[0])
        observed = roc_auc_score(y, p)
        if abs(observed - expected) > 0.00001:
            raise ValueError(f"{group} LOSO AUC mismatch: {observed} vs {expected}")

    for src, dst in [("Control", "ADHD"), ("ADHD", "Control")]:
        train, test = df[df.group == src], df[df.group == dst]
        m = clf().fit(train[FEATURES], train.y)
        curves[f"{src} → {dst}"] = (
            test.y.to_numpy(), m.predict_proba(test[FEATURES])[:, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    for group, color, ls in [
        ("Pooled", NAVY, "-"), ("ADHD", BLUE, "-"),
        ("Control", RUST, "-"), ("Control → ADHD", BLUE, "--"),
        ("ADHD → Control", RUST, "--")]:
        y, p = curves[group]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.8,
                label=f"{group} (AUC {auc(fpr,tpr):.3f})")
    ax.plot([0,1], [0,1], color="#ACB9C3", linestyle=":", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Corrected v2 participant-independent ROC curves")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    export(fig, "v2_roc_curves", folder)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(FEATURES))
    w = 0.36
    for group, offset, color in [
        ("ADHD", -w/2, BLUE), ("Control", w/2, RUST)]:
        coeff = coefs[group]
        means = coeff.mean(axis=0)
        sds = coeff.std(axis=0)
        ax.bar(x + offset, means, width=w, color=color, label=group)
        ax.errorbar(x + offset, means, yerr=sds,
                    linestyle="", color=NAVY, capsize=3, linewidth=1)
    ax.axhline(0, color="#9FAEBB", linewidth=1)
    ax.set_xticks(x, ["RT median", "RT CV", "RT slope",
                      "Omission rate", "Commission rate"],
                  rotation=15, ha="right")
    ax.set_ylabel("Standardized log-odds coefficient (mean ± fold SD)")
    ax.set_title("Corrected v2 LOSO coefficient stability")
    ax.legend(frameon=False)
    fig.text(0.12, .01, "Descriptive fold variation; not confidence intervals.",
             fontsize=9)
    export(fig, "v2_coefficients", folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path,
                        default=Path("results/v2_primary_performance.csv"))
    parser.add_argument("--robustness", type=Path,
                        default=Path("results/v2_robustness_verified_summary.csv"))
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("figures/v2_recomputed"))
    args = parser.parse_args()
    setup()
    # Preflight: do not modify any graphics unless all expected destinations are free.
    stems = ["v2_state_distribution", "v2_primary_auc_ci",
             "v2_time_on_task", "v2_subtype_exploratory"]
    if args.features:
        stems += ["v2_roc_curves", "v2_coefficients"]
    for stem in stems:
        for ext in ["svg", "pdf"]:
            p = args.output_dir / (stem + "." + ext)
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite {p}")
    primary = pd.read_csv(args.primary)
    robustness = pd.read_csv(args.robustness)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_figure(args.output_dir)
    auc_figure(args.output_dir, primary)
    time_figure(args.output_dir, robustness)
    subtype_figure(args.output_dir, robustness)
    if args.features:
        model_figures(args.output_dir, corrected_data(args.features), primary)
    else:
        print("ROC curves and coefficients skipped: supply --features CSV for these.")


if __name__ == "__main__":
    main()
