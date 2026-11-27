#!/usr/bin/env python3
"""Corrected v2 paired time-on-task incremental-value audit.

The LOSO logistic models are refit from raw_att_code-corrected v2 labels.
A participant bootstrap then compares out-of-sample AUC for probe-only,
behavior-only and combined models on the *same* observations.

Run:
    python3 scripts/v2_incremental_time_audit.py
Output:
    results/v2_recomputed/v2_incremental_time_results.csv

All resampling is at participant level, never at probe level. The bootstrap
CI describes an out-of-sample paired AUC difference; it is not a causal or
formal conditional-independence test. Existing committed results are untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FEATURES = ["rt_median", "rt_cv", "rt_slope", "omission_rate", "commission_rate"]


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    req = set(FEATURES + ["participant", "group", "probe_number", "raw_att_code"])
    if miss := sorted(req - set(d.columns)):
        raise ValueError(f"Missing columns: {miss}")
    if d.duplicated(["participant", "probe_number"]).any():
        raise ValueError("Duplicate probe keys")
    if (d.groupby("participant").group.nunique() != 1).any():
        raise ValueError("Participant ID spans multiple groups")
    if not d.group.isin(["ADHD", "Control"]).all():
        raise ValueError("Unexpected group")
    if not np.isfinite(d[FEATURES+["probe_number"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite features")
    scores = pd.to_numeric(d.raw_att_code, errors="raise")
    if scores.isna().any() or (scores != scores.astype(int)).any() or (scores == 0).any():
        raise ValueError("Invalid attention score")
    d["y"] = (scores >= 43).astype(int)  # corrected JSON: OT <43, MW-D >=43
    return d


def loso_predict(d: pd.DataFrame, features: list[str]) -> np.ndarray:
    X, y = d[features], d.y.to_numpy(dtype=int)
    people = d.participant.to_numpy()
    p = np.full(len(d), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, people):
        model = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced",
                                      max_iter=2000, random_state=42))
        ])
        model.fit(X.iloc[tr], y[tr])
        p[te] = model.predict_proba(X.iloc[te])[:, 1]
    if not np.isfinite(p).all():
        raise ValueError("Missing out-of-fold predictions")
    return p


def rank_auc(y: np.ndarray, p: np.ndarray) -> float:
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if not npos or not nneg:
        return np.nan
    return float((rankdata(p)[pos].sum() - npos*(npos+1)/2) / (npos*nneg))


def evaluate_group(d: pd.DataFrame, resamples: int, seed: int) -> dict:
    d = d.reset_index(drop=True)
    time = loso_predict(d, ["probe_number"])
    behavior = loso_predict(d, FEATURES)
    combined = loso_predict(d, FEATURES + ["probe_number"])
    y, people = d.y.to_numpy(), d.participant.to_numpy()
    ids = pd.unique(people)
    indices = {pid: np.flatnonzero(people == pid) for pid in ids}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        sample = rng.choice(ids, size=len(ids), replace=True)
        ix = np.concatenate([indices[pid] for pid in sample])
        a, b = rank_auc(y[ix], combined[ix]), rank_auc(y[ix], time[ix])
        if np.isfinite(a) and np.isfinite(b):
            differences.append(a-b)
    if len(differences) < resamples*0.8:
        raise ValueError("Too many invalid cluster-bootstrap samples")
    ci = np.percentile(differences, [2.5, 97.5])
    return dict(n_participants=len(ids), n_probes=len(d),
                auc_probe_only=roc_auc_score(y, time),
                auc_behavior_only=roc_auc_score(y, behavior),
                auc_behavior_plus_probe=roc_auc_score(y, combined),
                delta_auc_vs_probe_only=roc_auc_score(y, combined)-roc_auc_score(y, time),
                bootstrap_ci_low=float(ci[0]), bootstrap_ci_high=float(ci[1]),
                bootstrap_resamples=resamples, bootstrap_seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path,
                    default=Path("attlapse_behavior_features_v1.csv"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/v2_recomputed/v2_incremental_time_results.csv"))
    ap.add_argument("--resamples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.resamples < 20:
        ap.error("At least 20 bootstrap resamples are required")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    df = load(args.features)
    rows = []
    for group in ["ADHD", "Control", "Pooled"]:
        subset = df if group == "Pooled" else df.loc[df.group == group]
        row = dict(group=group, **evaluate_group(subset, args.resamples, args.seed))
        rows.append(row)
        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
