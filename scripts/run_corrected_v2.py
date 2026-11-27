#!/usr/bin/env python3
"""ATTLAPSE corrected-JSON (v2) modeling, using an existing v1 feature CSV.

The numerical behavioral features do not change; only the target label is
reconstructed from raw_att_code. Never overwrites v1 data/results.

Usage:
    python3 scripts/run_corrected_v2.py --features attlapse_behavior_features_v1.csv
Produces (unless overridden):
    results/v2_recomputed/v2_primary_performance.csv
    results/v2_recomputed/v2_paired_transfer_comparison.csv
    results/v2_recomputed/v2_label_audit.csv

Input is participant-level derived data. NEVER commit the input to GitHub.
The official corrected JSON describes OT <43, MW-D 43-52, MW-S >52.
An earlier email gave different boundaries. Version-specific provenance is
discussed in docs/att_state_scoring_audit.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, matthews_corrcoef
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = ["rt_20", "sd_rt_20", "ce_20"]
EXP = ["rt_median", "rt_cv", "rt_slope", "omission_rate", "commission_rate"]


def relabel(data: pd.DataFrame) -> pd.DataFrame:
    required = ["participant", "group", "probe_number", "raw_att_code", "state"] + BASE + EXP
    if missing := sorted(set(required) - set(data.columns)):
        raise ValueError(f"Missing columns: {missing}")
    df = data.copy()
    scores = pd.to_numeric(df["raw_att_code"], errors="raise")
    if scores.isna().any() or (scores != scores.astype(int)).any():
        raise ValueError("Invalid attention-state score")
    if (scores == 0).any():
        raise ValueError("Non-probe score 0 found among probe observations")
    if not df["group"].isin(["ADHD", "Control"]).all():
        raise ValueError("Invalid group values")
    if df.duplicated(["participant", "probe_number"]).any():
        raise ValueError("Duplicate participant/probe_number keys")
    if (df.groupby("participant")["group"].nunique() > 1).any():
        raise ValueError("Participant appears in both cohorts")
    if not np.isfinite(df[BASE + EXP].to_numpy(dtype=float)).all():
        raise ValueError("Missing or infinite model features")
    df["state_v2"] = np.select(
        [scores.lt(43), scores.between(43, 52), scores.gt(52)],
        ["OT", "MW-D", "MW-S"], default="UNMAPPED")
    if (df["state_v2"] == "UNMAPPED").any():
        raise ValueError("Unmapped state scores")
    df["y"] = (df["state_v2"] != "OT").astype(int)
    return df


def classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42))
    ])


def loso(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X, y, groups = df[features], df["y"], df["participant"]
    p = np.full(len(df), np.nan)
    pred = np.full(len(df), -1, dtype=int)
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        if y.iloc[train].nunique() < 2:
            raise ValueError("Single-class LOSO training fold")
        clf = classifier().fit(X.iloc[train], y.iloc[train])
        p[test] = clf.predict_proba(X.iloc[test])[:, 1]
        pred[test] = clf.predict(X.iloc[test])
    if not np.isfinite(p).all() or (pred < 0).any():
        raise ValueError("Missing model predictions")
    return pd.DataFrame({
        "participant": groups.to_numpy(),
        "y": y.to_numpy(),
        "p": p, "pred": pred})


def transfer(df: pd.DataFrame, features: list[str], src: str, tgt: str) -> pd.DataFrame:
    train, test = df[df.group == src], df[df.group == tgt]
    clf = classifier().fit(train[features], train.y)
    return pd.DataFrame({
        "participant": test.participant.to_numpy(),
        "y": test.y.to_numpy(),
        "p": clf.predict_proba(test[features])[:, 1],
        "pred": clf.predict(test[features])})


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    positive = y == 1
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return float("nan")
    ranks = rankdata(p, method="average")
    return float((ranks[positive].sum() - npos*(npos+1)/2) / (npos*nneg))


def cluster_ci(df: pd.DataFrame, n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    ppl, y, p = df.participant.to_numpy(), df.y.to_numpy(), df.p.to_numpy()
    ids = pd.unique(ppl)
    indices = {person: np.where(ppl == person)[0] for person in ids}
    values = []
    for _ in range(n):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        ix = np.concatenate([indices[person] for person in sampled])
        auc = fast_auc(y[ix], p[ix])
        if np.isfinite(auc):
            values.append(auc)
    if not values:
        raise ValueError("Bootstrap failed to generate a valid AUC")
    return np.percentile(values, [2.5, 97.5])


def paired_test(within: pd.DataFrame, cross: pd.DataFrame, n: int, seed: int = 42):
    if (not within.participant.equals(cross.participant)
            or not within.y.equals(cross.y)
            or len(within) != len(cross)
            or not within[["p"]].notna().all().all()
            or not cross[["p"]].notna().all().all()):
        raise ValueError("Participant, label or prediction alignment failed")
    rng = np.random.default_rng(seed)
    people = within.participant.to_numpy()
    ids = pd.unique(people)
    ix = {person: np.where(people == person)[0] for person in ids}
    y, a, b = within.y.to_numpy(), within.p.to_numpy(), cross.p.to_numpy()
    observed = fast_auc(y, a) - fast_auc(y, b)
    swap_dist, bootstrap_dist = [], []
    for _ in range(n):
        swap_people = ids[rng.random(len(ids)) < 0.5]
        mask = np.isin(people, swap_people)
        a_swap = np.where(mask, b, a)
        b_swap = np.where(mask, a, b)
        swap_dist.append(fast_auc(y, a_swap) - fast_auc(y, b_swap))
        sampled = rng.choice(ids, size=len(ids), replace=True)
        sampled_ix = np.concatenate([ix[person] for person in sampled])
        delta = fast_auc(y[sampled_ix], a[sampled_ix]) - fast_auc(y[sampled_ix], b[sampled_ix])
        if np.isfinite(delta):
            bootstrap_dist.append(delta)
    ci = np.percentile(bootstrap_dist, [2.5, 97.5])
    return {"within_auc": fast_auc(y, a), "transfer_auc": fast_auc(y, b),
            "delta_auc": observed,
            "p_paired_swap": (1 + sum(abs(z) >= abs(observed) for z in swap_dist)) / (n + 1),
            "ci_low": float(ci[0]), "ci_high": float(ci[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("attlapse_behavior_features_v1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v2_recomputed"))
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()
    if args.resamples < 20:
        parser.error("--resamples must be >=20")
    df = relabel(pd.read_csv(args.features))
    label_audit = (df.assign(changed=df.state.ne(df.state_v2))
                   .groupby(["group", "state", "state_v2", "changed"])
                   .size().rename("count").reset_index())
    cases = [
        ("Pooled LOSO", "Baseline Logistic", "Authors baseline", loso(df, BASE)),
        ("Pooled LOSO", "Primary Logistic", "Feature Set v1", loso(df, EXP)),
        ("ADHD-only LOSO", "Primary Logistic", "Feature Set v1", loso(df[df.group == "ADHD"], EXP)),
        ("Control-only LOSO", "Primary Logistic", "Feature Set v1", loso(df[df.group == "Control"], EXP)),
        ("Control → ADHD", "Primary Logistic", "Feature Set v1", transfer(df, EXP, "Control", "ADHD")),
        ("ADHD → Control", "Primary Logistic", "Feature Set v1", transfer(df, EXP, "ADHD", "Control")),
    ]
    rows = []
    for name, model, feature_set, r in cases:
        low, high = cluster_ci(r, args.resamples)
        rows.append({
            "Analysis": name, "Model": model, "Feature_Set": feature_set,
            "AUC": roc_auc_score(r.y, r.p),
            "AUC_CI_low": low, "AUC_CI_high": high,
            "Balanced_Accuracy": balanced_accuracy_score(r.y, r.pred),
            "AUPRC": average_precision_score(r.y, r.p),
            "MCC": matthews_corrcoef(r.y, r.pred)})
    paired = []
    for target, wi, cr in [("ADHD", cases[2][3], cases[4][3]),
                           ("Control", cases[3][3], cases[5][3])]:
        paired.append({"target": target, **paired_test(wi, cr, args.resamples)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("v2_primary_performance.csv", pd.DataFrame(rows)),
        ("v2_paired_transfer_comparison.csv", pd.DataFrame(paired)),
        ("v2_label_audit.csv", label_audit),
    ]
    for filename, table in outputs:
        path = args.output_dir / filename
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing v2 output: {path}")
    for filename, table in outputs:
        path = args.output_dir / filename
        table.to_csv(path, index=False)
        print(f"Saved {path}")
    print(pd.DataFrame(rows)[["Analysis", "AUC", "AUC_CI_low", "AUC_CI_high"]].to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == "__main__":
    main()
