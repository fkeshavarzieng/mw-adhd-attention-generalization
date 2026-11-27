#!/usr/bin/env python3
"""Corrected-label v2, participant-preserving LOSO permutation tests.

Usage (run from project root, after obtaining the excluded feature CSV):
    python3 scripts/run_v2_permutation_tests.py

Uses the corrected ATTLAPSE JSON: OT < 43, MW-D 43..52, MW-S > 52.
Recomputes the classifier for every LOSO fold in every permutation.
Shuffles labels *within participant* preserving each person's class counts.
The output contains aggregate statistics only; v1 files are not changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["rt_median", "rt_cv", "rt_slope", "omission_rate", "commission_rate"]


def validate_and_label(df: pd.DataFrame) -> pd.DataFrame:
    required = {"participant", "group", "probe_number", "raw_att_code"} | set(FEATURES)
    if missing := sorted(required - set(df.columns)):
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    scores = pd.to_numeric(df["raw_att_code"], errors="raise")
    if scores.isna().any() or (scores != scores.astype(int)).any() or (scores == 0).any():
        raise ValueError("Invalid probe-level attention score")
    if not df.group.isin(["ADHD", "Control"]).all():
        raise ValueError("Unexpected group label")
    if df.duplicated(["participant", "probe_number"]).any():
        raise ValueError("Duplicate participant/probe key")
    if (df.groupby("participant").group.nunique() != 1).any():
        raise ValueError("Participant ID appears in multiple groups")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Missing or infinite features")
    df["state_v2"] = np.select(
        [scores.lt(43), scores.between(43, 52), scores.gt(52)],
        ["OT", "MW-D", "MW-S"], default="UNMAPPED"
    )
    if (df.state_v2 == "UNMAPPED").any():
        raise ValueError("Unmapped state")
    df["y"] = (df.state_v2 != "OT").astype(int)
    return df


def loso_auc(X: np.ndarray, y: np.ndarray, folds: list[tuple]) -> float:
    p = np.empty(len(y))
    for tr, te in folds:
        if np.unique(y[tr]).size < 2:
            raise ValueError("LOSO training fold has one class")
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=42))
        ])
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p))


def permutation_test(df: pd.DataFrame, n: int, seed: int) -> list[dict]:
    rows = []
    for group in ["ADHD", "Control"]:
        sub = df.loc[df.group == group]
        X = sub[FEATURES].to_numpy(dtype=float)
        y = sub.y.to_numpy(dtype=int)
        ids = sub.participant.to_numpy()
        participants = pd.unique(ids)
        person_indices = [np.flatnonzero(ids == person) for person in participants]
        folds = list(LeaveOneGroupOut().split(X, y, ids))
        observed = loso_auc(X, y, folds)
        rng = np.random.default_rng(seed)
        null_aucs = np.empty(n)
        for i in range(n):
            permuted = y.copy()
            for ix in person_indices:
                permuted[ix] = rng.permutation(y[ix])
            null_aucs[i] = loso_auc(X, permuted, folds)
            if (i + 1) % 200 == 0:
                print(f"{group}: {i+1}/{n} permutations", flush=True)

        exceedances = int(np.sum(null_aucs >= observed))
        rows.append({
            "group": group,
            "n_participants": len(participants),
            "n_probes": len(y),
            "n_ot": int((y == 0).sum()),
            "n_mw": int((y == 1).sum()),
            "observed_auc": observed,
            "n_permutations": n,
            "null_auc_mean": float(null_aucs.mean()),
            "null_auc_sd": float(null_aucs.std(ddof=1)),
            "null_auc_q025": float(np.quantile(null_aucs, 0.025)),
            "null_auc_q975": float(np.quantile(null_aucs, 0.975)),
            "null_max": float(null_aucs.max()),
            "null_ge_observed": exceedances,
            "p_one_sided": (exceedances + 1) / (n + 1),
            "seed": seed,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=Path("attlapse_behavior_features_v1.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/v2_recomputed/v2_within_group_permutation.csv"))
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.permutations < 20:
        parser.error("--permutations must be >=20")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    df = validate_and_label(pd.read_csv(args.features))
    rows = permutation_test(df, args.permutations, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Saved aggregate results:", args.output)
    print("Caution: the p-value tests within-person label exchangeability, "
          "not clinical validity or independent-dataset replication.")


if __name__ == "__main__":
    main()
