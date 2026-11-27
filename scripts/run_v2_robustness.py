#!/usr/bin/env python3
"""Corrected-label v2 robustness from the private probe-level feature CSV.

python3 scripts/run_v2_robustness.py
python3 scripts/run_v2_robustness.py --include-forest

Outputs aggregate-only CSV to ignored results/v2_recomputed/ and refuses overwrite.
The one-trial-shifted window REQUIRES raw trial data and is NOT computed here.
This script follows the archived notebook's comparison models/settings.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, matthews_corrcoef
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

FEATURES = ["rt_median", "rt_cv", "rt_slope", "omission_rate", "commission_rate"]


def load_data(path):
    d = pd.read_csv(path)
    required = ["participant", "group", "probe_number", "raw_att_code", "state"] + FEATURES
    if missing := sorted(set(required) - set(d.columns)):
        raise ValueError(f"Missing columns: {missing}")
    if d.duplicated(["participant", "probe_number"]).any():
        raise ValueError("Duplicate participant/probe")
    if (d.groupby("participant").group.nunique() != 1).any():
        raise ValueError("Participant appears in both cohorts")
    if not d.group.isin(["ADHD", "Control"]).all():
        raise ValueError("Unexpected group")
    if not np.isfinite(d[FEATURES+["probe_number"]].to_numpy(dtype=float)).all():
        raise ValueError("Missing or infinite features")
    s = pd.to_numeric(d.raw_att_code, errors="raise")
    if s.isna().any() or (s != s.astype(int)).any() or (s == 0).any():
        raise ValueError("Non-integer or non-probe attention score")
    d["state_v2"] = np.select(
        [s.lt(43), s.between(43, 52), s.gt(52)],
        ["OT", "MW-D", "MW-S"], default="UNMAPPED")
    if (d.state_v2 == "UNMAPPED").any():
        raise ValueError("Unmapped state")
    d["y"] = (d.state_v2 != "OT").astype(int)
    return d


def logistic():
    return Pipeline([("scaler", StandardScaler()),
                     ("classifier", LogisticRegression(class_weight="balanced",
                                                        max_iter=2000, random_state=42))])


def model_options(include_forest=False):
    m = {
        "Logistic Regression": logistic(),
        "Linear SVM": Pipeline([("scaler", StandardScaler()),
                                ("classifier", LinearSVC(class_weight="balanced",
                                                          max_iter=10000, random_state=42))]),
    }
    if include_forest:
        m["Random Forest"] = RandomForestClassifier(
            n_estimators=500, max_depth=3, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=1)
    return m


def evaluate(r):
    return dict(AUC=roc_auc_score(r.y, r.score),
                Balanced_Accuracy=balanced_accuracy_score(r.y, r.prediction),
                AUPRC=average_precision_score(r.y, r.score),
                MCC=matthews_corrcoef(r.y, r.prediction))


def loso(d, features, model=None, detrend=False):
    if model is None: model = logistic()
    X = d[features].copy()
    y, people = d.y.to_numpy(), d.participant.to_numpy()
    scores, predictions = np.full(len(d), np.nan), np.full(len(d), -1)
    for tr, te in LeaveOneGroupOut().split(X, y, people):
        xt, xv = X.iloc[tr].copy(), X.iloc[te].copy()
        if detrend:
            time_train = d.iloc[tr][["probe_number"]]
            time_test = d.iloc[te][["probe_number"]]
            for feature in features:
                trend = LinearRegression().fit(time_train, xt[[feature]])
                xt[feature] = xt[feature] - trend.predict(time_train).ravel()
                xv[feature] = xv[feature] - trend.predict(time_test).ravel()
        trained = clone(model).fit(xt, y[tr])
        predictions[te] = trained.predict(xv)
        scores[te] = (trained.predict_proba(xv)[:, 1]
                      if hasattr(trained, "predict_proba") else trained.decision_function(xv))
    if not np.isfinite(scores).all() or (predictions < 0).any():
        raise ValueError("Incomplete LOSO predictions")
    return pd.DataFrame(dict(participant=people, y=y, score=scores, prediction=predictions))


def transfer(d, features, model, source, target):
    train, test = d[d.group == source], d[d.group == target]
    trained = clone(model).fit(train[features], train.y)
    scores = (trained.predict_proba(test[features])[:, 1]
              if hasattr(trained, "predict_proba") else trained.decision_function(test[features]))
    return pd.DataFrame(dict(participant=test.participant.to_numpy(),
                             y=test.y.to_numpy(), score=scores,
                             prediction=trained.predict(test[features])))


def analysis(d, include_forest=False):
    rows = []
    for group in ["ADHD", "Control"]:
        sub = d[d.group == group]
        baseline = loso(sub, FEATURES)
        auc = roc_auc_score(baseline.y, baseline.score)
        deleted_aucs = [
            roc_auc_score(baseline.loc[baseline.participant != pid, "y"],
                          baseline.loc[baseline.participant != pid, "score"])
            for pid in sub.participant.unique()
        ]
        for label, feats, detrend in [
            ("Behavior", FEATURES, False),
            ("Reduced feature set", [x for x in FEATURES if x != "rt_slope"], False),
            ("Probe number only", ["probe_number"], False),
            ("Behavior + probe number", FEATURES + ["probe_number"], False),
            ("Detrended behavior", FEATURES, True),
        ]:
            result = baseline if label == "Behavior" else loso(sub, feats, detrend=detrend)
            rows.append(dict(section="feature_time", group=group, analysis=label,
                             participants=sub.participant.nunique(), probes=len(sub), **evaluate(result)))
        rows.append(dict(section="participant_influence", group=group,
                         analysis="Posthoc deletion of held-out predictions (no refit)",
                         participants=sub.participant.nunique(), probes=len(sub), **evaluate(baseline),
                         auc_without_person_min=min(deleted_aucs),
                         auc_without_person_max=max(deleted_aucs),
                         max_abs_auc_change=max(abs(auc - x) for x in deleted_aucs)))
    for name, model in model_options(include_forest).items():
        for group in ["Pooled", "ADHD", "Control"]:
            sub = d if group == "Pooled" else d[d.group == group]
            r = loso(sub, FEATURES, model)
            rows.append(dict(section="model_comparison", group=group, analysis=name,
                             participants=sub.participant.nunique(), probes=len(sub), **evaluate(r)))
        for src, tgt in [("Control", "ADHD"), ("ADHD", "Control")]:
            r = transfer(d, FEATURES, model, src, tgt)
            rows.append(dict(section="model_comparison", group=f"{src} → {tgt}",
                             analysis=name, participants=d[d.group == tgt].participant.nunique(),
                             probes=len(r), **evaluate(r)))
        print("Completed", name, flush=True)
    mw = d[d.state_v2.isin(["MW-S", "MW-D"])].copy()
    counts = mw.groupby(["participant", "state_v2"]).size().unstack(fill_value=0)
    for label in ["MW-S", "MW-D"]:
        if label not in counts: counts[label] = 0
    for min_each in [1, 2]:
        selected = counts[(counts["MW-S"] >= min_each) & (counts["MW-D"] >= min_each)].index
        sub = mw[mw.participant.isin(selected)].copy()
        sub["y"] = (sub.state_v2 == "MW-S").astype(int)
        r = loso(sub, FEATURES)
        individual_auc = [roc_auc_score(chunk.y, chunk.score)
                          for _, chunk in r.groupby("participant") if chunk.y.nunique() == 2]
        rows.append(dict(section="subtype", group="Pooled MW subtype",
                         analysis=f">={min_each} of each subtype",
                         participants=len(selected), probes=len(sub),
                         n_MWS=int(sub.y.sum()), n_MWD=int((sub.y == 0).sum()),
                         participant_auc_mean=float(np.mean(individual_auc)),
                         participant_auc_median=float(np.median(individual_auc)),
                         participant_auc_std=float(np.std(individual_auc, ddof=1)),
                         **evaluate(r)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("attlapse_behavior_features_v1.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/v2_recomputed/v2_robustness_aggregate.csv"))
    parser.add_argument("--include-forest", action="store_true",
                        help="include original 500-tree Random Forest; slow on older machines")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    df = analysis(load_data(args.features), include_forest=args.include_forest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print("Saved", args.output)


if __name__ == "__main__":
    main()
