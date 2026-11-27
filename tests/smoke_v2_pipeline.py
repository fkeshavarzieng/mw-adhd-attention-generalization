#!/usr/bin/env python3
"""Synthetic-data smoke tests of the exact committed v2 entrypoints.

No original participant-level data are required or redistributed.
Run from the repository root:
    python3 tests/smoke_v2_pipeline.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
FEATURES = ["rt_median", "rt_cv", "rt_slope", "omission_rate", "commission_rate"]


def run(script: str, *args: str):
    cmd = [sys.executable, str(SCRIPT_DIR / script), *map(str, args)]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True,
                               timeout=180, check=False)
    if completed.returncode:
        raise AssertionError(
            f"{script} failed (exit {completed.returncode})\n"
            f"STDOUT:\n{completed.stdout[-5000:]}\nSTDERR:\n{completed.stderr[-5000:]}"
        )
    return completed.stdout


def synthetic_features(path: Path):
    # Eight fictitious participants: independent 15-probe sequences.
    rng = np.random.default_rng(2026)
    rows = []
    for person in range(8):
        group = "ADHD" if person < 4 else "Control"
        for probe in range(15):
            # Exercise the corrected score-43 boundary.
            code = [39, 43, 49, 56, 42][(probe + person) % 5]
            rt = float(0.35 + rng.normal(0, 0.045) + probe * 0.003)
            rows.append(dict(
                participant=f"synthetic-{person:02d}", group=group,
                probe_number=probe + 1, raw_att_code=code,
                state="OT" if code <= 43 else ("MW-D" if code <= 52 else "MW-S"),
                rt_20=rt, sd_rt_20=float(0.09+rng.random()*0.03),
                ce_20=float(rng.random()*.2),
                rt_median=rt, rt_cv=float(.15+rng.random()*.2),
                rt_slope=float(rng.normal(0,.01)),
                omission_rate=float(rng.random()*.2),
                commission_rate=float(rng.random()*.2)))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def check_csv(path: Path, columns: set[str], n: int | None = None):
    if not path.exists():
        raise AssertionError(f"Missing output {path}")
    df = pd.read_csv(path)
    if missing := columns - set(df.columns):
        raise AssertionError(f"Missing {missing} in {path}")
    if n is not None and len(df) != n:
        raise AssertionError(f"Expected {n} rows in {path}, got {len(df)}")
    return df


def main():
    with tempfile.TemporaryDirectory(prefix="mw-v2-smoke-") as dirname:
        folder = Path(dirname)
        data = folder / "synthetic.csv"
        synthetic_features(data)

        model_dir = folder / "primary"
        run("run_corrected_v2.py", "--features", data,
            "--output-dir", model_dir, "--resamples", "25")
        primary = check_csv(model_dir / "v2_primary_performance.csv",
                            {"Analysis", "AUC", "AUC_CI_low", "AUC_CI_high"}, 6)
        check_csv(model_dir / "v2_paired_transfer_comparison.csv",
                  {"target", "delta_auc", "p_paired_swap"}, 2)
        audit = check_csv(model_dir / "v2_label_audit.csv",
                          {"group", "state", "state_v2", "changed", "count"})
        if int(audit["count"].sum()) != 120:
            raise AssertionError("Synthetic probes were dropped")
        if not int(audit.loc[audit["changed"], "count"].sum()) > 0:
            raise AssertionError("Score-43 recoding was not triggered")

        perm = folder / "permutation.csv"
        run("run_v2_permutation_tests.py", "--features", data,
            "--output", perm, "--permutations", "20")
        p = check_csv(perm, {"group", "p_one_sided", "observed_auc"}, 2)
        if not p["p_one_sided"].between(0, 1).all():
            raise AssertionError("Permutation p outside [0,1]")

        robustness = folder / "robustness.csv"
        run("run_v2_robustness.py", "--features", data,
            "--output", robustness)
        r = check_csv(robustness, {"section", "analysis", "group", "AUC"})
        if not {"feature_time", "model_comparison", "subtype"}.issubset(set(r.section)):
            raise AssertionError("Robustness sections missing")

        incremental = folder / "incremental.csv"
        run("v2_incremental_time_audit.py", "--features", data,
            "--output", incremental, "--resamples", "25")
        check_csv(incremental, {"group", "delta_auc_vs_probe_only",
                               "bootstrap_ci_low", "bootstrap_ci_high"}, 3)

        figs = folder / "figures"
        run("generate_v2_figures.py", "--primary",
            model_dir / "v2_primary_performance.csv", "--robustness",
            robustness, "--features", data, "--output-dir", figs)
        for stem in ["v2_state_distribution", "v2_primary_auc_ci",
                     "v2_time_on_task", "v2_subtype_exploratory",
                     "v2_roc_curves", "v2_coefficients"]:
            for extension in ("svg", "pdf"):
                if not (figs / f"{stem}.{extension}").is_file():
                    raise AssertionError(f"Missing figure {stem}.{extension}")
        print("PASS: 5 v2 scripts, corrected-label boundary, six figure pairs")

    print("PASS: no participant-level ATTLAPSE files were required")


if __name__ == "__main__":
    main()
