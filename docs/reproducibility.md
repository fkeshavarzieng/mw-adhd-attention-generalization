# Reproducing the analysis

This repository provides corrected-label modeling and aggregate results for an independent secondary analysis of the ATTLAPSE SART dataset. Original participant-level records are not redistributed.

## Environment

Use Python 3.14 and the versions pinned in [`requirements.txt`](../requirements.txt):

```bash
python3 -m pip install -r requirements.txt
python3 tests/smoke_v2_pipeline.py
```

The smoke test generates synthetic participant-level observations in a temporary directory, runs all five analysis/figure scripts, and checks their output files. It does **not** substitute for reproducing the scientific results on the real dataset. The same test ran successfully in [GitHub Actions](https://github.com/fkeshavarzieng/mw-adhd-attention-generalization/actions/runs/46029776664).

## Input data

Obtain ATTLAPSE from its [Zenodo Version 2.0.1 record](https://doi.org/10.5281/zenodo.23592484), observing its [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

The corrected scripts take a local probe-level table named `attlapse_behavior_features_v1.csv` with these fields: `participant`, `group`, `probe_number`, `raw_att_code`, `state`, `rt_20`, `sd_rt_20`, `ce_20`, `rt_median`, `rt_cv`, `rt_slope`, `omission_rate`, and `commission_rate`. Additional fields may be present.

**Important:** The upstream raw-trial-to-feature workflow is not fully reproduced in this public repository. Downloading ATTLAPSE alone will not create the required feature table automatically. The scripts intentionally recalculate the binary label from `raw_att_code` to apply the updated score boundaries. The name of the input file retains its historical suffix; the models and public aggregate outputs use the corrected labels.

The private study table has 840 probes from 56 participants (28 ADHD, 28 controls). To reproduce the corrected aggregate results, the input must follow the same feature and probe-window specification described in [Methods](methods.md). Store the derived table locally and do not commit it.

## Running the corrected analyses

From the repository root:

```bash
python3 scripts/run_corrected_v2.py --features attlapse_behavior_features_v1.csv
python3 scripts/run_v2_permutation_tests.py --features attlapse_behavior_features_v1.csv
python3 scripts/run_v2_robustness.py --features attlapse_behavior_features_v1.csv
python3 scripts/v2_incremental_time_audit.py --features attlapse_behavior_features_v1.csv
python3 scripts/generate_v2_figures.py --features attlapse_behavior_features_v1.csv
```

All commands produce aggregate files in local, ignored `results/v2_recomputed/` or `figures/v2_recomputed/`. Existing outputs are not overwritten: use new output paths or remove prior local generated files before running again. The Random Forest comparisons are optionally enabled with `--include-forest` on the robustness script and can take considerably longer than the linear models.

The results already provided under [`results/`](../results/) are corrected v2 aggregate tables. The figures under [`figures/`](../figures/) are compact summaries; detailed ROC and coefficient plots can be regenerated locally by the figure script when the derived data are available.

## Verification and limitations

The six primary AUCs and their participant-bootstrap confidence intervals were independently recalculated from the private derived feature table. Additional corrected-label sensitivity and permutation results are recorded as aggregate tables. In a clean, pinned Python environment, synthetic-data tests exercised all five scripts and their output formats.

The full raw-trial-to-feature extraction was **not independently rerun**. The alternative shifted 20-trial-window check, some 500-tree Random Forest evaluations, and original-data regeneration of all graphics remain incomplete. The local corrected JSON used for score boundaries has not been independently matched by checksum to the exact cited archive. These are methodological limitations, not features of the automated smoke test.

The study is exploratory and has not been externally validated. See [Methods](methods.md) for the statistical design and interpretation.
