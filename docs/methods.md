# Methods and analysis notes

## Study question

This secondary analysis asks whether short windows of task behavior predict self-reported mind-wandering in participants not included in training, and whether a model learned from one cohort transfers to the other. The analysis is exploratory and is not intended for ADHD diagnosis.

## Data

The study draws on the ATTLAPSE Sustained Attention to Response Task (SART) dataset. The derived analysis table contains 840 thought probes from 56 participants, including 28 in the ADHD cohort and 28 in the control cohort. Each observation represents a thought probe and its associated behavioral window.

The five primary input features in the existing analysis table are median Go reaction time (`rt_median`), its coefficient of variation (`rt_cv`), reaction-time slope (`rt_slope`), omission rate (`omission_rate`), and commission rate (`commission_rate`). The original processing used 20-trial windows. The present repository provides corrected-label modeling from this derived table; it **does not provide or independently rebuild the complete trial-to-feature preprocessing pipeline**.

### Attention-state coding

The updated behavioral metadata used in this analysis defines:

| State | Score |
| --- | --- |
| On-task (OT) | below 43 |
| Deliberate mind-wandering (MW-D) | 43–52 |
| Spontaneous mind-wandering (MW-S) | above 52 |

The binary target is OT (0) versus MW-D or MW-S (1). The corrected labels contain 429 OT, 137 MW-D, and 274 MW-S probes. Seventeen observations with score 43 were reassigned from OT to MW-D compared with the earlier analysis, without changing their behavioral features.

The downloaded updated JSON describes these boundaries. Earlier correspondence from the data authors used slightly different endpoint wording, which did not explicitly account for scores 43 and 53. Our implementation follows the updated JSON definition; confirming a checksum match against the cited ATTLAPSE 2.0.1 archive remains a provenance check rather than a verified fact.

## Models and evaluation

The main model is a logistic-regression classifier with `class_weight="balanced"`, `max_iter=2000`, and `random_state=42`. Standardization is fitted **only on training observations** within each evaluation fold. `LeaveOneGroupOut` uses participant IDs to prevent data from the same participant appearing in training and testing for the primary pooled and within-cohort models.

The five principal analyses are pooled leave-one-participant-out (LOSO), ADHD-only LOSO, control-only LOSO, and transfers in both directions between the two cohorts. ROC AUC is the principal discrimination metric. Confidence intervals are obtained by resampling participants rather than individual probes; these intervals condition on the fixed held-out predictions and do not capture all model refitting or selection uncertainty.

The corrected results also include 1,000 within-participant label permutations per cohort, paired within-versus-transfer comparisons, feature reduction, linear-SVM comparison, time-on-task sensitivity, participant-influence checks, and an exploratory MW-D versus MW-S classification task. Model-selection and analysis decisions were not prospectively registered.

## Statistical interpretation

| Evaluation | ROC AUC | 95% participant-bootstrap CI |
| --- | ---: | --- |
| Pooled LOSO | 0.624 | 0.567–0.677 |
| ADHD LOSO | 0.643 | 0.536–0.747 |
| Control LOSO | 0.585 | 0.523–0.651 |
| Control → ADHD | 0.520 | 0.446–0.602 |
| ADHD → Control | 0.562 | 0.486–0.630 |

Cross-group transfer had lower descriptive AUCs, but paired comparisons did **not** establish a significant within-versus-transfer difference (ADHD target: ΔAUC 0.123, p=0.137; control target: ΔAUC 0.023, p=0.650). Within-participant label permutation p-values were 0.000999 for ADHD and 0.015984 for controls; they test a specific label-exchangeability null, not clinical validity.

Probe number alone gave AUC 0.585 for ADHD and 0.610 for controls. Adding behavior to probe number produced 0.658 and 0.633. A paired participant-bootstrap comparison of the combined model against probe number alone gave a pooled ΔAUC of 0.054 (95% CI 0.007–0.099). The ADHD and control confidence intervals included zero. This does not identify a causal attention-specific signal separate from task progression.

Alternative feature sets and linear SVMs gave broadly similar discrimination. Exploratory spontaneous-versus-deliberate mind-wandering prediction was near chance (AUC about 0.522–0.523). The complete alternative-window analysis and some 500-tree Random Forest evaluations remain incomplete; they are not represented as validated corrected-label results.

All results are in [`results/`](../results/), and the visualization inputs are the corresponding aggregate tables.

## Scope and limitations

This is an exploratory, single-dataset study based on 56 participants and self-reported thought probes from one task. Correlated observations, time-on-task effects, subtype imbalance, historical feature screening, and the lack of external validation limit what can be inferred. Models should not be used for clinical decision-making or presented as validated ADHD biomarkers. The complete raw-trial feature extraction and some secondary checks remain to be independently reproduced.

## Data source and license

Horáková, G., Weibel, S., Tissot, U., Pham, B.-T., Costache, M. E., & Bonnefond, A. (2026). *ATTLAPSE: Behavioral and EEG data of ADHD patients and neurotypical controls during a Go/No-Go task with embedded thought-probes* (Version 2.0.1) [Dataset]. Zenodo. [doi:10.5281/zenodo.23592484](https://doi.org/10.5281/zenodo.23592484).

ATTLAPSE is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original participant-level data are not included here. The repository's [MIT license](../LICENSE) applies to original project code, not to the underlying ATTLAPSE dataset.
