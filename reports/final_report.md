# Final Report

## Executive summary

This project reproduces and critically evaluates fraud-detection methodology
from the Fraud Detection Handbook using its public simulated transaction
dataset. The study focuses on temporal validation, fraud-appropriate metrics,
baseline modeling, and leakage-safe behavioral feature engineering.

The project shows that chronological validation and Average Precision / PR-AUC
are necessary for interpreting rare-event fraud models. It also shows that past
behavioral histories improve performance, but the strongest results depend on
prior fraud labels being available at scoring time.

The project does not prove real-world fraud-detection performance.

## Source and data

The selected source is the
[Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/).
The original implementation is the
[Fraud-Detection-Handbook repository](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook).
The raw simulated transaction dataset is obtained separately from
[simulated-data-raw](https://github.com/Fraud-Detection-Handbook/simulated-data-raw).

Raw data are not stored in this repository. The expected local placement is
documented in [../data/README.md](../data/README.md).

## Methodology

The analysis uses a chronological train/validation/test split. Validation data
are used for threshold selection only, and final metrics are reported on the
held-out chronological test period.

The project compares:

- a no-fraud prior baseline;
- logistic regression with class balancing;
- histogram gradient boosting with balanced sample weights;
- baseline transaction/time features;
- past-only behavioral features with no prior fraud labels;
- past behavioral features plus prior fraud-label history.

Behavioral features are computed from prior transactions only. Full-window
customer and terminal aggregates are not used.

## Main result

| Feature set | Best model | AP / PR-AUC | ROC-AUC | Precision | Recall | F1 |
|:---|:---|---:|---:|---:|---:|---:|
| Baseline transaction/time features | Logistic regression | 0.245 | 0.644 | 0.940 | 0.211 | 0.345 |
| Past-only behavioral features, no prior fraud labels | Histogram gradient boosting | 0.257 | 0.658 | 0.561 | 0.289 | 0.381 |
| Past behavioral features + prior fraud-label history | Histogram gradient boosting | 0.870 | 0.988 | 0.890 | 0.774 | 0.828 |

The conservative finding is that unlabeled behavioral histories provide a
modest improvement. Prior-label histories provide much larger gains, but their
validity depends on timely known labels.

## Error analysis

Error analysis is based on the held-out chronological test-set confusion counts
and alert-ranking behavior saved in the project result tables. The examples
below are not new experiments; they interpret the observed false positives,
false negatives, and alert volumes for the main feature sets.

| Feature set | Best model | FP | FN | TP | Alerts |
|:---|:---|---:|---:|---:|---:|
| Baseline transaction/time features | Logistic regression | 42 | 2,438 | 653 | 695 |
| Past-only behavioral features, no prior fraud labels | Histogram gradient boosting | 699 | 2,198 | 893 | 1,592 |
| Past behavioral features + prior fraud-label history | Histogram gradient boosting | 295 | 699 | 2,392 | 2,687 |

The baseline transaction/time model has AP / PR-AUC 0.245, precision 0.940,
recall 0.211, and F1 0.345. It is conservative: it produces very few false
positives, but misses many fraud cases. The past-only behavioral model has AP /
PR-AUC 0.257, precision 0.561, recall 0.289, and F1 0.381. It catches more
fraud but creates more false positives. The prior fraud-label history model has
AP / PR-AUC 0.870, precision 0.890, recall 0.774, and F1 0.828. It is the
strongest test result, but it depends on timely availability of prior fraud
labels.

False positives are legitimate transactions flagged as suspicious. In this
setting they create review cost, customer friction, and possible unnecessary
declined or blocked activity. A likely false-positive pattern is a legitimate
transaction that looks unusual in amount, timing, terminal behavior, or
customer history. The baseline model is conservative: it has high precision
but only 42 false positives, while missing 2,438 fraud cases.

False negatives are fraudulent transactions missed by the detector. They imply
direct fraud loss, delayed intervention, and continued compromised customer or
terminal activity. A likely false-negative pattern is fraud that resembles
normal customer behavior or appears before enough behavioral history has
accumulated. Past-only behavioral features catch more fraud than the baseline
but increase false positives, showing the review-burden trade-off.

The prior-label history model has the strongest test result, but it has
prior-label-specific failure modes. A first fraud in a new pattern may occur
before previous labels are known, producing false negatives. Conversely,
previous fraud history can make later legitimate behavior look suspicious,
producing false positives. These are plausible operational patterns consistent
with the feature design, not proven real-world failure modes.

The FP/FN trade-off is central: higher recall usually increases review burden,
while higher precision may miss more fraud. The best threshold therefore
depends on business and security costs, not only F1. This project uses
validation-set F1 thresholding as a reproducible benchmark, not as a final
deployment policy.

## Critical evaluation

The results support internal reproducibility on the synthetic benchmark. They
do not establish deployment readiness. Synthetic fraud rules may produce
clearer relationships than real payment data, and prior fraud labels may not be
available quickly enough in production.

False positives create investigation cost and customer friction. False
negatives create financial and security loss. A deployable system would require
cost-sensitive thresholding, calibration, delayed-label analysis, drift
monitoring, retraining policy, review-budget planning, and privacy/security
controls for customer histories.

## Detailed reports

- [Source selection](source_selection.md)
- [Reproducibility audit](reproducibility_audit.md)
- [EDA findings](eda_findings.md)
- [Baseline modeling](baseline_modeling.md)
- [Behavioral features](behavioral_features.md)
- [Final synthesis](final_synthesis.md)
- [Final submission checklist](final_submission_checklist.md)
