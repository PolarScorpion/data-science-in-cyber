# Final Report

## Executive summary

This project reproduces and critically evaluates fraud-detection methodology from the Fraud Detection Handbook using its public simulated transaction dataset. The study focuses on temporal validation, fraud-appropriate metrics, baseline modeling, and leakage-safe behavioral feature engineering.

The project shows that chronological validation and Average Precision / PR-AUC are necessary for interpreting rare-event fraud models. It also shows that past behavioral histories improve performance, but the strongest results depend on prior fraud labels being available at scoring time.

The project does not prove real-world fraud-detection performance.

## Source and data

The selected source is the Fraud Detection Handbook. The raw simulated transaction dataset is obtained separately from:

<https://github.com/Fraud-Detection-Handbook/simulated-data-raw>

Raw data are not stored in this repository. The expected local placement is documented in [../data/README.md](../data/README.md).

## Methodology

The analysis uses a chronological train/validation/test split. Validation data are used for threshold selection only, and final metrics are reported on the held-out chronological test period.

The project compares:

- a no-fraud prior baseline;
- logistic regression with class balancing;
- histogram gradient boosting with balanced sample weights;
- Phase 5 baseline features;
- Phase 6 unlabeled behavioral features;
- Phase 6 prior-label behavioral features.

Behavioral features are computed from prior transactions only. Full-window customer and terminal aggregates are not used.

## Main result

| Feature set | Best model | AP / PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Phase 5 baseline | Logistic regression | 0.245 | 0.644 | 0.940 | 0.211 | 0.345 |
| Phase 6 unlabeled behavioral features | Histogram gradient boosting | 0.257 | 0.658 | 0.561 | 0.289 | 0.381 |
| Phase 6 prior-label behavioral features | Histogram gradient boosting | 0.870 | 0.988 | 0.890 | 0.774 | 0.828 |

The conservative finding is that unlabeled behavioral histories provide a modest improvement. Prior-label histories provide much larger gains, but their validity depends on timely known labels.

## Critical evaluation

The results support internal reproducibility on the synthetic benchmark. They do not establish deployment readiness. Synthetic fraud rules may produce clearer relationships than real payment data, and prior fraud labels may not be available quickly enough in production.

False positives create investigation cost and customer friction. False negatives create financial and security loss. A deployable system would require cost-sensitive thresholding, calibration, delayed-label analysis, drift monitoring, retraining policy, review-budget planning, and privacy/security controls for customer histories.

## Detailed reports

- [Source selection](source_selection.md)
- [Reproducibility audit](reproducibility_audit.md)
- [EDA findings](eda_findings.md)
- [Baseline modeling](baseline_modeling.md)
- [Behavioral features](behavioral_features.md)
- [Final synthesis](final_synthesis.md)
- [Final submission checklist](final_submission_checklist.md)
