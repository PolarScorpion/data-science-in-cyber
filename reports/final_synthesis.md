# Final Synthesis and External-Validity Analysis

## Scope of the evidence

This project is a critical reproduction study of fraud-detection methodology using the Fraud Detection Handbook simulated transaction data. The analysis supports claims about internal reproducibility on this benchmark. It does not prove real-world fraud-detection performance, and it does not show that any model is ready for deployment in a payment system.

The central result is methodological rather than operational: chronological validation, precision-recall metrics, and past-only behavioral features materially change the interpretation of model quality. The strongest results come from prior fraud-label histories, but those features require an operational assumption that earlier labels are known before later transactions are scored.

## Evaluation design

The final experiments use a chronological train/validation/test split:

| Split | Dates | Rows | Fraud cases | Fraud rate |
|---|---:|---:|---:|---:|
| Train | 2018-04-01 to 2018-07-19 | 1,054,747 | 8,496 | 0.806% |
| Validation | 2018-07-20 to 2018-08-25 | 354,264 | 3,094 | 0.873% |
| Test | 2018-08-26 to 2018-09-30 | 345,144 | 3,091 | 0.896% |

Thresholds for logistic regression and histogram gradient boosting are selected only on the validation period by maximizing F1. The final test period is not used for fitting or threshold selection. The no-fraud prior baseline uses a fixed 0.5 threshold.

Average Precision, also interpreted as PR-AUC, is the main ranking metric because fraud prevalence in the test period is below 1%. Accuracy is misleading in this setting: a classifier that predicts every transaction as legitimate would be correct for more than 99% of test transactions while detecting no fraud.

## Phase 5 and Phase 6 comparison

| Feature set | Model | AP / PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 5 baseline | Logistic regression | 0.245 | 0.644 | 0.940 | 0.211 | 0.345 | 653 | 42 | 2,438 |
| Phase 5 baseline | Histogram gradient boosting | 0.131 | 0.633 | 0.514 | 0.227 | 0.315 | 703 | 666 | 2,388 |
| Phase 5 baseline | No-fraud prior | 0.009 | 0.500 | 0.000 | 0.000 | 0.000 | 0 | 0 | 3,091 |
| Behavioral, unlabeled | Logistic regression | 0.250 | 0.666 | 0.424 | 0.252 | 0.316 | 780 | 1,060 | 2,311 |
| Behavioral, unlabeled | Histogram gradient boosting | 0.257 | 0.658 | 0.561 | 0.289 | 0.381 | 893 | 699 | 2,198 |
| Behavioral, label history | Logistic regression | 0.823 | 0.984 | 0.821 | 0.754 | 0.787 | 2,332 | 507 | 759 |
| Behavioral, label history | Histogram gradient boosting | 0.870 | 0.988 | 0.890 | 0.774 | 0.828 | 2,392 | 295 | 699 |

The unlabeled behavioral features produce a modest but credible improvement. For histogram gradient boosting, AP increases from 0.131 to 0.257 and recall increases from 0.227 to 0.289. This improvement comes from features that could plausibly be available at authorization time: prior counts, prior amounts, prior frequencies, time gaps, and customer-terminal interaction histories.

The label-history feature set produces a much larger improvement. Histogram gradient boosting reaches AP 0.870 and recall 0.774. This result should be interpreted carefully. It shows that prior fraud-label histories are highly informative in the simulated benchmark, but it depends on earlier fraud labels being available with little delay. If real labels arrive after investigation, dispute resolution, or chargeback windows, the same feature set would not be valid without explicit lagging.

## Scientific interpretation

The project supports three limited claims.

First, temporal validation is necessary. The data have time-dependent fraud prevalence and entity histories, so random row splitting would risk mixing future behavior into training or validation.

Second, fraud-appropriate metrics are necessary. ROC-AUC is useful but insufficient because fraud is rare. A model can improve ROC-AUC while still producing weak alert precision or recall at realistic review budgets. Average Precision and Precision@k better reflect the ranking problem faced by fraud review teams.

Third, behavioral features matter, but their interpretation depends on what information is available at prediction time. Unlabeled behavior gives a moderate gain and is methodologically defensible. Label-history behavior gives a large gain, but its real-world validity depends on label latency and label quality.

## What the project does not prove

The project does not prove that the best model would perform well on real bank, merchant, or payment-processor data. It does not estimate production cost savings, customer harm, review workload, or fraud-loss reduction. It also does not prove that prior fraud labels are available fast enough to support label-history features in an online scoring system.

The simulated generator may create cleaner and stronger relationships than real fraud data. In particular, amount rules, terminal histories, and customer histories may align with the same mechanisms used to assign labels. Strong model performance can therefore reflect recovery of simulator rules rather than discovery of externally valid fraud behavior.

## External-validity risks

Synthetic data are useful for reproducibility, but they narrow the evidence. Real fraud behavior is adaptive, adversarial, and affected by changing merchant mix, customer behavior, detection controls, and reporting delays. A feature that works well against a known simulation rule may degrade when fraud patterns change.

Prior fraud labels are especially fragile. In production, fraud labels may arrive late, be revised, or represent different business processes. A transaction scored at authorization time may not have access to recent labels that appear available in a historical dataset.

Class imbalance also makes threshold choice cost-sensitive. False positives create investigation cost, customer friction, and possible payment declines. False negatives create financial loss and allow compromised accounts or terminals to remain active. F1 is a useful validation rule for this study, but it does not encode business costs or review capacity.

## Deployment limitations

A deployment-oriented study would need additional evidence before any operational claim could be made:

- label-delay simulation, with fraud histories lagged by realistic review or chargeback windows;
- probability calibration and Brier score analysis;
- monitoring for drift in fraud prevalence, amount distributions, terminal behavior, and customer activity;
- a retraining schedule tied to temporal degradation rather than only to model age;
- thresholding based on review budget, expected loss, and customer-impact constraints;
- privacy and security controls for customer and terminal histories;
- documentation of how missing, delayed, or corrected transaction records are handled.

None of these deployment requirements is solved by the present project.

## Methodological strengths

The project has several strengths as a reproducible benchmark study:

- it preserves chronological train/validation/test separation;
- it avoids random temporal leakage;
- it computes behavioral histories from past transactions only;
- it separates unlabeled histories from prior-label histories;
- it selects thresholds on validation data only;
- it reports final metrics on a held-out chronological test period;
- it uses reproducible scripts, notebooks, saved metric tables, and fixed random seeds;
- it avoids saving raw data and fitted model binaries in the repository.

These strengths improve internal validity. They do not remove the external-validity limits created by synthetic data.

## Future work

The most valuable next steps are analytical rather than simply adding larger models:

- cost-sensitive thresholding based on false-positive review cost and false-negative fraud loss;
- calibration curves and Brier score analysis;
- delayed-label simulation for prior fraud-history features;
- drift analysis across time windows;
- alert-budget optimization using Precision@k and Recall@k;
- ablation studies separating amount, time, customer, terminal, interaction, and label-history features;
- stronger models only after the same leakage checks are preserved.

## Final conclusion

The project reproduces a meaningful fraud-detection workflow on a public synthetic benchmark and demonstrates why temporal validation, precision-recall evaluation, and past-only feature construction are essential. The conservative result is that unlabeled behavioral histories improve fraud ranking modestly. The strongest result, using prior fraud-label histories, is internally impressive but externally conditional. It should be treated as evidence that the simulated benchmark contains strong label-history structure, not as evidence that a deployable fraud model has been built.
