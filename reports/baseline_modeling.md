# Baseline Fraud Modeling

## Scope

This phase establishes a leakage-controlled baseline for the Fraud Detection Handbook simulated transaction dataset. It uses the raw simulated transactions from 1 April through 30 September 2018 and evaluates models on a chronological holdout period. The phase does not implement behavioral customer or terminal histories; those features require a separate past-only construction step.

## Temporal validation design

Transactions are split by calendar day, not by random row assignment:

| Split | Dates | Rows | Fraud cases | Fraud rate |
|---|---:|---:|---:|---:|
| Train | 2018-04-01 to 2018-07-19 | 1,054,747 | 8,496 | 0.806% |
| Validation | 2018-07-20 to 2018-08-25 | 354,264 | 3,094 | 0.873% |
| Test | 2018-08-26 to 2018-09-30 | 345,144 | 3,091 | 0.896% |

The validation period is used only for threshold selection. Reported metrics use the final test period.

## Predictor set and leakage controls

The baseline feature matrix contains:

- transaction amount,
- log-transformed transaction amount,
- cyclic hour-of-day encodings,
- cyclic day-of-week encodings.

The model excludes `TX_FRAUD`, `TX_FRAUD_SCENARIO`, `TRANSACTION_ID`, `CUSTOMER_ID`, `TERMINAL_ID`, `TX_TIME_SECONDS`, and `TX_TIME_DAYS`. The scenario field is a direct description of the simulator rule and would leak the label. Customer and terminal identifiers are not used as numeric variables because their integer values are arbitrary. Full-period customer or terminal aggregates are also excluded because they would mix past and future information.

## Models

Three baselines were trained:

| Model | Imbalance handling | Role |
|---|---|---|
| No-fraud prior | Training prevalence score; fixed 0.5 threshold | Sanity-check baseline |
| Logistic regression | `class_weight="balanced"` | Linear baseline |
| Histogram gradient boosting | Balanced sample weights | Nonlinear baseline |

No fitted model binaries are saved.

## Test metrics

| Model | Average precision | ROC-AUC | Precision | Recall | F1 | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.245 | 0.644 | 0.940 | 0.211 | 0.345 | 42 | 2,438 | 653 |
| Histogram gradient boosting | 0.131 | 0.633 | 0.514 | 0.227 | 0.315 | 666 | 2,388 | 703 |
| No-fraud prior | 0.009 | 0.500 | 0.000 | 0.000 | 0.000 | 0 | 3,091 | 0 |

Accuracy is intentionally not used as a primary metric. A no-fraud classifier would be correct for more than 99% of test transactions while detecting no fraud.

## Precision@k

| Model | k | Alert rate | Fraud found | Precision@k | Recall@k |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 345 | 0.100% | 345 | 1.000 | 0.112 |
| Histogram gradient boosting | 345 | 0.100% | 165 | 0.478 | 0.053 |
| No-fraud prior | 345 | 0.100% | 3 | 0.009 | 0.001 |
| Logistic regression | 1,726 | 0.500% | 713 | 0.413 | 0.231 |
| Histogram gradient boosting | 1,726 | 0.500% | 711 | 0.412 | 0.230 |
| No-fraud prior | 1,726 | 0.500% | 21 | 0.012 | 0.007 |
| Logistic regression | 3,451 | 1.000% | 756 | 0.219 | 0.245 |
| Histogram gradient boosting | 3,451 | 1.000% | 752 | 0.218 | 0.243 |
| No-fraud prior | 3,451 | 1.000% | 36 | 0.010 | 0.012 |

The constant-score prior baseline has arbitrary tie ordering, so its Precision@k values are included only as a sanity check.

## Interpretation

Logistic regression is the strongest baseline by average precision. Its high precision at the validation-selected threshold reflects a conservative alerting behavior, but recall remains low: it identifies 653 of 3,091 fraud cases in the test period. Histogram gradient boosting identifies slightly more fraud cases at the selected threshold, but with more false positives and lower average precision.

These results support two limited conclusions. First, temporally valid evaluation produces a meaningful baseline that differs sharply from the no-fraud classifier despite the low fraud prevalence. Second, amount and time features alone are insufficient for broad fraud coverage. The next methodological step should test past-only behavioral features and compare their gains against this baseline.

## Limitations

The dataset is synthetic. Strong performance may reflect alignment with the simulator's fraud rules rather than externally valid fraud behavior. The current models also omit behavioral histories, merchant context, device information, and other operational signals that would normally affect fraud detection. The results should therefore be treated as internal benchmark evidence, not deployment evidence.

## Reproducible artifacts

- `results/baseline_split_summary.csv`
- `results/baseline_thresholds.csv`
- `results/baseline_test_metrics.csv`
- `results/baseline_precision_recall_at_k.csv`
- `figures/baseline_model_metrics.png`
- `figures/baseline_precision_recall_roc_curves.png`
- `figures/baseline_confusion_matrices.png`
