# Leakage-Safe Behavioral Feature Engineering

## Scope

This phase evaluates whether historical customer and terminal behavior improves fraud detection on the Fraud Detection Handbook simulated transaction dataset. The experiment keeps the Phase 5 chronological split and model families, then compares the original baseline feature set with two behavioral feature sets.

## Temporal validation

The split is unchanged from Phase 5:

| Split | Dates | Rows | Fraud cases | Fraud rate |
|---|---:|---:|---:|---:|
| Train | 2018-04-01 to 2018-07-19 | 1,054,747 | 8,496 | 0.806% |
| Validation | 2018-07-20 to 2018-08-25 | 354,264 | 3,094 | 0.873% |
| Test | 2018-08-26 to 2018-09-30 | 345,144 | 3,091 | 0.896% |

Validation is used for threshold selection only. Test metrics are computed on the final chronological holdout period.

## Feature sets

The baseline transaction/time feature set from Phase 5 contains transaction amount, log amount, cyclic hour-of-day encodings, and cyclic day-of-week encodings.

The past-only behavioral feature set from Phase 6 adds:

- customer prior transaction count,
- customer prior mean, median, and standard deviation of amount,
- customer transaction count in the prior 1 day and 7 days,
- customer time since previous transaction,
- terminal prior transaction count,
- terminal prior mean amount,
- terminal transaction count in the prior 1 day and 7 days,
- terminal time since previous transaction,
- prior customer-terminal interaction count.

The label-history behavioral feature set adds:

- customer prior fraud count,
- customer prior fraud rate,
- customer prior 7-day fraud count,
- terminal prior fraud count,
- terminal prior fraud rate,
- terminal prior 7-day fraud count.

These label-history variables are valid only under the operational assumption that earlier fraud labels are already known when later transactions are scored. Because this assumption may not hold in real fraud operations, the unlabeled feature set is the more conservative estimate of behavioral-feature value.

## Leakage-prevention strategy

Transactions are sorted by `TX_DATETIME` and `TRANSACTION_ID`. Behavioral histories are computed from prior transactions only. Entity-time blocks are used so transactions sharing the same entity and timestamp do not see one another as past events. The current transaction's label is never used to build its own features.

The direct leakage fields `TX_FRAUD` and `TX_FRAUD_SCENARIO` are not predictors. `TX_FRAUD_SCENARIO` remains excluded because it describes the simulator rule that produced the label. Full-window customer and terminal aggregates are not used.

## Test metrics

| Feature set | Model | AP / PR-AUC | ROC-AUC | Precision | Recall | F1 | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline transaction/time features | Logistic regression | 0.245 | 0.644 | 0.940 | 0.211 | 0.345 | 42 | 2,438 | 653 |
| Baseline transaction/time features | Histogram gradient boosting | 0.131 | 0.633 | 0.514 | 0.227 | 0.315 | 666 | 2,388 | 703 |
| Past-only behavioral features | Logistic regression | 0.250 | 0.666 | 0.424 | 0.252 | 0.316 | 1,060 | 2,311 | 780 |
| Past-only behavioral features | Histogram gradient boosting | 0.257 | 0.658 | 0.561 | 0.289 | 0.381 | 699 | 2,198 | 893 |
| Past behavioral features + prior fraud-label history | Logistic regression | 0.823 | 0.984 | 0.821 | 0.754 | 0.787 | 507 | 759 | 2,332 |
| Past behavioral features + prior fraud-label history | Histogram gradient boosting | 0.870 | 0.988 | 0.890 | 0.774 | 0.828 | 295 | 699 | 2,392 |
| No-fraud prior | All feature sets | 0.009 | 0.500 | 0.000 | 0.000 | 0.000 | 0 | 3,091 | 0 |

Accuracy is not reported as a main metric because the test fraud prevalence is only 0.896%. A no-fraud classifier would have high accuracy while detecting no fraud.

## Comparison with Phase 5

The unlabeled behavioral features produce modest improvements. For histogram gradient boosting, average precision increases from 0.131 to 0.257 and recall increases from 0.227 to 0.289. Logistic regression gains slightly in average precision and recall, but its F1 decreases because the validation-selected threshold produces more false positives.

The label-history feature set produces much larger improvements. Histogram gradient boosting reaches AP 0.870 and recall 0.774. This result is internally plausible for the simulated benchmark because the fraud generator creates customer and terminal histories that are highly informative. It is not equally strong evidence for external validity because real fraud labels may be delayed, noisy, disputed, or unavailable at authorization time.

## Precision@k

At a 0.5% alert budget on the test period (`k = 1,726`):

| Feature set | Model | Fraud found | Precision@k | Recall@k |
|---|---|---:|---:|---:|
| Baseline transaction/time features | Logistic regression | 713 | 0.413 | 0.231 |
| Baseline transaction/time features | Histogram gradient boosting | 711 | 0.412 | 0.230 |
| Past-only behavioral features | Logistic regression | 772 | 0.447 | 0.250 |
| Past-only behavioral features | Histogram gradient boosting | 913 | 0.529 | 0.295 |
| Past behavioral features + prior fraud-label history | Logistic regression | 1,597 | 0.925 | 0.517 |
| Past behavioral features + prior fraud-label history | Histogram gradient boosting | 1,666 | 0.965 | 0.539 |

The ranking view confirms the same pattern: unlabeled behavioral histories help, while label histories dominate under the immediate-label assumption.

## Interpretation

The conservative conclusion is that past transaction behavior improves the baseline, but not enough to solve the rare-event problem by itself. The best unlabeled result, histogram gradient boosting, increases AP by 0.126 and recall by 0.061 relative to its Phase 5 counterpart. This is a meaningful improvement because it uses only information that would be observable before the current transaction.

The label-history result is much stronger but must be treated separately. It shows that past fraud labels are highly predictive in this simulated benchmark. It does not prove that a real payment system would have timely, accurate labels available for feature updates. If labels arrive after manual review, chargeback windows, or investigation delays, these features would require lagging or exclusion.

## Limitations

The data are synthetic and produced by known fraud scenarios. Behavioral features may align with the same rules that created the labels. Strong gains, especially from prior fraud rates, may therefore reflect generator-rule recovery rather than general fraud-detection validity.

The experiment also omits label-delay modeling, transaction authorization latency, calibration, cost-sensitive threshold selection, and feature drift analysis. These omissions matter because operational fraud systems often choose thresholds based on review capacity and asymmetric costs rather than F1 alone.

## Reproducible artifacts

- `results/behavioral_split_summary.csv`
- `results/behavioral_feature_sets.csv`
- `results/behavioral_thresholds.csv`
- `results/behavioral_test_metrics.csv`
- `results/behavioral_metric_comparison.csv`
- `results/behavioral_precision_recall_at_k.csv`
- `results/behavioral_logistic_coefficients.csv`
- `figures/behavioral_model_comparison.png`
- `figures/behavioral_precision_recall_curves.png`
- `figures/behavioral_logistic_coefficients.png`
