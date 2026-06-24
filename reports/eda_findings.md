# Exploratory Data Analysis Findings

## Scope and evidential status

This note records Phase 4 exploratory findings for the Fraud Detection Handbook simulated transaction dataset. The analysis covers 1,754,155 transactions from 1 April through 30 September 2018 (183 calendar days). It describes the benchmark and identifies requirements for later experiments; it does not report predictive performance, and no estimator was trained.

The distinction between reproduced observations and external claims is material. Results below establish properties of the published simulation output. They do not establish that the same magnitudes, relationships, or operational trade-offs hold in a payment system whose fraud mechanisms are unknown.

## Data quality and distributions

All nine raw columns contain zero missing values. There are also no exact duplicate rows. This completeness reduces ambiguity in reproducing the benchmark, but it is a property of simulator output rather than evidence about production data quality. Real payment data may contain delayed events, missing attributes, inconsistent upstream records, or collection changes. Consequently, later methodology should specify how missing and late-arriving information would be handled even though this benchmark does not require imputation.

Transaction amount is right-skewed: its mean is 53.63, median is 44.64, 99th percentile is 171.59, and maximum is 2,628.00. The skewness estimate is 2.44. A temporary `log1p` view makes the central and upper portions of the distribution easier to compare, but the raw column remains unchanged. Scaling or a logarithmic transformation may be tested later for models that are sensitive to feature magnitude or skew; such a transformation must be estimated without using the evaluation period.

The source files store `TX_TIME_SECONDS` and `TX_TIME_DAYS` with object dtype despite integer-valued contents. Phase 4 converts them to numeric values only in a checked analytical copy. Their distributions span the full observation window and are not behavioral measurements by themselves. A permanent conversion, if required later, should be explicit and schema-tested.

The Tukey 1.5-IQR upper bound for amount is 160.86. A total of 27,449 transactions (1.565%) exceed this bound. Their fraud rate is 14.30%, substantially above the overall prevalence, but 23,524 of these observations are legitimate and 73.26% of all fraud observations do not exceed the bound. High amount is therefore neither a fraud definition nor a defensible deletion criterion. Removing these observations would discard legitimate upper-tail activity and a non-trivial share of fraud.

Fraud and legitimate amount distributions overlap. The fraud median is 72.22, compared with 44.49 for legitimate transactions. Fraud observations have a much longer upper tail: their 95th percentile is 470.00, whereas the legitimate 95th percentile is 129.72. The discontinuity is partly explained by the simulator: every transaction at or above the temporary 220 amount boundary in this dataset is labeled fraudulent. This relationship is useful for verifying that the generator output is internally coherent, but it is unusually deterministic and should not be treated as evidence for an operational threshold.

## Temporal behavior

Daily transaction volume is stable relative to the total scale, ranging from 9,285 to 9,789 transactions, with a median of 9,581. Fraud prevalence is not equally stable. Daily prevalence ranges from 0.032% to 1.139%; the first weeks show a pronounced increase before rates fluctuate around a higher level. This early-period structure may reflect initialization and history-dependent simulator rules. A random row split would distribute these phases across both training and evaluation data, producing an evaluation that does not represent prediction of later fraud from earlier observations.

Chronological validation is therefore required. Behavioral features must also be reconstructed at each transaction time from prior events only. A temporal split by itself would not prevent leakage if full-period customer or terminal aggregates were computed before splitting.

Hourly fraud prevalence varies from 0.770% at hour 03 to 0.979% at hour 01. These differences are modest relative to the amount-bin discontinuity, and the lowest-volume overnight hours have less exposure. Hour should be treated as cyclic if tested later; an ordinary integer encoding would incorrectly place hours 23 and 00 far apart.

## Customer, terminal, and scenario concentration

The data contain 4,990 customer identifiers and 10,000 terminal identifiers. Across the full observation window, 69.58% of customers and 42.66% of terminals have at least one fraud observation. Maximum full-period fraud rates reach 20.0% for a customer and 35.6% for a terminal. These aggregates demonstrate entity-level concentration, but they are descriptive only. They combine past and future labels and would constitute target leakage if supplied directly to a predictive model.

Any later behavioral variables must use lagged, rolling histories. Examples include prior transaction frequency, prior average amount, and prior fraud exposure, all calculated strictly before the transaction being scored. The simulation's fraud rules explicitly use customer and terminal histories; strong gains from similarly constructed variables may therefore indicate successful reproduction and generator-rule alignment at the same time.

`TX_FRAUD_SCENARIO` is deterministically related to the target: scenario 0 contains 1,739,474 legitimate transactions, while scenarios 1, 2, and 3 contain 973, 9,077, and 4,631 fraud observations respectively, with no cross-label cases. It is an explanatory label-generation field and must not be used as a predictor, preprocessing input, grouping feature for model selection, or source of thresholds.

## Correlation method and result

Spearman rank correlation was selected for the raw numerical analysis because amount is skewed, contains extreme values, and its relationship with fraud need not be linear. Pearson correlation measures standardized linear covariance and would be more sensitive to the upper amount tail. Kendall correlation has a useful concordance interpretation but is less practical for this dataset size and extensive ties. Spearman reduces, but does not eliminate, problems caused by outliers and non-normality; it also discards magnitude information and does not establish causation.

The selected matrix excludes identifiers because their numerical codes are arbitrary and excludes `TX_FRAUD_SCENARIO` because it leaks the label-generation rule. Spearman correlation between amount and fraud is 0.0491. This small global coefficient coexists with a deterministic upper-amount rule because fraud is rare and other scenarios occur at lower amounts. `TX_TIME_SECONDS` and `TX_TIME_DAYS` have correlation 1.0000, confirming that they are redundant encodings of transaction position in time. A high or low coefficient does not determine whether a feature is safe: temporal availability and generation logic must be examined separately.

## Class imbalance and error meaning

There are 14,681 fraud observations (0.83693%) and 1,739,474 legitimate observations (99.16307%). Predicting every transaction as legitimate would achieve 99.163% accuracy while detecting no fraud. Accuracy is therefore unsuitable as the primary evaluation measure.

Later experiments should report precision and recall, precision-recall ranking performance, and threshold-dependent alert volume under chronological validation. False negatives may permit financial loss and continuing account or terminal abuse. False positives may cause review cost, legitimate payment declines, and customer harm. The appropriate operating point depends on these asymmetric consequences and cannot be inferred from prevalence alone.

## Implications for later phases

- Preserve chronological order in all splits and compute every behavioral aggregate from past-only data.
- Exclude transaction identifiers and `TX_FRAUD_SCENARIO` from predictors; avoid redundant time encodings without a stated purpose.
- Retain high-amount observations. Test scaling or logarithmic amount representations within training data rather than as an irreversible cleaning step.
- Compare class-sensitive metrics and threshold-level alert burdens; do not optimize or select models by accuracy.
- Separate internal reproduction from external validity. Agreement with the handbook on this benchmark can show methodological reproducibility, while unusually strong effects may also reflect alignment between engineered variables and known simulation rules.
- Treat the absence of missing data and the deterministic high-amount boundary as limitations when discussing deployment relevance.

## Reproducible figures

- [`transaction_amount_distribution.png`](../figures/transaction_amount_distribution.png)
- [`transaction_amount_by_fraud_label.png`](../figures/transaction_amount_by_fraud_label.png)
- [`transactions_over_time.png`](../figures/transactions_over_time.png)
- [`fraud_prevalence_over_time.png`](../figures/fraud_prevalence_over_time.png)
- [`fraud_rate_by_hour.png`](../figures/fraud_rate_by_hour.png)
- [`fraud_rate_by_amount_bin.png`](../figures/fraud_rate_by_amount_bin.png)
- [`correlation_heatmap_spearman.png`](../figures/correlation_heatmap_spearman.png)

These findings delimit the next phase: feature construction and model evaluation must test whether temporal validation, metric choice, and behavioral features remain informative without treating simulator-specific regularities as general evidence.
