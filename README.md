# Data Science in Cybersecurity Final Project

## Project goal

This repository is a critical reproduction study of machine-learning methods for
payment-card fraud detection using the Fraud Detection Handbook simulated
transaction data.

The project asks whether conclusions about temporal validation, metric choice,
and behavioral feature engineering remain supported under the handbook's
simulated fraud scenarios, and how far those conclusions can generalize beyond
data produced by known simulation rules.

The project does not claim production readiness or real-world fraud performance.

## Source and dataset

- Selected source: [Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
- Original handbook repository: <https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook>
- Raw simulated transactions: <https://github.com/Fraud-Detection-Handbook/simulated-data-raw>
- Public transformed transactions used by the handbook notebooks: <https://github.com/Fraud-Detection-Handbook/simulated-data-transformed>

The analysis is limited to the simulated-data portions of Chapters 3-5:
transaction simulation, baseline modeling, temporal validation, performance
metrics, and behavioral feature engineering.

## Main result

The final test period contains 345,144 transactions and 3,091 fraud cases, for
0.896% fraud prevalence. Average Precision / PR-AUC is the main ranking metric
because accuracy is misleading under this class imbalance.

| Feature set | Best model | AP / PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Phase 5 baseline | Logistic regression | 0.245 | 0.644 | 0.940 | 0.211 | 0.345 |
| Phase 6 unlabeled behavioral features | Histogram gradient boosting | 0.257 | 0.658 | 0.561 | 0.289 | 0.381 |
| Phase 6 prior-label behavioral features | Histogram gradient boosting | 0.870 | 0.988 | 0.890 | 0.774 | 0.828 |

Unlabeled behavioral histories provide a modest improvement over the baseline.
Prior-label histories provide a much larger improvement, but only under the
strong assumption that earlier fraud labels are known at scoring time.

## Reproducibility

Tested local Python version: Python 3.12.13.

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Acquire the raw dataset from the project root:

```bash
git clone --depth 1 https://github.com/Fraud-Detection-Handbook/simulated-data-raw.git data/raw/simulated-data-raw
```

Expected local data layout:

```text
data/
|-- README.md
`-- raw/
    `-- simulated-data-raw/
        `-- data/
            |-- 2018-04-01.pkl
            |-- ...
            `-- 2018-09-30.pkl
```

The raw data directory is ignored by Git and must remain untracked.

To run the full analysis, open
[notebooks/main_analysis.ipynb](notebooks/main_analysis.ipynb), select the
project virtual environment as the kernel, and run the notebook from top to
bottom. The notebook loads the local raw pickles, reproduces the EDA, trains
the baseline and behavioral-feature models, saves small metric tables, and
regenerates figures.

Useful validation commands:

```bash
.venv/Scripts/python -m compileall src
.venv/Scripts/python -m pip check
```

## Expected outputs

The committed outputs are lightweight reproducibility artifacts:

- result tables under [results/](results/)
- figures under [figures/](figures/)
- phase reports under [reports/](reports/)

No fitted model binaries are required for the submitted analysis.

## Phase summary

| Phase | Output |
|---|---|
| 1 | Repository scaffold and ignore rules |
| 2 | Source selection and reproducibility audit |
| 3 | Data loading and initial inspection |
| 4 | Exploratory data analysis |
| 5 | Baseline fraud modeling |
| 6 | Leakage-safe behavioral feature engineering |
| 7 | Final synthesis and external-validity analysis |
| 8 | Submission-readiness audit and cleanup |

## Reports

- [Source selection](reports/source_selection.md)
- [Reproducibility audit](reports/reproducibility_audit.md)
- [EDA findings](reports/eda_findings.md)
- [Baseline modeling](reports/baseline_modeling.md)
- [Behavioral features](reports/behavioral_features.md)
- [Final synthesis](reports/final_synthesis.md)
- [Final report](reports/final_report.md)
- [Final submission checklist](reports/final_submission_checklist.md)

## Repository structure

```text
.
|-- data/
|   `-- README.md
|-- figures/
|-- notebooks/
|   `-- main_analysis.ipynb
|-- reports/
|-- results/
|-- src/
|-- README.md
|-- requirements.txt
`-- LICENSE
```

## Limitations

The dataset is synthetic, so strong results may partly reflect recovery of
simulator rules. Prior fraud-label features depend on timely label
availability, which may not hold in production. The project does not model
delayed labels, calibration, drift monitoring, retraining policy,
review-budget constraints, customer impact, or deployment security controls.
