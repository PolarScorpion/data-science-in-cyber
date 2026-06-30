# Data Science in Cybersecurity Final Project

This repository contains a critical reproduction study of machine-learning methods for payment-card fraud detection using the Fraud Detection Handbook simulated transaction data.

## Selected source

[Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)

## Original repository

<https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook>

## Dataset

- Raw simulated transactions: <https://github.com/Fraud-Detection-Handbook/simulated-data-raw>
- Public transformed transactions used by the handbook notebooks: <https://github.com/Fraud-Detection-Handbook/simulated-data-transformed>

## Current scope

The project is limited to the simulated-data portions of Chapters 3-5: transaction simulation, baseline modeling, temporal validation, performance metrics, and behavioral feature engineering. It does not reproduce the whole handbook or claim that results on synthetic data generalize to real payment systems.

The research question is whether the handbook's conclusions about temporal validation, metric choice, and behavioral feature engineering remain supported under its simulated fraud scenarios, and how far those conclusions extend beyond data produced under known fraud rules.

The completed analysis compares Phase 5 baseline models with Phase 6 leakage-safe behavioral features. The final synthesis distinguishes internal reproducibility on the simulated benchmark from external validity in real payment systems.

Main result: unlabeled behavioral histories provide a modest improvement over the baseline, while prior fraud-label histories provide a much larger improvement under the strong assumption that earlier fraud labels are already known at scoring time. The project does not claim production readiness or real-world fraud performance.

## Main reports

- [Source selection](reports/source_selection.md)
- [Reproducibility audit](reports/reproducibility_audit.md)
- [EDA findings](reports/eda_findings.md)
- [Baseline modeling](reports/baseline_modeling.md)
- [Behavioral features](reports/behavioral_features.md)
- [Final synthesis](reports/final_synthesis.md)

## Repository structure

```text
.
|-- data/
|-- figures/
|-- notebooks/
|-- reports/
|-- results/
|-- src/
|-- README.md
|-- requirements.txt
`-- LICENSE
```

Installation and execution instructions will be added after the reproducible environment is finalized.
