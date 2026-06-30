# Data Science in Cybersecurity Final Project

This repository contains a critical reproduction study of machine-learning methods for payment-card fraud detection. No experimental results are claimed yet.

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

Phase 5 adds leakage-controlled baseline fraud modeling with a chronological train/validation/test split, class-imbalance-aware metrics, and lightweight saved metric tables and figures. It does not yet implement behavioral customer or terminal features.

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
