# Source Selection

## Selected source

This project evaluates *Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook* by Yann-Ael Le Borgne, Wissam Siblini, Bertrand Lebichot, and Gianluca Bontempi.

- Official handbook: <https://fraud-detection-handbook.github.io/fraud-detection-handbook/>
- Original GitHub repository: <https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook>
- Raw simulated dataset: <https://github.com/Fraud-Detection-Handbook/simulated-data-raw>
- Public transformed dataset used by the modeling notebooks: <https://github.com/Fraud-Detection-Handbook/simulated-data-transformed>

The reproducibility audit examined handbook revision `81cf7d1714bb7b2f5b496407d9055d91dc68dc25`, raw-data revision `6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a`, and transformed-data revision `6e3ca5849b4681430388056d3f1dcfb41d4e8269`. These identifiers make the reviewed source state explicit; they are not claims that later revisions are equivalent.

## Research question

> Do the handbook's conclusions about temporal validation, metric choice, and behavioral feature engineering remain supported under its simulated fraud scenarios, and how much do those conclusions generalize beyond data produced under known fraud rules?

The project will separate three questions that the source sometimes places too close together:

1. Can the simulated experiments be reproduced internally?
2. Are the proposed validation and evaluation methods useful for fraud detection?
3. Do results on this simulator justify expectations about real payment systems?

A positive answer to the first question does not establish the third.

## Cybersecurity task

The cybersecurity task is payment-card fraud detection. Each transaction receives a binary fraud label. In operational terms, false positives consume investigator capacity and can interrupt legitimate payments, while false negatives allow fraudulent transactions and compromised cards to remain undetected.

## Machine-learning and data-science task

The source treats fraud detection as highly imbalanced, temporally ordered binary classification. The relevant Chapters 3-5 cover:

- simulation of legitimate and fraudulent transactions;
- time, customer-behavior, and terminal-risk feature engineering;
- baseline supervised classifiers;
- chronological training, feedback delay, and test periods;
- repeated holdout and prequential validation;
- ROC-AUC, average precision, and card precision at a fixed review budget.

This combination is a strong match for the course because it joins statistical evaluation with operational cybersecurity consequences.

## Why the source satisfies the assignment

The source defines a concrete security problem, provides an implemented data-science approach, publishes executable notebooks, and supplies both raw and transformed simulated data. The raw repository contains 183 daily pickle files covering 1 April through 30 September 2018. The source also exposes claims that can be tested rather than merely summarized: the usefulness of behavioral features, the limitations of ROC-AUC under imbalance, and the asserted robustness of prequential validation.

The source is unusually suitable for critical reproduction because it documents its simulator. The fraud labels are not mysterious ground truth: they are created by three explicit rules. That transparency makes internal reproduction feasible and also creates the central external-validity problem. A model may score well because its features closely reflect those rules, not because it has learned fraud patterns that transfer to a different payment system.

## Scope: Chapters 3-5 only

The project is deliberately limited to the simulated-data portions of Chapters 3-5:

- Chapter 3: simulator, baseline feature transformation, and baseline modeling;
- Chapter 4: threshold-based, threshold-free, and top-k performance metrics;
- Chapter 5: temporal validation and model selection.

The deep-learning and imbalance-learning chapters are excluded. They would broaden the project without improving the answer to the research question. The three `RealWorldData` notebooks within Chapters 3-5 are also excluded from experimental reproduction because the handbook explicitly states that their underlying transaction data are confidential. Their saved figures and summaries may provide context, but they cannot serve as independently verified evidence.

## Risks and limitations

The main risks are methodological rather than computational:

- The fraud generator contains only three scenarios. One is a direct amount threshold; the other two create temporarily compromised terminals or customers.
- The engineered features intentionally summarize transaction amount, customer spending, and terminal fraud history. These are closely aligned with the label-generating mechanisms.
- All simulated train, validation, and test periods come from the same generator and parameterization. A chronological split therefore does not test transfer to a new institution, population, attack strategy, or data-collection process.
- The simulator supplies complete labels and simplified feedback timing. Real fraud labels can be delayed, disputed, missing, or biased toward investigated cases.
- The original software stack is old and must be recreated or carefully modernized before exact notebook execution.
- The public data repositories do not state a dataset-specific license or provide checksums in their minimal README files.

These weaknesses do not make the source unsuitable. They are precisely the issues the project must measure and discuss without converting simulated benchmark performance into a deployment claim.

## Final justification over the alternatives

[EMBER](https://github.com/elastic/ember) was not selected because its 2018 archive alone is approximately 1.70 GB compressed, its feature extraction is sensitive to an old LIEF version, and its static PE benchmark would impose substantially greater storage and dependency risk. It remains academically strong, but it is less practical for a focused local reproduction.

[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) was not selected because the candidate implementation lacks a dependency manifest and the required preprocessed files, relies on obsolete machine-learning interfaces, and does not clearly connect each script to a published result. That project offers severe reproducibility weaknesses, but repairing them would displace the required analysis.

The Fraud Detection Handbook is the strongest choice because its data size is manageable, its temporal structure is explicit, its code and data are public for the simulated experiments, and its central claims can be tested through controlled comparisons. It also supports a meaningful negative conclusion if internal reproducibility succeeds while external validity remains unsupported.
