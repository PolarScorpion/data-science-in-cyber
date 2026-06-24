# Reproducibility Audit

## Audit scope and standard

This audit covers the simulated-data workflow in Chapters 3-5 of the Fraud Detection Handbook. It distinguishes file availability from actual reproducibility. A notebook being public, structurally valid, or accompanied by saved output does not prove that a clean environment can execute it from start to finish.

The following revisions were inspected on 24 June 2026:

- handbook: `81cf7d1714bb7b2f5b496407d9055d91dc68dc25`;
- raw simulated data: `6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a`;
- transformed simulated data: `6e3ca5849b4681430388056d3f1dcfb41d4e8269`.

No handbook notebook was executed end to end during this documentation phase, and no model result is claimed here. The audit combines repository inspection, notebook-structure validation, dependency review, code-path analysis, and successful retrieval of the public data repositories.

## Summary verdict

| Audit question | Verdict | Evidence and consequence |
|---|---|---|
| Is the code available? | Pass | The handbook repository contains the notebooks and shared functions. |
| Is the simulated data available? | Pass | Both raw and transformed datasets were successfully retrieved. |
| Is the whole scoped source reproducible? | Fail | Three `RealWorldData` notebooks depend on confidential data and cannot be independently rerun. |
| Are dependencies documented? | Partial | Versions are pinned, but the manifests disagree and target an old Python stack. |
| Are notebooks structurally executable? | Partial | All 11 scoped notebooks are valid notebook files, but clean end-to-end execution has not yet been demonstrated. |
| Are seeds controlled? | Mostly pass | The simulator and principal models use explicit seeds; complete cross-platform determinism is not established. |
| Is preprocessing explicit? | Mostly pass | Raw generation and feature transformations are shown, but later notebooks commonly bypass them by downloading precomputed transformed data. |
| Is temporal validation explicit? | Pass | Chronological training, delay, validation, and test intervals are implemented. |
| Is leakage ruled out? | Partial | No direct future-label use was found in the reviewed feature code, but generator-feature alignment and repeated test inspection remain serious risks. |
| Are external-validity claims supported? | Fail | A single simplified simulator cannot establish transfer to real payment systems. |

The defensible conclusion is narrow: the simulated workflow is reproducible in principle, subject to recreating its historical environment. The handbook as a whole is not fully reproducible, and the simulator cannot support broad real-world performance claims.

## Code availability

The original repository is public and includes notebook source, saved outputs, shared functions, configuration files, and dependency manifests. Eleven notebooks appear in Chapters 3-5. All eleven parse as notebook format version 4. Across them, 260 code cells were found and 157 code cells contain saved outputs.

Saved output is evidence that some environment executed the cells at some point. It is not a substitute for a clean rerun, and it does not prove that outputs correspond to the current notebook source. The repository does not provide continuous-integration execution of the notebooks or a machine-readable record tying every saved result to a source revision and data revision.

## Data availability and download verification

The approved raw-data repository was successfully cloned. It contains 183 daily pickle files from `2018-04-01.pkl` through `2018-09-30.pkl`, totaling approximately 102.2 MiB. A sample raw file could be read and contained the documented nine columns, including transaction identifiers, timestamps, amount, fraud label, and fraud-scenario label.

The later modeling and metric notebooks do not read the approved raw repository directly. They clone a second public repository, `simulated-data-transformed`, which contains 183 transformed pickle files totaling approximately 321.5 MiB. A sample transformed file was readable. This is an additional execution dependency and must be documented in the final project even though it is publicly available.

The two data repositories have nearly empty README files. They do not provide checksums, a schema, a release tag, or a dataset-specific license. The generator notebook supplies much of the missing technical explanation, but the downloadable artifacts themselves are poorly documented. Public accessibility does not eliminate this provenance and licensing weakness.

## Dependency documentation

Dependency documentation exists, but it is internally inconsistent:

- `requirements.txt` pins NumPy 1.19.5, pandas 1.3.5, scikit-learn 1.0.0, XGBoost 1.5.1, and related packages.
- `binder/environment.yml` pins Python 3.8.3 but uses XGBoost 1.3.3 and seaborn 0.11.1 rather than the versions in `requirements.txt`.
- The README reports a tested Jupyter Book version that differs from `requirements.txt`.
- No lock file or package hashes define one authoritative environment.

This is not a cosmetic issue. The notebooks use `DataFrame.append`, removed in pandas 2.0, and an older `make_scorer(..., needs_proba=True)` interface. Running the notebooks with current packages is therefore expected to fail without code changes. Exact reproduction should first target the documented Python 3.8 environment; modernization must be treated as a separate, recorded modification rather than silently mixed with reproduction.

## Notebook executability

The eight simulated-data notebooks in scope are structurally complete and contain relative data paths. Several include shell cells that clone the transformed dataset automatically. They also assume a particular working directory and access to sibling shared functions. Those assumptions are visible but fragile.

The three real-world notebooks are not executable as independent reproductions. `Baseline_RealWorldData.ipynb` states directly that the Belgian e-commerce transaction data cannot be shared for confidentiality reasons. The later real-world notebooks load saved figures or serialized performance summaries rather than reconstructing the underlying experiments. Their numerical claims cannot be independently audited from the public materials.

Accordingly, the handbook's broad statement that all techniques and results are reproducible is false as written. It is accurate only for the simulated-data subset, and even there a clean end-to-end rerun remains to be demonstrated in a compatible environment.

## Random seeds and determinism

The source makes a serious reproducibility effort. Customer and terminal profiles use fixed NumPy seeds; customer transaction generation seeds both Python and NumPy from the customer identifier; fraud scenarios seed sampling by day; and the baseline estimators generally set `random_state=0`.

That does not prove bit-for-bit determinism. Parallel estimators use `n_jobs=-1`, package versions differ between manifests, and platform-specific numerical behavior is not tested. The appropriate claim is that stochastic choices are substantially controlled, not that identical results are guaranteed on every system.

## Preprocessing and feature engineering transparency

The source exposes the entire simulated path:

1. generation of customer and terminal profiles;
2. generation of time-ordered transactions;
3. application of three fraud-labeling scenarios;
4. time-of-day and weekend features;
5. customer transaction-count and average-amount windows;
6. terminal transaction-count and delayed fraud-risk windows.

The customer rolling features include the current transaction. This is not target leakage because no current fraud label is used, but it must be documented when interpreting behavioral features. The terminal risk calculation subtracts the most recent delay period and is designed to use only labels old enough to be available. Static inspection found no direct future-label reference in that calculation.

Later notebooks usually download precomputed transformed data instead of regenerating it. This is practical, but it weakens provenance unless the transformed revision is pinned and the transformation is independently reproduced at least once.

## Temporal validation

Temporal handling is one of the source's strongest features. The baseline uses a chronological training week, a seven-day feedback delay, and a later test week. Known compromised cards are removed according to labels available before each test day. Chapter 5 implements repeated holdout and prequential folds rather than relying only on shuffled cross-validation.

The seven-day feedback delay is still a modeling assumption, not an empirical fact established by the simulator. Real reporting delays vary by customer, institution, fraud type, and investigation process. The project must therefore test the method under the source's assumption without presenting that assumption as universal.

## Leakage and evaluation risks

### Direct target leakage

No obvious direct target column is included among the listed model inputs. The terminal-risk features deliberately exclude labels from the recent feedback-delay period. The reviewed feature logic is temporally ordered rather than randomly computed across the full dataset.

This finding is provisional until the transformed data are regenerated and checked against the raw files. Precomputed transformed files should never be trusted solely because their column names look plausible.

### Generator-feature alignment

The larger risk is structural. Scenario 1 labels every transaction above a fixed amount as fraud. Scenario 2 compromises selected terminals. Scenario 3 compromises selected customers and multiplies some transaction amounts. The chosen predictors then include transaction amount, customer amount histories, and terminal fraud histories.

This is not conventional target leakage, because the label itself is not copied into the input matrix. It is nevertheless an unusually favorable alignment between the data-generating mechanism and the feature set. An ablation study is required later to determine how much performance comes from this alignment.

### Test-set reuse

The educational notebooks repeatedly plot validation and test performance over the same hyperparameter grids and discuss the test-optimal depths. This is useful exposition, but it is unsafe experimental practice if the test curves influence subsequent choices. The project must select models and thresholds using training and validation data only, then evaluate the final locked choices once on the test period.

### Population and temporal overlap

Customers, terminals, simulator rules, and parameter distributions persist across the time split. This resembles continued operation within one payment system, but it does not test transfer to new populations or attack strategies. A chronological split prevents obvious future-to-past mixing while leaving the generator unchanged.

## Synthetic-data limitations

The simulator is intentionally simple and transparent. That is useful for debugging methodology, but the limitations are severe:

- only three fraud scenarios exist;
- one scenario is an artificial amount threshold;
- fraud labels are complete and error-free;
- feedback delay is simplified;
- customer and terminal behavior follow fixed parametric rules;
- fraudsters do not adapt to the detector;
- institutional, geographic, regulatory, and collection differences are absent;
- the same generator controls every experimental period.

Consequently, internal performance estimates measure success on this simulator. They do not estimate deployment performance. The real-world notebook outputs cannot repair this gap because their underlying data and experiment cannot be independently rerun.

## Testability of the source's claims

The following claims are testable with public materials:

- the raw simulator can reproduce the published class structure and temporal data;
- behavioral transformations improve, harm, or leave unchanged performance on the simulated scenarios;
- ROC-AUC, average precision, and card precision rank models differently under class imbalance and a fixed review budget;
- prequential validation provides a closer or more stable estimate of later test performance than the simpler alternatives used in Chapter 5.

The following claims are not testable from the public materials:

- the reported real-world Belgian transaction results;
- the general effectiveness of the selected features across payment institutions;
- the general superiority of a validation strategy under fraud processes not represented by the simulator.

The project can therefore reproduce and criticize the handbook's internal methodology. It cannot validate deployment effectiveness, and it must not use simulated success as evidence that such effectiveness has been established.
