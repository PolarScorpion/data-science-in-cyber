# Final Submission Checklist

## Repository state checked

- Branch: `cyber-final-project`
- Pre-audit latest commit: `8c94154 docs: add final project synthesis`
- Relationship to remote before this cleanup:
  ahead of `origin/cyber-final-project` by 3 commits and not behind
- Working tree before this cleanup: clean

## Commits included before this checklist

| Commit | Purpose |
|---|---|
| `8ed842f` | Initial license commit |
| `eb70617` | Project scaffold |
| `a982ae5` | Source selection and reproducibility audit |
| `055ad26` | Data loading and initial inspection |
| `7e894f2` | Exploratory data analysis |
| `60b3ba9` | Baseline fraud modeling |
| `717f2d6` | Leakage-safe behavioral features |
| `8c94154` | Final project synthesis |

## Checks passed

- README includes project goal, dataset source, setup instructions, run
  instructions, expected outputs, phase summary, main result table, project
  structure, reports, and limitations.
- Raw data are documented as local-only under `data/raw/simulated-data-raw/`.
- Notebook headings are coherent and numbered from 1 through 42.
- The notebook parses successfully.
- Source files compile with `python -m compileall src`.
- Installed packages pass `python -m pip check`.
- Local README and report links resolve to existing files.
- Referenced result tables and figures exist.
- Raw data, local environments, fitted model binaries, sensitive files, and
  private files are not tracked.

## Known limitations retained

- The benchmark data are synthetic.
- Prior-label behavioral features require timely known labels.
- Label delay is not simulated.
- Probability calibration is not evaluated.
- Drift monitoring and retraining policy are not implemented.
- Review-budget and cost-sensitive thresholding are discussed but not optimized.
- The project does not claim production readiness.
