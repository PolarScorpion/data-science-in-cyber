# Data

## Dataset source

The project uses the Fraud Detection Handbook's public simulated transaction data:

<https://github.com/Fraud-Detection-Handbook/simulated-data-raw>

The dataset repository does not state a dataset-specific license. Do not redistribute its files through this project repository.

## Expected local structure

```text
data/
|-- README.md
|-- raw/
|   `-- simulated-data-raw/
|       `-- data/
|           |-- 2018-04-01.pkl
|           |-- ...
|           `-- 2018-09-30.pkl
`-- processed/
```

The official repository contains 183 daily pickle files. Raw and processed data directories are ignored by Git and must not be force-added.

## Acquisition

From the project root, run:

```bash
git clone --depth 1 https://github.com/Fraud-Detection-Handbook/simulated-data-raw.git data/raw/simulated-data-raw
```

Verify that files are available:

```bash
python -c "from src.data_loading import load_available_transaction_files; print(len(load_available_transaction_files()))"
```

The expected count is 183. Importing `src.data_loading` never retrieves data. If the files are missing, the loader reports this acquisition command and stops.

## File format and trust boundary

The daily files use Python's pickle format. Pickles can execute code during loading. Only load files obtained from the documented official repository, and do not replace them with files from an untrusted source.

No processed data are created during Phase 3. If later phases create processed artifacts, they belong under `data/processed/` and remain untracked unless a small file is explicitly reviewed and justified.
