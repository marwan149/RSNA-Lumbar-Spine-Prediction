# RSNA Lumbar Spine Prediction

This repository converts the notebook inference workflow from `rsna-lumbar-spine-prediction.ipynb` into a reusable Python package and CLI-based repo structure.

## Repository structure

- `src/rsna_lumbar_spine_prediction/`: package source code
- `scripts/run_predict.py`: inference entrypoint for the test dataset
- `requirements.txt`: runtime dependencies
- `pyproject.toml`: package metadata and optional development dependencies
- `tests/`: small smoke tests for repository imports and config defaults

## Installation

Install runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the package locally for development and CLI access:

```bash
python -m pip install -e .
```

You can also run the package directly via the installed console script:

```bash
rsna-lumbar-spine-prediction --help
```

## Usage

```bash
python scripts/run_predict.py \
  --data-dir /path/to/rsna-2024-lumbar-spine-degenerative-classification \
  --checkpoints /path/to/model_fold-0.pt /path/to/model_fold-1.pt /path/to/model_fold-2.pt \
  --output submission.csv
```

## Docker

Build the image:

```bash
docker build -t rsna-lumbar-spine-prediction .
```

Run inference inside the container:

```bash
docker run --rm -v /path/to/data:/data -v /path/to/checkpoints:/checkpoints rsna-lumbar-spine-prediction \
  --data-dir /data \
  --checkpoints /checkpoints/model_fold-0.pt /checkpoints/model_fold-1.pt /checkpoints/model_fold-2.pt \
  --output /data/submission.csv
```

## Notes

- The package expects the Kaggle-style dataset layout:
  - `test_series_descriptions.csv`
  - `sample_submission.csv`
  - `test_images/<study_id>/<series_id>/*.dcm`
- The model pipeline supports ensemble inference with multiple checkpoints.
- Output is saved to the file specified by `--output`.
