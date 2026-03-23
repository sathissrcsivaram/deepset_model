# DeepSet Model

Baseline DeepSets model for heatmap-based source localization, with synthetic event generation, training, and evaluation.

This repository contains only the baseline DeepSets model. It does not include the attention-based model.

## Overview

The goal of this project is to predict the source position on a `50 x 50` grid from a set of simulated detector events.

Each sample is stored as a CSV file containing many events. The model reads those events, converts each event into a feature vector, and predicts a 2D heatmap. The highest point in that heatmap is treated as the predicted source location.

## Current Workflow

The end-to-end workflow in this repository is:

1. Generate synthetic event CSV files with `generate_events`
2. Load each CSV as one training sample
3. Convert each event into 12 engineered features
4. Train the baseline DeepSets model to predict a `50 x 50` target heatmap
5. Evaluate the saved model on a held-out test split
6. Save metrics, plots, predictions, and preview heatmaps

## Repository Layout

```text
.
├── baseline_deepsets/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── heatmap.py
│   ├── metrics.py
│   ├── model.py
│   ├── train.py
│   └── README.md
├── generate_events/
│   ├── __init__.py
│   ├── gen_sim_events.py
│   └── params.py
├── README.md
├── pyproject.toml
└── .gitignore
```

### Layout Explanation

- `baseline_deepsets/`
  Main baseline-model package. This contains the model, feature processing, training, evaluation, and helper utilities.
- `generate_events/`
  Synthetic data generation package. This is used to create the CSV event files needed for training and evaluation.
- `README.md`
  Project documentation and usage guide.
- `pyproject.toml`
  Python project metadata and dependency definitions.
- `.gitignore`
  Excludes generated data, logs, cache files, and virtual environments from version control.

## Package Breakdown

### `generate_events/`

Generates synthetic detector-event CSV files.

- `gen_sim_events.py`: dataset generation script
- `params.py`: simulation constants and physics values

### `baseline_deepsets/`

Contains the complete baseline-only model pipeline.

- `model.py`: DeepSets baseline architecture
- `data.py`: CSV loading and feature engineering
- `train.py`: model training
- `evaluate.py`: test-time evaluation
- `heatmap.py`: Gaussian heatmap creation and argmax decoding
- `metrics.py`: metric summaries and inference timing
- `config.py`: paths and hyperparameters

### `baseline_deepsets/` File Roles

- `model.py`
  Defines the baseline DeepSets neural network.
- `data.py`
  Reads CSV files, selects the required columns, and converts raw event values into the 12 model input features.
- `train.py`
  Trains the model, logs metrics, and saves checkpoints and plots.
- `evaluate.py`
  Loads a trained checkpoint, runs prediction on the held-out test split, and saves preview outputs.
- `heatmap.py`
  Creates Gaussian target heatmaps and converts predicted heatmaps back to `(x, y)` coordinates.
- `metrics.py`
  Computes summary metrics such as accuracy, distance errors, and inference speed.
- `config.py`
  Central place for paths, training settings, and model hyperparameters.
- `__init__.py`
  Makes the package importable as `baseline_deepsets`.

## Baseline Model Architecture

The baseline model is a DeepSets-style network:

1. Each event is converted into a 12-dimensional feature vector
2. The same encoder MLP is applied to every event independently
3. Event embeddings are pooled with:
   - mean
   - max
   - standard deviation
4. The pooled representation is passed through a decoder
5. Two output heads predict:
   - x-axis logits
   - y-axis logits
6. These logits are combined into a `50 x 50` heatmap

Because the model uses set pooling, the order of events does not matter.

## Input Features

Each event is converted into these 12 input features:

1. normalized scatter x
2. normalized scatter y
3. normalized absorb x
4. normalized absorb y
5. normalized delta x
6. normalized delta y
7. normalized path length
8. x direction component
9. y direction component
10. z direction component
11. normalized scattering angle
12. electron energy

The true source position is not used as a model input. It is only used to create the target heatmap during training.

## Why 12 Input Features Instead of 8 Raw Inputs

I did not use only a small set of raw detector values directly. Instead, I used 12 engineered input features for each event.

The reason is that the baseline model should not have to learn all geometric relationships from scratch. Some important information can be made explicit before the data reaches the network.

### Why not use only raw inputs

If only raw inputs are used, the model must first learn basic geometric relationships such as:

- the difference between scatter and absorb positions
- the direction of travel between interaction points
- the total path length
- the normalized angular relationship of the event

These are physically meaningful quantities, and giving them directly to the model makes the baseline more informative and easier to train.

### Why 12 features make sense

The 12 input features include:

- scatter position information
- absorb position information
- relative displacement
- path length
- direction in 3D
- scattering angle
- electron energy

This gives the model both:
- where the event happened
- how the event moved
- what physical behavior the event represents

### Benefit of using engineered features

Using 12 engineered inputs instead of a smaller raw input set helps because:

- the model receives richer geometric information
- the model does not need to discover simple coordinate transforms by itself
- training becomes easier for a baseline architecture
- the representation stays compact and interpretable
- physically meaningful event structure is preserved

### Why this is suitable for a baseline model

This repository is designed around a baseline DeepSets model.

For a baseline, it is reasonable to use hand-crafted features that capture obvious physics and geometry.

A more complex model might learn these transformations automatically, but for a baseline approach, using engineered features is a practical design choice.

### Summary

I chose 12 input features instead of only 8 raw inputs because the extra derived features make the event geometry explicit.

This improves the baseline model by giving it:

- position information
- relative motion information
- directional information
- physics-based information

So the model can focus more on learning source localization, instead of spending capacity learning basic feature transformations.

## Installation

Requirements:

- Python 3.12+
- PyTorch
- NumPy
- Pandas
- Matplotlib

Install with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data Generation

The generator writes dataset files to:

```text
<repo-root>/data/idealData_100/
```

In this codebase, that path is configured in `generate_events/gen_sim_events.py` and matches the training path used by `baseline_deepsets/config.py`.

From the repository root, generate the full synthetic dataset with:

```bash
cd /path/to/deepset_model
python -m generate_events.gen_sim_events
```

This creates one CSV file per source position on the `50 x 50` source grid.

Expected dataset path:

```text
/path/to/deepset_model/data/idealData_100/
```

For the full `50 x 50` source grid, this produces `2500` CSV files.

You can verify the file count with:

```bash
find data/idealData_100 -maxdepth 1 -type f -name '*.csv' | wc -l
```

Expected output:

```text
2500
```

## Training

Training is the step where the baseline DeepSets model learns to map a set of events to a `50 x 50` source-location heatmap.

This is needed because the generated CSV files contain raw event information, but the model weights must be learned from many examples before the model can predict source locations reliably.

From the repository root, run:

```bash
cd /path/to/deepset_model
python -m baseline_deepsets.train
```

What training does:

1. Loads all CSV files from `data/idealData_100/`
2. Splits them into train, validation, and test sets
3. Converts each event into the 12 engineered input features
4. Trains the baseline DeepSets model on the training split
5. Monitors validation performance after every epoch
6. Saves the trained checkpoint and training artifacts

Training outputs are saved to:

```text
data/models/baseline_deepsets/
data/results/baseline_deepsets/
logs/baseline_deepsets/
```

Saved artifacts include:

- trained checkpoint
- epoch-wise metrics CSV
- training/validation loss curve
- detailed log file

## Evaluation

Evaluation is the step where the trained model is tested on held-out data that was not used for training.

This is important because training metrics alone do not tell you whether the model generalizes well. Evaluation shows how accurately the model predicts on unseen samples.

From the repository root, run:

```bash
cd /path/to/deepset_model
python -m baseline_deepsets.evaluate
```

What evaluation does:

1. Loads the saved checkpoint from `data/models/baseline_deepsets/`
2. Uses the held-out test split
3. Predicts a heatmap for each test sample
4. Converts predicted heatmaps into `(x, y)` source coordinates
5. Compares predictions against the true source positions
6. Saves metrics, prediction CSV output, and preview heatmaps

Evaluation outputs are saved under:

```text
data/results/baseline_deepsets/
```

Saved artifacts include:

- predictions CSV
- preview predicted-vs-target heatmap images
- printed test summary metrics

## Current Results

Using the full dataset of `2500` generated CSV files, the current baseline model produced:

### Best validation result

- Epoch: `37`
- Validation loss: `0.074012`
- Validation mean distance: `0.532`
- Validation accuracy: `0.533`

### Test result

- Test samples: `375`
- Test accuracy: `0.5253`
- Test mean distance: `0.5291`
- Test median distance: `0.0000`
- Test p95 distance: `1.4142`
- Test within 1 pixel accuracy: `0.9093`
- Test within 2 pixels accuracy: `0.9813`
- Inference time: `1.6364 ms/sample`
- Parameter count: `546532`

These numbers indicate that the model is learning and is a strong baseline for this task.

## Output Files

Examples of generated files:

- `data/models/baseline_deepsets/baseline_model.pth`
- `data/results/baseline_deepsets/baseline_train_metrics.csv`
- `data/results/baseline_deepsets/baseline_loss_curve.png`
- `data/results/baseline_deepsets/predictions.csv`
- `data/results/baseline_deepsets/result_0.png`

## Notes

- This repository contains only the baseline DeepSets model
- There is no attention model in this repository
- The model predicts a heatmap, not a direct `(x, y)` coordinate
- Generated data, logs, cache files, and virtual environments should not be committed
