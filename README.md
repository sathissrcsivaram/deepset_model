# DeepSet Model

Standalone baseline DeepSets model for heatmap-based source localization, packaged with the synthetic event generator used to create training data.

## What This Repo Contains

- `baseline_deepsets/`
  Baseline-only DeepSets implementation with model definition, training, evaluation, dataset loading, heatmap utilities, and metrics.
- `generate_events/`
  Synthetic event generation code for producing CSV datasets consumed by the model.

## Repository Layout

```text
.
├── baseline_deepsets/
├── generate_events/
├── README.md
├── pyproject.toml
└── .gitignore
```

## Requirements

- Python 3.12+
- `pip` or another Python package installer

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data Generation

Generate the full synthetic dataset:

```bash
python -m generate_events.gen_sim_events
```

Output is written to:

```text
data/idealData_100/
```

The generator creates one CSV per source location on a `50 x 50` grid.

## Training

Train the baseline DeepSets model:

```bash
python -m baseline_deepsets.train
```

Outputs are written to:

```text
data/models/baseline_deepsets/
data/results/baseline_deepsets/
logs/baseline_deepsets/
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python -m baseline_deepsets.evaluate
```

This saves preview images and a predictions CSV under:

```text
data/results/baseline_deepsets/
```

## Notes

- This repo contains only the baseline DeepSets model path. There is no attention-based model here.
- Generated data, trained weights, logs, and virtual environments are excluded by `.gitignore`.
