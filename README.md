# Baseline DeepSets Repo

This repository contains a standalone baseline DeepSets heatmap model and the event-generation code used to produce training data.

## Structure

- `baseline_deepsets/`: baseline model, training, evaluation, dataset, heatmap, and metrics code
- `generate_events/`: synthetic event generation package

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Generate data

```bash
python -m generate_events.gen_sim_events
```

This writes CSV files under `data/idealData_100/`.

## Train the baseline model

```bash
python -m baseline_deepsets.train
```

## Evaluate the baseline model

```bash
python -m baseline_deepsets.evaluate
```

