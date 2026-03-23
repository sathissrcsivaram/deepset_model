# Baseline DeepSets

This folder contains a standalone baseline DeepSets heatmap model package.

Files:
- `model.py`: baseline DeepSets architecture
- `train.py`: baseline-only training entrypoint
- `evaluate.py`: baseline-only evaluation entrypoint
- `config.py`: paths and hyperparameters
- `data.py`: dataset loading and feature construction
- `heatmap.py`: Gaussian heatmap utilities
- `metrics.py`: evaluation summary helpers

Run:
- `python -m baseline_deepsets.train`
- `python -m baseline_deepsets.evaluate`
