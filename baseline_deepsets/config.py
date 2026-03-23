from pathlib import Path

import torch

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent

DATA_DIR = REPO_ROOT / "data" / "idealData_100"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "baseline_deepsets"
MODEL_DIR = REPO_ROOT / "data" / "models" / "baseline_deepsets"
LOGS_DIR = REPO_ROOT / "logs" / "baseline_deepsets"

PATTERN = "events_*.csv"
MODEL_NAME = "baseline"

SEED = 42
NUM_EVENTS = 1000
HEATMAP_SIZE = (50, 50)
SIGMA = 1.5
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 50
EMBED_DIM = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

