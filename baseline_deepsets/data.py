import glob
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from baseline_deepsets.heatmap import make_gaussian_heatmap


REQUIRED_COLUMNS = [
    "scat_x",
    "scat_y",
    "scat_z",
    "absorb_x",
    "absorb_y",
    "absorb_z",
    "theta",
    "e_energy",
    "source_x",
    "source_y",
]


def derive_event_features(df: pd.DataFrame) -> np.ndarray:
    dx = df["absorb_x"].values - df["scat_x"].values
    dy = df["absorb_y"].values - df["scat_y"].values
    dz = df["absorb_z"].values - df["scat_z"].values

    norm = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-8

    theta_rad = np.deg2rad(df["theta"].values)
    scatter_x = ((df["scat_x"].values - 25.0) / 25.0).astype(np.float32)
    scatter_y = ((df["scat_y"].values - 25.0) / 25.0).astype(np.float32)
    absorb_x = ((df["absorb_x"].values - 22.5) / 22.5).astype(np.float32)
    absorb_y = ((df["absorb_y"].values - 22.5) / 22.5).astype(np.float32)
    delta_x = (dx / 45.0).astype(np.float32)
    delta_y = (dy / 45.0).astype(np.float32)
    path_len = (norm / 100.0).astype(np.float32)
    theta_norm = (theta_rad / np.pi).astype(np.float32)

    return np.stack(
        [
            scatter_x,
            scatter_y,
            absorb_x,
            absorb_y,
            delta_x,
            delta_y,
            path_len,
            (dx / norm).astype(np.float32),
            (dy / norm).astype(np.float32),
            (dz / norm).astype(np.float32),
            theta_norm,
            df["e_energy"].values.astype(np.float32),
        ],
        axis=1,
    )


class EventHeatmapDataset(Dataset):
    def __init__(
        self,
        data_dir,
        pattern="events_*.csv",
        files=None,
        num_events=1000,
        heatmap_size=(50, 50),
        sigma=1.5,
        seed=42,
    ):
        self.num_events = num_events
        self.heatmap_h, self.heatmap_w = heatmap_size
        self.sigma = sigma
        self.seed = seed

        if files is not None:
            self.files = files
        else:
            self.files = sorted(glob.glob(str(data_dir / pattern)))

        if not self.files:
            raise FileNotFoundError("No CSV files found.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_csv(self.files[idx])[REQUIRED_COLUMNS].dropna().copy()

        if len(df) >= self.num_events:
            df = df.sample(n=self.num_events, random_state=self.seed)
        else:
            df = df.sample(n=self.num_events, replace=True, random_state=self.seed)

        events = derive_event_features(df)

        source_x = float(df["source_x"].iloc[0])
        source_y = float(df["source_y"].iloc[0])
        target = make_gaussian_heatmap(
            source_x,
            source_y,
            height=self.heatmap_h,
            width=self.heatmap_w,
            sigma=self.sigma,
        )

        return torch.tensor(events), torch.tensor(target)


def split_files(data_dir, pattern="events_*.csv", seed=42):
    files = sorted(glob.glob(str(data_dir / pattern)))
    random.Random(seed).shuffle(files)

    n = len(files)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    return train_files, val_files, test_files

