import argparse
import csv
import logging
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline_deepsets.config import (
    BATCH_SIZE,
    DATA_DIR,
    DEVICE,
    EPOCHS,
    EMBED_DIM,
    HEATMAP_SIZE,
    LOGS_DIR,
    LR,
    MODEL_DIR,
    MODEL_NAME,
    NUM_EVENTS,
    PATTERN,
    RESULTS_DIR,
    SEED,
    SIGMA,
)
from baseline_deepsets.data import EventHeatmapDataset, split_files
from baseline_deepsets.heatmap import heatmap_to_xy
from baseline_deepsets.model import DeepSetsHeatmap


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_grad_norm(model):
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total += grad_norm**2
    return total**0.5


def configure_logger(log_path: Path, console_level: str, file_level: str):
    logger = logging.getLogger("baseline_deepsets.train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, console_level.upper()))
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def write_metrics_csv(csv_path: Path, metrics_history):
    if not metrics_history:
        return

    fieldnames = list(metrics_history[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_history)


def plot_loss_curves(plot_path: Path, metrics_history):
    if not metrics_history:
        return

    epochs = [row["epoch"] for row in metrics_history]
    train_losses = [row["train_loss"] for row in metrics_history]
    val_losses = [row["val_loss"] for row in metrics_history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def describe_dataset(name, dataset):
    sample_events, sample_target = dataset[0]
    sample_xy = heatmap_to_xy(sample_target.numpy())
    return (
        f"{name}: samples={len(dataset)} events_shape={tuple(sample_events.shape)} "
        f"target_shape={tuple(sample_target.shape)} sample_target_xy={sample_xy} "
        f"target_mean={sample_target.float().mean().item():.6f} target_max={sample_target.max().item():.6f}"
    )


def normalize_heatmaps(heatmaps):
    flat = heatmaps.view(heatmaps.size(0), -1)
    return flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-8)


def log_epoch_health(logger, epoch, train_metrics, val_metrics):
    if train_metrics["avg_grad_norm"] < 1e-6:
        logger.warning(
            "Epoch %03d | gradient norm is near zero (%.8f). Training may be stalled.",
            epoch,
            train_metrics["avg_grad_norm"],
        )
    if train_metrics["pred_std"] < 1e-5 and val_metrics["pred_std"] < 1e-5:
        logger.warning(
            "Epoch %03d | predicted heatmaps are nearly constant (train_pred_std=%.8f val_pred_std=%.8f).",
            epoch,
            train_metrics["pred_std"],
            val_metrics["pred_std"],
        )
    if val_metrics["mean_dist"] > 20:
        logger.warning(
            "Epoch %03d | validation localization error is still high (val_dist=%.3f).",
            epoch,
            val_metrics["mean_dist"],
        )


def run_epoch(model, loader, criterion, logger, split_name, optimizer=None, batch_log_interval=0):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_count = 0
    total_grad_norm = 0.0
    grad_steps = 0
    all_pred_xy = []
    all_true_xy = []
    pred_means = []
    pred_stds = []
    pred_maxes = []
    target_means = []
    target_maxes = []

    batch_count = len(loader)

    for batch_idx, (events, target) in enumerate(loader, start=1):
        events = events.to(DEVICE)
        target = target.to(DEVICE)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(events)
            logits_flat = logits.view(logits.size(0), -1)
            target_probs = normalize_heatmaps(target)
            log_probs = torch.log_softmax(logits_flat, dim=1)
            loss = criterion(log_probs, target_probs)

            if train:
                loss.backward()
                grad_norm = compute_grad_norm(model)
                total_grad_norm += grad_norm
                grad_steps += 1
                optimizer.step()
            else:
                grad_norm = None

        total_loss += loss.item() * events.size(0)
        total_count += events.size(0)

        pred_probs = torch.softmax(logits_flat.detach(), dim=1).view(-1, *HEATMAP_SIZE)
        pred_cpu = pred_probs.cpu()
        target_cpu = target.detach().cpu()
        pred_means.append(pred_cpu.mean().item())
        pred_stds.append(pred_cpu.std().item())
        pred_maxes.append(pred_cpu.max().item())
        target_means.append(target_cpu.mean().item())
        target_maxes.append(target_cpu.max().item())

        for pred_map, target_map in zip(pred_cpu.numpy(), target_cpu.numpy()):
            all_pred_xy.append(heatmap_to_xy(pred_map))
            all_true_xy.append(heatmap_to_xy(target_map))

        if batch_log_interval and (batch_idx == 1 or batch_idx % batch_log_interval == 0 or batch_idx == batch_count):
            pred_xy = all_pred_xy[-1]
            true_xy = all_true_xy[-1]
            extra = ""
            if grad_norm is not None:
                extra = f" grad_norm={grad_norm:.6f}"
            logger.debug(
                "%s batch %03d/%03d | loss=%.6f pred_mean=%.6f pred_std=%.6f pred_max=%.6f target_xy=%s pred_xy=%s%s",
                split_name,
                batch_idx,
                batch_count,
                loss.item(),
                pred_means[-1],
                pred_stds[-1],
                pred_maxes[-1],
                true_xy,
                pred_xy,
                extra,
            )

    pred_xy_arr = np.array(all_pred_xy, dtype=np.float32)
    true_xy_arr = np.array(all_true_xy, dtype=np.float32)
    mean_dist = np.sqrt(((pred_xy_arr - true_xy_arr) ** 2).sum(axis=1)).mean()
    exact_acc = np.mean(np.all(pred_xy_arr == true_xy_arr, axis=1))

    return {
        "loss": total_loss / total_count,
        "mean_dist": float(mean_dist),
        "exact_acc": float(exact_acc),
        "avg_grad_norm": (total_grad_norm / grad_steps) if grad_steps else 0.0,
        "pred_mean": float(np.mean(pred_means)),
        "pred_std": float(np.mean(pred_stds)),
        "pred_max": float(np.mean(pred_maxes)),
        "target_mean": float(np.mean(target_means)),
        "target_max": float(np.mean(target_maxes)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the baseline DeepSets heatmap model.")
    parser.add_argument(
        "--batch-log-interval",
        type=int,
        default=0,
        help="Log batch-level metrics every N batches. Disabled when set to 0.",
    )
    parser.add_argument(
        "--console-log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for console output.",
    )
    parser.add_argument(
        "--file-log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for the persisted log file under logs/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{MODEL_NAME}_model.pth"
    csv_path = RESULTS_DIR / f"{MODEL_NAME}_train_metrics.csv"
    loss_plot_path = RESULTS_DIR / f"{MODEL_NAME}_loss_curve.png"
    log_path = LOGS_DIR / f"{MODEL_NAME}_train.log"

    logger = configure_logger(log_path, args.console_log_level, args.file_log_level)

    train_files, val_files, _ = split_files(DATA_DIR, PATTERN, SEED)

    train_dataset = EventHeatmapDataset(
        DATA_DIR,
        pattern=PATTERN,
        files=train_files,
        num_events=NUM_EVENTS,
        heatmap_size=HEATMAP_SIZE,
        sigma=SIGMA,
        seed=SEED,
    )
    val_dataset = EventHeatmapDataset(
        DATA_DIR,
        pattern=PATTERN,
        files=val_files,
        num_events=NUM_EVENTS,
        heatmap_size=HEATMAP_SIZE,
        sigma=SIGMA,
        seed=SEED,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepSetsHeatmap(embed_dim=EMBED_DIM, heatmap_size=HEATMAP_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.KLDivLoss(reduction="batchmean")

    best_val_loss = float("inf")
    best_state = None
    metrics_history = []

    logger.info(
        "Training configuration | model=%s device=%s epochs=%d batch_size=%d lr=%.6f console_log_level=%s file_log_level=%s",
        MODEL_NAME,
        DEVICE,
        EPOCHS,
        BATCH_SIZE,
        LR,
        args.console_log_level,
        args.file_log_level,
    )
    logger.info("Persisted log file: %s", log_path)
    logger.info(describe_dataset("train", train_dataset))
    logger.info(describe_dataset("val", val_dataset))

    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            logger=logger,
            split_name="train",
            optimizer=optimizer,
            batch_log_interval=args.batch_log_interval,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            logger=logger,
            split_name="val",
            batch_log_interval=args.batch_log_interval,
        )
        epoch_time_s = time.perf_counter() - epoch_start

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": epoch_time_s,
            "train_loss": train_metrics["loss"],
            "train_mean_dist": train_metrics["mean_dist"],
            "train_accuracy": train_metrics["exact_acc"],
            "train_avg_grad_norm": train_metrics["avg_grad_norm"],
            "train_pred_mean": train_metrics["pred_mean"],
            "train_pred_std": train_metrics["pred_std"],
            "train_pred_max": train_metrics["pred_max"],
            "train_target_mean": train_metrics["target_mean"],
            "train_target_max": train_metrics["target_max"],
            "val_loss": val_metrics["loss"],
            "val_mean_dist": val_metrics["mean_dist"],
            "validation_accuracy": val_metrics["exact_acc"],
            "val_pred_mean": val_metrics["pred_mean"],
            "val_pred_std": val_metrics["pred_std"],
            "val_pred_max": val_metrics["pred_max"],
            "val_target_mean": val_metrics["target_mean"],
            "val_target_max": val_metrics["target_max"],
            "best_val_loss": best_val_loss,
        }
        metrics_history.append(row)

        logger.info(
            "Epoch %03d | train_loss=%.6f train_dist=%.3f train_accuracy=%.3f grad=%.6f pred_std=%.6f | "
            "val_loss=%.6f val_dist=%.3f validation_accuracy=%.3f pred_std=%.6f | epoch_time=%.2fs",
            epoch + 1,
            train_metrics["loss"],
            train_metrics["mean_dist"],
            train_metrics["exact_acc"],
            train_metrics["avg_grad_norm"],
            train_metrics["pred_std"],
            val_metrics["loss"],
            val_metrics["mean_dist"],
            val_metrics["exact_acc"],
            val_metrics["pred_std"],
            epoch_time_s,
        )
        log_epoch_health(logger, epoch + 1, train_metrics, val_metrics)

    if best_state is not None:
        model.load_state_dict(best_state)

    write_metrics_csv(csv_path, metrics_history)
    plot_loss_curves(loss_plot_path, metrics_history)
    torch.save(model.state_dict(), model_path)
    logger.info("Saved metrics CSV: %s", csv_path)
    logger.info("Saved loss curve plot: %s", loss_plot_path)
    logger.info("Saved model: %s", model_path)


if __name__ == "__main__":
    main()
