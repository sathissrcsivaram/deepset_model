import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baseline_deepsets.config import MODEL_NAME, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Generate result plots for the baseline DeepSets model.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=RESULTS_DIR / f"{MODEL_NAME}_train_metrics.csv",
        help="Path to the training metrics CSV.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=RESULTS_DIR / "predictions.csv",
        help="Path to the evaluation predictions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where the plots will be saved.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float_array(rows, key):
    return np.array([float(row[key]) for row in rows], dtype=np.float32)


def to_int_array(rows, key):
    return np.array([int(float(row[key])) for row in rows], dtype=np.int32)


def save_distance_histogram(prediction_rows, output_dir: Path):
    distances = to_float_array(prediction_rows, "distance")
    plt.figure(figsize=(8, 5))
    bins = np.arange(-0.125, max(2.5, float(distances.max()) + 0.25), 0.25)
    plt.hist(distances, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.85)
    plt.axvline(float(distances.mean()), color="#E45756", linestyle="--", linewidth=2, label=f"Mean = {distances.mean():.3f}")
    plt.xlabel("Distance Error (pixels)")
    plt.ylabel("Number of Test Samples")
    plt.title("Baseline Distance Error Histogram")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "distance_error_histogram.png", dpi=150)
    plt.close()


def save_xy_residual_histograms(prediction_rows, output_dir: Path):
    x_errors = to_float_array(prediction_rows, "x_error")
    y_errors = to_float_array(prediction_rows, "y_error")
    bins = np.arange(-2.5, 3.0, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    axes[0].hist(x_errors, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.85)
    axes[0].axvline(float(x_errors.mean()), color="#E45756", linestyle="--", linewidth=2, label=f"Mean = {x_errors.mean():.3f}")
    axes[0].set_title("X Residuals")
    axes[0].set_xlabel("pred_x - true_x")
    axes[0].set_ylabel("Number of Test Samples")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].hist(y_errors, bins=bins, color="#72B7B2", edgecolor="black", alpha=0.85)
    axes[1].axvline(float(y_errors.mean()), color="#E45756", linestyle="--", linewidth=2, label=f"Mean = {y_errors.mean():.3f}")
    axes[1].set_title("Y Residuals")
    axes[1].set_xlabel("pred_y - true_y")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("Baseline X/Y Residual Histograms")
    fig.tight_layout()
    fig.savefig(output_dir / "xy_residual_histograms.png", dpi=150)
    plt.close(fig)


def save_spatial_error_heatmap(prediction_rows, output_dir: Path):
    true_x = to_int_array(prediction_rows, "true_x")
    true_y = to_int_array(prediction_rows, "true_y")
    distances = to_float_array(prediction_rows, "distance")

    grid_size = max(int(true_x.max()), int(true_y.max())) + 1
    error_sum = np.zeros((grid_size, grid_size), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)

    for x, y, distance in zip(true_x, true_y, distances):
        error_sum[y, x] += distance
        counts[y, x] += 1.0

    mean_error = np.divide(
        error_sum,
        counts,
        out=np.full_like(error_sum, np.nan),
        where=counts > 0,
    )

    plt.figure(figsize=(7, 6))
    image = plt.imshow(mean_error, origin="lower", cmap="magma")
    plt.colorbar(image, label="Mean Distance Error (pixels)")
    plt.xlabel("True X")
    plt.ylabel("True Y")
    plt.title("Spatial Heatmap of Mean Localization Error")
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_error_heatmap.png", dpi=150)
    plt.close()


def save_prediction_mean_std_plot(metrics_rows, output_dir: Path):
    epochs = to_int_array(metrics_rows, "epoch")
    train_pred_mean = to_float_array(metrics_rows, "train_pred_mean")
    val_pred_mean = to_float_array(metrics_rows, "val_pred_mean")
    train_pred_std = to_float_array(metrics_rows, "train_pred_std")
    val_pred_std = to_float_array(metrics_rows, "val_pred_std")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, train_pred_mean, label="Train Pred Mean", color="#4C78A8", linewidth=2)
    axes[0].plot(epochs, val_pred_mean, label="Val Pred Mean", color="#F58518", linewidth=2)
    axes[0].set_title("Prediction Mean by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Predicted Heatmap Value")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_pred_std, label="Train Pred Std", color="#54A24B", linewidth=2)
    axes[1].plot(epochs, val_pred_std, label="Val Pred Std", color="#E45756", linewidth=2)
    axes[1].set_title("Prediction Std by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Std of Predicted Heatmap Values")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("Prediction Mean and Std Across Training")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_mean_std_by_epoch.png", dpi=150)
    plt.close(fig)


def save_accuracy_distance_plot(metrics_rows, output_dir: Path):
    epochs = to_int_array(metrics_rows, "epoch")
    train_accuracy = to_float_array(metrics_rows, "train_accuracy")
    val_accuracy = to_float_array(metrics_rows, "validation_accuracy")
    train_mean_dist = to_float_array(metrics_rows, "train_mean_dist")
    val_mean_dist = to_float_array(metrics_rows, "val_mean_dist")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, train_accuracy, label="Train Accuracy", color="#4C78A8", linewidth=2)
    axes[0].plot(epochs, val_accuracy, label="Val Accuracy", color="#F58518", linewidth=2)
    axes[0].set_title("Exact-Match Accuracy by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_mean_dist, label="Train Mean Distance", color="#54A24B", linewidth=2)
    axes[1].plot(epochs, val_mean_dist, label="Val Mean Distance", color="#E45756", linewidth=2)
    axes[1].set_title("Mean Distance Error by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Distance Error (pixels)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("Accuracy and Distance Across Training")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_distance_by_epoch.png", dpi=150)
    plt.close(fig)


def save_error_mean_std_plot(prediction_rows, output_dir: Path):
    x_errors = to_float_array(prediction_rows, "x_error")
    y_errors = to_float_array(prediction_rows, "y_error")
    distances = to_float_array(prediction_rows, "distance")

    labels = ["X Error", "Y Error", "Distance Error"]
    means = np.array([x_errors.mean(), y_errors.mean(), distances.mean()], dtype=np.float32)
    stds = np.array([x_errors.std(), y_errors.std(), distances.std()], dtype=np.float32)
    x_positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(x_positions, means, color=["#4C78A8", "#72B7B2", "#F58518"], edgecolor="black", alpha=0.85)
    axes[0].set_xticks(x_positions, labels)
    axes[0].set_ylabel("Mean Error (pixels)")
    axes[0].set_title("Mean X/Y/Distance Error")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x_positions, stds, color=["#54A24B", "#E45756", "#B279A2"], edgecolor="black", alpha=0.85)
    axes[1].set_xticks(x_positions, labels)
    axes[1].set_ylabel("Error Std (pixels)")
    axes[1].set_title("Std of X/Y/Distance Error")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Prediction Error Mean and Standard Deviation")
    fig.tight_layout()
    fig.savefig(output_dir / "error_mean_std.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = read_csv_rows(args.metrics_csv)
    prediction_rows = read_csv_rows(args.predictions_csv)

    save_distance_histogram(prediction_rows, args.output_dir)
    save_xy_residual_histograms(prediction_rows, args.output_dir)
    save_spatial_error_heatmap(prediction_rows, args.output_dir)
    save_prediction_mean_std_plot(metrics_rows, args.output_dir)
    save_accuracy_distance_plot(metrics_rows, args.output_dir)
    save_error_mean_std_plot(prediction_rows, args.output_dir)

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
