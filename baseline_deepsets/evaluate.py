import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_deepsets.config import (
    BATCH_SIZE,
    DATA_DIR,
    DEVICE,
    HEATMAP_SIZE,
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
from baseline_deepsets.metrics import benchmark_inference_ms, count_parameters, summarize_prediction_records
from baseline_deepsets.model import DeepSetsHeatmap
from baseline_deepsets.report_plots import (
    save_distance_histogram,
    save_error_mean_std_plot,
    save_xy_residual_histograms,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_rows_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the baseline DeepSets heatmap model.")
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print predictions for all test samples instead of only a preview.",
    )
    parser.add_argument(
        "--max-preview",
        type=int,
        default=8,
        help="Number of prediction examples to print and save when --print-all is not set.",
    )
    parser.add_argument(
        "--skip-report-plots",
        action="store_true",
        help="Skip generating summary report plots from the saved metrics and prediction CSV files.",
    )
    return parser.parse_args()


def save_preview_image(output_dir, index, pred_map, target_map, px, py, tx, ty):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(target_map, origin="lower")
    plt.title(f"True ({tx},{ty})")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(pred_map, origin="lower")
    plt.title(f"Pred ({px},{py})")
    plt.colorbar()

    plt.tight_layout()
    save_path = output_dir / f"true_vs_predicted_heatmap_{index}.png"
    plt.savefig(save_path)
    plt.close()


def print_summary(summary):
    print(f"\nTest summary for {MODEL_NAME}:")
    print(f"Samples: {summary['num_test_samples']}")
    print(f"Test accuracy: {summary['test_accuracy']:.4f}")
    print(f"Test mean distance: {summary['test_mean_distance']:.4f}")
    print(f"Test median distance: {summary['test_median_distance']:.4f}")
    print(f"Test p95 distance: {summary['test_p95_distance']:.4f}")
    print(f"Test within 1px accuracy: {summary['test_within_1px_accuracy']:.4f}")
    print(f"Test within 2px accuracy: {summary['test_within_2px_accuracy']:.4f}")
    print(f"Test x MAE: {summary['test_x_mae']:.4f}")
    print(f"Test y MAE: {summary['test_y_mae']:.4f}")
    print(f"Test x bias: {summary['test_x_bias']:.4f}")
    print(f"Test y bias: {summary['test_y_bias']:.4f}")
    print(f"Mean peak confidence: {summary['test_mean_peak_confidence']:.4f}")
    print(f"Inference ms/sample: {summary['inference_ms_per_sample']:.4f}")
    print(f"Parameter count: {summary['parameter_count']}")


def test_dataset_sample(test_loader):
    events, _ = next(iter(test_loader))
    return events[:1]


def generate_report_plots(results_dir: Path):
    prediction_rows = list(prediction_rows_from_csv(results_dir / "predictions.csv"))
    save_distance_histogram(prediction_rows, results_dir)
    save_xy_residual_histograms(prediction_rows, results_dir)
    save_error_mean_std_plot(prediction_rows, results_dir)


def prediction_rows_from_csv(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        yield from reader


def main():
    args = parse_args()
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = MODEL_DIR / f"{MODEL_NAME}_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _, _, test_files = split_files(DATA_DIR, PATTERN, SEED)
    test_dataset = EventHeatmapDataset(
        DATA_DIR,
        PATTERN,
        files=test_files,
        num_events=NUM_EVENTS,
        heatmap_size=HEATMAP_SIZE,
        sigma=SIGMA,
        seed=SEED,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeepSetsHeatmap(heatmap_size=HEATMAP_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    print(f"\nEvaluating model: {MODEL_NAME}")
    print(f"Checkpoint: {checkpoint_path}")

    prediction_rows = []
    saved_examples = 0
    max_saved_examples = len(test_dataset) if args.print_all else max(args.max_preview, 0)
    inference_ms_per_sample = benchmark_inference_ms(model, test_dataset_sample(test_loader), timed_steps=10)
    parameter_count = count_parameters(model)

    with torch.no_grad():
        sample_offset = 0
        for events, target in test_loader:
            events = events.to(DEVICE)
            logits = model(events)
            pred = torch.softmax(logits.view(logits.size(0), -1), dim=1)
            pred = pred.view(-1, HEATMAP_SIZE[0], HEATMAP_SIZE[1]).cpu().numpy()
            target = target.numpy()

            for i in range(pred.shape[0]):
                px, py = heatmap_to_xy(pred[i])
                tx, ty = heatmap_to_xy(target[i])
                distance = float(np.sqrt((px - tx) ** 2 + (py - ty) ** 2))
                peak_confidence = float(pred[i].max())
                prediction_rows.append(
                    {
                        "model": MODEL_NAME,
                        "sample_index": sample_offset + i,
                        "true_x": tx,
                        "true_y": ty,
                        "pred_x": px,
                        "pred_y": py,
                        "x_error": px - tx,
                        "y_error": py - ty,
                        "distance": distance,
                        "exact_match": int(px == tx and py == ty),
                        "within_1px": int(distance <= 1.0),
                        "within_2px": int(distance <= 2.0),
                        "peak_confidence": peak_confidence,
                    }
                )

                if saved_examples < max_saved_examples:
                    print(f"Pred: {px} {py}   True: {tx} {ty}")
                    save_preview_image(RESULTS_DIR, saved_examples, pred[i], target[i], px, py, tx, ty)
                    saved_examples += 1
            sample_offset += pred.shape[0]

    prediction_csv = RESULTS_DIR / "predictions.csv"
    write_rows_csv(prediction_csv, prediction_rows)
    summary = summarize_prediction_records(prediction_rows)
    summary["model"] = MODEL_NAME
    summary["prediction_csv"] = str(prediction_csv)
    summary["inference_ms_per_sample"] = inference_ms_per_sample
    summary["parameter_count"] = parameter_count
    print_summary(summary)
    print(f"\nSaved predictions CSV: {prediction_csv}")
    if not args.skip_report_plots:
        generate_report_plots(RESULTS_DIR)
        print(f"Saved summary plots under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
