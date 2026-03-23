import time

import numpy as np
import torch

from baseline_deepsets.config import DEVICE


def summarize_prediction_records(records):
    if not records:
        raise ValueError("No prediction records provided.")

    distances = np.array([row["distance"] for row in records], dtype=np.float32)
    pred_x = np.array([row["pred_x"] for row in records], dtype=np.float32)
    pred_y = np.array([row["pred_y"] for row in records], dtype=np.float32)
    true_x = np.array([row["true_x"] for row in records], dtype=np.float32)
    true_y = np.array([row["true_y"] for row in records], dtype=np.float32)
    exact = np.array([row["exact_match"] for row in records], dtype=np.float32)
    within_1px = np.array([row["within_1px"] for row in records], dtype=np.float32)
    within_2px = np.array([row["within_2px"] for row in records], dtype=np.float32)
    peak_confidence = np.array([row["peak_confidence"] for row in records], dtype=np.float32)

    return {
        "num_test_samples": int(len(records)),
        "test_accuracy": float(exact.mean()),
        "test_mean_distance": float(distances.mean()),
        "test_median_distance": float(np.median(distances)),
        "test_p90_distance": float(np.percentile(distances, 90)),
        "test_p95_distance": float(np.percentile(distances, 95)),
        "test_within_1px_accuracy": float(within_1px.mean()),
        "test_within_2px_accuracy": float(within_2px.mean()),
        "test_x_mae": float(np.abs(pred_x - true_x).mean()),
        "test_y_mae": float(np.abs(pred_y - true_y).mean()),
        "test_x_bias": float((pred_x - true_x).mean()),
        "test_y_bias": float((pred_y - true_y).mean()),
        "test_mean_peak_confidence": float(peak_confidence.mean()),
        "test_correct_mean_peak_confidence": float(peak_confidence[exact == 1].mean()) if np.any(exact == 1) else 0.0,
        "test_incorrect_mean_peak_confidence": float(peak_confidence[exact == 0].mean()) if np.any(exact == 0) else 0.0,
    }


def count_parameters(model):
    return int(sum(param.numel() for param in model.parameters()))


def benchmark_inference_ms(model, sample_events, warmup_steps=5, timed_steps=20):
    device = DEVICE
    model.eval()
    sample_events = sample_events.to(device)

    with torch.no_grad():
        for _ in range(warmup_steps):
            model(sample_events)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(timed_steps):
            model(sample_events)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start

    return float(elapsed_s * 1000.0 / timed_steps / sample_events.size(0))

