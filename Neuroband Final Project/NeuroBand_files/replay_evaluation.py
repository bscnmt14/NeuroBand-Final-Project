"""Replay recorded EMG through the realtime inference pipeline.

Replay preserves chronological window order so that offline evaluation reflects
the behavior experienced in the GUI. The module applies model-specific filtering,
feature extraction, confidence rejection, and temporal decisions, then writes
predictions, confusion matrices, class metrics, timing information, and summary
statistics for model selection and regression testing.

"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from classifier_adapter import FeatureExtractor
from model_training import (
    EMG_COLUMNS,
    SENSOR_LOCATIONS,
    SENSOR_ORDER,
    SPECTRUM_COLUMNS,
    estimate_sample_rate_hz,
    multiclass_scores,
    read_inter_subject_data,
    write_json,
)
from model_training import positive_class_scores


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEFAULT_OUTPUT_DIR = APP_DIR / "realtime_replay_results"

warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed` should be used")


def parse_replay_args() -> argparse.Namespace:
    """Perform the parse args operation used by the replay realtime evaluator workflow."""
    parser = argparse.ArgumentParser(
        description="Replay recorded uMyo CSV files through a trained classifier as realtime windows."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--majority-windows", type=int, default=3)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max-timeline-segments", type=int, default=4)
    return parser.parse_args()


def load_model(path: Path) -> dict[str, object]:
    """Load and validate model for the current replay realtime evaluator workflow."""
    with path.open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, dict):
        raise TypeError(f"Expected a model artifact dict, got {type(model)}")
    if model.get("artifact_type") not in {
        "hierarchical_inter_subject_fused",
        "hierarchical_personal_stage_fused_v2",
    }:
        raise ValueError(f"Unsupported artifact type: {model.get('artifact_type')}")
    force_single_worker(model.get("binary_model"))
    force_single_worker(model.get("gesture_model"))
    return model


def force_single_worker(model: object) -> None:
    """Perform the force single worker operation used by the replay realtime evaluator workflow."""
    if model is None:
        return
    try:
        model.set_params(model__n_jobs=1)
        return
    except Exception:
        pass
    try:
        model.set_params(n_jobs=1)
    except Exception:
        pass


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Perform the sigmoid operation used by the replay realtime evaluator workflow."""
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))


def softmax(values: np.ndarray) -> np.ndarray:
    """Perform the softmax operation used by the replay realtime evaluator workflow."""
    values = np.asarray(values, dtype=float)
    values = values - np.nanmax(values)
    exp = np.exp(np.clip(values, -50, 50))
    total = np.sum(exp)
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(exp) / len(exp)
    return exp / total


def aligned_realtime_windows(
    data: pd.DataFrame,
    window_ms: int,
    overlap: float,
    fs: float,
) -> list[dict[str, object]]:
    """Perform the aligned realtime windows operation used by the replay realtime evaluator workflow."""
    window_samples = max(8, int(round(fs * window_ms / 1000.0)))
    step_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    records: list[dict[str, object]] = []

    segment_cols = ["source_file", "trial_index", "gesture_label"]
    for (source_file, trial_index, label), segment in data.groupby(segment_cols, sort=True):
        sensor_payloads: dict[str, list[dict[str, object]]] = {}
        segment_start = float(segment["timestamp"].min())
        segment_end = float(segment["timestamp"].max())
        for sensor_id, group in segment.groupby("unit_id", sort=False):
            if sensor_id not in SENSOR_ORDER:
                continue
            ordered = group.sort_values("timestamp").reset_index(drop=True)
            emg_flat = ordered[EMG_COLUMNS].to_numpy(dtype=float).reshape(-1)
            spectrum = ordered[["sp0", *SPECTRUM_COLUMNS]].to_numpy(dtype=float) if "sp0" in ordered.columns else np.column_stack(
                [np.zeros(len(ordered)), ordered[SPECTRUM_COLUMNS].to_numpy(dtype=float)]
            )
            windows: list[dict[str, object]] = []
            if emg_flat.size < window_samples:
                continue
            for window_index, start in enumerate(range(0, emg_flat.size - window_samples + 1, step_samples)):
                end = start + window_samples
                packet_start = start // 8
                packet_end = min(len(ordered), int(math.ceil(end / 8.0)))
                if packet_start >= packet_end:
                    continue
                windows.append(
                    {
                        "emg": emg_flat[start:end],
                        "spectrum": spectrum[packet_start:packet_end],
                        "start_time": float(ordered.loc[packet_start, "timestamp"]),
                        "end_time": float(ordered.loc[packet_end - 1, "timestamp"]),
                    }
                )
            sensor_payloads[str(sensor_id)] = windows

        if not all(sensor_id in sensor_payloads for sensor_id in SENSOR_ORDER):
            continue
        count = min(len(sensor_payloads[sensor_id]) for sensor_id in SENSOR_ORDER)
        if count == 0:
            continue
        for idx in range(count):
            sensor_windows = {sensor_id: sensor_payloads[sensor_id][idx] for sensor_id in SENSOR_ORDER}
            start_time = min(float(payload["start_time"]) for payload in sensor_windows.values())
            end_time = max(float(payload["end_time"]) for payload in sensor_windows.values())
            records.append(
                {
                    "source_file": str(source_file),
                    "trial_index": int(trial_index),
                    "true_label": str(label),
                    "window_index_in_segment": int(idx),
                    "segment_window_count": int(count),
                    "relative_position": float((idx + 0.5) / count),
                    "segment_start_time": segment_start,
                    "segment_end_time": segment_end,
                    "window_start_time": start_time,
                    "window_end_time": end_time,
                    "sensor_windows": sensor_windows,
                }
            )
    records.sort(key=lambda row: (row["source_file"], row["trial_index"], row["segment_start_time"], row["window_index_in_segment"]))
    return records


def predict_one(
    model: dict[str, object],
    extractor: FeatureExtractor,
    sensor_windows: dict[str, dict[str, np.ndarray]],
) -> tuple[str, float, dict[str, float], float]:
    """Predict one for the current replay realtime evaluator workflow."""
    x_values = extract_model_features(model, extractor, sensor_windows)
    binary_model = model["binary_model"]
    binary_threshold = float(model.get("binary_threshold", 0.5))
    gesture_model = model["gesture_model"]
    gesture_encoder = model["gesture_encoder"]
    gesture_thresholds = np.asarray(model["gesture_thresholds"], dtype=float)
    classes = list(model["classes"])

    binary_scores = positive_class_scores(binary_model, x_values, positive_label=1)
    binary_score = float(binary_scores[0])

    probabilities = {label: 0.0 for label in classes}
    if hasattr(binary_model, "predict_proba"):
        binary_proba = np.asarray(binary_model.predict_proba(x_values))[0]
        binary_classes = list(binary_model.classes_)
        rest_prob = float(binary_proba[binary_classes.index(0)])
        any_gesture_prob = float(binary_proba[binary_classes.index(1)])
    else:
        any_gesture_prob = float(sigmoid(np.asarray([binary_score]))[0])
        rest_prob = 1.0 - any_gesture_prob
    probabilities["at_rest"] = rest_prob

    if binary_score < binary_threshold:
        return "at_rest", float(max(probabilities.values())), probabilities, binary_score

    gesture_scores = multiclass_scores(gesture_model, x_values, len(gesture_encoder.classes_))[0]
    adjusted = gesture_scores - gesture_thresholds.reshape(-1)
    best_idx = int(np.argmax(adjusted))
    gesture_classes = [str(label) for label in gesture_encoder.inverse_transform(np.arange(len(gesture_encoder.classes_)))]
    raw_label = gesture_classes[best_idx]

    if hasattr(gesture_model, "predict_proba"):
        gesture_values = np.asarray(gesture_model.predict_proba(x_values))[0]
        model_classes = list(gesture_model.classes_)
        class_probs = np.zeros(len(gesture_encoder.classes_), dtype=float)
        for output_idx, class_idx in enumerate(model_classes):
            class_probs[int(class_idx)] = float(gesture_values[output_idx])
    else:
        class_probs = softmax(gesture_scores)

    for label, prob in zip(gesture_classes, class_probs):
        probabilities[label] = any_gesture_prob * float(prob)
    return raw_label, float(max(probabilities.values())), probabilities, binary_score


def extract_model_features(
    model: dict[str, object],
    extractor: FeatureExtractor,
    sensor_windows: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Extract model features for the current replay realtime evaluator workflow."""
    feature_columns = list(model["feature_columns"])
    normalization_stats = model.get("raw_emg_normalization_stats", {})
    if model.get("artifact_type") == "hierarchical_personal_stage_fused_v2":
        extractor.model_sensor_order = list(model.get("sensor_order", sensor_windows))
        return extractor.extract_personal_stage_fused_features(
            sensor_windows,
            feature_columns,
            normalization_stats,
            str(model.get("feature_set", "baseline")),
        )
    return extractor.extract_inter_subject_fused_features(sensor_windows, feature_columns, normalization_stats)


def predict_batch(
    model: dict[str, object],
    x_values: pd.DataFrame,
) -> tuple[list[str], list[float], list[dict[str, float]], list[float]]:
    """Predict batch for the current replay realtime evaluator workflow."""
    binary_model = model["binary_model"]
    binary_threshold = float(model.get("binary_threshold", 0.5))
    gesture_model = model["gesture_model"]
    gesture_encoder = model["gesture_encoder"]
    gesture_thresholds = np.asarray(model["gesture_thresholds"], dtype=float)
    classes = list(model["classes"])
    gesture_classes = [str(label) for label in gesture_encoder.inverse_transform(np.arange(len(gesture_encoder.classes_)))]

    binary_scores = positive_class_scores(binary_model, x_values, positive_label=1)
    if hasattr(binary_model, "predict_proba"):
        binary_proba = np.asarray(binary_model.predict_proba(x_values))
        binary_classes = list(binary_model.classes_)
        rest_probs = binary_proba[:, binary_classes.index(0)].astype(float)
        any_gesture_probs = binary_proba[:, binary_classes.index(1)].astype(float)
    else:
        any_gesture_probs = sigmoid(binary_scores)
        rest_probs = 1.0 - any_gesture_probs

    raw_predictions = np.array(["at_rest"] * len(x_values), dtype=object)
    probabilities: list[dict[str, float]] = []
    for rest_prob in rest_probs:
        row_probs = {label: 0.0 for label in classes}
        row_probs["at_rest"] = float(rest_prob)
        probabilities.append(row_probs)

    is_gesture = np.asarray(binary_scores, dtype=float) >= binary_threshold
    if np.any(is_gesture):
        gesture_x = x_values.loc[is_gesture].reset_index(drop=True)
        gesture_scores = multiclass_scores(gesture_model, gesture_x, len(gesture_encoder.classes_))
        adjusted = gesture_scores - gesture_thresholds.reshape(1, -1)
        best_indices = np.argmax(adjusted, axis=1)
        gesture_predictions = [gesture_classes[int(idx)] for idx in best_indices]
        raw_predictions[is_gesture] = gesture_predictions

        if hasattr(gesture_model, "predict_proba"):
            gesture_values = np.asarray(gesture_model.predict_proba(gesture_x))
            model_classes = list(gesture_model.classes_)
            class_probs = np.zeros((len(gesture_x), len(gesture_encoder.classes_)), dtype=float)
            for output_idx, class_idx in enumerate(model_classes):
                class_probs[:, int(class_idx)] = gesture_values[:, output_idx]
        else:
            class_probs = np.vstack([softmax(row) for row in gesture_scores])

        original_indices = np.flatnonzero(is_gesture)
        for local_idx, original_idx in enumerate(original_indices):
            for label, prob in zip(gesture_classes, class_probs[local_idx]):
                probabilities[int(original_idx)][label] = float(any_gesture_probs[int(original_idx)] * prob)

    confidences = [float(max(row.values())) if row else 0.0 for row in probabilities]
    return [str(label) for label in raw_predictions], confidences, probabilities, [float(x) for x in binary_scores]


def majority_vote(labels: list[str], index: int, width: int) -> str:
    """Perform the majority vote operation used by the replay realtime evaluator workflow."""
    if width <= 1:
        return labels[index]
    start = max(0, index - width + 1)
    window = labels[start : index + 1]
    counts = Counter(window)
    return max(counts.items(), key=lambda item: (item[1], window[::-1].index(item[0]) * -1))[0]


def add_phase(relative_position: float) -> str:
    """Add phase for the current replay realtime evaluator workflow."""
    if relative_position <= 0.25:
        return "early"
    if relative_position >= 0.75:
        return "late"
    return "middle"


def threshold_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    """Perform the threshold sweep operation used by the replay realtime evaluator workflow."""
    rows = []
    labels = sorted(frame["true_label"].unique())
    for threshold in np.round(np.arange(0.0, 0.96, 0.05), 2):
        displayed = np.where(frame["confidence"].to_numpy() < threshold, "Uncertain", frame["raw_prediction"].to_numpy())
        known_mask = displayed != "Uncertain"
        overall_accuracy = float(np.mean(displayed == frame["true_label"].to_numpy()))
        known_accuracy = float(np.mean(displayed[known_mask] == frame.loc[known_mask, "true_label"].to_numpy())) if np.any(known_mask) else np.nan
        unknown_rate = float(np.mean(~known_mask))
        side_flex_rate = float(np.mean(displayed == "side_flex"))
        try:
            ba = float(balanced_accuracy_score(frame["true_label"], displayed))
        except Exception:
            ba = np.nan
        rows.append(
            {
                "confidence_threshold": threshold,
                "overall_accuracy_unknown_wrong": overall_accuracy,
                "known_accuracy_excluding_uncertain": known_accuracy,
                "unknown_rate": unknown_rate,
                "side_flex_display_rate": side_flex_rate,
                "balanced_accuracy_unknown_as_class": ba,
                "label_count": len(labels),
            }
        )
    return pd.DataFrame(rows)


def run_length_summary(labels: list[str]) -> pd.DataFrame:
    """Run length summary for the current replay realtime evaluator workflow."""
    if not labels:
        return pd.DataFrame()
    runs = []
    current = labels[0]
    length = 1
    for label in labels[1:]:
        if label == current:
            length += 1
        else:
            runs.append({"label": current, "run_length_windows": length})
            current = label
            length = 1
    runs.append({"label": current, "run_length_windows": length})
    frame = pd.DataFrame(runs)
    return (
        frame.groupby("label")
        .agg(
            run_count=("run_length_windows", "count"),
            mean_run_length=("run_length_windows", "mean"),
            max_run_length=("run_length_windows", "max"),
        )
        .reset_index()
        .sort_values("max_run_length", ascending=False)
    )


def plot_outputs(predictions: pd.DataFrame, threshold_sweep_frame: pd.DataFrame, output_dir: Path, max_segments: int) -> None:
    """Perform the plot outputs operation used by the replay realtime evaluator workflow."""
    plt.figure(figsize=(9, 5))
    for label, group in predictions.groupby("true_label"):
        plt.hist(group["confidence"], bins=30, alpha=0.45, label=label)
    plt.xlabel("Confidence")
    plt.ylabel("Window count")
    plt.title("Confidence distribution by true label")
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_sweep_frame["confidence_threshold"], threshold_sweep_frame["unknown_rate"], label="Unknown rate")
    plt.plot(
        threshold_sweep_frame["confidence_threshold"],
        threshold_sweep_frame["known_accuracy_excluding_uncertain"],
        label="Known accuracy",
    )
    plt.plot(
        threshold_sweep_frame["confidence_threshold"],
        threshold_sweep_frame["side_flex_display_rate"],
        label="side_flex display rate",
    )
    plt.xlabel("Confidence threshold")
    plt.ylabel("Rate")
    plt.title("Confidence threshold tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_tradeoff.png", dpi=160)
    plt.close()

    ranked_segments = (
        predictions.assign(correct=predictions["raw_prediction"] == predictions["true_label"])
        .groupby(["source_file", "trial_index", "true_label"])
        .agg(raw_accuracy=("correct", "mean"), windows=("correct", "count"))
        .reset_index()
        .sort_values(["raw_accuracy", "windows"], ascending=[True, False])
        .head(max_segments)
    )
    label_to_idx = {
        label: idx
        for idx, label in enumerate(
            sorted(
                set(predictions["true_label"])
                | set(predictions["raw_prediction"])
                | set(predictions["displayed_prediction"])
            )
        )
    }
    for plot_idx, segment in enumerate(ranked_segments.itertuples(index=False), start=1):
        mask = (
            (predictions["source_file"] == segment.source_file)
            & (predictions["trial_index"] == segment.trial_index)
            & (predictions["true_label"] == segment.true_label)
        )
        group = predictions.loc[mask].sort_values("window_index_in_segment")
        plt.figure(figsize=(10, 4))
        plt.plot(group["window_index_in_segment"], [label_to_idx[x] for x in group["raw_prediction"]], label="raw prediction")
        plt.plot(group["window_index_in_segment"], [label_to_idx[x] for x in group["displayed_prediction"]], label="displayed")
        plt.axhline(label_to_idx[segment.true_label], color="black", linestyle="--", linewidth=1, label="true")
        plt.yticks(list(label_to_idx.values()), list(label_to_idx.keys()), fontsize=8)
        plt.xlabel("Window index in segment")
        plt.title(f"Worst segment {plot_idx}: {segment.true_label}, trial {segment.trial_index}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"timeline_worst_segment_{plot_idx}.png", dpi=160)
        plt.close()


def run_replay() -> None:
    """Run the module's command-line or graphical application entry point."""
    args = parse_replay_args()
    start_time = time.perf_counter()
    run_name = args.name or args.model_path.stem
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path)
    data = read_inter_subject_data(args.data_dir)
    fs = float(model.get("sampling_rate_hz") or estimate_sample_rate_hz(data))
    window_ms = int(model.get("window_ms", 500))
    extractor = FeatureExtractor(fs=fs)

    windows = aligned_realtime_windows(data, window_ms, args.overlap, fs)
    feature_rows = [extract_model_features(model, extractor, row["sensor_windows"]) for row in windows]
    x_values = pd.concat(feature_rows, ignore_index=True)
    raw_predictions, confidences, probability_rows, binary_scores = predict_batch(model, x_values)

    rows: list[dict[str, object]] = []
    for row, raw_prediction, confidence, probabilities, binary_score in zip(
        windows,
        raw_predictions,
        confidences,
        probability_rows,
        binary_scores,
    ):
        displayed = "Uncertain" if confidence < args.confidence_threshold else raw_prediction
        top2 = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:2]
        rows.append(
            {
                "source_file": row["source_file"],
                "trial_index": row["trial_index"],
                "true_label": row["true_label"],
                "window_index_in_segment": row["window_index_in_segment"],
                "segment_window_count": row["segment_window_count"],
                "relative_position": row["relative_position"],
                "phase": add_phase(float(row["relative_position"])),
                "window_start_time": row["window_start_time"],
                "window_end_time": row["window_end_time"],
                "raw_prediction": raw_prediction,
                "displayed_prediction": displayed,
                "confidence": confidence,
                "binary_score": binary_score,
                "top1_label": top2[0][0] if top2 else "",
                "top1_probability": top2[0][1] if top2 else 0.0,
                "top2_label": top2[1][0] if len(top2) > 1 else "",
                "top2_probability": top2[1][1] if len(top2) > 1 else 0.0,
                **{f"prob_{label}": prob for label, prob in probabilities.items()},
            }
        )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No replay windows were created.")
    stable_labels = [
        majority_vote(list(predictions["displayed_prediction"]), idx, args.majority_windows)
        for idx in range(len(predictions))
    ]
    predictions["majority_prediction"] = stable_labels
    predictions["raw_correct"] = predictions["raw_prediction"] == predictions["true_label"]
    predictions["displayed_correct"] = predictions["displayed_prediction"] == predictions["true_label"]
    predictions["majority_correct"] = predictions["majority_prediction"] == predictions["true_label"]
    predictions.to_csv(output_dir / "replay_predictions.csv", index=False)

    labels = sorted(predictions["true_label"].unique())
    phase_summary = (
        predictions.groupby(["true_label", "phase"])
        .agg(
            windows=("true_label", "count"),
            raw_accuracy=("raw_correct", "mean"),
            displayed_accuracy=("displayed_correct", "mean"),
            majority_accuracy=("majority_correct", "mean"),
            mean_confidence=("confidence", "mean"),
            uncertain_rate=("displayed_prediction", lambda values: float(np.mean(values == "Uncertain"))),
            side_flex_rate=("raw_prediction", lambda values: float(np.mean(values == "side_flex"))),
        )
        .reset_index()
    )
    phase_summary.to_csv(output_dir / "phase_summary.csv", index=False)

    label_summary = (
        predictions.groupby("true_label")
        .agg(
            windows=("true_label", "count"),
            raw_accuracy=("raw_correct", "mean"),
            displayed_accuracy=("displayed_correct", "mean"),
            majority_accuracy=("majority_correct", "mean"),
            mean_confidence=("confidence", "mean"),
            uncertain_rate=("displayed_prediction", lambda values: float(np.mean(values == "Uncertain"))),
            side_flex_rate=("raw_prediction", lambda values: float(np.mean(values == "side_flex"))),
        )
        .reset_index()
        .sort_values("raw_accuracy")
    )
    label_summary.to_csv(output_dir / "label_summary.csv", index=False)

    segment_summary = (
        predictions.groupby(["source_file", "trial_index", "true_label"])
        .agg(
            windows=("true_label", "count"),
            raw_accuracy=("raw_correct", "mean"),
            displayed_accuracy=("displayed_correct", "mean"),
            majority_accuracy=("majority_correct", "mean"),
            mean_confidence=("confidence", "mean"),
            dominant_raw_prediction=("raw_prediction", lambda values: Counter(values).most_common(1)[0][0]),
            dominant_displayed_prediction=("displayed_prediction", lambda values: Counter(values).most_common(1)[0][0]),
        )
        .reset_index()
        .sort_values("raw_accuracy")
    )
    segment_summary.to_csv(output_dir / "segment_summary.csv", index=False)

    threshold_frame = threshold_sweep(predictions)
    threshold_frame.to_csv(output_dir / "confidence_threshold_sweep.csv", index=False)
    run_lengths = run_length_summary(list(predictions["displayed_prediction"]))
    run_lengths.to_csv(output_dir / "displayed_run_lengths.csv", index=False)

    cm = pd.DataFrame(
        confusion_matrix(predictions["true_label"], predictions["raw_prediction"], labels=labels),
        index=labels,
        columns=labels,
    )
    cm.to_csv(output_dir / "raw_confusion_matrix.csv")

    plot_outputs(predictions, threshold_frame, output_dir, args.max_timeline_segments)

    summary = {
        "name": run_name,
        "data_dir": str(args.data_dir),
        "model_path": str(args.model_path),
        "model_type": model.get("model_type"),
        "model_params": model.get("model_params"),
        "window_ms": window_ms,
        "sampling_rate_hz": fs,
        "confidence_threshold": args.confidence_threshold,
        "majority_windows": args.majority_windows,
        "window_count": int(len(predictions)),
        "raw_accuracy": float(predictions["raw_correct"].mean()),
        "displayed_accuracy_unknown_wrong": float(predictions["displayed_correct"].mean()),
        "majority_accuracy_unknown_wrong": float(predictions["majority_correct"].mean()),
        "raw_balanced_accuracy": float(balanced_accuracy_score(predictions["true_label"], predictions["raw_prediction"])),
        "displayed_balanced_accuracy_unknown_as_class": float(
            balanced_accuracy_score(predictions["true_label"], predictions["displayed_prediction"])
        ),
        "unknown_rate": float(np.mean(predictions["displayed_prediction"] == "Uncertain")),
        "side_flex_raw_rate": float(np.mean(predictions["raw_prediction"] == "side_flex")),
        "side_flex_displayed_rate": float(np.mean(predictions["displayed_prediction"] == "side_flex")),
        "mean_confidence": float(predictions["confidence"].mean()),
        "lowest_accuracy_labels": label_summary.head(4).to_dict(orient="records"),
        "worst_segments": segment_summary.head(8).to_dict(orient="records"),
        "elapsed_seconds": time.perf_counter() - start_time,
        "sensor_order": SENSOR_ORDER,
        "sensor_locations": SENSOR_LOCATIONS,
        "outputs": [
            "replay_predictions.csv",
            "label_summary.csv",
            "phase_summary.csv",
            "segment_summary.csv",
            "confidence_threshold_sweep.csv",
            "displayed_run_lengths.csv",
            "raw_confusion_matrix.csv",
            "confidence_histogram.png",
            "threshold_tradeoff.png",
        ],
    }
    write_json(output_dir / "replay_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Saved replay results to: {ascii(output_dir)}")


# Temporal decision-strategy comparison

def parse_decision_args() -> argparse.Namespace:
    """Perform the parse args operation used by the compare replay decision strategies workflow."""
    parser = argparse.ArgumentParser(description="Compare realtime decision mechanisms on replay predictions.")
    parser.add_argument("--replay-dir", type=Path, required=True, help="Folder containing replay_predictions.csv.")
    parser.add_argument("--step-ms", type=float, default=250.0, help="Realtime step between decisions, default 500 ms window with 50%% overlap.")
    return parser.parse_args()


def decision_majority_vote(labels: list[str], width: int) -> list[str]:
    """Perform the majority vote operation used by the compare replay decision strategies workflow."""
    if width <= 1:
        return labels
    output: list[str] = []
    for index in range(len(labels)):
        window = labels[max(0, index - width + 1) : index + 1]
        counts = Counter(window)
        output.append(max(counts.items(), key=lambda item: (item[1], -window[::-1].index(item[0])))[0])
    return output


def consecutive_switch(labels: list[str], required: int) -> list[str]:
    """Perform the consecutive switch operation used by the compare replay decision strategies workflow."""
    current = "Uncertain"
    candidate = ""
    candidate_count = 0
    output: list[str] = []
    for label in labels:
        if label == "Uncertain":
            current = "Uncertain"
            candidate = ""
            candidate_count = 0
            output.append(current)
            continue
        if current in {"", "Uncertain"} or label == current:
            current = label
            candidate = ""
            candidate_count = 0
            output.append(current)
            continue
        if label == candidate:
            candidate_count += 1
        else:
            candidate = label
            candidate_count = 1
        if candidate_count >= required:
            current = candidate
            candidate = ""
            candidate_count = 0
        output.append(current)
    return output


def hysteresis(raw: list[str], confidence: np.ndarray, enter: float, stay: float) -> list[str]:
    """Perform the hysteresis operation used by the compare replay decision strategies workflow."""
    current = "Uncertain"
    output: list[str] = []
    for label, conf in zip(raw, confidence):
        if current == "Uncertain":
            current = label if conf >= enter else "Uncertain"
        elif label == current:
            current = current if conf >= stay else "Uncertain"
        elif conf >= enter:
            current = label
        elif conf < stay:
            current = "Uncertain"
        output.append(current)
    return output


def threshold(raw: np.ndarray, confidence: np.ndarray, value: float) -> list[str]:
    """Perform the threshold operation used by the compare replay decision strategies workflow."""
    return ["Uncertain" if conf < value else label for label, conf in zip(raw, confidence)]


def first_correct_lag_seconds(frame: pd.DataFrame, prediction_col: str, step_ms: float) -> float:
    """Perform the first correct lag seconds operation used by the compare replay decision strategies workflow."""
    lags: list[float] = []
    group_cols = ["source_file", "trial_index", "true_label"]
    for (_, _, true_label), group in frame.groupby(group_cols, sort=False):
        ordered = group.sort_values("window_index_in_segment")
        if true_label == "at_rest":
            continue
        matches = np.flatnonzero(ordered[prediction_col].to_numpy() == true_label)
        if len(matches):
            lags.append(float(matches[0]) * step_ms / 1000.0)
    return float(np.mean(lags)) if lags else float("nan")


def metrics(frame: pd.DataFrame, name: str, labels: list[str], step_ms: float) -> dict[str, object]:
    """Perform the metrics operation used by the compare replay decision strategies workflow."""
    true = frame["true_label"].to_numpy()
    pred = frame[name].to_numpy()
    known_mask = pred != "Uncertain"
    switches = int(np.sum(pred[1:] != pred[:-1])) if len(pred) > 1 else 0
    try:
        ba = float(balanced_accuracy_score(true, pred))
    except Exception:
        ba = float("nan")
    return {
        "strategy": name,
        "accuracy_unknown_wrong": float(np.mean(pred == true)),
        "balanced_accuracy_unknown_as_class": ba,
        "known_accuracy_excluding_uncertain": float(np.mean(pred[known_mask] == true[known_mask])) if np.any(known_mask) else float("nan"),
        "unknown_rate": float(np.mean(pred == "Uncertain")),
        "switch_count": switches,
        "switches_per_minute": float(switches / max(1e-9, len(pred) * step_ms / 60000.0)),
        "mean_first_correct_lag_s": first_correct_lag_seconds(frame, name, step_ms),
        "false_active_during_rest_rate": float(np.mean((true == "at_rest") & ~np.isin(pred, ["at_rest", "Uncertain"]))),
        "label_count": len(labels),
    }


def compare_decision_strategies() -> None:
    """Run the module's command-line or graphical application entry point."""
    args = parse_decision_args()
    predictions_path = args.replay_dir / "replay_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    frame = pd.read_csv(predictions_path)
    raw = frame["raw_prediction"].astype(str).to_numpy()
    confidence = frame["confidence"].to_numpy(dtype=float)
    labels = sorted(frame["true_label"].astype(str).unique())

    frame["raw_no_gate"] = raw
    for threshold_value in [0.45, 0.55]:
        base_name = f"threshold_{threshold_value:.2f}"
        frame[base_name] = threshold(raw, confidence, threshold_value)
        frame[f"{base_name}_majority_3"] = decision_majority_vote(list(frame[base_name]), 3)
        frame[f"{base_name}_consecutive_2"] = consecutive_switch(list(frame[base_name]), 2)
    for enter, stay in [(0.55, 0.40), (0.65, 0.45)]:
        name = f"hysteresis_enter_{enter:.2f}_stay_{stay:.2f}"
        frame[name] = hysteresis(list(raw), confidence, enter, stay)

    strategy_cols = [col for col in frame.columns if col.startswith(("raw_no_gate", "threshold_", "hysteresis_"))]
    rows = [metrics(frame, col, labels, args.step_ms) for col in strategy_cols]
    results = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy_unknown_as_class", "accuracy_unknown_wrong", "switches_per_minute"],
        ascending=[False, False, True],
    )

    output_dir = args.replay_dir / "decision_strategy_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "predictions_with_strategies.csv", index=False)
    results.to_csv(output_dir / "decision_strategy_results.csv", index=False)
    summary = {
        "replay_dir": str(args.replay_dir),
        "step_ms": args.step_ms,
        "best_strategy": results.iloc[0].to_dict() if not results.empty else None,
        "top_10": results.head(10).to_dict(orient="records"),
    }
    (output_dir / "decision_strategy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def main() -> None:
    """Dispatch replay generation or decision-strategy comparison."""
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "replay"
    if command == "replay":
        run_replay()
    elif command == "compare":
        compare_decision_strategies()
    else:
        raise SystemExit(f"Unknown replay command: {command}")


if __name__ == "__main__":
    main()
