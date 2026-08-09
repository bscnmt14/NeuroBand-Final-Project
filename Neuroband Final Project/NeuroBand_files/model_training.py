"""Signal processing and hierarchical model training workflows.

This module defines sensor identities, EMG filtering, classical time-domain feature
extraction, synchronized window construction, model fitting, probability handling,
ROC-based threshold selection, and evaluation helpers. The functions are reused by
the current personal-training and replay pipelines even though the earliest command-
line experiment also supported inter-subject datasets.

"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, tf2sos
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data" / "inter_subject_data"
DEFAULT_OUTPUT_DIR = APP_DIR / "inter_subject_results"

SENSOR_ORDER = ["B0DAC7E9", "ED7A78C8", "37ED348F"]
SENSOR_LOCATIONS = {
    "B0DAC7E9": "Ventral forearm",
    "ED7A78C8": "Dorsal forearm",
    "37ED348F": "Inner forearm side",
}
EMG_COLUMNS = [f"emg_{idx}" for idx in range(8)]
SPECTRUM_COLUMNS = ["sp1", "sp2", "sp3"]
DISABLED_GESTURES = {"side_flex"}
TRAIN_TRIALS = {1, 2, 3, 4, 5, 6}
TEST_TRIALS = {7, 8, 9, 10}
LABEL_MAP = {
    "open_palm": "open_hand",
    "rest": "at_rest",
}


@dataclass
class HierarchicalResult:
    """Represent the HierarchicalResult component and keep its related state and behavior together."""
    window_ms: int
    model_type: str
    model_params: str
    train_balanced_accuracy: float
    validation_balanced_accuracy: float
    test_balanced_accuracy: float
    train_windows: int
    validation_windows: int
    test_windows: int
    binary_threshold: float
    output_dir: str


def parse_args() -> argparse.Namespace:
    """Perform the parse args operation used by the train inter subject hierarchical models workflow."""
    parser = argparse.ArgumentParser(
        description="Train hierarchical gesture classifiers on the new inter-subject uMyo data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-ms", type=int, nargs="+", default=[200, 300, 400, 500])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--trim-edge-ms", type=float, default=100.0)
    parser.add_argument("--sampling-rate-hz", type=float, default=None)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-best-model", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_inter_subject_data(data_dir: Path) -> pd.DataFrame:
    """Read and parse inter subject data for the current train inter subject hierarchical models workflow."""
    frames = []
    for path in sorted(data_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    data = pd.concat(frames, ignore_index=True)
    required = ["gesture_label", "trial_index", "unit_id", "timestamp", *EMG_COLUMNS, *SPECTRUM_COLUMNS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data[data["gesture_label"].astype(str).str.lower() != "beginning"].copy()
    data["gesture_label"] = (
        data["gesture_label"].astype(str).str.strip().str.lower().replace(LABEL_MAP)
    )
    data = data[~data["gesture_label"].isin(DISABLED_GESTURES)].copy()
    data["unit_id"] = data["unit_id"].astype(str).str.upper().str.strip()
    data = data[data["unit_id"].isin(SENSOR_ORDER)].copy()
    data["trial_index"] = pd.to_numeric(data["trial_index"], errors="coerce").astype("Int64")
    data["timestamp"] = pd.to_numeric(data["timestamp"], errors="coerce")
    for column in [*EMG_COLUMNS, *SPECTRUM_COLUMNS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["gesture_label", "trial_index", "unit_id", "timestamp", *EMG_COLUMNS, *SPECTRUM_COLUMNS])
    data["trial_index"] = data["trial_index"].astype(int)
    return data


def estimate_sample_rate_hz(data: pd.DataFrame) -> float:
    """Perform the estimate sample rate hz operation used by the train inter subject hierarchical models workflow."""
    estimates = []
    for _, group in data.groupby(["source_file", "trial_index", "gesture_label", "unit_id"], sort=False):
        ordered = group.sort_values("timestamp")
        diffs = np.diff(ordered["timestamp"].to_numpy(dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            estimates.append(8.0 / float(np.median(diffs)))
    if not estimates:
        return 1100.0
    return float(np.median(estimates))


def make_bandpass(fs: float) -> np.ndarray:
    """Create and configure bandpass for the current train inter subject hierarchical models workflow."""
    nyquist = 0.5 * fs
    low_hz = 35.0
    high_hz = min(500.0, nyquist - 1.0)
    if high_hz <= low_hz:
        raise ValueError(f"Sampling rate {fs:.1f} Hz is too low for a 35-500 Hz bandpass.")
    bandpass_sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    if 50.0 >= nyquist:
        return bandpass_sos
    notch_b, notch_a = iirnotch(50.0 / nyquist, 30.0)
    notch_sos = tf2sos(notch_b, notch_a)
    return np.vstack([bandpass_sos, notch_sos])


def filter_signal(values: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Filter signal for the current train inter subject hierarchical models workflow."""
    values = np.asarray(values, dtype=float)
    if values.size < 24:
        return sosfilt(sos, values)
    try:
        return sosfiltfilt(sos, values)
    except ValueError:
        return sosfilt(sos, values)


def apply_bpf_and_standard_normalization(
    data: pd.DataFrame,
    fs: float,
    train_trials: Iterable[int],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Apply bpf and standard normalization for the current train inter subject hierarchical models workflow."""
    processed = data.copy()
    processed[EMG_COLUMNS] = processed[EMG_COLUMNS].astype(float)
    sos = make_bandpass(fs)

    for key, group in processed.groupby(["source_file", "trial_index", "gesture_label", "unit_id"], sort=False):
        flat = group.sort_values("timestamp")[EMG_COLUMNS].to_numpy(dtype=float).reshape(-1)
        filtered = filter_signal(flat, sos).reshape(-1, 8)
        processed.loc[group.sort_values("timestamp").index, EMG_COLUMNS] = filtered

    stats: dict[str, dict[str, float]] = {}
    train_trial_set = set(train_trials)
    for sensor_id in SENSOR_ORDER:
        sensor_mask = processed["unit_id"] == sensor_id
        train_mask = sensor_mask & processed["trial_index"].isin(train_trial_set)
        values = processed.loc[train_mask, EMG_COLUMNS].to_numpy(dtype=float).reshape(-1)
        mean = float(np.mean(values))
        std = float(np.std(values))
        if not math.isfinite(std) or std == 0:
            std = 1.0
        processed.loc[sensor_mask, EMG_COLUMNS] = (processed.loc[sensor_mask, EMG_COLUMNS] - mean) / std
        stats[sensor_id] = {"mean_after_bpf": mean, "std_after_bpf": std}

    return processed, stats


def zero_crossings(values: np.ndarray) -> float:
    """Perform the zero crossings operation used by the train inter subject hierarchical models workflow."""
    signs = np.sign(values)
    signs[signs == 0] = 1
    return float(np.sum(signs[:-1] * signs[1:] < 0))


def slope_sign_changes(values: np.ndarray) -> float:
    """Perform the slope sign changes operation used by the train inter subject hierarchical models workflow."""
    if len(values) < 3:
        return 0.0
    diffs = np.diff(values)
    return float(np.sum(diffs[:-1] * diffs[1:] < 0))


def emg_features(values: np.ndarray) -> dict[str, float]:
    """Perform the emg features operation used by the train inter subject hierarchical models workflow."""
    values = np.nan_to_num(np.asarray(values, dtype=float).reshape(-1))
    return {
        "emg_rms": float(np.sqrt(np.mean(np.square(values)))),
        "emg_mav": float(np.mean(np.abs(values))),
        "emg_zc": zero_crossings(values),
        "emg_ssc": slope_sign_changes(values),
        "emg_wl": float(np.sum(np.abs(np.diff(values)))),
        "emg_var": float(np.var(values)),
    }


def extract_device_windows(
    data: pd.DataFrame,
    window_ms: int,
    overlap: float,
    fs: float,
    trim_edge_ms: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Extract device windows for the current train inter subject hierarchical models workflow."""
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    metadata: list[dict[str, object]] = []
    window_samples = max(8, int(round(fs * window_ms / 1000.0)))
    step_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    trim_samples = max(0, int(round(fs * trim_edge_ms / 1000.0)))

    group_cols = ["source_file", "trial_index", "gesture_label", "unit_id"]
    for (source_file, trial_index, label, sensor_id), group in data.groupby(group_cols, sort=True):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        emg_matrix = ordered[EMG_COLUMNS].to_numpy(dtype=float)
        emg_flat = emg_matrix.reshape(-1)
        spectrum = ordered[SPECTRUM_COLUMNS].to_numpy(dtype=float)
        if emg_flat.size < window_samples:
            continue
        for window_index, start in enumerate(range(0, emg_flat.size - window_samples + 1, step_samples)):
            end = start + window_samples
            if trim_samples and (start < trim_samples or end > emg_flat.size - trim_samples):
                continue
            packet_start = start // 8
            packet_end = min(len(ordered), int(math.ceil(end / 8.0)))
            if packet_start >= packet_end:
                continue
            row = emg_features(emg_flat[start:end])
            spectrum_window = spectrum[packet_start:packet_end]
            row.update(
                {
                    "spectrum_1_mean": float(np.mean(spectrum_window[:, 0])),
                    "spectrum_2_mean": float(np.mean(spectrum_window[:, 1])),
                    "spectrum_3_mean": float(np.mean(spectrum_window[:, 2])),
                }
            )
            rows.append(row)
            labels.append(str(label))
            metadata.append(
                {
                    "source_file": source_file,
                    "trial_index": int(trial_index),
                    "gesture_label": str(label),
                    "unit_id": str(sensor_id),
                    "window_index_in_segment": int(window_index),
                    "window_start_sample": int(start),
                    "window_end_sample": int(end),
                    "window_start_time": float(ordered.loc[packet_start, "timestamp"]),
                    "window_end_time": float(ordered.loc[packet_end - 1, "timestamp"]),
                }
            )

    if not rows:
        raise ValueError(f"No windows were created for {window_ms} ms.")
    return pd.DataFrame(rows), pd.Series(labels, name="gesture_label"), pd.DataFrame(metadata)


def fuse_device_windows(
    x_all: pd.DataFrame,
    y_all: pd.Series,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Perform the fuse device windows operation used by the train inter subject hierarchical models workflow."""
    key_cols = ["source_file", "trial_index", "gesture_label", "window_index_in_segment"]
    frame = pd.concat([metadata[key_cols + ["unit_id"]].reset_index(drop=True), x_all.reset_index(drop=True)], axis=1)
    fused_rows: list[dict[str, float]] = []
    fused_labels: list[str] = []
    fused_meta: list[dict[str, object]] = []

    for key_values, group in frame.groupby(key_cols, sort=True):
        available = set(group["unit_id"].astype(str))
        if not set(SENSOR_ORDER).issubset(available):
            continue
        row: dict[str, float] = {}
        for sensor_id in SENSOR_ORDER:
            sensor_row = group[group["unit_id"] == sensor_id].iloc[0]
            for feature_name in x_all.columns:
                row[f"{sensor_id}_{feature_name}"] = float(sensor_row[feature_name])
        key_dict = dict(zip(key_cols, key_values))
        fused_rows.append(row)
        fused_labels.append(str(key_dict["gesture_label"]))
        fused_meta.append({**key_dict, "unit_id": "fused", "sensor_order": ",".join(SENSOR_ORDER)})

    if not fused_rows:
        raise ValueError("No fused windows were created. Check sensor alignment and trial labels.")
    return pd.DataFrame(fused_rows), pd.Series(fused_labels, name="gesture_label"), pd.DataFrame(fused_meta)


def split_train_validation(y_train_labels: pd.Series, validation_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform the split train validation operation used by the train inter subject hierarchical models workflow."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
    train_idx, validation_idx = next(splitter.split(np.zeros(len(y_train_labels)), y_train_labels))
    return train_idx, validation_idx


def model_specs(random_state: int) -> list[tuple[str, str, object]]:
    """Perform the model specs operation used by the train inter subject hierarchical models workflow."""
    return [
        (
            "knn",
            "n_neighbors=5;weights=distance",
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5, weights="distance"))]),
        ),
        (
            "svm_rbf",
            "C=10;gamma=scale",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced", probability=True)),
                ]
            ),
        ),
        (
            "random_forest",
            "n_estimators=300;max_depth=10;min_samples_leaf=1;class_weight=balanced",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=10,
                            min_samples_leaf=1,
                            class_weight="balanced",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees",
            "n_estimators=300;max_depth=10;min_samples_leaf=1;class_weight=balanced",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=300,
                            max_depth=10,
                            min_samples_leaf=1,
                            class_weight="balanced",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logistic_regression",
            "class_weight=balanced",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state),
                    ),
                ]
            ),
        ),
        (
            "gradient_boosting",
            "n_estimators=120;learning_rate=0.05;max_depth=3",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=120,
                            learning_rate=0.05,
                            max_depth=3,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    ]


def scores_for_positive_class(model: object, x_values: pd.DataFrame, positive_label: int = 1) -> np.ndarray:
    """Perform the scores for positive class operation used by the train inter subject hierarchical models workflow."""
    proba = model.predict_proba(x_values)
    classes = list(model.classes_)
    return proba[:, classes.index(positive_label)]


def best_binary_threshold(y_true: np.ndarray, positive_scores: np.ndarray) -> tuple[float, float]:
    """Perform the best binary threshold operation used by the train inter subject hierarchical models workflow."""
    fpr, tpr, thresholds = roc_curve(y_true, positive_scores)
    scores = (tpr + (1.0 - fpr)) / 2.0
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def multiclass_scores(model: object, x_values: pd.DataFrame, class_count: int) -> np.ndarray:
    """Perform the multiclass scores operation used by the train inter subject hierarchical models workflow."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x_values))
        scores = np.zeros((len(x_values), class_count), dtype=float)
        for output_idx, class_idx in enumerate(model.classes_):
            scores[:, int(class_idx)] = proba[:, output_idx]
        return scores
    scores = np.asarray(model.decision_function(x_values))
    if class_count == 2 and scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores


def best_multiclass_thresholds(y_true: np.ndarray, scores: np.ndarray, class_count: int) -> tuple[np.ndarray, dict[str, float]]:
    """Perform the best multiclass thresholds operation used by the train inter subject hierarchical models workflow."""
    thresholds = np.zeros(class_count, dtype=float)
    qualities: dict[str, float] = {}
    for class_idx in range(class_count):
        binary_true = (y_true == class_idx).astype(int)
        fpr, tpr, roc_thresholds = roc_curve(binary_true, scores[:, class_idx])
        balanced_scores = (tpr + (1.0 - fpr)) / 2.0
        best_idx = int(np.argmax(balanced_scores))
        thresholds[class_idx] = float(roc_thresholds[best_idx])
        qualities[str(class_idx)] = float(balanced_scores[best_idx])
    return thresholds, qualities


def predict_multiclass_with_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Predict multiclass with thresholds for the current train inter subject hierarchical models workflow."""
    return np.argmax(scores - thresholds.reshape(1, -1), axis=1)


def fit_hierarchical(
    model_template: object,
    x_train: pd.DataFrame,
    y_train_labels: pd.Series,
    x_validation: pd.DataFrame,
    y_validation_labels: pd.Series,
) -> dict[str, object]:
    """Fit hierarchical for the current train inter subject hierarchical models workflow."""
    y_train_binary = (y_train_labels.to_numpy() != "at_rest").astype(int)
    y_validation_binary = (y_validation_labels.to_numpy() != "at_rest").astype(int)

    binary_model = clone(model_template)
    binary_model.fit(x_train, y_train_binary)
    binary_scores = scores_for_positive_class(binary_model, x_validation, positive_label=1)
    binary_threshold, binary_roc_balanced = best_binary_threshold(y_validation_binary, binary_scores)

    train_active = y_train_labels != "at_rest"
    validation_active = y_validation_labels != "at_rest"
    gesture_encoder = LabelEncoder()
    y_train_gesture = gesture_encoder.fit_transform(y_train_labels.loc[train_active])
    y_validation_gesture = gesture_encoder.transform(y_validation_labels.loc[validation_active])

    gesture_model = clone(model_template)
    gesture_model.fit(x_train.loc[train_active].reset_index(drop=True), y_train_gesture)
    validation_gesture_scores = multiclass_scores(
        gesture_model,
        x_validation.loc[validation_active].reset_index(drop=True),
        len(gesture_encoder.classes_),
    )
    gesture_thresholds, gesture_threshold_quality = best_multiclass_thresholds(
        y_validation_gesture,
        validation_gesture_scores,
        len(gesture_encoder.classes_),
    )
    return {
        "binary_model": binary_model,
        "binary_threshold": binary_threshold,
        "binary_roc_balanced": binary_roc_balanced,
        "gesture_model": gesture_model,
        "gesture_encoder": gesture_encoder,
        "gesture_thresholds": gesture_thresholds,
        "gesture_threshold_quality": gesture_threshold_quality,
    }


def predict_hierarchical(bundle: dict[str, object], x_values: pd.DataFrame) -> np.ndarray:
    """Predict hierarchical for the current train inter subject hierarchical models workflow."""
    binary_model = bundle["binary_model"]
    binary_threshold = float(bundle["binary_threshold"])
    gesture_model = bundle["gesture_model"]
    gesture_encoder: LabelEncoder = bundle["gesture_encoder"]
    gesture_thresholds = np.asarray(bundle["gesture_thresholds"], dtype=float)

    binary_scores = scores_for_positive_class(binary_model, x_values, positive_label=1)
    is_gesture = binary_scores >= binary_threshold
    predictions = np.array(["at_rest"] * len(x_values), dtype=object)
    if np.any(is_gesture):
        gesture_scores = multiclass_scores(gesture_model, x_values.loc[is_gesture].reset_index(drop=True), len(gesture_encoder.classes_))
        gesture_pred = predict_multiclass_with_thresholds(gesture_scores, gesture_thresholds)
        predictions[is_gesture] = gesture_encoder.inverse_transform(gesture_pred)
    return predictions


def evaluate_bundle(bundle: dict[str, object], x_values: pd.DataFrame, y_true: pd.Series) -> tuple[np.ndarray, float]:
    """Evaluate bundle for the current train inter subject hierarchical models workflow."""
    y_pred = predict_hierarchical(bundle, x_values)
    return y_pred, float(balanced_accuracy_score(y_true, y_pred))


def write_json(path: Path, payload: object) -> None:
    """Save json for the current train inter subject hierarchical models workflow."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run() -> list[HierarchicalResult]:
    """Perform the run operation used by the train inter subject hierarchical models workflow."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_inter_subject_data(args.data_dir)
    fs = float(args.sampling_rate_hz or estimate_sample_rate_hz(raw))
    data, normalization_stats = apply_bpf_and_standard_normalization(raw, fs, TRAIN_TRIALS)

    dataset_summary = {
        "data_dir": str(args.data_dir),
        "source_files": sorted(data["source_file"].unique()),
        "rows_after_dropping_beginning": int(len(data)),
        "labels": data["gesture_label"].value_counts().sort_index().to_dict(),
        "trial_indices": sorted(map(int, data["trial_index"].unique())),
        "sensor_order": SENSOR_ORDER,
        "sensor_locations": SENSOR_LOCATIONS,
        "estimated_sampling_rate_hz": fs,
        "preprocessing": {
            "bpf": "Butterworth bandpass order 4, 35-500 Hz",
            "raw_emg_standard_normalization": "Per sensor, mean/std from train trials 1-6 after BPF",
            "feature_standard_normalization": "StandardScaler inside every model pipeline",
            "spectrum_0": "Dropped; using sp1-sp3 means",
            "trim_edge_ms": args.trim_edge_ms,
        },
        "normalization_stats": normalization_stats,
        "split": {"train_trials": sorted(TRAIN_TRIALS), "test_trials": sorted(TEST_TRIALS)},
    }
    write_json(args.output_dir / "dataset_summary.json", dataset_summary)

    results: list[HierarchicalResult] = []
    all_rows: list[dict[str, object]] = []
    best_result: tuple[HierarchicalResult, dict[str, object], list[str]] | None = None

    for window_ms in args.window_ms:
        x_device, y_device, device_meta = extract_device_windows(
            data=data,
            window_ms=window_ms,
            overlap=args.overlap,
            fs=fs,
            trim_edge_ms=args.trim_edge_ms,
        )
        x_all, y_all, metadata = fuse_device_windows(x_device, y_device, device_meta)
        window_dir = args.output_dir / f"window_{window_ms}ms"
        window_dir.mkdir(parents=True, exist_ok=True)
        x_all.to_csv(window_dir / "window_features.csv", index=False)
        metadata.assign(gesture_label=y_all).to_csv(window_dir / "window_metadata.csv", index=False)

        train_mask = metadata["trial_index"].isin(TRAIN_TRIALS).to_numpy()
        test_mask = metadata["trial_index"].isin(TEST_TRIALS).to_numpy()
        x_train_trials = x_all.loc[train_mask].reset_index(drop=True)
        y_train_trials = y_all.loc[train_mask].reset_index(drop=True)
        x_test = x_all.loc[test_mask].reset_index(drop=True)
        y_test = y_all.loc[test_mask].reset_index(drop=True)
        if "at_rest" not in set(y_train_trials):
            raise ValueError("Hierarchical training requires at_rest labels in train trials.")

        train_idx, validation_idx = split_train_validation(y_train_trials, args.validation_size, args.random_state)
        x_train = x_train_trials.iloc[train_idx].reset_index(drop=True)
        y_train = y_train_trials.iloc[train_idx].reset_index(drop=True)
        x_validation = x_train_trials.iloc[validation_idx].reset_index(drop=True)
        y_validation = y_train_trials.iloc[validation_idx].reset_index(drop=True)

        for model_type, model_params, model in model_specs(args.random_state):
            bundle = fit_hierarchical(model, x_train, y_train, x_validation, y_validation)
            _, train_ba = evaluate_bundle(bundle, x_train, y_train)
            y_validation_pred, validation_ba = evaluate_bundle(bundle, x_validation, y_validation)
            y_test_pred, test_ba = evaluate_bundle(bundle, x_test, y_test)

            result = HierarchicalResult(
                window_ms=window_ms,
                model_type=model_type,
                model_params=model_params,
                train_balanced_accuracy=train_ba,
                validation_balanced_accuracy=validation_ba,
                test_balanced_accuracy=test_ba,
                train_windows=len(x_train),
                validation_windows=len(x_validation),
                test_windows=len(x_test),
                binary_threshold=float(bundle["binary_threshold"]),
                output_dir=str(window_dir),
            )
            results.append(result)
            all_rows.append(result.__dict__)

            pred_frame = metadata.loc[test_mask].reset_index(drop=True).copy()
            pred_frame["true_label"] = y_test
            pred_frame["predicted_label"] = y_test_pred
            pred_frame.to_csv(window_dir / f"test_predictions_{model_type}.csv", index=False)

            labels = sorted(y_all.unique())
            cm = confusion_matrix(y_test, y_test_pred, labels=labels)
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(window_dir / f"confusion_matrix_{model_type}.csv")

            write_json(
                window_dir / f"summary_{model_type}.json",
                {
                    **result.__dict__,
                    "labels": labels,
                    "binary_roc_balanced_on_validation": float(bundle["binary_roc_balanced"]),
                    "gesture_threshold_quality": bundle["gesture_threshold_quality"],
                },
            )

            if best_result is None or test_ba > best_result[0].test_balanced_accuracy:
                best_result = (result, bundle, list(x_all.columns))

    summary = pd.DataFrame(all_rows).sort_values("test_balanced_accuracy", ascending=False)
    summary.to_csv(args.output_dir / "hierarchical_model_comparison.csv", index=False)
    write_json(args.output_dir / "best_result.json", summary.iloc[0].to_dict())

    if args.save_best_model and best_result is not None:
        result, bundle, feature_columns = best_result
        artifact = {
            "artifact_type": "hierarchical_inter_subject_fused",
            "model_type": result.model_type,
            "model_params": result.model_params,
            "window_ms": result.window_ms,
            "sampling_rate_hz": fs,
            "sensor_order": SENSOR_ORDER,
            "sensor_locations": SENSOR_LOCATIONS,
            "feature_columns": feature_columns,
            "classes": sorted(data["gesture_label"].unique()),
            "binary_model": bundle["binary_model"],
            "binary_threshold": bundle["binary_threshold"],
            "gesture_model": bundle["gesture_model"],
            "gesture_encoder": bundle["gesture_encoder"],
            "gesture_thresholds": bundle["gesture_thresholds"],
            "raw_emg_normalization_stats": normalization_stats,
            "preprocessing": dataset_summary["preprocessing"],
            "note": "Trained/evaluated with train trials 1-6, ROC threshold calibration on validation split, test trials 7-10.",
        }
        with (args.output_dir / "best_hierarchical_model.pkl").open("wb") as file:
            pickle.dump(artifact, file)

    print(summary.to_string(index=False))
    print(f"\nSaved results to: {args.output_dir}")
    return results


# Rolling-trial grid workflow

def parse_rolling_args() -> argparse.Namespace:
    """Parse arguments for the historical rolling-trial grid experiment."""
    parser = argparse.ArgumentParser(description="Rolling split grid search for staged uMyo recordings.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "Data" / "inter_subject_data")
    parser.add_argument("--output-dir", type=Path, default=APP_DIR / "hierarchical_rolling_grid_results")
    parser.add_argument("--window-ms", type=int, nargs="+", default=[200, 300, 400, 500])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--trim-edge-ms", type=float, default=100.0)
    parser.add_argument("--sampling-rate-hz", type=float, default=None)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-best-model", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def rolling_splits(trials: list[int]) -> list[dict[str, object]]:
    """Perform the rolling splits operation used by the hierarchical model workflow."""
    specs = [(2, 1), (3, 2), (4, 3)]
    splits: list[dict[str, object]] = []
    trial_set = set(trials)
    for train_count, test_count in specs:
        for start in trials:
            train_trials = list(range(start, start + train_count))
            test_trials = list(range(start + train_count, start + train_count + test_count))
            if set(train_trials).issubset(trial_set) and set(test_trials).issubset(trial_set):
                splits.append(
                    {
                        "split_name": f"{train_count}_{test_count}_start_{start}",
                        "ratio": f"{train_count}/{test_count}",
                        "train_trials": train_trials,
                        "test_trials": test_trials,
                    }
                )
    return splits


def apply_bpf_only(data: pd.DataFrame, fs: float) -> pd.DataFrame:
    """Apply bpf only for the current hierarchical model workflow."""
    processed = data.copy()
    processed[EMG_COLUMNS] = processed[EMG_COLUMNS].astype(float)
    sos = make_bandpass(fs)
    group_cols = ["source_file", "trial_index", "gesture_label", "unit_id"]
    for _, group in processed.groupby(group_cols, sort=False):
        ordered = group.sort_values("timestamp")
        flat = ordered[EMG_COLUMNS].to_numpy(dtype=float).reshape(-1)
        filtered = filter_signal(flat, sos).reshape(-1, 8)
        processed.loc[ordered.index, EMG_COLUMNS] = filtered
    return processed


def normalize_for_split(
    filtered_data: pd.DataFrame,
    train_trials: list[int],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Perform the normalize for split operation used by the hierarchical model workflow."""
    normalized = filtered_data.copy()
    normalized[EMG_COLUMNS] = normalized[EMG_COLUMNS].astype(float)
    stats: dict[str, dict[str, float]] = {}
    train_trial_set = set(train_trials)
    for sensor_id in SENSOR_ORDER:
        sensor_mask = normalized["unit_id"] == sensor_id
        train_mask = sensor_mask & normalized["trial_index"].isin(train_trial_set)
        values = normalized.loc[train_mask, EMG_COLUMNS].to_numpy(dtype=float).reshape(-1)
        mean = float(np.mean(values))
        std = float(np.std(values))
        if not math.isfinite(std) or std == 0:
            std = 1.0
        normalized.loc[sensor_mask, EMG_COLUMNS] = (normalized.loc[sensor_mask, EMG_COLUMNS] - mean) / std
        stats[sensor_id] = {"mean_after_bpf": mean, "std_after_bpf": std}
    return normalized, stats


def rolling_model_specs(random_state: int) -> list[tuple[str, str, object]]:
    """Perform the model specs operation used by the hierarchical model workflow."""
    specs: list[tuple[str, str, object]] = []

    for n_neighbors in [3, 5, 7]:
        for weights in ["uniform", "distance"]:
            specs.append(
                (
                    "knn",
                    f"n_neighbors={n_neighbors};weights={weights}",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("model", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
                        ]
                    ),
                )
            )

    for c_value in [1.0, 10.0, 100.0]:
        specs.append(
            (
                "svm_rbf",
                f"C={c_value:g};gamma=scale",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            SVC(
                                C=c_value,
                                kernel="rbf",
                                gamma="scale",
                                class_weight="balanced",
                                probability=False,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            )
        )

    for max_depth in [5, 10, None]:
        for min_samples_leaf in [1, 3]:
            specs.append(
                (
                    "random_forest",
                    f"n_estimators=250;max_depth={max_depth};min_samples_leaf={min_samples_leaf}",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                RandomForestClassifier(
                                    n_estimators=250,
                                    max_depth=max_depth,
                                    min_samples_leaf=min_samples_leaf,
                                    class_weight="balanced",
                                    random_state=random_state,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                )
            )
            specs.append(
                (
                    "extra_trees",
                    f"n_estimators=250;max_depth={max_depth};min_samples_leaf={min_samples_leaf}",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                ExtraTreesClassifier(
                                    n_estimators=250,
                                    max_depth=max_depth,
                                    min_samples_leaf=min_samples_leaf,
                                    class_weight="balanced",
                                    random_state=random_state,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                )
            )

    for c_value in [0.1, 1.0, 10.0]:
        specs.append(
            (
                "logistic_regression",
                f"C={c_value:g};class_weight=balanced",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=c_value,
                                class_weight="balanced",
                                max_iter=3000,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            )
        )

    for learning_rate in [0.03, 0.1]:
        for max_depth in [2, 3]:
            specs.append(
                (
                    "gradient_boosting",
                    f"n_estimators=120;learning_rate={learning_rate:g};max_depth={max_depth}",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                GradientBoostingClassifier(
                                    n_estimators=120,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    random_state=random_state,
                                ),
                            ),
                        ]
                    ),
                )
            )

    return specs


def positive_class_scores(model: object, x_values: pd.DataFrame, positive_label: int = 1) -> np.ndarray:
    """Perform the positive class scores operation used by the hierarchical model workflow."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x_values))
        classes = list(model.classes_)
        return proba[:, classes.index(positive_label)]
    scores = np.asarray(model.decision_function(x_values))
    if scores.ndim == 1:
        classes = list(model.classes_)
        positive_sign = 1.0 if classes[-1] == positive_label else -1.0
        return positive_sign * scores
    classes = list(model.classes_)
    return scores[:, classes.index(positive_label)]


def fit_rolling_hierarchical(
    model_template: object,
    x_train: pd.DataFrame,
    y_train_labels: pd.Series,
    x_validation: pd.DataFrame,
    y_validation_labels: pd.Series,
) -> dict[str, object]:
    """Fit hierarchical for the current hierarchical model workflow."""
    from sklearn.preprocessing import LabelEncoder

    y_train_binary = (y_train_labels.to_numpy() != "at_rest").astype(int)
    y_validation_binary = (y_validation_labels.to_numpy() != "at_rest").astype(int)

    binary_model = clone(model_template)
    binary_model.fit(x_train, y_train_binary)
    binary_scores = positive_class_scores(binary_model, x_validation, positive_label=1)
    binary_threshold, binary_roc_balanced = best_binary_threshold(y_validation_binary, binary_scores)

    train_active = y_train_labels != "at_rest"
    validation_active = y_validation_labels != "at_rest"
    gesture_encoder = LabelEncoder()
    y_train_gesture = gesture_encoder.fit_transform(y_train_labels.loc[train_active])
    y_validation_gesture = gesture_encoder.transform(y_validation_labels.loc[validation_active])

    gesture_model = clone(model_template)
    gesture_model.fit(x_train.loc[train_active].reset_index(drop=True), y_train_gesture)
    validation_gesture_scores = multiclass_scores(
        gesture_model,
        x_validation.loc[validation_active].reset_index(drop=True),
        len(gesture_encoder.classes_),
    )
    gesture_thresholds, gesture_threshold_quality = best_multiclass_thresholds(
        y_validation_gesture,
        validation_gesture_scores,
        len(gesture_encoder.classes_),
    )
    return {
        "binary_model": binary_model,
        "binary_threshold": binary_threshold,
        "binary_roc_balanced": binary_roc_balanced,
        "gesture_model": gesture_model,
        "gesture_encoder": gesture_encoder,
        "gesture_thresholds": gesture_thresholds,
        "gesture_threshold_quality": gesture_threshold_quality,
    }


def predict_hierarchical_local(bundle: dict[str, object], x_values: pd.DataFrame) -> np.ndarray:
    """Predict hierarchical local for the current hierarchical model workflow."""
    binary_model = bundle["binary_model"]
    binary_threshold = float(bundle["binary_threshold"])
    gesture_model = bundle["gesture_model"]
    gesture_encoder = bundle["gesture_encoder"]
    gesture_thresholds = np.asarray(bundle["gesture_thresholds"], dtype=float)

    binary_scores = positive_class_scores(binary_model, x_values, positive_label=1)
    is_gesture = binary_scores >= binary_threshold
    predictions = np.array(["at_rest"] * len(x_values), dtype=object)
    if np.any(is_gesture):
        gesture_scores = multiclass_scores(
            gesture_model,
            x_values.loc[is_gesture].reset_index(drop=True),
            len(gesture_encoder.classes_),
        )
        gesture_pred = predict_multiclass_with_thresholds(gesture_scores, gesture_thresholds)
        predictions[is_gesture] = gesture_encoder.inverse_transform(gesture_pred)
    return predictions


def evaluate_bundle_local(bundle: dict[str, object], x_values: pd.DataFrame, y_true: pd.Series) -> tuple[np.ndarray, float]:
    """Evaluate bundle local for the current hierarchical model workflow."""
    y_pred = predict_hierarchical_local(bundle, x_values)
    return y_pred, float(balanced_accuracy_score(y_true, y_pred))


def run_rolling_grid() -> None:
    """Perform the run_rolling_grid operation used by the hierarchical model workflow."""
    args = parse_rolling_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_inter_subject_data(args.data_dir)
    fs = float(args.sampling_rate_hz or estimate_sample_rate_hz(raw))
    filtered = apply_bpf_only(raw, fs)
    trials = [trial for trial in sorted(map(int, filtered["trial_index"].unique())) if trial > 0]
    splits = rolling_splits(trials)
    models = rolling_model_specs(args.random_state)

    dataset_summary = {
        "data_dir": str(args.data_dir),
        "source_files": sorted(filtered["source_file"].unique()),
        "rows_after_dropping_beginning": int(len(filtered)),
        "labels": filtered["gesture_label"].value_counts().sort_index().to_dict(),
        "trial_indices": trials,
        "rolling_splits": splits,
        "model_grid_count": len(models),
        "window_ms": args.window_ms,
        "estimated_sampling_rate_hz": fs,
        "sensor_order": SENSOR_ORDER,
        "sensor_locations": SENSOR_LOCATIONS,
        "preprocessing": {
            "bpf": "Butterworth bandpass order 4, 35-500 Hz plus 50 Hz notch filter",
            "emg_samples": "emg_0..emg_7 are flattened as sequential time samples",
            "raw_emg_standard_normalization": "Per split, per sensor, mean/std from train trials only after BPF",
            "feature_standard_normalization": "StandardScaler inside every model pipeline",
            "spectrum_0": "Dropped; using sp1-sp3 means",
            "trim_edge_ms": args.trim_edge_ms,
            "label_map": LABEL_MAP,
        },
    }
    write_json(args.output_dir / "dataset_summary.json", dataset_summary)

    result_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for split in splits:
        train_trials = list(split["train_trials"])
        test_trials = list(split["test_trials"])
        normalized, normalization_stats = normalize_for_split(filtered, train_trials)
        for window_ms in args.window_ms:
            print(f"Split {split['split_name']} | window {window_ms}ms", flush=True)
            x_device, y_device, device_meta = extract_device_windows(
                data=normalized,
                window_ms=window_ms,
                overlap=args.overlap,
                fs=fs,
                trim_edge_ms=args.trim_edge_ms,
            )
            x_all, y_all, metadata = fuse_device_windows(x_device, y_device, device_meta)
            train_mask = metadata["trial_index"].isin(train_trials).to_numpy()
            test_mask = metadata["trial_index"].isin(test_trials).to_numpy()
            x_train_trials = x_all.loc[train_mask].reset_index(drop=True)
            y_train_trials = y_all.loc[train_mask].reset_index(drop=True)
            x_test = x_all.loc[test_mask].reset_index(drop=True)
            y_test = y_all.loc[test_mask].reset_index(drop=True)

            train_idx, validation_idx = split_train_validation(
                y_train_trials,
                args.validation_size,
                args.random_state,
            )
            x_train = x_train_trials.iloc[train_idx].reset_index(drop=True)
            y_train = y_train_trials.iloc[train_idx].reset_index(drop=True)
            x_validation = x_train_trials.iloc[validation_idx].reset_index(drop=True)
            y_validation = y_train_trials.iloc[validation_idx].reset_index(drop=True)

            for model_type, model_params, model in models:
                try:
                    bundle = fit_rolling_hierarchical(model, x_train, y_train, x_validation, y_validation)
                    _, train_ba = evaluate_bundle_local(bundle, x_train, y_train)
                    _, validation_ba = evaluate_bundle_local(bundle, x_validation, y_validation)
                    y_test_pred, test_ba = evaluate_bundle_local(bundle, x_test, y_test)
                    row = {
                        "ratio": split["ratio"],
                        "split_name": split["split_name"],
                        "train_trials": ",".join(map(str, train_trials)),
                        "test_trials": ",".join(map(str, test_trials)),
                        "window_ms": window_ms,
                        "model_type": model_type,
                        "model_params": model_params,
                        "train_balanced_accuracy": train_ba,
                        "validation_balanced_accuracy": validation_ba,
                        "test_balanced_accuracy": test_ba,
                        "train_windows": int(len(x_train)),
                        "validation_windows": int(len(x_validation)),
                        "test_windows": int(len(x_test)),
                    }
                    result_rows.append(row)
                    if best is None or test_ba > float(best["row"]["test_balanced_accuracy"]):
                        best = {
                            "row": row,
                            "bundle": bundle,
                            "feature_columns": list(x_all.columns),
                            "normalization_stats": normalization_stats,
                            "labels": sorted(y_all.unique()),
                            "confusion_matrix": confusion_matrix(y_test, y_test_pred, labels=sorted(y_all.unique())).tolist(),
                        }
                except Exception as exc:
                    result_rows.append(
                        {
                            "ratio": split["ratio"],
                            "split_name": split["split_name"],
                            "train_trials": ",".join(map(str, train_trials)),
                            "test_trials": ",".join(map(str, test_trials)),
                            "window_ms": window_ms,
                            "model_type": model_type,
                            "model_params": model_params,
                            "train_balanced_accuracy": np.nan,
                            "validation_balanced_accuracy": np.nan,
                            "test_balanced_accuracy": np.nan,
                            "train_windows": int(len(x_train)),
                            "validation_windows": int(len(x_validation)),
                            "test_windows": int(len(x_test)),
                            "error": str(exc),
                        }
                    )

    results = pd.DataFrame(result_rows)
    results.to_csv(args.output_dir / "rolling_grid_all_results.csv", index=False)

    valid = results.dropna(subset=["test_balanced_accuracy"]).copy()
    summary = (
        valid.groupby(["ratio", "window_ms", "model_type", "model_params"], dropna=False)
        .agg(
            split_count=("test_balanced_accuracy", "count"),
            test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
            test_balanced_accuracy_std=("test_balanced_accuracy", "std"),
            test_balanced_accuracy_min=("test_balanced_accuracy", "min"),
            test_balanced_accuracy_max=("test_balanced_accuracy", "max"),
            validation_balanced_accuracy_mean=("validation_balanced_accuracy", "mean"),
        )
        .reset_index()
        .sort_values("test_balanced_accuracy_mean", ascending=False)
    )
    summary.to_csv(args.output_dir / "rolling_grid_summary.csv", index=False)

    best_by_ratio = (
        summary.sort_values("test_balanced_accuracy_mean", ascending=False)
        .groupby("ratio", as_index=False)
        .head(1)
        .sort_values("ratio")
    )
    best_by_ratio.to_csv(args.output_dir / "best_by_ratio.csv", index=False)

    if best is not None:
        write_json(args.output_dir / "best_single_split_result.json", best["row"])
        if args.save_best_model:
            row = best["row"]
            artifact = {
                "artifact_type": "hierarchical_inter_subject_fused",
                "model_type": row["model_type"],
                "model_params": row["model_params"],
                "window_ms": int(row["window_ms"]),
                "sampling_rate_hz": fs,
                "sensor_order": SENSOR_ORDER,
                "sensor_locations": SENSOR_LOCATIONS,
                "feature_columns": best["feature_columns"],
                "classes": best["labels"],
                "binary_model": best["bundle"]["binary_model"],
                "binary_threshold": best["bundle"]["binary_threshold"],
                "gesture_model": best["bundle"]["gesture_model"],
                "gesture_encoder": best["bundle"]["gesture_encoder"],
                "gesture_thresholds": best["bundle"]["gesture_thresholds"],
                "raw_emg_normalization_stats": best["normalization_stats"],
                "preprocessing": dataset_summary["preprocessing"],
                "training_split": {
                    "ratio": row["ratio"],
                    "split_name": row["split_name"],
                    "train_trials": row["train_trials"],
                    "test_trials": row["test_trials"],
                },
                "note": "Best single split model from rolling grid experiment. Use summary CSV for robust mean performance.",
            }
            with (args.output_dir / "best_single_split_model.pkl").open("wb") as file:
                pickle.dump(artifact, file)

    print("\nTop 20 mean results:")
    print(summary.head(20).to_string(index=False))
    print(f"\nSaved results to: {args.output_dir}")


def main() -> None:
    """Dispatch one of the retained hierarchical-training command-line workflows."""
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "inter-subject"
    if command == "inter-subject":
        run()
    elif command == "rolling":
        run_rolling_grid()
    else:
        raise SystemExit(f"Unknown model-training command: {command}")


if __name__ == "__main__":
    main()
