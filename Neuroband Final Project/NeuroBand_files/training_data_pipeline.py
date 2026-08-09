"""Shared data-loading, preprocessing, and feature-building utilities.

Although originally introduced for broad model experiments, this module now
provides core functions used by personal training. It reads staged uMyo CSV files,
interprets packet EMG values as consecutive time samples, applies the configured
filters and normalization, synchronizes the three sensor locations, and constructs
window-level feature matrices and metadata.

"""

from __future__ import annotations

import argparse
import json
import math
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from model_training import fit_hierarchical, predict_hierarchical


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data" / "calibration_sessions"
DEFAULT_OUTPUT_DIR = APP_DIR / "broad_existing_recordings_experiment"
SENSORS = ["B0DAC7E9", "ED7A78C8", "37ED348F"]
EMG_COLUMNS = [f"emg_{index}" for index in range(8)]
SP_COLUMNS = ["sp1", "sp2", "sp3"]
MOUSE_LABEL_MAP = {
    "at_rest": "at_rest",
    "open_hand": "movement",
    "pointing": "movement",
    "fist": "left_click",
    "pinch": "left_click",
    "like": "right_click",
    "wrist_extension": "scroll_up",
    "wrist_flexion": "scroll_down",
}


def parse_broad_args() -> argparse.Namespace:
    """Perform the parse args operation used by the run broad existing recordings experiment workflow."""
    parser = argparse.ArgumentParser(description="Broad offline experiment using existing V2 recordings only.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-ms", type=int, nargs="+", default=[200, 300, 400, 500])
    parser.add_argument("--normalizations", nargs="+", default=["none", "sensor_standard", "sensor_robust", "rest_relative", "window_energy"])
    parser.add_argument("--feature-sets", nargs="+", default=["baseline", "extended", "wavelet", "all"])
    parser.add_argument("--label-modes", nargs="+", default=["gesture", "mouse_action"])
    parser.add_argument("--architectures", nargs="+", default=["direct", "hierarchical"])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--trim-edge-ms", type=float, default=100.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Run a reduced smoke-test sweep.")
    return parser.parse_args()


def read_existing_recordings(root: Path) -> pd.DataFrame:
    """Read and parse existing recordings for the current run broad existing recordings experiment workflow."""
    frames = []
    for path in sorted(root.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        unit_col = "unit_id" if "unit_id" in frame else "device_id" if "device_id" in frame else None
        required = {"timestamp", "trial_index", "gesture_label", *EMG_COLUMNS}
        if unit_col is None or not required.issubset(frame.columns):
            continue
        frame = frame.rename(columns={unit_col: "unit_id"})
        for column in SP_COLUMNS:
            if column not in frame:
                frame[column] = 0.0
        frame["source_file"] = str(path.relative_to(root))
        frames.append(frame[["source_file", "timestamp", "trial_index", "gesture_label", "unit_id", *EMG_COLUMNS, *SP_COLUMNS]])
    if not frames:
        raise FileNotFoundError(f"No compatible recording CSV files found under {root}")
    data = pd.concat(frames, ignore_index=True)
    data["gesture_label"] = data["gesture_label"].astype(str).str.strip().str.lower().replace({"rest": "at_rest", "open_palm": "open_hand"})
    data = data[~data["gesture_label"].isin({"beginning", "side_flex"})].copy()
    data["unit_id"] = data["unit_id"].astype(str).str.upper().str.strip()
    data = data[data["unit_id"].isin(SENSORS)].copy()
    for column in ["timestamp", "trial_index", *EMG_COLUMNS, *SP_COLUMNS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna().reset_index(drop=True)


def estimate_fs(data: pd.DataFrame) -> float:
    """Perform the estimate fs operation used by the run broad existing recordings experiment workflow."""
    estimates = []
    for _, group in data.groupby(["source_file", "unit_id"], sort=False):
        diffs = np.diff(group.sort_values("timestamp")["timestamp"].to_numpy(float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            estimates.append(8.0 / np.median(diffs))
    return float(np.median(estimates)) if estimates else 620.0


def filter_all(data: pd.DataFrame, fs: float) -> pd.DataFrame:
    """Filter all for the current run broad existing recordings experiment workflow."""
    result = data.copy()
    nyq = fs / 2.0
    high = min(500.0, nyq - 1.0)
    sos = signal.butter(4, [35.0 / nyq, high / nyq], btype="bandpass", output="sos")
    if 50.0 < nyq:
        b, a = signal.iirnotch(50.0 / nyq, 30.0)
        sos = np.vstack([sos, signal.tf2sos(b, a)])
    # Match scipy.signal.sosfiltfilt's default padding requirement. Very short
    # transition fragments cannot be zero-phase filtered, but they should not
    # abort an otherwise valid personal calibration session.
    ntaps = 2 * len(sos) + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    padlen = 3 * ntaps
    for _, group in result.groupby(["source_file", "trial_index", "gesture_label", "unit_id"], sort=False):
        ordered = group.sort_values("timestamp")
        flat = ordered[EMG_COLUMNS].to_numpy(float).reshape(-1)
        filtered = signal.sosfiltfilt(sos, flat) if flat.size > padlen else signal.sosfilt(sos, flat)
        result.loc[ordered.index, EMG_COLUMNS] = filtered.reshape(-1, 8)
    return result


def normalize_raw(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Perform the normalize raw operation used by the run broad existing recordings experiment workflow."""
    result = data.copy()
    for sensor_id in SENSORS:
        mask = result["unit_id"] == sensor_id
        values = result.loc[mask, EMG_COLUMNS].to_numpy(float)
        flat = values.reshape(-1)
        if mode == "none" or mode == "window_energy":
            continue
        if mode == "sensor_robust":
            center = np.median(flat)
            scale = np.percentile(flat, 75) - np.percentile(flat, 25)
        elif mode == "rest_relative":
            rest = result.loc[mask & (result["gesture_label"] == "at_rest"), EMG_COLUMNS].to_numpy(float).reshape(-1)
            center = np.mean(rest) if rest.size else np.mean(flat)
            scale = np.std(rest) if rest.size else np.std(flat)
        else:
            center, scale = np.mean(flat), np.std(flat)
        result.loc[mask, EMG_COLUMNS] = (values - center) / max(float(scale), 1e-9)
    return result


def zero_crossings(values: np.ndarray) -> float:
    """Perform the zero crossings operation used by the run broad existing recordings experiment workflow."""
    return float(np.sum(np.signbit(values[:-1]) != np.signbit(values[1:])))


def haar_features(values: np.ndarray, levels: int = 3) -> dict[str, float]:
    """Perform the haar features operation used by the run broad existing recordings experiment workflow."""
    current = np.asarray(values, dtype=float)
    output: dict[str, float] = {}
    for level in range(1, levels + 1):
        if current.size < 4:
            output[f"wavelet_d{level}_energy"] = 0.0
            output[f"wavelet_d{level}_std"] = 0.0
            continue
        if current.size % 2:
            current = current[:-1]
        approx = (current[0::2] + current[1::2]) / math.sqrt(2.0)
        detail = (current[0::2] - current[1::2]) / math.sqrt(2.0)
        output[f"wavelet_d{level}_energy"] = float(np.mean(detail ** 2))
        output[f"wavelet_d{level}_std"] = float(np.std(detail))
        current = approx
    output["wavelet_approx_energy"] = float(np.mean(current ** 2)) if current.size else 0.0
    return output


def signal_features(values: np.ndarray, fs: float, feature_set: str, window_energy: bool) -> dict[str, float]:
    """Perform the signal features operation used by the run broad existing recordings experiment workflow."""
    values = np.nan_to_num(np.asarray(values, dtype=float))
    if window_energy:
        values = values / max(float(np.sqrt(np.mean(values ** 2))), 1e-9)
    diff = np.diff(values)
    rms = float(np.sqrt(np.mean(values ** 2)))
    features = {
        "rms": rms,
        "mav": float(np.mean(np.abs(values))),
        "zc": zero_crossings(values),
        "ssc": float(np.sum(diff[:-1] * diff[1:] < 0)) if diff.size > 1 else 0.0,
        "wl": float(np.sum(np.abs(diff))),
        "var": float(np.var(values)),
    }
    if feature_set in {"extended", "all"}:
        spectrum = np.abs(np.fft.rfft(values)) ** 2
        freqs = np.fft.rfftfreq(values.size, 1.0 / fs)
        total = max(float(np.sum(spectrum)), 1e-12)
        cumulative = np.cumsum(spectrum)
        features.update(
            {
                "iemg": float(np.sum(np.abs(values))),
                "log_detector": float(np.exp(np.mean(np.log(np.abs(values) + 1e-9)))),
                "wamp": float(np.sum(np.abs(diff) > 0.1 * max(rms, 1e-9))),
                "peak_to_peak": float(np.ptp(values)),
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                "hjorth_mobility": float(np.sqrt(np.var(diff) / max(np.var(values), 1e-12))),
                "spectral_entropy": float(-np.sum((spectrum / total) * np.log(spectrum / total + 1e-12))),
                "mean_frequency": float(np.sum(freqs * spectrum) / total),
                "median_frequency": float(freqs[min(len(freqs) - 1, int(np.searchsorted(cumulative, total / 2.0)))]),
            }
        )
    if feature_set in {"wavelet", "all"}:
        features.update(haar_features(values))
    return features


def extract_fused_windows(data: pd.DataFrame, fs: float, window_ms: int, overlap: float, trim_edge_ms: float, feature_set: str, normalization: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Extract fused windows for the current run broad existing recordings experiment workflow."""
    window = max(8, int(round(fs * window_ms / 1000.0)))
    step = max(1, int(round(window * (1.0 - overlap))))
    trim = max(0, int(round(fs * trim_edge_ms / 1000.0)))
    packet_interval_s = 8.0 / fs
    step_s = step / fs
    # Host timestamps describe packet delivery, which can arrive in short bursts
    # under Windows. Only treat a pause as a discontinuity at the established
    # serious-gap boundary; shorter scheduling pauses do not prove sample loss.
    continuity_gap_s = max(0.25, 6.0 * packet_interval_s)
    stage_keys = ["source_file", "trial_index", "gesture_label"]
    stage_starts = data.groupby(stage_keys, sort=False)["timestamp"].min().to_dict()
    device_rows = []
    for key, group in data.groupby(["source_file", "trial_index", "gesture_label", "unit_id"], sort=True):
        source, trial, label, sensor_id = key
        ordered = group.sort_values("timestamp")
        flat = ordered[EMG_COLUMNS].to_numpy(float).reshape(-1)
        spectrum = ordered[SP_COLUMNS].to_numpy(float)
        packet_times = ordered["timestamp"].to_numpy(float)
        stage_start = float(stage_starts[(source, trial, label)])
        for sequential_index, start in enumerate(range(0, flat.size - window + 1, step)):
            end = start + window
            if trim and (start < trim or end > flat.size - trim):
                continue
            packet_start, packet_end = start // 8, min(len(spectrum), int(math.ceil(end / 8)))
            times = packet_times[packet_start:packet_end]
            if not times.size or np.any(~np.isfinite(times)):
                continue
            timestamp_diffs = np.diff(times)
            if timestamp_diffs.size and (np.any(timestamp_diffs <= 0.0) or float(np.max(timestamp_diffs)) > continuity_gap_s):
                continue
            time_index = int(round((float(times[0]) - stage_start) / max(step_s, 1e-9)))
            if time_index < 0:
                time_index = sequential_index
            row = signal_features(flat[start:end], fs, feature_set, normalization == "window_energy")
            for sp_index, name in enumerate(SP_COLUMNS):
                row[f"{name}_mean"] = float(np.mean(spectrum[packet_start:packet_end, sp_index]))
            device_rows.append({
                **row,
                "source_file": source,
                "trial_index": int(trial),
                "gesture_label": label,
                "unit_id": sensor_id,
                "window_index": time_index,
                "window_start_time": float(times[0]),
            })
    frame = pd.DataFrame(device_rows)
    feature_columns = [column for column in frame.columns if column not in {"source_file", "trial_index", "gesture_label", "unit_id", "window_index", "window_start_time"}]
    fused, labels, meta = [], [], []
    meta_columns = ["source_file", "trial_index", "gesture_label", "window_index", "window_start_time"]
    if frame.empty:
        return pd.DataFrame(), pd.Series(dtype=str), pd.DataFrame(columns=meta_columns)

    # Sensor packet streams are asynchronous. Match valid windows by their real
    # timestamps instead of requiring identical rounded time buckets.
    alignment_tolerance_s = max(2.0 * packet_interval_s, 0.55 * step_s)
    for stage_key, stage_group in frame.groupby(stage_keys, sort=True):
        source, trial, label = stage_key
        by_sensor = {
            sensor_id: stage_group[stage_group["unit_id"] == sensor_id].sort_values("window_start_time")
            for sensor_id in SENSORS
        }
        if any(sensor_rows.empty for sensor_rows in by_sensor.values()):
            continue
        used_indices = {sensor_id: set() for sensor_id in SENSORS[1:]}
        fused_index = 0
        for reference_index, reference in by_sensor[SENSORS[0]].iterrows():
            matched = {SENSORS[0]: reference}
            reference_time = float(reference["window_start_time"])
            for sensor_id in SENSORS[1:]:
                candidates = by_sensor[sensor_id].loc[
                    ~by_sensor[sensor_id].index.isin(used_indices[sensor_id])
                ]
                if candidates.empty:
                    break
                distances = (candidates["window_start_time"] - reference_time).abs()
                candidate_index = distances.idxmin()
                if float(distances.loc[candidate_index]) > alignment_tolerance_s:
                    break
                matched[sensor_id] = candidates.loc[candidate_index]
            if len(matched) != len(SENSORS):
                continue
            for sensor_id in SENSORS[1:]:
                used_indices[sensor_id].add(matched[sensor_id].name)
            row = {}
            for sensor_id in SENSORS:
                sensor = matched[sensor_id]
                row.update({f"{sensor_id}_{column}": float(sensor[column]) for column in feature_columns})
            for first, second in [(SENSORS[0], SENSORS[1]), (SENSORS[0], SENSORS[2]), (SENSORS[1], SENSORS[2])]:
                row[f"{first}_{second}_rms_diff"] = row[f"{first}_rms"] - row[f"{second}_rms"]
                row[f"{first}_{second}_rms_ratio"] = row[f"{first}_rms"] / max(abs(row[f"{second}_rms"]), 1e-9)
            fused.append(row)
            labels.append(str(label))
            meta.append({
                "source_file": source,
                "trial_index": int(trial),
                "gesture_label": label,
                "window_index": fused_index,
                "window_start_time": float(np.median([float(sensor["window_start_time"]) for sensor in matched.values()])),
            })
            fused_index += 1
    return pd.DataFrame(fused), pd.Series(labels, dtype=str), pd.DataFrame(meta, columns=meta_columns)


def split_indices(y: pd.Series, groups: pd.Series, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform the split indices operation used by the run broad existing recordings experiment workflow."""
    if groups.nunique() >= 3:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
        train_val, test = next(outer.split(np.zeros(len(y)), y, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + 1)
        train_local, val_local = next(inner.split(np.zeros(len(train_val)), y.iloc[train_val], groups.iloc[train_val]))
        return train_val[train_local], train_val[val_local], test
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_val, test = next(outer.split(np.zeros(len(y)), y))
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + 1)
    train_local, val_local = next(inner.split(np.zeros(len(train_val)), y.iloc[train_val]))
    return train_val[train_local], train_val[val_local], test


def model_specs(random_state: int, quick: bool) -> list[tuple[str, object]]:
    """Perform the model specs operation used by the run broad existing recordings experiment workflow."""
    specs = [
        ("extra_trees", ExtraTreesClassifier(n_estimators=250, class_weight="balanced", n_jobs=-1, random_state=random_state)),
        ("random_forest", RandomForestClassifier(n_estimators=250, class_weight="balanced", n_jobs=-1, random_state=random_state)),
        ("svm_rbf", SVC(C=10, kernel="rbf", class_weight="balanced", probability=True)),
        ("logistic_regression", LogisticRegression(C=10, class_weight="balanced", max_iter=3000)),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=random_state)),
        ("gaussian_nb", GaussianNB()),
    ]
    return specs[:2] if quick else specs


def run_broad_experiment() -> int:
    """Run the module's command-line or graphical application entry point."""
    args = parse_broad_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    raw = read_existing_recordings(args.data_dir)
    fs = estimate_fs(raw)
    filtered = filter_all(raw, fs)
    rows = []
    best = None
    windows = args.window_ms[:1] if args.quick else args.window_ms
    norms = args.normalizations[:1] if args.quick else args.normalizations
    features = args.feature_sets[:2] if args.quick else args.feature_sets
    for normalization in norms:
        normalized = normalize_raw(filtered, normalization)
        for window_ms in windows:
            for feature_set in features:
                x, original_y, meta = extract_fused_windows(normalized, fs, window_ms, args.overlap, args.trim_edge_ms, feature_set, normalization)
                for label_mode in args.label_modes:
                    y = original_y.map(MOUSE_LABEL_MAP) if label_mode == "mouse_action" else original_y
                    valid = y.notna()
                    x_mode, y_mode, meta_mode = x.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), meta.loc[valid].reset_index(drop=True)
                    train_idx, val_idx, test_idx = split_indices(y_mode, meta_mode["source_file"], args.random_state)
                    for scaler_name, scaler in [("standard", StandardScaler()), ("robust", RobustScaler())]:
                        for model_name, model in model_specs(args.random_state, args.quick):
                            for architecture in args.architectures:
                                pipeline = Pipeline([("scaler", clone(scaler)), ("model", clone(model))])
                                try:
                                    if architecture == "hierarchical":
                                        bundle = fit_hierarchical(pipeline, x_mode.iloc[train_idx], y_mode.iloc[train_idx], x_mode.iloc[val_idx], y_mode.iloc[val_idx])
                                        prediction = predict_hierarchical(bundle, x_mode.iloc[test_idx])
                                    else:
                                        pipeline.fit(x_mode.iloc[train_idx], y_mode.iloc[train_idx])
                                        prediction = pipeline.predict(x_mode.iloc[test_idx])
                                    score = balanced_accuracy_score(y_mode.iloc[test_idx], prediction)
                                    row = {
                                        "normalization": normalization, "window_ms": window_ms, "feature_set": feature_set,
                                        "label_mode": label_mode, "feature_scaler": scaler_name, "model": model_name,
                                        "architecture": architecture, "balanced_accuracy": float(score),
                                        "features": x_mode.shape[1], "train_windows": len(train_idx), "test_windows": len(test_idx),
                                    }
                                    rows.append(row)
                                    if best is None or score > best["balanced_accuracy"]:
                                        best = {**row, "labels": sorted(y_mode.unique()), "confusion_matrix": confusion_matrix(y_mode.iloc[test_idx], prediction, labels=sorted(y_mode.unique())).tolist()}
                                except Exception as exc:
                                    rows.append({"normalization": normalization, "window_ms": window_ms, "feature_set": feature_set, "label_mode": label_mode, "feature_scaler": scaler_name, "model": model_name, "architecture": architecture, "balanced_accuracy": np.nan, "error": str(exc)})
    results = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False, na_position="last")
    results.to_csv(args.output_dir / "all_results.csv", index=False)
    summary = {"data_dir": str(args.data_dir), "sampling_rate_hz": fs, "candidate_count": len(results), "best": best, "elapsed_seconds": time.perf_counter() - start_time}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(results.head(30).to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


# Strict intra-subject evaluation workflow

def parse_intra_args() -> argparse.Namespace:
    """Perform the parse args operation used by the run intra subject existing recordings experiment workflow."""
    parser = argparse.ArgumentParser(description="Leakage-safe intra-subject experiment for every V2 calibration recording.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=APP_DIR / "intra_subject_existing_recordings_results")
    parser.add_argument("--window-ms", type=int, nargs="+", default=[200, 300, 400, 500])
    parser.add_argument("--normalizations", nargs="+", default=["none", "sensor_standard", "sensor_robust", "rest_relative", "window_energy"])
    parser.add_argument("--feature-sets", nargs="+", default=["baseline", "extended", "wavelet", "all"])
    parser.add_argument("--feature-scalers", nargs="+", choices=["standard", "robust"], default=["standard", "robust"])
    parser.add_argument("--label-modes", nargs="+", default=["gesture", "mouse_action"])
    parser.add_argument("--architectures", nargs="+", default=["direct", "hierarchical"])
    parser.add_argument("--models", nargs="+", default=["extra_trees", "random_forest", "svm_rbf", "logistic_regression", "gradient_boosting", "gaussian_nb"])
    parser.add_argument("--sessions", nargs="+", help="Optional calibration-session folder names to evaluate.")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--trim-edge-ms", type=float, default=100.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def intra_stage_split(data: pd.DataFrame, random_state: int) -> tuple[set[int], set[int], set[int]]:
    """Perform the stage split operation used by the run intra subject existing recordings experiment workflow."""
    stages = data[["trial_index", "gesture_label"]].drop_duplicates()
    train: set[int] = set()
    validation: set[int] = set()
    test: set[int] = set()
    for label_index, (label, group) in enumerate(stages.groupby("gesture_label", sort=True)):
        trials = group["trial_index"].astype(int).to_numpy(copy=True)
        rng = np.random.default_rng(random_state + 1009 * label_index)
        rng.shuffle(trials)
        if len(trials) < 3:
            raise ValueError(f"Gesture {label!r} has only {len(trials)} protocol stages; at least three are required.")
        test_count = max(1, int(round(0.20 * len(trials))))
        validation_count = max(1, int(round(0.20 * len(trials))))
        while len(trials) - test_count - validation_count < 1:
            test_count = max(1, test_count - 1)
            validation_count = max(1, validation_count - 1)
        test.update(map(int, trials[:test_count]))
        validation.update(map(int, trials[test_count : test_count + validation_count]))
        train.update(map(int, trials[test_count + validation_count :]))
    return train, validation, test


def normalize_from_training_stages(data: pd.DataFrame, mode: str, train_trials: set[int]) -> pd.DataFrame:
    """Perform the normalize from training stages operation used by the run intra subject existing recordings experiment workflow."""
    result = data.copy()
    if mode in {"none", "window_energy"}:
        return result
    for sensor_id in SENSORS:
        sensor_mask = result["unit_id"] == sensor_id
        train_mask = sensor_mask & result["trial_index"].isin(train_trials)
        source = result.loc[train_mask]
        if mode == "rest_relative":
            rest = source[source["gesture_label"] == "at_rest"]
            source = rest if not rest.empty else source
        flat = source[EMG_COLUMNS].to_numpy(float).reshape(-1)
        if not flat.size:
            raise ValueError(f"No training samples available for sensor {sensor_id}.")
        if mode == "sensor_robust":
            center = float(np.median(flat))
            scale = float(np.percentile(flat, 75) - np.percentile(flat, 25))
        else:
            center = float(np.mean(flat))
            scale = float(np.std(flat))
        values = result.loc[sensor_mask, EMG_COLUMNS].to_numpy(float)
        result.loc[sensor_mask, EMG_COLUMNS] = (values - center) / max(scale, 1e-9)
    return result


def load_protocol_kinds(session_dir: Path) -> dict[int, str]:
    """Load and validate protocol kinds for the current run intra subject existing recordings experiment workflow."""
    path = session_dir / "session_protocol.json"
    if not path.exists():
        return {}
    protocol = json.loads(path.read_text(encoding="utf-8"))
    return {index: str(stage.get("kind", "unknown")) for index, stage in enumerate(protocol)}


def stage_split_indices(meta: pd.DataFrame, train_trials: set[int], validation_trials: set[int], test_trials: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform the split indices operation used by the run intra subject existing recordings experiment workflow."""
    train = np.flatnonzero(meta["trial_index"].isin(train_trials).to_numpy())
    validation = np.flatnonzero(meta["trial_index"].isin(validation_trials).to_numpy())
    test = np.flatnonzero(meta["trial_index"].isin(test_trials).to_numpy())
    return train, validation, test


def safe_balanced_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Perform the safe balanced accuracy operation used by the run intra subject existing recordings experiment workflow."""
    return float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else math.nan


def run_intra_subject_experiment() -> int:
    """Run the module's command-line or graphical application entry point."""
    args = parse_intra_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    raw_all = read_existing_recordings(args.data_dir)
    if args.sessions:
        requested_sessions = set(args.sessions)
        raw_all = raw_all[
            raw_all["source_file"].map(lambda value: Path(value).parts[0] in requested_sessions)
        ].copy()
        if raw_all.empty:
            raise ValueError(f"No compatible recordings found for requested sessions: {sorted(requested_sessions)}")
    model_lookup = {name: model for name, model in model_specs(args.random_state, quick=False)}
    requested_models = [(name, model_lookup[name]) for name in args.models]
    sessions = sorted(raw_all["source_file"].unique())

    if args.quick:
        args.window_ms = [500]
        args.normalizations = ["sensor_standard"]
        args.feature_sets = ["baseline", "extended", "wavelet"]
        args.label_modes = ["gesture"]
        args.architectures = ["hierarchical"]
        requested_models = [(name, model_lookup[name]) for name in ["extra_trees", "svm_rbf"]]

    all_rows: list[dict[str, object]] = []
    best_by_session: dict[str, dict[str, object]] = {}
    for session_number, source_file in enumerate(sessions, start=1):
        session_start = time.perf_counter()
        session_raw = raw_all[raw_all["source_file"] == source_file].copy().reset_index(drop=True)
        session_name = Path(source_file).parts[0]
        session_dir = args.data_dir / session_name
        stage_kinds = load_protocol_kinds(session_dir)
        train_trials, validation_trials, test_trials = intra_stage_split(session_raw, args.random_state)
        fs = estimate_fs(session_raw)
        filtered = filter_all(session_raw, fs)
        best: dict[str, object] | None = None

        for normalization in args.normalizations:
            normalized = normalize_from_training_stages(filtered, normalization, train_trials)
            for window_ms in args.window_ms:
                for feature_set in args.feature_sets:
                    x, original_y, meta = extract_fused_windows(
                        normalized, fs, window_ms, args.overlap, args.trim_edge_ms, feature_set, normalization
                    )
                    meta["stage_kind"] = meta["trial_index"].map(stage_kinds).fillna("unknown")
                    train_idx, validation_idx, test_idx = stage_split_indices(meta, train_trials, validation_trials, test_trials)
                    for label_mode in args.label_modes:
                        y = original_y.map(MOUSE_LABEL_MAP) if label_mode == "mouse_action" else original_y
                        valid = y.notna().to_numpy()
                        scaler_lookup = {"standard": StandardScaler(), "robust": RobustScaler()}
                        for scaler_name in args.feature_scalers:
                            scaler = scaler_lookup[scaler_name]
                            for model_name, model in requested_models:
                                for architecture in args.architectures:
                                    pipeline = Pipeline([("scaler", clone(scaler)), ("model", clone(model))])
                                    row: dict[str, object] = {
                                        "session": session_name,
                                        "normalization": normalization,
                                        "window_ms": window_ms,
                                        "feature_set": feature_set,
                                        "label_mode": label_mode,
                                        "feature_scaler": scaler_name,
                                        "model": model_name,
                                        "architecture": architecture,
                                        "features": int(x.shape[1]),
                                    }
                                    try:
                                        tr = train_idx[valid[train_idx]]
                                        va = validation_idx[valid[validation_idx]]
                                        te = test_idx[valid[test_idx]]
                                        if architecture == "hierarchical":
                                            bundle = fit_hierarchical(pipeline, x.iloc[tr], y.iloc[tr], x.iloc[va], y.iloc[va])
                                            validation_prediction = predict_hierarchical(bundle, x.iloc[va])
                                            prediction = predict_hierarchical(bundle, x.iloc[te])
                                        else:
                                            pipeline.fit(x.iloc[tr], y.iloc[tr])
                                            validation_prediction = pipeline.predict(x.iloc[va])
                                            prediction = pipeline.predict(x.iloc[te])
                                        validation_score = safe_balanced_accuracy(y.iloc[va], validation_prediction)
                                        score = safe_balanced_accuracy(y.iloc[te], prediction)
                                        row.update(
                                            {
                                                "validation_balanced_accuracy": validation_score,
                                                "balanced_accuracy": score,
                                                "train_windows": len(tr),
                                                "validation_windows": len(va),
                                                "test_windows": len(te),
                                                "test_transition_accuracy": float(np.mean(prediction[meta.iloc[te]["stage_kind"].to_numpy() == "transition_hold"] == y.iloc[te][meta.iloc[te]["stage_kind"].to_numpy() == "transition_hold"])) if np.any(meta.iloc[te]["stage_kind"].to_numpy() == "transition_hold") else math.nan,
                                            }
                                        )
                                        if best is None or validation_score > float(best["validation_balanced_accuracy"]):
                                            labels = sorted(y.iloc[te].unique())
                                            best = {
                                                **row,
                                                "labels": labels,
                                                "confusion_matrix": confusion_matrix(y.iloc[te], prediction, labels=labels).tolist(),
                                                "train_trials": sorted(train_trials),
                                                "validation_trials": sorted(validation_trials),
                                                "test_trials": sorted(test_trials),
                                                "sampling_rate_hz": fs,
                                            }
                                    except Exception as exc:
                                        row.update({"balanced_accuracy": math.nan, "error": str(exc)})
                                    all_rows.append(row)

        if best is not None:
            best_by_session[session_name] = best
        pd.DataFrame(all_rows).to_csv(args.output_dir / "all_results.csv", index=False)
        (args.output_dir / "best_by_session.json").write_text(json.dumps(best_by_session, indent=2), encoding="utf-8")
        best_text = "none" if best is None else f"{float(best['balanced_accuracy']):.4f}"
        print(f"[{session_number}/{len(sessions)}] {session_name}: best={best_text}, elapsed={time.perf_counter() - session_start:.1f}s", flush=True)

    results = pd.DataFrame(all_rows).sort_values("balanced_accuracy", ascending=False, na_position="last")
    results.to_csv(args.output_dir / "all_results.csv", index=False)
    valid_results = results.dropna(subset=["balanced_accuracy"])
    config_columns = ["normalization", "window_ms", "feature_set", "label_mode", "feature_scaler", "model", "architecture"]
    aggregate = (
        valid_results.groupby(config_columns, dropna=False)["balanced_accuracy"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
        .sort_values(["mean", "min"], ascending=False)
    )
    aggregate.to_csv(args.output_dir / "aggregate_config_results.csv", index=False)
    factor_rows = []
    for factor in ["normalization", "window_ms", "feature_set", "label_mode", "feature_scaler", "model", "architecture"]:
        summary = valid_results.groupby(factor)["balanced_accuracy"].agg(["mean", "std", "median", "count"]).reset_index()
        summary.insert(0, "factor", factor)
        summary = summary.rename(columns={factor: "value"})
        factor_rows.append(summary)
    pd.concat(factor_rows, ignore_index=True).to_csv(args.output_dir / "factor_summary.csv", index=False)
    summary = {
        "evaluation": "strictly intra-subject; every session trained, validated, and tested independently",
        "split_unit": "protocol stage / trial_index; overlapping windows from one stage never cross splits",
        "normalization": "raw normalization statistics fitted on training stages only",
        "session_count": len(sessions),
        "candidate_count": len(results),
        "elapsed_seconds": time.perf_counter() - start_time,
        "best_by_session": best_by_session,
        "best_aggregate_config": None if aggregate.empty else aggregate.iloc[0].to_dict(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"best_aggregate_config": summary["best_aggregate_config"], "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))
    return 0


def main() -> int:
    """Dispatch the broad or strict intra-subject offline experiment."""
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "intra-subject"
    if command == "broad":
        return run_broad_experiment()
    if command == "intra-subject":
        return run_intra_subject_experiment()
    raise SystemExit(f"Unknown data-pipeline command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
