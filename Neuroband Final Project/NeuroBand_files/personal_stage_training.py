"""Train a personal hierarchical gesture model from a calibration session.

The training workflow reads one user's staged recording, performs quality-aware
preprocessing, builds synchronized three-sensor windows, and separates trials into
training, validation, and held-out test sets. Candidate estimators, window sizes,
normalization options, and feature configurations are compared on validation data.
The selected configuration is refitted and saved with all metadata required for
realtime preprocessing and decision control.

"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC

import training_data_pipeline as broad
from training_data_pipeline import intra_stage_split as stage_split, load_protocol_kinds
from model_training import (
    SENSOR_LOCATIONS,
    SENSOR_ORDER,
    fit_rolling_hierarchical as fit_hierarchical,
    predict_hierarchical_local,
    write_json,
)
from recording_quality_gate import audit_session_for_training


FAST_NORMALIZATIONS = ["none", "sensor_standard", "sensor_robust", "rest_relative"]
FULL_NORMALIZATIONS = ["none", "sensor_standard", "sensor_robust"]
WINDOWS_MS = [200, 300, 400, 500]
TRIM_EDGE_MS = 100.0


def parse_args(default_mode: str | None = None) -> argparse.Namespace:
    """Perform the parse args operation used by the personal stage training workflow."""
    parser = argparse.ArgumentParser(description="Leakage-safe personal calibration training.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["fast", "full"], default=default_mode or "fast")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overlap", type=float, default=0.5)
    return parser.parse_args()


def model_specs(mode: str, random_state: int) -> list[tuple[str, str, object]]:
    """Perform the model specs operation used by the personal stage training workflow."""
    specs = [
        (
            "svm_rbf",
            "C=10;kernel=rbf;class_weight=balanced",
            Pipeline(
                [
                    ("scaler", RobustScaler()),
                    ("model", SVC(C=10, kernel="rbf", class_weight="balanced", probability=False, random_state=random_state)),
                ]
            ),
        )
    ]
    if mode == "fast":
        return specs
    specs.extend(
        [
            (
                "extra_trees",
                "n_estimators=250;class_weight=balanced",
                Pipeline(
                    [
                        ("scaler", RobustScaler()),
                        ("model", ExtraTreesClassifier(n_estimators=250, class_weight="balanced", n_jobs=-1, random_state=random_state)),
                    ]
                ),
            ),
            (
                "gradient_boosting",
                "n_estimators=120;learning_rate=0.05;max_depth=2",
                Pipeline(
                    [
                        ("scaler", RobustScaler()),
                        ("model", GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=random_state)),
                    ]
                ),
            ),
            (
                "logistic_regression",
                "C=10;class_weight=balanced",
                Pipeline(
                    [
                        ("scaler", RobustScaler()),
                        ("model", LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=random_state)),
                    ]
                ),
            ),
            (
                "logistic_regression_pca99",
                "C=10;class_weight=balanced;pca_variance=0.99",
                Pipeline(
                    [
                        ("scaler", RobustScaler()),
                        ("pca", PCA(n_components=0.99, random_state=random_state)),
                        ("model", LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=random_state)),
                    ]
                ),
            ),
        ]
    )
    return specs


def normalization_stats(data: pd.DataFrame, mode: str, trials: set[int]) -> dict[str, dict[str, float | str]]:
    """Perform the normalization stats operation used by the personal stage training workflow."""
    stats: dict[str, dict[str, float | str]] = {}
    for sensor_id in broad.SENSORS:
        source = data[(data["unit_id"] == sensor_id) & data["trial_index"].isin(trials)]
        if mode == "rest_relative":
            rest = source[source["gesture_label"] == "at_rest"]
            source = rest if not rest.empty else source
        flat = source[broad.EMG_COLUMNS].to_numpy(float).reshape(-1)
        if not flat.size:
            raise ValueError(f"No normalization data for sensor {sensor_id}.")
        if mode == "sensor_robust":
            center = float(np.median(flat))
            scale = float(np.percentile(flat, 75) - np.percentile(flat, 25))
        elif mode in {"sensor_standard", "rest_relative"}:
            center = float(np.mean(flat))
            scale = float(np.std(flat))
        else:
            center, scale = 0.0, 1.0
        stats[sensor_id] = {"mode": mode, "center": center, "scale": max(scale, 1e-9)}
    return stats


def apply_normalization(data: pd.DataFrame, stats: dict[str, dict[str, float | str]]) -> pd.DataFrame:
    """Apply normalization for the current personal stage training workflow."""
    result = data.copy()
    for sensor_id, sensor_stats in stats.items():
        mask = result["unit_id"] == sensor_id
        center = float(sensor_stats["center"])
        scale = float(sensor_stats["scale"])
        values = result.loc[mask, broad.EMG_COLUMNS].to_numpy(float)
        result.loc[mask, broad.EMG_COLUMNS] = (values - center) / scale
    return result


def make_features(
    filtered: pd.DataFrame,
    fs: float,
    normalization: str,
    fit_trials: set[int],
    window_ms: int,
    feature_set: str,
    overlap: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, dict[str, float | str]]]:
    """Create and configure features for the current personal stage training workflow."""
    stats = normalization_stats(filtered, normalization, fit_trials)
    normalized = apply_normalization(filtered, stats)
    x, y, meta = broad.extract_fused_windows(
        normalized,
        fs,
        window_ms,
        overlap,
        TRIM_EDGE_MS,
        feature_set,
        normalization,
    )
    if feature_set == "baseline_no_var":
        x = x.loc[:, [column for column in x.columns if not column.endswith("_var")]]
    return x, y, meta, stats


def indices_for(meta: pd.DataFrame, trials: set[int]) -> np.ndarray:
    """Perform the indices for operation used by the personal stage training workflow."""
    return np.flatnonzero(meta["trial_index"].isin(trials).to_numpy())


def refit_with_selected_thresholds(
    model_template: object,
    x: pd.DataFrame,
    y: pd.Series,
    selected_bundle: dict[str, object],
) -> dict[str, object]:
    """Perform the refit with selected thresholds operation used by the personal stage training workflow."""
    binary_model = clone(model_template)
    binary_model.fit(x, (y.to_numpy() != "at_rest").astype(int))

    active = y != "at_rest"
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y.loc[active])
    gesture_model = clone(model_template)
    gesture_model.fit(x.loc[active].reset_index(drop=True), encoded)

    selected_encoder = selected_bundle["gesture_encoder"]
    selected_thresholds = np.asarray(selected_bundle["gesture_thresholds"], dtype=float)
    threshold_by_label = {
        str(label): float(selected_thresholds[index])
        for index, label in enumerate(selected_encoder.classes_)
    }
    thresholds = np.asarray([threshold_by_label[str(label)] for label in encoder.classes_], dtype=float)
    return {
        "binary_model": binary_model,
        "binary_threshold": float(selected_bundle["binary_threshold"]),
        "binary_roc_balanced": float(selected_bundle.get("binary_roc_balanced", 0.0)),
        "gesture_model": gesture_model,
        "gesture_encoder": encoder,
        "gesture_thresholds": thresholds,
        "gesture_threshold_quality": selected_bundle.get("gesture_threshold_quality", {}),
    }


def artifact(
    *,
    row: dict[str, object],
    bundle: dict[str, object],
    feature_columns: list[str],
    stats: dict[str, dict[str, float | str]],
    fs: float,
    split: dict[str, object],
    role: str,
) -> dict[str, object]:
    """Perform the artifact operation used by the personal stage training workflow."""
    return {
        "artifact_type": "hierarchical_personal_stage_fused_v2",
        "artifact_role": role,
        "model_type": row["model_type"],
        "model_params": row["model_params"],
        "window_ms": int(row["window_ms"]),
        "trim_edge_ms": TRIM_EDGE_MS,
        "feature_set": row["feature_set"],
        "raw_normalization": row["normalization"],
        "sampling_rate_hz": fs,
        "sensor_order": SENSOR_ORDER,
        "sensor_locations": SENSOR_LOCATIONS,
        "feature_columns": feature_columns,
        "classes": sorted(bundle["gesture_encoder"].classes_.tolist() + ["at_rest"]),
        "binary_model": bundle["binary_model"],
        "binary_threshold": bundle["binary_threshold"],
        "gesture_model": bundle["gesture_model"],
        "gesture_encoder": bundle["gesture_encoder"],
        "gesture_thresholds": bundle["gesture_thresholds"],
        "raw_emg_normalization_stats": stats,
        "preprocessing": {
            "bpf": "Butterworth bandpass order 4, 35-500 Hz plus 50 Hz notch filter",
            "emg_samples": "emg_0..emg_7 flattened as sequential time samples",
            "spectrum_0": "Dropped; using sp1-sp3 means",
            "feature_scaler": row["feature_scaler"],
            "dimensionality_reduction": "PCA retaining 99% training variance" if row["model_type"] == "logistic_regression_pca99" else "none",
            "feature_set": row["feature_set"],
            "raw_normalization": row["normalization"],
            "trim_edge_ms": TRIM_EDGE_MS,
            "side_flex": "Disabled and excluded",
        },
        "training_split": split,
        "note": f"Personal calibration {role} model using complete protocol-stage splits.",
    }


def save_pickle(path: Path, value: object) -> None:
    """Save pickle for the current personal stage training workflow."""
    with path.open("wb") as file:
        pickle.dump(value, file)


def run_training(session_dir: Path, mode: str, random_state: int = 42, overlap: float = 0.5) -> dict[str, object]:
    """Run training for the current personal stage training workflow."""
    started = time.perf_counter()
    quality_audit = audit_session_for_training(session_dir, SENSOR_ORDER)
    if not quality_audit["passed"]:
        raise ValueError("Training blocked by recording quality: " + "; ".join(quality_audit["blockers"]))
    raw_dir = session_dir / "raw_recordings"
    output_dir = session_dir / "trained_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = broad.read_existing_recordings(raw_dir)
    fs = broad.estimate_fs(raw)
    filtered = broad.filter_all(raw, fs)
    train_trials, validation_trials, test_trials = stage_split(filtered, random_state)
    stage_kinds = load_protocol_kinds(session_dir)

    normalizations = FAST_NORMALIZATIONS if mode == "fast" else FULL_NORMALIZATIONS
    feature_sets = ["baseline"] if mode == "fast" else ["baseline_no_var"]
    models = model_specs(mode, random_state)
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for normalization in normalizations:
        for window_ms in WINDOWS_MS:
            for feature_set in feature_sets:
                x, y, meta, stats = make_features(
                    filtered, fs, normalization, train_trials, window_ms, feature_set, overlap
                )
                if x.empty or meta.empty or "trial_index" not in meta:
                    raise ValueError(
                        "No synchronized three-sensor windows for "
                        f"window={window_ms} ms, normalization={normalization}, feature_set={feature_set}."
                    )
                tr, va = indices_for(meta, train_trials), indices_for(meta, validation_trials)
                for model_type, model_params, model in models:
                    bundle = fit_hierarchical(model, x.iloc[tr], y.iloc[tr], x.iloc[va], y.iloc[va])
                    train_prediction = predict_hierarchical_local(bundle, x.iloc[tr])
                    validation_prediction = predict_hierarchical_local(bundle, x.iloc[va])
                    row = {
                        "model_type": model_type,
                        "model_params": model_params,
                        "window_ms": window_ms,
                        "trim_edge_ms": TRIM_EDGE_MS,
                        "feature_set": feature_set,
                        "normalization": normalization,
                        "feature_scaler": "robust+pca99" if model_type == "logistic_regression_pca99" else "robust",
                        "train_balanced_accuracy": float(balanced_accuracy_score(y.iloc[tr], train_prediction)),
                        "validation_balanced_accuracy": float(balanced_accuracy_score(y.iloc[va], validation_prediction)),
                        "train_windows": len(tr),
                        "validation_windows": len(va),
                    }
                    rows.append(row)
                    if best is None or row["validation_balanced_accuracy"] > best["row"]["validation_balanced_accuracy"]:
                        best = {
                            "row": row,
                            "bundle": bundle,
                            "feature_columns": list(x.columns),
                            "selection_stats": stats,
                        }

    if best is None:
        raise RuntimeError("No personal calibration model was trained.")

    best_row = best["row"]
    train_validation_trials = train_trials | validation_trials
    x_eval, y_eval, meta_eval, eval_stats = make_features(
        filtered,
        fs,
        str(best_row["normalization"]),
        train_validation_trials,
        int(best_row["window_ms"]),
        str(best_row["feature_set"]),
        overlap,
    )
    train_validation_idx = indices_for(meta_eval, train_validation_trials)
    test_idx = indices_for(meta_eval, test_trials)
    selected_template = next(model for name, _params, model in models if name == best_row["model_type"])
    evaluation_bundle = refit_with_selected_thresholds(
        selected_template,
        x_eval.iloc[train_validation_idx],
        y_eval.iloc[train_validation_idx],
        best["bundle"],
    )
    test_prediction = predict_hierarchical_local(evaluation_bundle, x_eval.iloc[test_idx])
    test_ba = float(balanced_accuracy_score(y_eval.iloc[test_idx], test_prediction))
    transition_mask = meta_eval.iloc[test_idx]["trial_index"].map(stage_kinds).eq("transition_hold").to_numpy()
    transition_accuracy = (
        float(np.mean(test_prediction[transition_mask] == y_eval.iloc[test_idx].to_numpy()[transition_mask]))
        if np.any(transition_mask)
        else float("nan")
    )

    all_trials = train_trials | validation_trials | test_trials
    x_deploy, y_deploy, _meta_deploy, deploy_stats = make_features(
        filtered,
        fs,
        str(best_row["normalization"]),
        all_trials,
        int(best_row["window_ms"]),
        str(best_row["feature_set"]),
        overlap,
    )
    deployment_bundle = refit_with_selected_thresholds(selected_template, x_deploy, y_deploy, best["bundle"])

    split_summary = {
        "method": "complete protocol stages; overlapping windows never cross splits",
        "train_trials": sorted(train_trials),
        "validation_trials": sorted(validation_trials),
        "test_trials": sorted(test_trials),
    }
    deployment_artifact = artifact(
        row=best_row,
        bundle=deployment_bundle,
        feature_columns=list(x_deploy.columns),
        stats=deploy_stats,
        fs=fs,
        split={**split_summary, "model_fit_trials": sorted(all_trials), "role": "realtime deployment"},
        role="deployment",
    )
    deployment_name = "personal_fast_model.pkl" if mode == "fast" else "personal_model.pkl"
    save_pickle(output_dir / deployment_name, deployment_artifact)
    for legacy_name in ("personal_evaluation_model.pkl", "personal_grid_best_model.pkl"):
        (output_dir / legacy_name).unlink(missing_ok=True)

    results_name = "personal_training_results.csv" if mode == "fast" else "personal_grid_results.csv"
    summary_name = "personal_training_summary.json" if mode == "fast" else "personal_grid_summary.json"
    pd.DataFrame(rows).sort_values("validation_balanced_accuracy", ascending=False).to_csv(output_dir / results_name, index=False)
    labels = sorted(y_eval.iloc[test_idx].unique())
    summary = {
        "training_architecture": "stage_split_select_evaluate_then_retrain_all",
        "mode": mode,
        "session_dir": str(session_dir),
        "candidate_count": len(rows),
        "sampling_rate_hz": fs,
        "best_result": {
            **best_row,
            "test_balanced_accuracy": test_ba,
            "test_transition_accuracy": transition_accuracy,
            "test_windows": len(test_idx),
        },
        "split": split_summary,
        "recording_quality_audit": quality_audit,
        "test_labels": labels,
        "test_confusion_matrix": confusion_matrix(y_eval.iloc[test_idx], test_prediction, labels=labels).tolist(),
        "artifacts": {
            "deployment_model": str(output_dir / deployment_name),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / summary_name, summary)
    return summary


def main(default_mode: str | None = None) -> None:
    """Run the module's command-line or graphical application entry point."""
    args = parse_args(default_mode)
    summary = run_training(args.session_dir, args.mode, args.random_state, args.overlap)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
