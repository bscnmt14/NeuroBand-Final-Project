"""Adapt an existing personal model using a short recalibration recording.

The update workflow preserves the original model architecture while recalculating
current-condition normalization statistics and fitting with a mixture of original
and new calibration examples. Validation safeguards compare the updated model with
the base model before saving a versioned artifact and recalibration report.

"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, recall_score


def parse_args() -> argparse.Namespace:
    """Perform the parse args operation used by the update personal model workflow."""
    parser = argparse.ArgumentParser(description="Short current-condition adaptation for an existing personal model.")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--update-session", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--old-windows-per-class", type=int, default=60)
    parser.add_argument("--current-data-fraction", type=float, default=0.35)
    parser.add_argument("--minimum-ba-improvement", type=float, default=0.02)
    parser.add_argument("--minimum-candidate-ba", type=float, default=0.60)
    parser.add_argument("--maximum-recall-regression", type=float, default=0.15)
    return parser.parse_args()


def read_pickle(path: Path) -> dict[str, object]:
    """Read and parse pickle for the current update personal model workflow."""
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or "binary_model" not in value or "gesture_model" not in value:
        raise ValueError("The selected file is not a supported hierarchical personal model.")
    return value


def protocol_trials(session_dir: Path) -> tuple[set[int], set[int]]:
    """Perform the protocol trials operation used by the update personal model workflow."""
    protocol_path = session_dir / "session_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    train_trials = {
        index for index, stage in enumerate(protocol) if str(stage.get("protocol_condition", "")) == "adapt_train"
    }
    validation_trials = {
        index for index, stage in enumerate(protocol) if str(stage.get("protocol_condition", "")) == "adapt_validate"
    }
    if not train_trials or not validation_trials:
        raise ValueError("The update recording does not contain independent adaptation and validation rounds.")
    return train_trials, validation_trials


def current_normalization_stats(personal, filtered: pd.DataFrame, mode: str, trials: set[int]) -> dict[str, dict[str, float | str]]:
    """Perform the current normalization stats operation used by the update personal model workflow."""
    if mode == "none":
        return personal.normalization_stats(filtered, "none", trials)
    return personal.normalization_stats(filtered, mode, trials)


def extract_features(broad, personal, filtered: pd.DataFrame, model: dict[str, object], stats: dict[str, dict[str, float | str]]):
    """Extract features for the current update personal model workflow."""
    normalized = personal.apply_normalization(filtered, stats)
    x, y, meta = broad.extract_fused_windows(
        normalized,
        float(model["sampling_rate_hz"]),
        int(model["window_ms"]),
        0.5,
        float(model.get("trim_edge_ms", 100.0)),
        str(model.get("feature_set", "baseline")),
        str(model.get("raw_normalization", "none")),
    )
    expected = list(model.get("feature_columns") or x.columns)
    missing = [column for column in expected if column not in x.columns]
    if missing:
        raise ValueError(f"Update feature extraction is missing model columns: {missing[:5]}")
    return x.loc[:, expected], y.reset_index(drop=True), meta.reset_index(drop=True)


def representative_rows(x: pd.DataFrame, y: pd.Series, per_class: int, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    """Perform the representative rows operation used by the update personal model workflow."""
    rng = np.random.default_rng(random_state)
    indexes = []
    for label, group_indexes in y.groupby(y).groups.items():
        values = np.asarray(list(group_indexes), dtype=int)
        if len(values) > per_class:
            values = rng.choice(values, size=per_class, replace=False)
        indexes.extend(values.tolist())
    indexes = sorted(indexes)
    return x.iloc[indexes].reset_index(drop=True), y.iloc[indexes].reset_index(drop=True)


def mix_old_and_current(
    old_x: pd.DataFrame,
    old_y: pd.Series,
    current_x: pd.DataFrame,
    current_y: pd.Series,
    current_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, dict[str, int]]]:
    """Perform the mix old and current operation used by the update personal model workflow."""
    rng = np.random.default_rng(random_state)
    x_parts = []
    y_parts = []
    counts: dict[str, dict[str, int]] = {}
    labels = sorted(set(old_y) | set(current_y))
    for label in labels:
        old_indexes = np.flatnonzero(old_y.to_numpy() == label)
        current_indexes = np.flatnonzero(current_y.to_numpy() == label)
        if not len(old_indexes):
            raise ValueError(f"The original calibration recording does not contain {label!r}.")
        if not len(current_indexes):
            # Keep the original class representation when one short adaptation
            # stage is unusable. Its independent validation stage remains held
            # out, so the update is still evaluated without leakage.
            x_parts.append(old_x.iloc[old_indexes])
            y_parts.append(old_y.iloc[old_indexes])
            counts[str(label)] = {"original": int(len(old_indexes)), "current": 0}
            continue
        current_target = max(len(current_indexes), int(round(len(old_indexes) * current_fraction / max(1e-6, 1.0 - current_fraction))))
        chosen_current = rng.choice(current_indexes, size=current_target, replace=current_target > len(current_indexes))
        x_parts.extend([old_x.iloc[old_indexes], current_x.iloc[chosen_current]])
        y_parts.extend([old_y.iloc[old_indexes], current_y.iloc[chosen_current]])
        counts[str(label)] = {"original": int(len(old_indexes)), "current": int(current_target)}
    mixed_x = pd.concat(x_parts, ignore_index=True)
    mixed_y = pd.concat(y_parts, ignore_index=True)
    order = rng.permutation(len(mixed_x))
    return mixed_x.iloc[order].reset_index(drop=True), mixed_y.iloc[order].reset_index(drop=True), counts


def model_template(base: dict[str, object]):
    """Perform the model template operation used by the update personal model workflow."""
    return clone(base["gesture_model"])


def prediction_metrics(predict_hierarchical, bundle: dict[str, object], x: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """Perform the prediction metrics operation used by the update personal model workflow."""
    prediction = predict_hierarchical(bundle, x)
    labels = sorted(y.unique())
    recalls = recall_score(y, prediction, labels=labels, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "per_class_recall": {str(label): float(value) for label, value in zip(labels, recalls)},
        "predictions": prediction,
    }


def rest_rms(filtered: pd.DataFrame, train_trials: set[int], sensors: list[str], emg_columns: list[str]) -> dict[str, float]:
    """Perform the rest rms operation used by the update personal model workflow."""
    result = {}
    rest = filtered[(filtered["trial_index"].isin(train_trials)) & (filtered["gesture_label"] == "at_rest")]
    for sensor in sensors:
        values = rest.loc[rest["unit_id"] == sensor, emg_columns].to_numpy(float).reshape(-1)
        result[sensor] = float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan")
    return result


def main() -> None:
    """Run the module's command-line or graphical application entry point."""
    args = parse_args()
    app_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(app_dir))
    import personal_stage_training as personal
    import training_data_pipeline as broad
    from model_training import predict_hierarchical_local

    base_path = args.base_model.resolve()
    update_session = args.update_session.resolve()
    base = read_pickle(base_path)
    original_session = base_path.parent.parent
    original_raw_dir = original_session / "raw_recordings"
    if not original_raw_dir.exists():
        raise FileNotFoundError("The original calibration recording is required for safe mixed-data adaptation.")

    train_trials, validation_trials = protocol_trials(update_session)
    fs = float(base.get("sampling_rate_hz", 620.0))
    original_raw = broad.read_existing_recordings(original_raw_dir)
    update_raw = broad.read_existing_recordings(update_session / "raw_recordings")
    original_filtered = broad.filter_all(original_raw, fs)
    update_filtered = broad.filter_all(update_raw, fs)
    mode = str(base.get("raw_normalization", "none"))
    update_stats = current_normalization_stats(personal, update_filtered, mode, train_trials)
    base_stats = base.get("raw_emg_normalization_stats") or current_normalization_stats(
        personal, original_filtered, mode, set(original_filtered["trial_index"].astype(int).unique())
    )

    original_x_current, original_y, _original_meta = extract_features(broad, personal, original_filtered, base, update_stats)
    update_x_current, update_y, update_meta = extract_features(broad, personal, update_filtered, base, update_stats)
    update_x_base, update_y_base, update_meta_base = extract_features(broad, personal, update_filtered, base, base_stats)

    train_mask = update_meta["trial_index"].isin(train_trials).to_numpy()
    validation_mask = update_meta["trial_index"].isin(validation_trials).to_numpy()
    validation_mask_base = update_meta_base["trial_index"].isin(validation_trials).to_numpy()
    if not np.any(train_mask) or not np.any(validation_mask):
        raise ValueError("No fused windows were generated for one of the update rounds.")

    old_x, old_y = representative_rows(original_x_current, original_y, args.old_windows_per_class, args.random_state)
    mixed_x, mixed_y, mix_counts = mix_old_and_current(
        old_x,
        old_y,
        update_x_current.loc[train_mask].reset_index(drop=True),
        update_y.loc[train_mask].reset_index(drop=True),
        args.current_data_fraction,
        args.random_state,
    )
    candidate_evaluation = personal.refit_with_selected_thresholds(model_template(base), mixed_x, mixed_y, base)
    base_metrics = prediction_metrics(
        predict_hierarchical_local,
        base,
        update_x_base.loc[validation_mask_base].reset_index(drop=True),
        update_y_base.loc[validation_mask_base].reset_index(drop=True),
    )
    candidate_metrics = prediction_metrics(
        predict_hierarchical_local,
        candidate_evaluation,
        update_x_current.loc[validation_mask].reset_index(drop=True),
        update_y.loc[validation_mask].reset_index(drop=True),
    )

    recall_regressions = {
        label: base_metrics["per_class_recall"].get(label, 0.0) - candidate_metrics["per_class_recall"].get(label, 0.0)
        for label in base_metrics["per_class_recall"]
    }
    max_regression = max(recall_regressions.values(), default=0.0)
    ba_improvement = float(candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"])
    recommended = (
        candidate_metrics["balanced_accuracy"] >= args.minimum_candidate_ba
        and ba_improvement >= args.minimum_ba_improvement
        and max_regression <= args.maximum_recall_regression
    )

    all_update_x = update_x_current.reset_index(drop=True)
    all_update_y = update_y.reset_index(drop=True)
    deploy_x, deploy_y, deploy_counts = mix_old_and_current(
        old_x,
        old_y,
        all_update_x,
        all_update_y,
        args.current_data_fraction,
        args.random_state + 1,
    )
    deployment_bundle = personal.refit_with_selected_thresholds(model_template(base), deploy_x, deploy_y, base)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"personal_model_update_{stamp}.pkl"
    artifact = copy.deepcopy(base)
    artifact.update(deployment_bundle)
    artifact["artifact_role"] = "adapted_deployment"
    artifact["parent_model_path"] = str(base_path)
    artifact["raw_emg_normalization_stats"] = update_stats
    artifact["adaptation"] = {
        "timestamp": stamp,
        "update_session": str(update_session),
        "base_model_validation": {key: value for key, value in base_metrics.items() if key != "predictions"},
        "candidate_validation": {key: value for key, value in candidate_metrics.items() if key != "predictions"},
        "balanced_accuracy_improvement": ba_improvement,
        "maximum_per_class_recall_regression": max_regression,
        "recommended": recommended,
        "mixture_counts_evaluation": mix_counts,
        "mixture_counts_deployment": deploy_counts,
        "current_condition_rest_rms": rest_rms(update_filtered, train_trials, broad.SENSORS, broad.EMG_COLUMNS),
    }
    output_dir = update_session / "trained_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    session_model_path = output_dir / model_name
    library_model_path = base_path.parent / model_name
    for destination in (session_model_path, library_model_path):
        with destination.open("wb") as handle:
            pickle.dump(artifact, handle)

    summary = {
        "base_model": str(base_path),
        "original_session": str(original_session),
        "update_session": str(update_session),
        "candidate_model": str(library_model_path),
        "session_model_copy": str(session_model_path),
        "base_validation": {key: value for key, value in base_metrics.items() if key != "predictions"},
        "candidate_validation": {key: value for key, value in candidate_metrics.items() if key != "predictions"},
        "balanced_accuracy_improvement": ba_improvement,
        "maximum_per_class_recall_regression": max_regression,
        "promotion_rule": {
            "minimum_candidate_balanced_accuracy": args.minimum_candidate_ba,
            "minimum_ba_improvement": args.minimum_ba_improvement,
            "maximum_recall_regression": args.maximum_recall_regression,
        },
        "missing_current_training_labels": sorted(
            label for label, counts in mix_counts.items() if int(counts.get("current", 0)) == 0
        ),
        "recommended": recommended,
        "note": "The original model was not overwritten. The candidate was retrained on representative original windows plus current-condition data.",
    }
    (update_session / "model_update_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
