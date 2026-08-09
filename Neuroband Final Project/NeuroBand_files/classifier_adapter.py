"""Runtime adapter for trained NeuroBand gesture-classification models.

This module presents one inference interface for the model formats produced during
the project. It validates model metadata, applies the same EMG filtering and
normalization used during training, extracts fused features from the three uMyo
sensors, and converts estimator outputs into gesture probabilities. Keeping this
logic in one place prevents training and realtime preprocessing from drifting apart.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import math
import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, lfilter, sosfilt, sosfiltfilt, tf2sos

try:
    import joblib
except Exception:  # pragma: no cover - GUI can still run without model support.
    joblib = None


class IdentityScaler:
    """Represent the IdentityScaler component and keep its related state and behavior together."""
    def transform(self, values):
        """Perform the transform operation used by the IdentityScaler workflow."""
        return values


DISABLED_GESTURES = {"side_flex"}


@dataclass
class PredictionResult:
    """Represent the PredictionResult component and keep its related state and behavior together."""
    gesture: str = "No model"
    confidence: float = 0.0
    stable_gesture: str = "No model"
    is_uncertain: bool = True
    error: str = ""
    debug_info: str = ""
    probabilities: dict[str, float] | None = None


class FeatureExtractor:
    """Represent the FeatureExtractor component and keep its related state and behavior together."""
    def __init__(self, fs: float = 1100.0, selected_channels: Iterable[int] | None = None):
        """Initialize the FeatureExtractor instance and its runtime state."""
        self.fs = fs
        self.selected_channels = list(selected_channels or range(8))

    def update_channels(self, selected_channels: Iterable[int]) -> None:
        """Refresh channels for the current FeatureExtractor workflow."""
        self.selected_channels = list(selected_channels)

    def _bandpass(self, data: np.ndarray) -> np.ndarray:
        """Perform the bandpass operation used by the FeatureExtractor workflow."""
        high_cut = min(500.0, (0.5 * self.fs) - 1.0)
        low_cut = 35.0
        if high_cut <= low_cut:
            return data
        nyq = 0.5 * self.fs
        b, a = butter(4, [low_cut / nyq, high_cut / nyq], btype="band")
        filtered = lfilter(b, a, data)
        b_notch, a_notch = iirnotch(50.0 / nyq, 30)
        return lfilter(b_notch, a_notch, filtered)

    def make_signal(self, emg_window: np.ndarray) -> np.ndarray:
        """Create and configure signal for the current FeatureExtractor workflow."""
        if emg_window.ndim == 1:
            signal = emg_window.astype(float)
        else:
            valid_channels = [ch for ch in self.selected_channels if 0 <= ch < emg_window.shape[1]]
            if not valid_channels:
                valid_channels = list(range(emg_window.shape[1]))
            signal = np.nanmean(emg_window[:, valid_channels].astype(float), axis=1)
        return self._bandpass(np.nan_to_num(signal))

    def extract(self, emg_window: np.ndarray) -> np.ndarray:
        """Perform the extract operation used by the FeatureExtractor workflow."""
        signal = self.make_signal(emg_window)
        if signal.size < 2:
            return np.zeros((1, 5), dtype=float)
        rms = float(np.sqrt(np.mean(signal ** 2)))
        std = float(np.std(signal))
        max_abs = float(np.max(np.abs(signal)))
        zero_crossings = float(((signal[:-1] * signal[1:]) < 0).sum())
        waveform_length = float(np.sum(np.abs(np.diff(signal))))
        return np.array([[rms, std, max_abs, zero_crossings, waveform_length]], dtype=float)

    @staticmethod
    def _zero_crossings(values: np.ndarray) -> float:
        """Perform the zero crossings operation used by the FeatureExtractor workflow."""
        signs = np.sign(values)
        signs[signs == 0] = 1
        return float(np.sum(signs[:-1] * signs[1:] < 0))

    @staticmethod
    def _slope_sign_changes(values: np.ndarray) -> float:
        """Perform the slope sign changes operation used by the FeatureExtractor workflow."""
        if len(values) < 3:
            return 0.0
        diffs = np.diff(values)
        return float(np.sum(diffs[:-1] * diffs[1:] < 0))

    @classmethod
    def _emg_features(cls, values: np.ndarray) -> dict[str, float]:
        """Perform the emg features operation used by the FeatureExtractor workflow."""
        values = np.nan_to_num(np.asarray(values, dtype=float).reshape(-1))
        if values.size < 2:
            values = np.pad(values, (0, max(0, 2 - values.size)))
        return {
            "emg_rms": float(np.sqrt(np.mean(np.square(values)))),
            "emg_mav": float(np.mean(np.abs(values))),
            "emg_zc": cls._zero_crossings(values),
            "emg_ssc": cls._slope_sign_changes(values),
            "emg_wl": float(np.sum(np.abs(np.diff(values)))),
            "emg_var": float(np.var(values)),
        }

    def _preprocess_inter_subject_emg(
        self,
        sensor_id: str,
        values: np.ndarray,
        normalization_stats: dict[str, dict[str, float]],
    ) -> np.ndarray:
        """Perform the preprocess inter subject emg operation used by the FeatureExtractor workflow."""
        values = np.nan_to_num(np.asarray(values, dtype=float).reshape(-1))
        values = self._bandpass_inter_subject(values)
        stats = normalization_stats.get(sensor_id, {})
        mean = float(stats.get("mean_after_bpf", 0.0))
        std = float(stats.get("std_after_bpf", 1.0))
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        return (values - mean) / std

    def _bandpass_inter_subject(self, data: np.ndarray) -> np.ndarray:
        """Perform the bandpass inter subject operation used by the FeatureExtractor workflow."""
        high_cut = min(500.0, (0.5 * self.fs) - 1.0)
        low_cut = 35.0
        if high_cut <= low_cut:
            return data
        nyq = 0.5 * self.fs
        sos = butter(4, [low_cut / nyq, high_cut / nyq], btype="bandpass", output="sos")
        if 50.0 < nyq:
            notch_b, notch_a = iirnotch(50.0 / nyq, 30.0)
            sos = np.vstack([sos, tf2sos(notch_b, notch_a)])
        if data.size < 24:
            return sosfilt(sos, data)
        try:
            return sosfiltfilt(sos, data)
        except ValueError:
            return sosfilt(sos, data)

    @staticmethod
    def _spectrum_features(values: np.ndarray) -> dict[str, float]:
        """Perform the spectrum features operation used by the FeatureExtractor workflow."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.size == 0:
            values = np.zeros((1, 4), dtype=float)
        padded = np.zeros((values.shape[0], 4), dtype=float)
        padded[:, : min(values.shape[1], 4)] = values[:, :4]
        return {
            "spectrum_1_mean": float(np.mean(padded[:, 1])),
            "spectrum_2_mean": float(np.mean(padded[:, 2])),
            "spectrum_3_mean": float(np.mean(padded[:, 3])),
        }

    def extract_fused_sensor_features(
        self,
        sensor_windows: dict[str, dict[str, np.ndarray]],
        feature_columns: list[str],
    ) -> pd.DataFrame:
        """Extract fused sensor features for the current FeatureExtractor workflow."""
        row: dict[str, float] = {}
        for sensor_id, payload in sensor_windows.items():
            emg_features = self._emg_features(payload.get("emg", np.array([], dtype=float)))
            spectrum_features = self._spectrum_features(payload.get("spectrum", np.zeros((1, 4), dtype=float)))
            for name, value in {**emg_features, **spectrum_features}.items():
                row[f"{sensor_id}_{name}"] = value
        return pd.DataFrame([{column: float(row.get(column, 0.0)) for column in feature_columns}])

    def extract_inter_subject_fused_features(
        self,
        sensor_windows: dict[str, dict[str, np.ndarray]],
        feature_columns: list[str],
        normalization_stats: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """Extract inter subject fused features for the current FeatureExtractor workflow."""
        row: dict[str, float] = {}
        for sensor_id, payload in sensor_windows.items():
            emg_values = self._preprocess_inter_subject_emg(
                sensor_id,
                payload.get("emg", np.array([], dtype=float)),
                normalization_stats,
            )
            emg_features = self._emg_features(emg_values)
            spectrum_features = self._spectrum_features(payload.get("spectrum", np.zeros((1, 4), dtype=float)))
            for name, value in {**emg_features, **spectrum_features}.items():
                row[f"{sensor_id}_{name}"] = value
        return pd.DataFrame([{column: float(row.get(column, 0.0)) for column in feature_columns}])

    @staticmethod
    def _haar_features(values: np.ndarray, levels: int = 3) -> dict[str, float]:
        """Perform the haar features operation used by the FeatureExtractor workflow."""
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

    @classmethod
    def _personal_stage_signal_features(cls, values: np.ndarray, feature_set: str) -> dict[str, float]:
        """Perform the personal stage signal features operation used by the FeatureExtractor workflow."""
        values = np.nan_to_num(np.asarray(values, dtype=float).reshape(-1))
        diff = np.diff(values)
        features = {
            "rms": float(np.sqrt(np.mean(values ** 2))),
            "mav": float(np.mean(np.abs(values))),
            "zc": cls._zero_crossings(values),
            "ssc": float(np.sum(diff[:-1] * diff[1:] < 0)) if diff.size > 1 else 0.0,
            "wl": float(np.sum(np.abs(diff))),
            "var": float(np.var(values)),
        }
        if feature_set == "wavelet":
            features.update(cls._haar_features(values))
        return features

    def extract_personal_stage_fused_features(
        self,
        sensor_windows: dict[str, dict[str, np.ndarray]],
        feature_columns: list[str],
        normalization_stats: dict[str, dict[str, float]],
        feature_set: str,
    ) -> pd.DataFrame:
        """Extract personal stage fused features for the current FeatureExtractor workflow."""
        row: dict[str, float] = {}
        for sensor_id, payload in sensor_windows.items():
            values = self._bandpass_inter_subject(payload.get("emg", np.array([], dtype=float)))
            stats = normalization_stats.get(sensor_id, {})
            center = float(stats.get("center", 0.0))
            scale = max(float(stats.get("scale", 1.0)), 1e-9)
            values = (values - center) / scale
            for name, value in self._personal_stage_signal_features(values, feature_set).items():
                row[f"{sensor_id}_{name}"] = value

            spectrum = np.asarray(payload.get("spectrum", np.zeros((1, 4), dtype=float)), dtype=float)
            if spectrum.ndim == 1:
                spectrum = spectrum.reshape(1, -1)
            for index, name in enumerate(["sp1", "sp2", "sp3"], start=1):
                row[f"{sensor_id}_{name}_mean"] = float(np.mean(spectrum[:, index])) if spectrum.shape[1] > index else 0.0

        sensor_ids = list(self.model_sensor_order if hasattr(self, "model_sensor_order") else sensor_windows)
        for first, second in [(sensor_ids[0], sensor_ids[1]), (sensor_ids[0], sensor_ids[2]), (sensor_ids[1], sensor_ids[2])]:
            first_rms = float(row.get(f"{first}_rms", 0.0))
            second_rms = float(row.get(f"{second}_rms", 0.0))
            row[f"{first}_{second}_rms_diff"] = first_rms - second_rms
            row[f"{first}_{second}_rms_ratio"] = first_rms / max(abs(second_rms), 1e-9)
        return pd.DataFrame([{column: float(row.get(column, 0.0)) for column in feature_columns}])


class GestureClassifierAdapter:
    """Represent the GestureClassifierAdapter component and keep its related state and behavior together."""
    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        fs: float = 1100.0,
        selected_channels: Iterable[int] | None = None,
        confidence_threshold: float = 0.55,
        stability_count: int = 3,
    ):
        """Initialize the GestureClassifierAdapter instance and its runtime state."""
        self.model = None
        self.scaler = None
        self.model_path: Optional[Path] = None
        self.scaler_path: Optional[Path] = None
        self.last_error = ""
        self.confidence_threshold = confidence_threshold
        self.extractor = FeatureExtractor(fs=fs, selected_channels=selected_channels)
        self.stability_count = stability_count
        self._recent: list[str] = []
        if model_path:
            self.load(model_path, scaler_path)

    @property
    def is_loaded(self) -> bool:
        """Determine whether loaded for the current GestureClassifierAdapter workflow."""
        return self.model is not None

    @staticmethod
    def _load_artifact(path: Path):
        """Load and validate artifact for the current GestureClassifierAdapter workflow."""
        if joblib is not None:
            try:
                return joblib.load(path)
            except Exception:
                if path.suffix.lower() not in {".pkl", ".pickle"}:
                    raise
        with path.open("rb") as file:
            return pickle.load(file)

    def load(self, model_path: str | Path, scaler_path: str | Path | None = None) -> bool:
        """Perform the load operation used by the GestureClassifierAdapter workflow."""
        self.last_error = ""
        try:
            model_path = Path(model_path).expanduser().resolve()
            if not model_path.exists():
                self.last_error = f"model file not found: {model_path}"
                self.model = None
                return False
            self.model = self._load_artifact(model_path)
            self.model_path = model_path
            self._recent.clear()
            if isinstance(self.model, dict) and self.model.get("sampling_rate_hz"):
                self.extractor.fs = float(self.model["sampling_rate_hz"])
            self.scaler = None
            if scaler_path:
                scaler_path = Path(scaler_path).expanduser().resolve()
                if scaler_path.exists():
                    self.scaler = self._load_artifact(scaler_path)
                    self.scaler_path = scaler_path
            return True
        except Exception as exc:
            self.model = None
            self.scaler = None
            self.last_error = str(exc)
            return False

    def window_ms(self, default: int = 500) -> int:
        """Perform the window ms operation used by the GestureClassifierAdapter workflow."""
        if isinstance(self.model, dict):
            try:
                return int(self.model.get("window_ms", default))
            except (TypeError, ValueError):
                return default
        return default

    def sampling_rate_hz(self, default: float = 1100.0) -> float:
        """Perform the sampling rate hz operation used by the GestureClassifierAdapter workflow."""
        if isinstance(self.model, dict):
            try:
                return float(self.model.get("sampling_rate_hz", default))
            except (TypeError, ValueError):
                return default
        return default

    def update_settings(self, selected_channels: Iterable[int], confidence_threshold: float) -> None:
        """Refresh settings for the current GestureClassifierAdapter workflow."""
        self.extractor.update_channels(selected_channels)
        self.confidence_threshold = confidence_threshold

    def predict(self, emg_window: np.ndarray) -> PredictionResult:
        """Perform the predict operation used by the GestureClassifierAdapter workflow."""
        if not self.is_loaded:
            return PredictionResult(error=self.last_error)
        try:
            probabilities = None
            if isinstance(self.model, dict) and self.model.get("artifact_type") == "hierarchical_extra_trees_fused":
                return self._predict_hierarchical_extra_trees(emg_window)
            if isinstance(self.model, dict) and self.model.get("artifact_type") in {
                "hierarchical_inter_subject_fused",
                "hierarchical_personal_stage_fused_v2",
            }:
                return self._predict_hierarchical_inter_subject(emg_window)

            X = self.extractor.extract(np.asarray(emg_window))
            if self.scaler is not None:
                X = self.scaler.transform(X)

            if hasattr(self.model, "predict_proba"):
                raw = self.model.predict_proba(X)[0]
                classes = [str(c) for c in getattr(self.model, "classes_", range(len(raw)))]
                probabilities = dict(zip(classes, [float(p) for p in raw]))
                best_idx = int(np.argmax(raw))
                gesture = classes[best_idx]
                confidence = float(raw[best_idx])
            else:
                gesture = str(self.model.predict(X)[0])
                confidence = 1.0

            uncertain = confidence < self.confidence_threshold
            displayed = "Uncertain" if uncertain else gesture
            stable = self._stable(displayed)
            return PredictionResult(
                gesture=displayed,
                confidence=confidence,
                stable_gesture=stable,
                is_uncertain=uncertain,
                probabilities=probabilities,
            )
        except Exception as exc:
            return PredictionResult(gesture="Error", stable_gesture="Error", error=str(exc))

    def _predict_hierarchical_extra_trees(self, sensor_windows) -> PredictionResult:
        """Predict hierarchical extra trees for the current GestureClassifierAdapter workflow."""
        if not isinstance(sensor_windows, dict):
            return PredictionResult(
                gesture="Error",
                stable_gesture="Error",
                error="hierarchical model expects fused sensor windows",
            )
        feature_columns = list(self.model["feature_columns"])
        X = self.extractor.extract_fused_sensor_features(sensor_windows, feature_columns)
        binary_model = self.model["binary_model"]
        gesture_model = self.model["gesture_model"]
        gesture_encoder = self.model["gesture_encoder"]
        classes = list(self.model["classes"])

        binary_proba = binary_model.predict_proba(X)[0]
        binary_classes = list(binary_model.classes_)
        rest_idx = binary_classes.index(0)
        gesture_idx = binary_classes.index(1)
        rest_prob = float(binary_proba[rest_idx])
        any_gesture_prob = float(binary_proba[gesture_idx])

        gesture_proba = gesture_model.predict_proba(X)[0]
        gesture_classes = [str(label) for label in gesture_encoder.inverse_transform(gesture_model.classes_)]
        probabilities = {label: 0.0 for label in classes}
        probabilities["at_rest"] = rest_prob
        for label, prob in zip(gesture_classes, gesture_proba):
            probabilities[label] = any_gesture_prob * float(prob)

        gesture = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[gesture])
        if gesture in DISABLED_GESTURES:
            gesture = "Uncertain"
            uncertain = True
            displayed = "Uncertain"
            stable = self._stable(displayed)
            return PredictionResult(
                gesture=displayed,
                confidence=confidence,
                stable_gesture=stable,
                is_uncertain=uncertain,
                probabilities=probabilities,
            )
        uncertain = confidence < self.confidence_threshold
        displayed = "Uncertain" if uncertain else gesture
        stable = self._stable(displayed)
        return PredictionResult(
            gesture=displayed,
            confidence=confidence,
            stable_gesture=stable,
            is_uncertain=uncertain,
            probabilities=probabilities,
        )

    def _predict_hierarchical_inter_subject(self, sensor_windows) -> PredictionResult:
        """Predict hierarchical inter subject for the current GestureClassifierAdapter workflow."""
        if not isinstance(sensor_windows, dict):
            return PredictionResult(
                gesture="Error",
                stable_gesture="Error",
                error="inter-subject hierarchical model expects fused sensor windows",
            )
        feature_columns = list(self.model["feature_columns"])
        normalization_stats = self.model.get("raw_emg_normalization_stats", {})
        if self.model.get("artifact_type") == "hierarchical_personal_stage_fused_v2":
            self.extractor.model_sensor_order = list(self.model.get("sensor_order", sensor_windows))
            X = self.extractor.extract_personal_stage_fused_features(
                sensor_windows,
                feature_columns,
                normalization_stats,
                str(self.model.get("feature_set", "baseline")),
            )
        else:
            X = self.extractor.extract_inter_subject_fused_features(sensor_windows, feature_columns, normalization_stats)

        binary_model = self.model["binary_model"]
        binary_threshold = float(self.model.get("binary_threshold", 0.5))
        gesture_model = self.model["gesture_model"]
        gesture_encoder = self.model["gesture_encoder"]
        gesture_thresholds = np.asarray(self.model["gesture_thresholds"], dtype=float)
        classes = list(self.model["classes"])

        binary_score, rest_prob, any_gesture_prob = self._binary_score_and_display_probabilities(
            binary_model,
            X,
            positive_label=1,
        )
        probabilities = {label: 0.0 for label in classes}
        probabilities["at_rest"] = rest_prob

        if binary_score < binary_threshold:
            gesture = "at_rest"
        else:
            gesture_scores, gesture_proba = self._multiclass_scores_and_display_probabilities(
                gesture_model,
                X,
                len(gesture_encoder.classes_),
            )
            gesture_classes = [str(label) for label in gesture_encoder.inverse_transform(np.arange(len(gesture_encoder.classes_)))]
            adjusted = gesture_scores - gesture_thresholds.reshape(-1)
            best_idx = int(np.argmax(adjusted))
            gesture = gesture_classes[best_idx]
            for label, prob in zip(gesture_classes, gesture_proba):
                probabilities[label] = any_gesture_prob * float(prob)

        confidence = float(max(probabilities.values())) if probabilities else 0.0
        uncertain = confidence < self.confidence_threshold or gesture in DISABLED_GESTURES
        displayed = "Uncertain" if uncertain else gesture
        stable = self._stable(displayed)
        debug_info = self._format_probability_debug(probabilities)
        return PredictionResult(
            gesture=displayed,
            confidence=confidence,
            stable_gesture=stable,
            is_uncertain=uncertain,
            debug_info=debug_info,
            probabilities=probabilities,
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Perform the sigmoid operation used by the GestureClassifierAdapter workflow."""
        return float(1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0))))

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        """Perform the softmax operation used by the GestureClassifierAdapter workflow."""
        values = np.asarray(values, dtype=float)
        values = values - np.nanmax(values)
        exp_values = np.exp(np.clip(values, -50.0, 50.0))
        total = float(np.sum(exp_values))
        if not np.isfinite(total) or total <= 0.0:
            return np.ones_like(exp_values, dtype=float) / max(1, len(exp_values))
        return exp_values / total

    @classmethod
    def _binary_score_and_display_probabilities(
        cls,
        model: object,
        x_values: pd.DataFrame,
        positive_label: int = 1,
    ) -> tuple[float, float, float]:
        """Perform the binary score and display probabilities operation used by the GestureClassifierAdapter workflow."""
        classes = list(getattr(model, "classes_", [0, 1]))
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(x_values))[0]
            positive_idx = classes.index(positive_label)
            positive_prob = float(proba[positive_idx])
            negative_prob = float(1.0 - positive_prob)
            negative_label = 0 if positive_label == 1 else 1
            if negative_label in classes:
                negative_prob = float(proba[classes.index(negative_label)])
            return positive_prob, negative_prob, positive_prob

        scores = np.asarray(model.decision_function(x_values))
        if scores.ndim == 0:
            score = float(scores)
        elif scores.ndim == 1:
            score = float(scores[0])
            if classes and classes[-1] != positive_label:
                score *= -1.0
        else:
            score = float(scores[0, classes.index(positive_label)])
        positive_prob = cls._sigmoid(score)
        return score, 1.0 - positive_prob, positive_prob

    @classmethod
    def _multiclass_scores_and_display_probabilities(
        cls,
        model: object,
        x_values: pd.DataFrame,
        class_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform the multiclass scores and display probabilities operation used by the GestureClassifierAdapter workflow."""
        scores = np.zeros(class_count, dtype=float)
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(x_values))[0]
            probabilities = np.zeros(class_count, dtype=float)
            for output_idx, class_idx in enumerate(getattr(model, "classes_", range(len(proba)))):
                scores[int(class_idx)] = float(proba[output_idx])
                probabilities[int(class_idx)] = float(proba[output_idx])
            return scores, probabilities

        raw_scores = np.asarray(model.decision_function(x_values))
        if raw_scores.ndim == 1:
            if class_count == 2:
                aligned = np.array([-float(raw_scores[0]), float(raw_scores[0])], dtype=float)
            else:
                aligned = np.asarray(raw_scores, dtype=float)
        else:
            aligned = np.zeros(class_count, dtype=float)
            for output_idx, class_idx in enumerate(getattr(model, "classes_", range(raw_scores.shape[1]))):
                aligned[int(class_idx)] = float(raw_scores[0, output_idx])
        return aligned, cls._softmax(aligned)

    @staticmethod
    def _probability_gap_confidence(probabilities: dict[str, float]) -> float:
        """Perform the probability gap confidence operation used by the GestureClassifierAdapter workflow."""
        values = np.asarray(list(probabilities.values()), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        values = np.sort(values)[::-1]
        top = float(values[0])
        runner_up = float(values[1]) if values.size > 1 else 0.0
        return float(np.clip(top - runner_up, 0.0, 1.0))

    @staticmethod
    def _format_probability_debug(probabilities: dict[str, float], limit: int = 3) -> str:
        """Perform the format probability debug operation used by the GestureClassifierAdapter workflow."""
        if not probabilities:
            return ""
        top_items = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:limit]
        return "Top: " + ", ".join(f"{label} {prob * 100:.0f}%" for label, prob in top_items)

    def _stable(self, gesture: str) -> str:
        """Perform the stable operation used by the GestureClassifierAdapter workflow."""
        if gesture in {"Uncertain", "Error", "No model"}:
            self._recent.clear()
            return gesture
        self._recent.append(gesture)
        self._recent = self._recent[-self.stability_count:]
        if len(self._recent) < self.stability_count:
            return gesture
        counts = {candidate: self._recent.count(candidate) for candidate in set(self._recent)}
        winner, votes = max(counts.items(), key=lambda item: (item[1], self._recent[::-1].index(item[0]) * -1))
        if votes >= (self.stability_count // 2) + 1:
            return winner
        return gesture
