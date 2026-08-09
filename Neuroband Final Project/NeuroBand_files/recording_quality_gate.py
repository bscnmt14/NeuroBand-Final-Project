"""Assess signal and recording quality for training and realtime operation.

Quality checks combine sensor availability, packet continuity, signal amplitude,
resting RMS, clipping, and other recording diagnostics. During calibration the
module marks questionable intervals for later review instead of repeatedly stopping
the protocol. During realtime use it provides a safety indication that can suppress
actions when the measured signal is unreliable.

"""

from __future__ import annotations

import json
import time
import csv
from pathlib import Path

import numpy as np
from scipy import signal


class RecordingQualityGate:
    """Stage-level recording integrity and rest-noise quality checks."""

    def __init__(
        self,
        sensor_order: list[str],
        sensor_locations: dict[str, str],
        sampling_rate_hz: float = 620.0,
    ) -> None:
        """Initialize the RecordingQualityGate instance and its runtime state."""
        self.sensor_order = list(sensor_order)
        self.sensor_locations = dict(sensor_locations)
        self.fs = float(sampling_rate_hz)
        self.session_dir: Path | None = None
        self.output_path: Path | None = None
        self.manifest_path: Path | None = None
        self.current_stage_index = -1
        self.current_stage: dict[str, object] | None = None
        self.stage_attempts: dict[int, int] = {}
        self.buffers: dict[str, dict[str, list[object]]] = {}
        self.history: list[dict[str, object]] = []
        self.accepted_by_stage: dict[int, dict[str, object]] = {}

    def start(self, session_dir: Path) -> None:
        """Perform the start operation used by the RecordingQualityGate workflow."""
        self.session_dir = Path(session_dir)
        self.output_path = self.session_dir / "session_quality.json"
        self.manifest_path = self.session_dir / "quality_exclusions.json"
        self.current_stage_index = -1
        self.current_stage = None
        self.stage_attempts = {}
        self.buffers = {}
        self.history = []
        self.accepted_by_stage = {}
        self._save()

    def begin_stage(self, index: int, stage: dict[str, object]) -> None:
        """Perform the begin stage operation used by the RecordingQualityGate workflow."""
        self.current_stage_index = int(index)
        self.current_stage = dict(stage)
        self.stage_attempts[index] = self.stage_attempts.get(index, 0) + 1
        self.buffers = {
            sensor_id: {"emg": [], "arrival_times": [], "rssi": [], "battery_mv": [], "yaw": [], "pitch": [], "roll": []}
            for sensor_id in self.sensor_order
        }

    def add_snapshots(self, snapshots: list[object]) -> None:
        """Add snapshots for the current RecordingQualityGate workflow."""
        if self.current_stage is None:
            return
        arrival_time = time.perf_counter()
        for snapshot in snapshots:
            sensor_id = f"{int(snapshot.unit_id):08X}"
            if sensor_id not in self.buffers:
                continue
            values = np.asarray(snapshot.emg, dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            sensor = self.buffers[sensor_id]
            sensor["emg"].append(values.copy())
            sensor["arrival_times"].append(arrival_time)
            sensor["rssi"].append(float(snapshot.rssi))
            sensor["battery_mv"].append(float(snapshot.battery_mv))
            sensor["yaw"].append(float(snapshot.yaw))
            sensor["pitch"].append(float(snapshot.pitch))
            sensor["roll"].append(float(snapshot.roll))

    @staticmethod
    def _longest_equal_run(values: np.ndarray) -> int:
        """Perform the longest equal run operation used by the RecordingQualityGate workflow."""
        if values.size == 0:
            return 0
        changes = np.flatnonzero(np.diff(values) != 0) + 1
        boundaries = np.concatenate(([0], changes, [len(values)]))
        return int(np.max(np.diff(boundaries)))

    def _filtered_metrics(self, values: np.ndarray) -> dict[str, float]:
        """Perform the filtered metrics operation used by the RecordingQualityGate workflow."""
        nyquist = self.fs / 2.0
        high_hz = min(500.0, nyquist - 1.0)
        bandpass = signal.butter(4, [35.0 / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
        notch_b, notch_a = signal.iirnotch(50.0 / nyquist, 30.0)
        notch = signal.tf2sos(notch_b, notch_a)
        minimum = 3 * (2 * (len(bandpass) + len(notch)) + 1)
        if len(values) > minimum:
            bandpassed = signal.sosfiltfilt(bandpass, values)
            filtered = signal.sosfiltfilt(notch, bandpassed)
        else:
            bandpassed = signal.sosfilt(bandpass, values)
            filtered = signal.sosfilt(notch, bandpassed)
        rms = float(np.sqrt(np.mean(filtered**2))) if filtered.size else 0.0
        third = max(1, len(filtered) // 3)
        start_rms = float(np.sqrt(np.mean(filtered[:third] ** 2)))
        end_rms = float(np.sqrt(np.mean(filtered[-third:] ** 2)))
        drift_ratio = abs(end_rms - start_rms) / max(rms, 1e-9)
        frequencies, power = signal.welch(bandpassed, fs=self.fs, nperseg=min(256, len(bandpassed)))
        useful = (frequencies >= 35.0) & (frequencies <= high_hz)
        mains = (frequencies >= 48.0) & (frequencies <= 52.0)
        line_ratio = float(np.sum(power[mains]) / max(float(np.sum(power[useful])), 1e-12))
        return {
            "filtered_rms": rms,
            "start_rms": start_rms,
            "end_rms": end_rms,
            "rms_drift_ratio": drift_ratio,
            "pre_notch_50hz_power_ratio": line_ratio,
        }

    def _rest_references(self) -> dict[str, list[float]]:
        """Perform the rest references operation used by the RecordingQualityGate workflow."""
        references = {sensor_id: [] for sensor_id in self.sensor_order}
        for record in self.accepted_by_stage.values():
            if not record.get("quiet_rest") or record.get("status") != "PASS" or record.get("decision") != "accepted":
                continue
            for sensor_id, metrics in record.get("sensor_metrics", {}).items():
                if sensor_id in references:
                    references[sensor_id].append(float(metrics["filtered_rms"]))
        return references

    def _rest_balance_references(self) -> dict[str, list[float]]:
        """Perform the rest balance references operation used by the RecordingQualityGate workflow."""
        references = {sensor_id: [] for sensor_id in self.sensor_order}
        for record in self.accepted_by_stage.values():
            if not record.get("quiet_rest") or record.get("status") != "PASS" or record.get("decision") != "accepted":
                continue
            metrics = record.get("sensor_metrics", {})
            if not all(sensor_id in metrics and "filtered_rms" in metrics[sensor_id] for sensor_id in self.sensor_order):
                continue
            rms_values = np.asarray([max(float(metrics[sensor_id]["filtered_rms"]), 1e-9) for sensor_id in self.sensor_order])
            geometric_mean = float(np.exp(np.mean(np.log(rms_values))))
            for sensor_id, rms in zip(self.sensor_order, rms_values):
                references[sensor_id].append(float(np.log(rms / geometric_mean)))
        return references

    @staticmethod
    def _robust_limit(values: list[float]) -> tuple[float, float, float] | None:
        """Perform the robust limit operation used by the RecordingQualityGate workflow."""
        if len(values) < 3:
            return None
        array = np.asarray(values, dtype=float)
        median = float(np.median(array))
        robust_sigma = 1.4826 * float(np.median(np.abs(array - median)))
        robust_sigma = max(robust_sigma, 0.10 * max(abs(median), 1.0), 1.0)
        return median, robust_sigma, median + 3.0 * robust_sigma

    def evaluate_current_stage(self, expected_duration_s: float) -> dict[str, object] | None:
        """Evaluate current stage for the current RecordingQualityGate workflow."""
        if self.current_stage is None or self.current_stage_index < 0:
            return None
        label = str(self.current_stage.get("gesture_label", "at_rest"))
        stage_kind = str(self.current_stage.get("kind", ""))
        quiet_rest = stage_kind == "rest" or stage_kind.endswith("_rest")
        imu_motion_stage = stage_kind.startswith(("pointer_", "rest_roll_", "fist_roll_", "fast_pitch_", "fast_yaw_", "normal_non_command_"))
        expected_samples = max(1, int(round(self.fs * float(expected_duration_s))))
        references = self._rest_references()
        balance_references = self._rest_balance_references()
        hard_failures: list[str] = []
        exclusion_reasons: list[str] = []
        review_reasons: list[str] = []
        informational_reasons: list[str] = []
        soft_failures: list[str] = []
        warnings: list[str] = []
        sensor_metrics: dict[str, dict[str, object]] = {}

        for sensor_id in self.sensor_order:
            sensor = self.buffers.get(sensor_id, {})
            chunks = sensor.get("emg", [])
            values = np.concatenate(chunks) if chunks else np.empty(0, dtype=float)
            arrivals = np.asarray(sensor.get("arrival_times", []), dtype=float)
            location = self.sensor_locations.get(sensor_id, sensor_id)
            if not values.size:
                hard_failures.append(f"{location}: no EMG samples")
                sensor_metrics[sensor_id] = {"location": location, "sample_count": 0}
                continue

            arrival_diffs = np.diff(arrivals)
            positive_diffs = arrival_diffs[arrival_diffs > 0]
            median_gap = float(np.median(positive_diffs)) if positive_diffs.size else 0.0
            max_gap = float(np.max(positive_diffs)) if positive_diffs.size else 0.0
            gap_limit = max(0.25, 6.0 * median_gap)
            longest_equal_run = self._longest_equal_run(values)
            clipping_fraction = float(np.mean(np.abs(values) >= 32760.0))
            metrics: dict[str, object] = {
                "location": location,
                "sample_count": int(values.size),
                "expected_sample_count": expected_samples,
                "sample_coverage": float(values.size / expected_samples),
                "median_arrival_gap_s": median_gap,
                "max_arrival_gap_s": max_gap,
                "arrival_gap_limit_s": gap_limit,
                "longest_equal_run": longest_equal_run,
                "unique_value_count": int(np.unique(values).size),
                "clipping_fraction": clipping_fraction,
                "mean_rssi": float(np.mean(sensor.get("rssi", [0.0]))),
                "mean_battery_mv": float(np.mean(sensor.get("battery_mv", [0.0]))),
                **self._filtered_metrics(values),
            }
            imu_ranges = {}
            for axis in ("yaw", "pitch", "roll"):
                axis_values = np.asarray(sensor.get(axis, []), dtype=float)
                finite_axis = axis_values[np.isfinite(axis_values)]
                imu_ranges[axis] = float(np.ptp(finite_axis)) if finite_axis.size else 0.0
                metrics[f"imu_{axis}_range"] = imu_ranges[axis]
                if axis_values.size and finite_axis.size != axis_values.size:
                    hard_failures.append(f"{location}: invalid {axis} values")
                if finite_axis.size and np.any(np.abs(finite_axis) >= 32760.0):
                    hard_failures.append(f"{location}: {axis} saturation")
            sensor_metrics[sensor_id] = metrics

            coverage = float(values.size / expected_samples)
            if coverage < 0.50:
                review_reasons.append(f"{location}: severely reduced sample coverage {coverage:.0%}")
            elif coverage < 0.70:
                review_reasons.append(f"{location}: reduced sample coverage {coverage:.0%}")
            if max_gap > 1.0:
                informational_reasons.append(f"{location}: long GUI delivery gap {max_gap * 1000.0:.0f} ms")
            elif max_gap > 0.25:
                informational_reasons.append(f"{location}: GUI delivery gap {max_gap * 1000.0:.0f} ms")
            if longest_equal_run >= max(64, int(0.05 * values.size)) or (values.size >= 64 and np.unique(values).size <= 3):
                hard_failures.append(f"{location}: frozen or repeated EMG stream")
            if clipping_fraction >= 0.01:
                hard_failures.append(f"{location}: ADC clipping {clipping_fraction:.1%}")

            if quiet_rest:
                rms = float(metrics["filtered_rms"])
                reference_limit = self._robust_limit(references[sensor_id])
                if reference_limit is None:
                    metrics["rest_reference_status"] = f"collecting ({len(references[sensor_id])}/3 accepted rest stages)"
                    if rms > 195.0:
                        informational_reasons.append(f"{location}: high preliminary rest RMS {rms:.0f}")
                else:
                    center, spread, limit = reference_limit
                    metrics.update({"rest_reference_median": center, "rest_reference_robust_sigma": spread, "rest_rms_limit": limit})
                    if rms > limit:
                        informational_reasons.append(f"{location}: rest RMS {rms:.0f} exceeds personal limit {limit:.0f}")
                if float(metrics["pre_notch_50hz_power_ratio"]) > 0.35:
                    soft_failures.append(f"{location}: strong 50 Hz component")
                if float(metrics["rms_drift_ratio"]) > 0.75:
                    informational_reasons.append(f"{location}: variable rest RMS")

        if quiet_rest and all("filtered_rms" in sensor_metrics.get(sensor_id, {}) for sensor_id in self.sensor_order):
            rms_values = np.asarray([max(float(sensor_metrics[sensor_id]["filtered_rms"]), 1e-9) for sensor_id in self.sensor_order])
            geometric_mean = float(np.exp(np.mean(np.log(rms_values))))
            balance_outliers = []
            for sensor_id, rms in zip(self.sensor_order, rms_values):
                balance_value = float(np.log(rms / geometric_mean))
                sensor_metrics[sensor_id]["log_rms_balance"] = balance_value
                reference_limit = self._robust_limit(balance_references[sensor_id])
                if reference_limit is None:
                    continue
                center, spread, _upper = reference_limit
                balance_z = abs(balance_value - center) / spread
                sensor_metrics[sensor_id]["rms_balance_robust_z"] = balance_z
                if balance_z > 3.5:
                    balance_outliers.append(f"{self.sensor_locations.get(sensor_id, sensor_id)} z={balance_z:.1f}")
            if balance_outliers:
                informational_reasons.append("unusual cross-sensor rest balance: " + ", ".join(balance_outliers))

        if imu_motion_stage:
            reference_metrics = sensor_metrics.get("ED7A78C8", {})
            expected_axis = "roll" if "roll" in stage_kind else "pitch" if "pitch" in stage_kind or "vertical" in stage_kind else "yaw" if "yaw" in stage_kind or "horizontal" in stage_kind else "combined"
            if expected_axis == "combined":
                movement_range = max(float(reference_metrics.get("imu_yaw_range", 0.0)), float(reference_metrics.get("imu_pitch_range", 0.0)))
            else:
                movement_range = float(reference_metrics.get(f"imu_{expected_axis}_range", 0.0))
            if movement_range < 5.0:
                soft_failures.append(f"Dorsal forearm IMU: insufficient {expected_axis} movement range ({movement_range:.1f})")

        review_reasons.extend(soft_failures)
        review_reasons.extend(warnings)
        if hard_failures:
            status = "CRITICAL"
            recommended_action = "repeat_now"
            quality_score = 0
        elif exclusion_reasons:
            status = "EXCLUDE"
            recommended_action = "exclude_stage"
            quality_score = max(10, 45 - 10 * len(exclusion_reasons) - 3 * len(review_reasons))
        elif review_reasons:
            status = "REVIEW"
            recommended_action = "review"
            quality_score = max(50, 85 - 5 * len(review_reasons))
        else:
            status = "PASS"
            recommended_action = "keep"
            quality_score = 100
        repeat_recommended = bool(hard_failures)
        return {
            "stage_index": self.current_stage_index,
            "attempt": self.stage_attempts.get(self.current_stage_index, 1),
            "gesture_label": label,
            "stage_kind": stage_kind,
            "quiet_rest": quiet_rest,
            "protocol_block": str(self.current_stage.get("protocol_block", "standard")),
            "expected_duration_s": float(expected_duration_s),
            "status": status,
            "quality_score": quality_score,
            "recommended_action": recommended_action,
            "repeat_recommended": repeat_recommended,
            "hard_failures": hard_failures,
            "exclusion_reasons": exclusion_reasons,
            "review_reasons": review_reasons,
            "informational_reasons": informational_reasons,
            "soft_failures": soft_failures,
            "warnings": warnings,
            "sensor_metrics": sensor_metrics,
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def record_decision(self, result: dict[str, object], decision: str) -> None:
        """Record decision for the current RecordingQualityGate workflow."""
        saved = dict(result)
        saved["decision"] = decision
        self.history.append(saved)
        if decision in {"accepted", "override"}:
            self.accepted_by_stage[int(result["stage_index"])] = saved
        self._save()

    def invalidate_from_stage(self, first_stage_index: int) -> None:
        """Perform the invalidate from stage operation used by the RecordingQualityGate workflow."""
        for stage_index in list(self.accepted_by_stage):
            if stage_index >= first_stage_index:
                del self.accepted_by_stage[stage_index]
        self._save()

    def summary(self) -> dict[str, object]:
        """Perform the summary operation used by the RecordingQualityGate workflow."""
        actions = {"keep": 0, "review": 0, "exclude_stage": 0}
        per_gesture: dict[str, dict[str, int]] = {}
        for record in self.accepted_by_stage.values():
            action = str(record.get("recommended_action", "keep"))
            label = str(record.get("gesture_label", "unknown"))
            actions[action] = actions.get(action, 0) + 1
            counts = per_gesture.setdefault(label, {"total": 0, "retained_after_exclusion": 0})
            counts["total"] += 1
            if action != "exclude_stage":
                counts["retained_after_exclusion"] += 1
        minimum_retained = min((row["retained_after_exclusion"] for row in per_gesture.values()), default=0)
        return {"actions": actions, "per_gesture": per_gesture, "minimum_retained_per_gesture": minimum_retained}

    def _save(self) -> None:
        """Perform the save operation used by the RecordingQualityGate workflow."""
        if self.output_path is None:
            return
        accepted = sorted(self.accepted_by_stage)
        payload = {
            "schema_version": 2,
            "sampling_rate_hz": self.fs,
            "policy": {
                "scope": "non-interrupting stage annotation; raw CSV is preserved",
                "hard_rule": "repeat only after catastrophic sensor loss, frozen data, invalid values, clipping, or saturation",
                "review_rule": "reduced coverage, strong 50 Hz noise, and insufficient IMU range are marked without interruption",
                "information_rule": "ordinary rest RMS level, drift, and cross-sensor balance are logged but do not change stage status",
                "exclusion_rule": "whole-stage exclusion is reserved for offline proof that no synchronized windows are usable",
                "rest_reference": "per-sensor median + 3 robust standard deviations from accepted at_rest stages",
            },
            "accepted_stage_indices": accepted,
            "history": self.history,
        }
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self.manifest_path is None:
            return
        final_stages = []
        counts_by_label: dict[str, dict[str, int]] = {}
        for stage_index, record in sorted(self.accepted_by_stage.items()):
            action = str(record.get("recommended_action", "keep"))
            label = str(record.get("gesture_label", "unknown"))
            row = {
                "stage_index": stage_index,
                "gesture_label": label,
                "stage_kind": record.get("stage_kind", ""),
                "protocol_block": record.get("protocol_block", "standard"),
                "quality_score": int(record.get("quality_score", 100)),
                "status": record.get("status", "PASS"),
                "recommended_action": action,
                "exclusion_reasons": record.get("exclusion_reasons", []),
                "review_reasons": record.get("review_reasons", []),
                "informational_reasons": record.get("informational_reasons", []),
            }
            final_stages.append(row)
            label_counts = counts_by_label.setdefault(label, {"total": 0, "keep": 0, "review": 0, "exclude_stage": 0})
            label_counts["total"] += 1
            label_counts[action] = label_counts.get(action, 0) + 1
        manifest = {
            "schema_version": 1,
            "raw_data_modified": False,
            "default_training_policy": "unfiltered until an explicit filtered-vs-unfiltered experiment is requested",
            "thresholds": {
                "keep_min_sample_coverage": 0.70,
                "online_delivery_gap": "informational only; true continuity is checked from saved packet timestamps",
                "whole_stage_exclusion": "not assigned online; use timestamp-aware synchronized-window analysis",
            },
            "keep_stage_indices": [row["stage_index"] for row in final_stages if row["recommended_action"] == "keep"],
            "review_stage_indices": [row["stage_index"] for row in final_stages if row["recommended_action"] == "review"],
            "exclude_stage_indices": [row["stage_index"] for row in final_stages if row["recommended_action"] == "exclude_stage"],
            "counts_by_gesture": counts_by_label,
            "stages": final_stages,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def audit_session_for_training(
    session_dir: Path,
    sensor_order: list[str],
    sampling_rate_hz: float = 620.0,
) -> dict[str, object]:
    """Validate complete stages before starting a personal-model training job."""
    session_dir = Path(session_dir)
    protocol_path = session_dir / "session_protocol.json"
    csv_path = session_dir / "raw_recordings" / "calibration_recording.csv"
    quality_path = session_dir / "session_quality.json"
    manifest_path = session_dir / "quality_exclusions.json"
    blockers: list[str] = []
    warnings: list[str] = []
    integrity_issues: list[str] = []
    if not protocol_path.exists():
        blockers.append("session_protocol.json is missing")
        protocol = []
    else:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not csv_path.exists():
        blockers.append("calibration_recording.csv is missing")
        return {"passed": False, "blockers": blockers, "warnings": warnings, "valid_stage_indices": []}

    counts: dict[int, dict[str, int]] = {}
    labels: dict[int, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            try:
                stage_index = int(float(row.get("trial_index", -1)))
            except (TypeError, ValueError):
                continue
            sensor_id = str(row.get("unit_id") or row.get("device_id") or "").upper()
            if sensor_id not in sensor_order:
                continue
            counts.setdefault(stage_index, {}).setdefault(sensor_id, 0)
            counts[stage_index][sensor_id] += 1
            labels[stage_index] = str(row.get("gesture_label", ""))

    valid_stages: list[int] = []
    for stage_index, stage in enumerate(protocol):
        sensor_counts = counts.get(stage_index, {})
        missing = [sensor_id for sensor_id in sensor_order if sensor_counts.get(sensor_id, 0) == 0]
        if missing:
            integrity_issues.append(f"stage {stage_index + 1}: missing sensors {', '.join(missing)}")
            continue
        duration = float(stage.get("duration_s", 0.0))
        expected_rows = max(1.0, duration * sampling_rate_hz / 8.0)
        low = [sensor_id for sensor_id in sensor_order if sensor_counts.get(sensor_id, 0) < 0.50 * expected_rows]
        if low:
            integrity_issues.append(f"stage {stage_index + 1}: insufficient samples from {', '.join(low)}")
            continue
        valid_stages.append(stage_index)

    if quality_path.exists():
        warnings.extend(integrity_issues)
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        accepted = {int(value) for value in quality.get("accepted_stage_indices", [])}
        missing_approval = sorted(set(valid_stages) - accepted)
        if missing_approval:
            shown = ", ".join(str(index + 1) for index in missing_approval[:10])
            suffix = "..." if len(missing_approval) > 10 else ""
            blockers.append(f"stages without a final quality decision: {shown}{suffix}")
        for record in quality.get("history", []):
            if record.get("decision") != "override":
                continue
            stage_number = int(record.get("stage_index", -1)) + 1
            if record.get("hard_failures"):
                blockers.append(f"stage {stage_number}: hard quality failure was overridden")
            elif record.get("soft_failures"):
                warnings.append(f"stage {stage_number}: soft quality warning was overridden")
    else:
        warnings.extend(integrity_issues)
        warnings.append("legacy recording: no session_quality.json; raw stage integrity checks were used")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    exclude_indices = {int(value) for value in manifest.get("exclude_stage_indices", [])}
    review_indices = {int(value) for value in manifest.get("review_stage_indices", [])}
    if exclude_indices:
        warnings.append(f"quality manifest recommends excluding {len(exclude_indices)} complete stages in a filtered run")
    if review_indices:
        warnings.append(f"quality manifest marks {len(review_indices)} stages for review")

    stages_per_label: dict[str, int] = {}
    for stage_index in valid_stages:
        label = labels.get(stage_index) or str(protocol[stage_index].get("gesture_label", ""))
        stages_per_label[label] = stages_per_label.get(label, 0) + 1
    for label in sorted({str(stage.get("gesture_label", "")) for stage in protocol}):
        if label and stages_per_label.get(label, 0) < 3:
            blockers.append(f"{label}: only {stages_per_label.get(label, 0)} valid stages; at least 3 are required")

    retained_per_label: dict[str, int] = {}
    for stage_index in valid_stages:
        if stage_index in exclude_indices:
            continue
        label = labels.get(stage_index) or str(protocol[stage_index].get("gesture_label", ""))
        retained_per_label[label] = retained_per_label.get(label, 0) + 1
    for label, count in sorted(retained_per_label.items()):
        if count < 3:
            warnings.append(f"filtered run would leave only {count} stages for {label}; do not filter automatically")

    return {
        "passed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "valid_stage_indices": valid_stages,
        "valid_stage_count": len(valid_stages),
        "protocol_stage_count": len(protocol),
        "stages_per_label": stages_per_label,
        "filtered_stages_per_label": retained_per_label,
        "recommended_exclude_stage_indices": sorted(exclude_indices),
        "review_stage_indices": sorted(review_indices),
    }


class RealtimeSignalSafetyGate:
    """Lightweight stream-integrity gate with delayed recovery."""

    def __init__(self, sensor_order: list[str], stale_limit_ms: float = 600.0, recovery_s: float = 0.75) -> None:
        """Initialize the RealtimeSignalSafetyGate instance and its runtime state."""
        self.sensor_order = list(sensor_order)
        self.stale_limit_ms = float(stale_limit_ms)
        self.recovery_s = float(recovery_s)
        self.safe = False
        self.reason = "waiting for all sensors"
        self.clean_since: float | None = None
        self.high_noise_since: float | None = None
        self.high_noise_latched = False

    def update(self, snapshots: list[object], noise_snapshot: dict[str, object], rest_expected: bool) -> dict[str, object]:
        """Perform the update operation used by the RealtimeSignalSafetyGate workflow."""
        now = time.monotonic()
        by_sensor = {f"{int(snapshot.unit_id):08X}": snapshot for snapshot in snapshots}
        reasons = []
        for sensor_id in self.sensor_order:
            snapshot = by_sensor.get(sensor_id)
            if snapshot is None:
                reasons.append(f"{sensor_id} missing")
                continue
            if float(snapshot.age_ms) > self.stale_limit_ms:
                reasons.append(f"{sensor_id} stale ({float(snapshot.age_ms):.0f} ms)")
            values = np.asarray(snapshot.emg, dtype=float).reshape(-1)
            finite = values[np.isfinite(values)]
            if values.size and finite.size != values.size:
                reasons.append(f"{sensor_id} invalid EMG values")
            if finite.size and float(np.mean(np.abs(finite) >= 32760.0)) >= 0.01:
                reasons.append(f"{sensor_id} clipping")

        high_noise_label = str(noise_snapshot.get("label", "OK")) == "High noise"
        if self.high_noise_latched:
            if high_noise_label:
                reasons.append("sustained high rest noise")
            else:
                self.high_noise_latched = False
                self.high_noise_since = None
        elif rest_expected and high_noise_label:
            if self.high_noise_since is None:
                self.high_noise_since = now
            elif now - self.high_noise_since >= 0.75:
                self.high_noise_latched = True
                reasons.append("sustained high rest noise")
        else:
            self.high_noise_since = None

        previous_safe = self.safe
        if reasons:
            self.safe = False
            self.reason = "; ".join(reasons)
            self.clean_since = None
        else:
            if self.clean_since is None:
                self.clean_since = now
            clean_time = now - self.clean_since
            if clean_time >= self.recovery_s:
                self.safe = True
                self.reason = "clean"
            else:
                self.safe = False
                self.reason = f"recovering ({clean_time:.1f}/{self.recovery_s:.1f} s)"
        return {"safe": self.safe, "reason": self.reason, "changed": self.safe != previous_safe}
