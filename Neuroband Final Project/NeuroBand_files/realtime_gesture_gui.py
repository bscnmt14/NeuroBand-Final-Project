"""Main NeuroBand realtime gesture-recognition and calibration interface.

The GUI coordinates serial acquisition, filtered EMG plots, model inference,
gesture probabilities, confidence-based uncertainty, temporal decision logic,
recording quality feedback, personal calibration, model testing, and short
recalibration. It also launches the mouse-control interface while keeping sensor
streaming and user-interface updates responsive and clearly separated.

"""

from __future__ import annotations

import json
import sys
import time
import csv
import re
import ctypes
import pickle
from collections import Counter, deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from scipy import signal
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from classifier_adapter import GestureClassifierAdapter, PredictionResult
from mouse_game_control import MouseControlWindow
from recording_quality_gate import RecordingQualityGate, RealtimeSignalSafetyGate, audit_session_for_training
from umyo_stream import DeviceSnapshot, UmyoSerialReader

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
PICTURES_DIR = PROJECT_ROOT / "pictures"
CALIBRATION_DIR = PROJECT_ROOT / "Data" / "calibration_sessions"
MODEL_TEST_DIR = PROJECT_ROOT / "Data" / "model_tests"
MODEL_UPDATE_DIR = PROJECT_ROOT / "Data" / "model_updates"
DEFAULT_NOISE_LIMITS = {
    "green_max_rms": 132.0,
    "yellow_max_rms": 195.0,
    "red_min_rms": 195.0,
}
CONFUSION_PROTECTED_PAIRS = {
    frozenset(("open_hand", "pointing")),
    frozenset(("open_hand", "wrist_extension")),
    frozenset(("open_hand", "pinch")),
    frozenset(("pinch", "wrist_extension")),
    frozenset(("pinch", "pointing")),
    frozenset(("fist", "like")),
}
DEFAULT_CONFUSION_THRESHOLD = 0.65

SENSOR_LOCATIONS = {
    "B0DAC7E9": "Ventral forearm",
    "ED7A78C8": "Dorsal forearm",
    "37ED348F": "Inner forearm side",
}
SENSOR_ORDER = ["B0DAC7E9", "ED7A78C8", "37ED348F"]
ACTIVE_GESTURES = [
    "fist",
    "like",
    "open_hand",
    "pinch",
    "pointing",
    "wrist_extension",
    "wrist_flexion",
]
PROBLEM_GESTURES = {"open_hand", "pinch", "pointing", "wrist_extension"}
EASY_GESTURES = set(ACTIVE_GESTURES) - PROBLEM_GESTURES
DISPLAY_GESTURES = ["at_rest", *ACTIVE_GESTURES]
GESTURE_DISPLAY_NAMES = {
    "at_rest": "At Rest",
    "fist": "Fist",
    "like": "Like",
    "open_hand": "Open Hand",
    "pinch": "Pinch",
    "pointing": "Pointing",
    "wrist_extension": "Wrist Extension",
    "wrist_flexion": "Wrist Flexion",
}
GESTURE_HEBREW_NAMES = {
    "at_rest": "מנוחה",
    "fist": "אגרוף",
    "like": "לייק",
    "open_hand": "כף יד פתוחה",
    "pinch": "צביטה",
    "pointing": "הצבעה",
    "wrist_extension": "פשיטת שורש כף היד",
    "wrist_flexion": "כיפוף שורש כף היד",
}
GESTURE_IMAGES = {
    "at_rest": "rest.jpeg",
    "fist": "fist.jpeg",
    "like": "like.jpeg",
    "open_hand": "open_palm.jpeg",
    "pinch": "pinch.jpeg",
    "pointing": "pointing.jpeg",
    "wrist_flexion": "wirst_flextion.jpeg",
    "wrist_extension": "wrist_extention.jpeg",
}
PLACEMENT_IMAGES = ["צמיד על היד 1.jpeg", "הצמיד על היד 2.jpeg"]
LOGO_IMAGE = "LOGO.png"
CUE_LEAD_SECONDS = 2.2
SENSOR_READY_MAX_AGE_S = 1.5
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
PROCESS_POWER_THROTTLING = 4
PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4
_timer_resolution_active = False


class ProcessPowerThrottlingState(ctypes.Structure):
    """Represent the ProcessPowerThrottlingState component and keep its related state and behavior together."""
    _fields_ = [
        ("Version", ctypes.c_ulong),
        ("ControlMask", ctypes.c_ulong),
        ("StateMask", ctypes.c_ulong),
    ]


def keep_display_awake(enable: bool) -> None:
    """Perform the keep display awake operation used by the realtime gesture gui workflow."""
    if sys.platform != "win32":
        return
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if enable else ES_CONTINUOUS
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def configure_realtime_priority() -> dict[str, str]:
    """Perform the configure realtime priority operation used by the realtime gesture gui workflow."""
    global _timer_resolution_active
    result = {"process": "normal", "gui_thread": "normal", "power_throttling": "unsupported", "timer_resolution": "default"}
    if sys.platform != "win32":
        return result
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessInformation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        kernel32.SetProcessInformation.restype = ctypes.c_int
        state = ProcessPowerThrottlingState(
            Version=PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask=PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
            StateMask=0,
        )
        success = kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(),
            PROCESS_POWER_THROTTLING,
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
        result["power_throttling"] = "disabled" if success else f"failed:{ctypes.get_last_error()}"
    except Exception as exc:
        result["power_throttling"] = f"failed:{exc}"
    try:
        if ctypes.windll.winmm.timeBeginPeriod(1) == 0:
            _timer_resolution_active = True
            result["timer_resolution"] = "1ms"
        else:
            result["timer_resolution"] = "failed"
    except Exception as exc:
        result["timer_resolution"] = f"failed:{exc}"
    return result


def release_realtime_priority() -> None:
    """Perform the release realtime priority operation used by the realtime gesture gui workflow."""
    global _timer_resolution_active
    if sys.platform == "win32" and _timer_resolution_active:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        _timer_resolution_active = False


def load_latest_noise_profile() -> dict[str, object]:
    """Load and validate latest noise profile for the current realtime gesture gui workflow."""
    profiles = sorted(CALIBRATION_DIR.glob("*/noise_profile.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not profiles:
        return {"global": DEFAULT_NOISE_LIMITS, "sensors": {}}
    try:
        return json.loads(profiles[0].read_text(encoding="utf-8"))
    except Exception:
        return {"global": DEFAULT_NOISE_LIMITS, "sensors": {}}


def classify_noise_rms(rms: float, limits: dict[str, object] | None = None) -> tuple[str, str]:
    """Perform the classify noise rms operation used by the realtime gesture gui workflow."""
    limits = limits or DEFAULT_NOISE_LIMITS
    green = float(limits.get("green_max_rms", DEFAULT_NOISE_LIMITS["green_max_rms"]))
    yellow = float(limits.get("yellow_max_rms", DEFAULT_NOISE_LIMITS["yellow_max_rms"]))
    if rms <= green:
        return "OK", "#166534"
    if rms <= yellow:
        return "Elevated", "#a16207"
    return "High noise", "#991b1b"


def parse_decision_strategy(strategy: str) -> dict[str, object]:
    """Perform the parse decision strategy operation used by the realtime gesture gui workflow."""
    strategy = (strategy or "raw_no_gate").strip()
    if strategy == "raw_no_gate":
        return {"method": "raw", "threshold": 0.0, "window": 1, "enter": 0.55, "stay": 0.40, "confusion_threshold": DEFAULT_CONFUSION_THRESHOLD}
    match = re.fullmatch(r"threshold_([0-9.]+)", strategy)
    if match:
        return {"method": "threshold", "threshold": float(match.group(1)), "window": 1, "enter": 0.55, "stay": 0.40, "confusion_threshold": DEFAULT_CONFUSION_THRESHOLD}
    match = re.fullmatch(r"threshold_([0-9.]+)_majority_(\d+)", strategy)
    if match:
        return {
            "method": "majority",
            "threshold": float(match.group(1)),
            "window": int(match.group(2)),
            "enter": 0.55,
            "stay": 0.40,
            "confusion_threshold": DEFAULT_CONFUSION_THRESHOLD,
        }
    match = re.fullmatch(r"threshold_([0-9.]+)_consecutive_(\d+)", strategy)
    if match:
        return {
            "method": "consecutive",
            "threshold": float(match.group(1)),
            "window": int(match.group(2)),
            "enter": 0.55,
            "stay": 0.40,
            "confusion_threshold": DEFAULT_CONFUSION_THRESHOLD,
        }
    match = re.fullmatch(r"hysteresis_enter_([0-9.]+)_stay_([0-9.]+)", strategy)
    if match:
        return {
            "method": "hysteresis",
            "threshold": float(match.group(1)),
            "window": 1,
            "enter": float(match.group(1)),
            "stay": float(match.group(2)),
            "confusion_threshold": DEFAULT_CONFUSION_THRESHOLD,
        }
    return {"method": "threshold", "threshold": 0.55, "window": 1, "enter": 0.55, "stay": 0.40, "confusion_threshold": DEFAULT_CONFUSION_THRESHOLD}


class DecisionSmoother:
    """Represent the DecisionSmoother component and keep its related state and behavior together."""
    def __init__(self) -> None:
        """Initialize the DecisionSmoother instance and its runtime state."""
        self.config: dict[str, object] = parse_decision_strategy("raw_no_gate")
        self.reset()

    def reset(self) -> None:
        """Perform the reset operation used by the DecisionSmoother workflow."""
        self.history: list[str] = []
        self.current = "Uncertain"
        self.candidate = ""
        self.candidate_count = 0

    def configure(self, config: dict[str, object]) -> None:
        """Perform the configure operation used by the DecisionSmoother workflow."""
        self.config = dict(config)
        self.reset()

    def apply(self, label: str, confidence: float) -> tuple[str, bool]:
        """Perform the apply operation used by the DecisionSmoother workflow."""
        method = str(self.config.get("method", "raw"))
        threshold = float(self.config.get("threshold", 0.55))
        previous = self.current
        if label in {"Error", "No model"}:
            self.reset()
            return label, True
        if method == "raw":
            output = label
            return self._apply_confusion_gate(previous, output, label, confidence), False
        if method == "threshold":
            output = "Uncertain" if confidence < threshold else label
            output = self._apply_confusion_gate(previous, output, label, confidence)
            return output, output == "Uncertain"
        if method == "majority":
            gated = "Uncertain" if confidence < threshold else label
            width = max(1, int(self.config.get("window", 3)))
            self.history.append(gated)
            self.history = self.history[-width:]
            counts = Counter(self.history)
            output = max(counts.items(), key=lambda item: (item[1], -self.history[::-1].index(item[0])))[0]
            output = self._apply_confusion_gate(previous, output, label, confidence)
            return output, output == "Uncertain"
        if method == "consecutive":
            gated = "Uncertain" if confidence < threshold else label
            required = max(1, int(self.config.get("window", 2)))
            if gated == "Uncertain":
                self.current = "Uncertain"
                self.candidate = ""
                self.candidate_count = 0
                return self.current, True
            if self.current in {"", "Uncertain"} or gated == self.current:
                self.current = gated
                self.candidate = ""
                self.candidate_count = 0
                return self.current, False
            if gated == self.candidate:
                self.candidate_count += 1
            else:
                self.candidate = gated
                self.candidate_count = 1
            if self.candidate_count >= required:
                self.current = self.candidate
                self.candidate = ""
                self.candidate_count = 0
            self.current = self._apply_confusion_gate(previous, self.current, label, confidence)
            return self.current, self.current == "Uncertain"
        if method == "hysteresis":
            enter = float(self.config.get("enter", 0.55))
            stay = float(self.config.get("stay", 0.40))
            if self.current == "Uncertain":
                self.current = label if confidence >= enter else "Uncertain"
            elif label == self.current:
                self.current = self.current if confidence >= stay else "Uncertain"
            elif confidence >= enter:
                self.current = label
            elif confidence < stay:
                self.current = "Uncertain"
            self.current = self._apply_confusion_gate(previous, self.current, label, confidence)
            return self.current, self.current == "Uncertain"
        return label, False

    def _apply_confusion_gate(self, previous: str, output: str, raw_label: str, confidence: float) -> str:
        """Apply confusion gate for the current DecisionSmoother workflow."""
        if output in {"Uncertain", "Error", "No model"}:
            return output
        if previous in {"", "Uncertain", "Error", "No model"}:
            self.current = output
            return output
        if output == previous:
            self.current = output
            return output
        pair = frozenset((previous, output))
        if pair not in CONFUSION_PROTECTED_PAIRS:
            self.current = output
            return output
        threshold = float(self.config.get("confusion_threshold", DEFAULT_CONFUSION_THRESHOLD))
        if raw_label == output and confidence >= threshold:
            self.current = output
            return output
        self.current = previous
        return previous


def safe_user_name(name: str) -> str:
    """Perform the safe user name operation used by the realtime gesture gui workflow."""
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip())
    return cleaned.strip("_") or "user"


def build_training_protocol() -> list[dict[str, object]]:
    """Create and configure training protocol for the current realtime gesture gui workflow."""
    protocol: list[dict[str, object]] = []

    def add(kind: str, label: str, duration: float, title: str = "") -> None:
        """Perform the add operation used by the build training protocol workflow."""
        protocol.append(
            {
                "kind": kind,
                "gesture_label": label,
                "duration_s": duration,
                "title": title or label,
            }
        )

    add("rest", "at_rest", 5.0, "Baseline rest")

    for gesture in ACTIVE_GESTURES:
        long_hold_s = 5.0 if gesture in PROBLEM_GESTURES else 3.0
        add("rest", "at_rest", 2.0, "Rest before gesture")
        add("hold_short", gesture, 3.0, f"Short hold: {gesture}")
        add("rest", "at_rest", 2.0, "Rest before long hold")
        add("hold_long", gesture, long_hold_s, f"Long hold: {gesture}")

    for first in ACTIVE_GESTURES:
        for second in ACTIVE_GESTURES:
            if first == second:
                continue
            add("rest", "at_rest", 1.5, "Rest before transition")
            add("transition_hold", first, 2.0, f"Transition start: {first}")
            add("transition_hold", second, 2.0, f"Switch to: {second}")

    add("rest", "at_rest", 6.0, "Final rest")
    return protocol


def build_focused_training_protocol() -> list[dict[str, object]]:
    """Create and configure focused training protocol for the current realtime gesture gui workflow."""
    protocol: list[dict[str, object]] = []
    difficult_gestures = ("fist", "open_hand", "wrist_extension")
    confusing_pairs = {
        frozenset(("open_hand", "pointing")),
        frozenset(("open_hand", "wrist_extension")),
        frozenset(("open_hand", "pinch")),
        frozenset(("pinch", "wrist_extension")),
        frozenset(("pinch", "pointing")),
        frozenset(("fist", "like")),
    }

    def add(
        block: str,
        kind: str,
        label: str,
        duration: float,
        title: str,
        effort: str = "natural",
        condition: str = "standard",
    ) -> None:
        """Perform the add operation used by the build focused training protocol workflow."""
        protocol.append(
            {
                "protocol_block": block,
                "kind": kind,
                "gesture_label": label,
                "duration_s": duration,
                "title": title,
                "effort_level": effort,
                "protocol_condition": condition,
            }
        )

    # Stage 1: repeated rest baselines capture natural repositioning and rest variability.
    for repetition in range(3):
        add("rest_baselines", "rest", "at_rest", 8.0, f"Rest baseline {repetition + 1} of 3: relax naturally")
        if repetition < 2:
            add("rest_baselines", "rest", "at_rest", 3.0, "Reposition the forearm slightly, then relax")

    # Stage 2: two standard holds per gesture, with longer holds for the difficult gestures.
    for repetition in range(2):
        for gesture in ACTIVE_GESTURES:
            duration = 5.0 if gesture in difficult_gestures else 3.0
            add("standard_holds", "rest", "at_rest", 2.0, f"Prepare {gesture}, repetition {repetition + 1}")
            add("standard_holds", "hold_standard", gesture, duration, f"Natural hold: {gesture}")

    # Stage 4: every directed transition once; repeat known confusing pairs once more.
    for first in ACTIVE_GESTURES:
        for second in ACTIVE_GESTURES:
            if first == second:
                continue
            repetitions = 2 if frozenset((first, second)) in confusing_pairs else 1
            for repetition in range(repetitions):
                suffix = f", repeat {repetition + 1}" if repetitions > 1 else ""
                add("gesture_transitions", "transition_hold", first, 1.5, f"Transition start: {first}{suffix}")
                add("gesture_transitions", "transition_hold", second, 1.5, f"Switch directly to: {second}{suffix}")
                add("gesture_transitions", "rest", "at_rest", 1.0, "Brief recovery")

    # Stage 6: repeat the difficult gestures at the end to capture fatigue and sensor drift.
    add("end_repeat_drift", "rest", "at_rest", 8.0, "End baseline: relax naturally")
    for gesture in difficult_gestures:
        add("end_repeat_drift", "rest", "at_rest", 2.0, f"Prepare final repeat: {gesture}")
        add("end_repeat_drift", "hold_repeat", gesture, 5.0, f"Final natural repeat: {gesture}")
    add("end_repeat_drift", "rest", "at_rest", 8.0, "Final recovery rest")
    return protocol


def build_imu_function_protocol() -> list[dict[str, object]]:
    """Create and configure imu function protocol for the current realtime gesture gui workflow."""
    protocol: list[dict[str, object]] = []

    def add(kind: str, gesture: str, duration: float, title: str) -> None:
        """Perform the add operation used by the build imu function protocol workflow."""
        protocol.append({"kind": kind, "gesture_label": gesture, "duration_s": duration, "title": title})

    add("imu_rest", "at_rest", 6.0, "IMU stage: relax naturally")
    add("pointer_horizontal_precise", "at_rest", 8.0, "Move the forearm slowly left and right")
    add("imu_rest", "at_rest", 3.0, "Return to the neutral position")
    add("pointer_vertical_precise", "at_rest", 8.0, "Move the forearm slowly up and down")
    add("imu_rest", "at_rest", 3.0, "Return to the neutral position")
    add("pointer_diagonal_precise", "at_rest", 8.0, "Make small, precise diagonal movements")
    add("imu_rest", "at_rest", 3.0, "Return to the neutral position")
    add("rest_roll_right_left_toggle", "at_rest", 5.0, "Quickly roll right, then quickly roll left")
    add("imu_rest", "at_rest", 3.0, "Return to the neutral position")
    add("fist_roll_right_left_drag_toggle", "fist", 5.0, "Hold a fist: quickly roll right, then left")
    add("imu_rest", "at_rest", 3.0, "Return to the neutral position")
    add("fast_pitch_positive", "at_rest", 3.0, "Perform a quick upward pitch movement")
    add("imu_rest", "at_rest", 3.0, "Return to the neutral position")
    add("fast_pitch_negative", "at_rest", 3.0, "Perform a quick downward pitch movement")
    add("imu_rest", "at_rest", 3.0, "Return to the neutral position")
    add("fast_yaw_positive", "at_rest", 3.0, "Perform a quick yaw movement to the right")
    add("imu_rest", "at_rest", 3.0, "Return to the neutral position")
    add("fast_yaw_negative", "at_rest", 3.0, "Perform a quick yaw movement to the left")
    add("normal_non_command_motion", "at_rest", 8.0, "Make normal everyday movements without commands")
    add("imu_rest", "at_rest", 5.0, "Final natural rest")
    return protocol


def build_model_test_protocol() -> list[dict[str, object]]:
    """Create and configure model test protocol for the current realtime gesture gui workflow."""
    protocol: list[dict[str, object]] = []

    def add(kind: str, label: str, duration: float, title: str) -> None:
        """Perform the add operation used by the build model test protocol workflow."""
        protocol.append(
            {
                "protocol_block": "model_test",
                "kind": kind,
                "gesture_label": label,
                "duration_s": duration,
                "title": title,
            }
        )

    add("test_rest", "at_rest", 5.0, "Test baseline: relax naturally")
    sequences = [
        ACTIVE_GESTURES,
        ["open_hand", "pointing", "pinch", "wrist_extension", "fist", "like", "wrist_flexion"],
    ]
    for round_index, sequence in enumerate(sequences, start=1):
        for gesture in sequence:
            add("test_rest", "at_rest", 2.0, f"Prepare {gesture}, round {round_index}")
            add("test_hold", gesture, 2.5, f"Test gesture: {gesture}")
    add("test_rest", "at_rest", 5.0, "Final test rest")
    return protocol


def build_short_model_update_protocol() -> list[dict[str, object]]:
    """Create and configure short model update protocol for the current realtime gesture gui workflow."""
    protocol: list[dict[str, object]] = []
    difficult = {"fist", "open_hand", "wrist_extension"}

    def add(kind: str, label: str, duration: float, title: str, condition: str) -> None:
        """Perform the add operation used by the build short model update protocol workflow."""
        protocol.append(
            {
                "protocol_block": "short_model_update",
                "kind": kind,
                "gesture_label": label,
                "duration_s": duration,
                "title": title,
                "protocol_condition": condition,
            }
        )

    add("update_train_rest", "at_rest", 6.0, "Current-condition baseline: relax naturally", "adapt_train")
    for round_index, condition in enumerate(("adapt_train", "adapt_validate"), start=1):
        if round_index == 2:
            add("update_validation_rest", "at_rest", 3.0, "Independent validation round: relax", condition)
        for gesture in ACTIVE_GESTURES:
            add(f"update_round_{round_index}_rest", "at_rest", 1.5, f"Prepare {gesture}", condition)
            duration = 2.5 if gesture in difficult else 2.0
            add(f"update_round_{round_index}_hold", gesture, duration, f"Short update: {gesture}", condition)
    add("update_validation_rest", "at_rest", 4.0, "Final current-condition rest", "adapt_validate")
    return protocol


class DeviceEmgPlot(QtWidgets.QWidget):
    """Represent the DeviceEmgPlot component and keep its related state and behavior together."""
    def __init__(self, sensor_id: str, location: str, samples: int = 1800, fs: float = 1100.0, parent=None):
        """Initialize the DeviceEmgPlot instance and its runtime state."""
        super().__init__(parent)
        self.sensor_id = sensor_id.upper()
        self.location = location
        self.samples = samples
        self.buffer = np.zeros(samples, dtype=float)
        self.raw_buffer = np.zeros(samples, dtype=float)
        self.spectrum_buffer = np.zeros((max(16, samples // 4), 4), dtype=float)
        self.last_data_id: int | None = None
        self.packet_count = 0
        self.dirty = False
        self.waiting_title = False
        self.filter_b, self.filter_a = self._make_display_filter(fs)
        self.filter_zi = np.zeros(max(len(self.filter_a), len(self.filter_b)) - 1, dtype=float)
        self.notch_b, self.notch_a = self._make_notch_filter(fs)
        self.notch_zi = np.zeros(max(len(self.notch_a), len(self.notch_b)) - 1, dtype=float)
        self._build_ui()

    @staticmethod
    def rssi_quality(rssi: int) -> float:
        """Perform the rssi quality operation used by the DeviceEmgPlot workflow."""
        if rssi <= 0:
            return 0.0
        return max(0.0, min(100.0, (90.0 - rssi) * 1.6))

    @staticmethod
    def battery_percent(batt_mv: int) -> float:
        """Perform the battery percent operation used by the DeviceEmgPlot workflow."""
        if batt_mv <= 0:
            return 0.0
        return max(0.0, min(100.0, (batt_mv - 3100.0) / 10.0))

    @staticmethod
    def _make_display_filter(fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Create and configure display filter for the current DeviceEmgPlot workflow."""
        nyquist = 0.5 * fs
        low_hz = 35.0
        high_hz = min(500.0, nyquist - 1.0)
        if high_hz <= low_hz:
            return np.array([1.0]), np.array([1.0])
        return signal.butter(4, [low_hz / nyquist, high_hz / nyquist], btype="bandpass")

    @staticmethod
    def _make_notch_filter(fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Create and configure notch filter for the current DeviceEmgPlot workflow."""
        nyquist = 0.5 * fs
        if 50.0 >= nyquist:
            return np.array([1.0]), np.array([1.0])
        return signal.iirnotch(50.0 / nyquist, 30.0)

    def _build_ui(self) -> None:
        """Create and configure ui for the current DeviceEmgPlot workflow."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        bars = QtWidgets.QHBoxLayout()
        bars.setSpacing(3)
        bars.setContentsMargins(0, 18, 0, 0)
        rssi_column = QtWidgets.QVBoxLayout()
        rssi_column.setSpacing(2)
        battery_column = QtWidgets.QVBoxLayout()
        battery_column.setSpacing(2)
        rssi_label = QtWidgets.QLabel("RSSI")
        battery_label = QtWidgets.QLabel("Battery")
        for label in [rssi_label, battery_label]:
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("font-size: 9px; font-weight: 600; color: #334155;")
        self.rssi_bar = QtWidgets.QProgressBar()
        self.rssi_bar.setOrientation(QtCore.Qt.Vertical)
        self.rssi_bar.setRange(0, 100)
        self.rssi_bar.setValue(0)
        self.rssi_bar.setFormat("%p%")
        self.rssi_bar.setTextVisible(True)
        self.rssi_bar.setFixedWidth(36)
        self.battery_bar = QtWidgets.QProgressBar()
        self.battery_bar.setOrientation(QtCore.Qt.Vertical)
        self.battery_bar.setRange(0, 100)
        self.battery_bar.setValue(0)
        self.battery_bar.setFormat("%p%")
        self.battery_bar.setTextVisible(True)
        self.battery_bar.setFixedWidth(36)
        rssi_column.addWidget(rssi_label)
        rssi_column.addWidget(self.rssi_bar, stretch=1)
        battery_column.addWidget(battery_label)
        battery_column.addWidget(self.battery_bar, stretch=1)
        bars.addLayout(rssi_column)
        bars.addLayout(battery_column)
        layout.addLayout(bars)

        plot_layout = QtWidgets.QVBoxLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)
        title = f"{self.location} | {self.sensor_id}"
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setStyleSheet("font-weight: 700; color: #1f2937;")
        plot_layout.addWidget(self.title_label)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "EMG")
        self.plot.setLabel("bottom", "Samples")
        self.plot.setYRange(-3000, 3000)
        self.plot.setMinimumHeight(128)
        self.curve = self.plot.plot(pen=pg.mkPen("#18a058", width=1.5))
        self.curve.setClipToView(True)
        self.curve.setDownsampling(auto=True, method="peak")
        plot_layout.addWidget(self.plot)
        layout.addLayout(plot_layout, stretch=1)

    def update_link_status(self, rssi: int, battery_mv: int) -> None:
        """Refresh link status for the current DeviceEmgPlot workflow."""
        self.rssi_bar.setValue(int(self.rssi_quality(rssi)))
        self.battery_bar.setValue(int(self.battery_percent(battery_mv)))

    def append_packet(self, values: np.ndarray, spectrum: np.ndarray) -> None:
        """Add packet for the current DeviceEmgPlot workflow."""
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 0:
            return
        spectrum = np.asarray(spectrum, dtype=float)
        if spectrum.ndim == 1:
            spectrum = spectrum.reshape(1, -1)
        spectrum_rows = np.zeros((max(1, spectrum.shape[0]), 4), dtype=float)
        if spectrum.size:
            spectrum_rows[:, : min(4, spectrum.shape[1])] = spectrum[:, :4]

        if values.size >= self.samples:
            self.raw_buffer[:] = values[-self.samples:]
        else:
            self.raw_buffer = np.roll(self.raw_buffer, -values.size)
            self.raw_buffer[-values.size:] = values

        filtered, self.filter_zi = signal.lfilter(self.filter_b, self.filter_a, values, zi=self.filter_zi)
        filtered, self.notch_zi = signal.lfilter(self.notch_b, self.notch_a, filtered, zi=self.notch_zi)
        if filtered.size >= self.samples:
            self.buffer[:] = filtered[-self.samples:]
        else:
            self.buffer = np.roll(self.buffer, -filtered.size)
            self.buffer[-filtered.size:] = filtered

        rows = min(len(spectrum_rows), len(self.spectrum_buffer))
        self.spectrum_buffer = np.roll(self.spectrum_buffer, -rows, axis=0)
        self.spectrum_buffer[-rows:, :] = spectrum_rows[-rows:, :]
        self.packet_count += 1
        self.dirty = True

    def refresh_plot(self) -> None:
        """Refresh plot for the current DeviceEmgPlot workflow."""
        if self.dirty:
            self.curve.setData(self.buffer)
            self.dirty = False

    def window(self, window_samples: int) -> np.ndarray:
        """Perform the window operation used by the DeviceEmgPlot workflow."""
        return self.raw_buffer[-window_samples:].copy()

    def noise_rms(self, window_samples: int = 256) -> float:
        """Perform the noise rms operation used by the DeviceEmgPlot workflow."""
        values = self.buffer[-min(window_samples, len(self.buffer)) :]
        return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0

    def spectrum_window(self, window_samples: int) -> np.ndarray:
        """Perform the spectrum window operation used by the DeviceEmgPlot workflow."""
        packet_count = max(1, int(np.ceil(window_samples / 8.0)))
        return self.spectrum_buffer[-packet_count:, :].copy()

    def clear_if_stale(self, age_ms: float) -> None:
        """Reset if stale for the current DeviceEmgPlot workflow."""
        is_waiting = age_ms > 3000
        if is_waiting == self.waiting_title:
            return
        self.waiting_title = is_waiting
        if is_waiting:
            self.title_label.setText(f"{self.location} | {self.sensor_id} | waiting")
        else:
            self.title_label.setText(f"{self.location} | {self.sensor_id}")


class ThreeSensorEmgPanel(QtWidgets.QWidget):
    """Represent the ThreeSensorEmgPanel component and keep its related state and behavior together."""
    def __init__(self, samples: int = 1800, fs: float = 1100.0, parent=None):
        """Initialize the ThreeSensorEmgPanel instance and its runtime state."""
        super().__init__(parent)
        self.samples = samples
        self.fs = fs
        self.plots: dict[str, DeviceEmgPlot] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        """Create and configure ui for the current ThreeSensorEmgPanel workflow."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        for sensor_id in SENSOR_ORDER:
            plot = DeviceEmgPlot(sensor_id, SENSOR_LOCATIONS[sensor_id], self.samples, fs=self.fs)
            self.plots[sensor_id] = plot
            layout.addWidget(plot, stretch=1)

    def update_from_snapshots(self, snapshots: list[DeviceSnapshot]) -> int:
        """Refresh from snapshots for the current ThreeSensorEmgPanel workflow."""
        updated = 0
        for snap in snapshots:
            sensor_id = f"{snap.unit_id:08X}"
            if sensor_id not in self.plots:
                continue
            plot = self.plots[sensor_id]
            plot.clear_if_stale(snap.age_ms)
            plot.update_link_status(snap.rssi, snap.battery_mv)
            if snap.emg.size > 0:
                plot.last_data_id = snap.data_id
                updated += 1
                plot.append_packet(snap.emg, snap.spectrum)
        return updated

    def refresh_plots(self) -> None:
        """Refresh plots for the current ThreeSensorEmgPanel workflow."""
        for plot in self.plots.values():
            plot.refresh_plot()

    def classifier_window(self, sensor_ids: list[str], window_samples: int) -> dict[str, dict[str, np.ndarray]]:
        """Perform the classifier window operation used by the ThreeSensorEmgPanel workflow."""
        windows: dict[str, dict[str, np.ndarray]] = {}
        for sensor_id in sensor_ids:
            plot = self.plots.get(sensor_id)
            if plot is not None:
                windows[sensor_id] = {
                    "emg": plot.window(window_samples),
                    "spectrum": plot.spectrum_window(window_samples),
                }
        if not windows:
            for sensor_id, plot in self.plots.items():
                windows[sensor_id] = {
                    "emg": plot.window(window_samples),
                    "spectrum": plot.spectrum_window(window_samples),
                }
        return windows

    def noise_snapshot(self, profile: dict[str, object]) -> dict[str, object]:
        """Perform the noise snapshot operation used by the ThreeSensorEmgPanel workflow."""
        sensors_profile = profile.get("sensors", {}) if isinstance(profile, dict) else {}
        rows = []
        worst_rank = 0
        worst_label = "OK"
        worst_color = "#166534"
        for sensor_id, plot in self.plots.items():
            rms = plot.noise_rms()
            limits = sensors_profile.get(sensor_id, DEFAULT_NOISE_LIMITS) if isinstance(sensors_profile, dict) else DEFAULT_NOISE_LIMITS
            label, color = classify_noise_rms(rms, limits)
            rank = {"OK": 0, "Elevated": 1, "High noise": 2}.get(label, 0)
            if rank >= worst_rank:
                worst_rank = rank
                worst_label = label
                worst_color = color
            rows.append({"sensor_id": sensor_id, "location": SENSOR_LOCATIONS[sensor_id], "rms": rms, "label": label, "color": color})
        return {"label": worst_label, "color": worst_color, "sensors": rows}


class CalibrationRecorder:
    """Represent the CalibrationRecorder component and keep its related state and behavior together."""
    def __init__(self) -> None:
        """Initialize the CalibrationRecorder instance and its runtime state."""
        self.active = False
        self.session_dir: Path | None = None
        self.csv_file = None
        self.writer: csv.DictWriter | None = None
        self.started_at = 0.0
        self.current_stage: dict[str, object] | None = None
        self.current_stage_index = -1
        self.rows_written = 0
        self.csv_path: Path | None = None
        self.fieldnames: list[str] = []
        self.quality_gate = RecordingQualityGate(SENSOR_ORDER, SENSOR_LOCATIONS)

    def start(self, session_dir: Path, protocol: list[dict[str, object]]) -> None:
        """Perform the start operation used by the CalibrationRecorder workflow."""
        self.stop()
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "raw_recordings").mkdir(exist_ok=True)
        (session_dir / "trained_model").mkdir(exist_ok=True)
        (session_dir / "replay_report").mkdir(exist_ok=True)
        (session_dir / "session_protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fieldnames = [
            "timestamp",
            "trial_index",
            "protocol_step",
            "stage_kind",
            "protocol_block",
            "effort_level",
            "protocol_condition",
            "gesture_label",
            "device_id",
            "unit_id",
            *[f"emg_{idx}" for idx in range(8)],
            "sp0",
            "sp1",
            "sp2",
            "sp3",
            "rssi",
            "battery_mv",
            "ax",
            "ay",
            "az",
            "yaw",
            "pitch",
            "roll",
        ]
        self.fieldnames = fieldnames
        self.csv_path = session_dir / "raw_recordings" / "calibration_recording.csv"
        self.csv_file = self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.session_dir = session_dir
        self.quality_gate.start(session_dir)
        self.started_at = time.time()
        self.rows_written = 0
        self.active = True

    def pause(self) -> None:
        """Perform the pause operation used by the CalibrationRecorder workflow."""
        self.current_stage = None
        if self.csv_file is not None:
            self.csv_file.flush()

    def discard_from_stage(self, first_stage_index: int) -> None:
        """Perform the discard from stage operation used by the CalibrationRecorder workflow."""
        if self.csv_path is None or not self.csv_path.exists():
            return
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
        with self.csv_path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = [row for row in reader if int(float(row.get("trial_index", -1))) < first_stage_index]
        with self.csv_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.rows_written = len(rows)
        self.quality_gate.invalidate_from_stage(first_stage_index)
        self.current_stage = None
        self.current_stage_index = -1
        self.active = True

    def set_stage(self, index: int, stage: dict[str, object]) -> None:
        """Set stage for the current CalibrationRecorder workflow."""
        self.current_stage_index = index
        self.current_stage = stage
        self.quality_gate.begin_stage(index, stage)

    def add_snapshots(self, snapshots: list[DeviceSnapshot]) -> None:
        """Add snapshots for the current CalibrationRecorder workflow."""
        if not self.active or self.writer is None or self.current_stage is None:
            return
        self.quality_gate.add_snapshots(snapshots)
        label = str(self.current_stage.get("gesture_label", "at_rest"))
        kind = str(self.current_stage.get("kind", ""))
        block = str(self.current_stage.get("protocol_block", "standard"))
        effort = str(self.current_stage.get("effort_level", "natural"))
        condition = str(self.current_stage.get("protocol_condition", "standard"))
        trial_index = int(self.current_stage_index)
        timestamp_base = time.time() - self.started_at
        for snap in snapshots:
            sensor_id = f"{snap.unit_id:08X}"
            if sensor_id not in SENSOR_ORDER or snap.emg.size == 0:
                continue
            emg_values = np.asarray(snap.emg, dtype=float).reshape(-1)
            spectra = np.asarray(snap.spectrum, dtype=float)
            if spectra.ndim == 1:
                spectra = spectra.reshape(1, -1)
            packet_count = max(1, int(np.ceil(emg_values.size / 8.0)))
            for packet_idx in range(packet_count):
                packet = emg_values[packet_idx * 8 : (packet_idx + 1) * 8]
                if packet.size < 8:
                    packet = np.pad(packet, (0, 8 - packet.size), constant_values=np.nan)
                spectrum_idx = min(packet_idx, max(0, len(spectra) - 1))
                spectrum = spectra[spectrum_idx] if spectra.size else np.zeros(4, dtype=float)
                padded_spectrum = np.zeros(4, dtype=float)
                padded_spectrum[: min(4, spectrum.size)] = spectrum[:4]
                row = {
                    "timestamp": timestamp_base + packet_idx * 8.0 / 620.0,
                    "trial_index": trial_index,
                    "protocol_step": trial_index,
                    "stage_kind": kind,
                    "protocol_block": block,
                    "effort_level": effort,
                    "protocol_condition": condition,
                    "gesture_label": label,
                    "device_id": sensor_id,
                    "unit_id": sensor_id,
                    "rssi": snap.rssi,
                    "battery_mv": snap.battery_mv,
                    "ax": snap.ax,
                    "ay": snap.ay,
                    "az": snap.az,
                    "yaw": snap.yaw,
                    "pitch": snap.pitch,
                    "roll": snap.roll,
                    "sp0": padded_spectrum[0],
                    "sp1": padded_spectrum[1],
                    "sp2": padded_spectrum[2],
                    "sp3": padded_spectrum[3],
                }
                for idx, value in enumerate(packet):
                    row[f"emg_{idx}"] = value
                self.writer.writerow(row)
                self.rows_written += 1

    def evaluate_current_stage(self, expected_duration_s: float) -> dict[str, object] | None:
        """Evaluate current stage for the current CalibrationRecorder workflow."""
        return self.quality_gate.evaluate_current_stage(expected_duration_s)

    def record_quality_decision(self, result: dict[str, object], decision: str) -> None:
        """Record quality decision for the current CalibrationRecorder workflow."""
        self.quality_gate.record_decision(result, decision)

    def quality_summary(self) -> dict[str, object]:
        """Perform the quality summary operation used by the CalibrationRecorder workflow."""
        return self.quality_gate.summary()

    def stop(self) -> None:
        """Perform the stop operation used by the CalibrationRecorder workflow."""
        if self.csv_file is not None:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception:
                pass
        self.csv_file = None
        self.writer = None
        self.active = False
        self.current_stage = None
        self.current_stage_index = -1


class GestureCueLane(QtWidgets.QWidget):
    """Represent the GestureCueLane component and keep its related state and behavior together."""
    def __init__(self, parent=None):
        """Initialize the GestureCueLane instance and its runtime state."""
        super().__init__(parent)
        self.current_stage: dict[str, object] | None = None
        self.next_stage: dict[str, object] | None = None
        self.remaining_s = 0.0
        self.lead_s = CUE_LEAD_SECONDS
        self._cache: dict[str, QtGui.QPixmap] = {}
        self.setMinimumHeight(185)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_cues(
        self,
        current_stage: dict[str, object] | None,
        next_stage: dict[str, object] | None,
        remaining_s: float,
    ) -> None:
        """Set cues for the current GestureCueLane workflow."""
        self.current_stage = current_stage
        self.next_stage = next_stage
        self.remaining_s = max(0.0, remaining_s)
        self.update()

    def _pixmap_for(self, gesture: str, size: QtCore.QSize) -> QtGui.QPixmap | None:
        """Perform the pixmap for operation used by the GestureCueLane workflow."""
        image_name = GESTURE_IMAGES.get(gesture)
        if not image_name:
            return None
        path = PICTURES_DIR / image_name
        key = f"{gesture}:{size.width()}x{size.height()}"
        if key not in self._cache and path.exists():
            pixmap = QtGui.QPixmap(str(path))
            if not pixmap.isNull():
                self._cache[key] = pixmap.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        return self._cache.get(key)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Render the widget using its current state."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QtGui.QColor("#f8fafc"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1"), 1))
        painter.drawRoundedRect(rect, 8, 8)

        image_size = QtCore.QSize(112, 112)
        target_frame = QtCore.QRect(
            rect.left() + 18,
            rect.center().y() - image_size.height() // 2 - 8,
            image_size.width() + 16,
            image_size.height() + 16,
        )
        target_x = target_frame.center().x()
        painter.setPen(QtGui.QPen(QtGui.QColor("#0f766e"), 4))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(target_frame, 8, 8)
        painter.setPen(QtGui.QPen(QtGui.QColor("#0f172a"), 1))
        painter.drawText(target_frame.left() + 18, target_frame.bottom() + 18, "Perform")

        y = rect.center().y() - image_size.height() // 2
        if self.current_stage is not None:
            gesture = str(self.current_stage.get("gesture_label", "at_rest"))
            pixmap = self._pixmap_for(gesture, image_size)
            if pixmap is not None:
                painter.drawPixmap(target_x - image_size.width() // 2, y, pixmap)
            painter.setPen(QtGui.QPen(QtGui.QColor("#111827"), 1))
            title = str(self.current_stage.get("title", gesture)).replace("_", " ")
            painter.drawText(target_frame.left(), rect.top() + 18, title)

        if self.next_stage is not None:
            gesture = str(self.next_stage.get("gesture_label", "at_rest"))
            progress = 0.0
            if self.remaining_s <= self.lead_s:
                progress = 1.0 - (self.remaining_s / max(0.1, self.lead_s))
            start_x = rect.right() - image_size.width() // 2
            next_x = int(start_x + (target_x - start_x) * max(0.0, min(1.0, progress)))
            pixmap = self._pixmap_for(gesture, image_size)
            if pixmap is not None:
                painter.setOpacity(0.55 + 0.45 * progress)
                painter.drawPixmap(next_x - image_size.width() // 2, y, pixmap)
                painter.setOpacity(1.0)
            painter.setPen(QtGui.QPen(QtGui.QColor("#475569"), 1))
            painter.drawText(next_x - 52, rect.bottom() - 28, "Next")


class CalibrationDialog(QtWidgets.QDialog):
    """Represent the CalibrationDialog component and keep its related state and behavior together."""
    train_requested = QtCore.Signal(str, str)

    def __init__(self, parent=None):
        """Initialize the CalibrationDialog instance and its runtime state."""
        super().__init__(parent)
        self.general_protocol = build_training_protocol()
        self.protocol = self.general_protocol
        self.protocol_mode = "general"
        self.session_dir: Path | None = None
        self.current_index = -1
        self.stage_started_at = 0.0
        self.running = False
        self.paused = False
        self.finished = False
        self.imu_finished = False
        self.user_locked = False
        self.latest_sensor_seen: dict[str, float] = {}
        self.recorder = CalibrationRecorder()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._build_ui()

    def _build_ui(self) -> None:
        """Create and configure ui for the current CalibrationDialog workflow."""
        self.setWindowTitle("Personal Calibration")
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowMinMaxButtonsHint | QtCore.Qt.WindowCloseButtonHint)
        self.resize(1000, 820)
        self.setMinimumSize(720, 580)
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 6, 8, 6)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        header = QtWidgets.QHBoxLayout()
        logo = QtWidgets.QLabel()
        logo_path = PICTURES_DIR / LOGO_IMAGE
        if logo_path.exists():
            pixmap = QtGui.QPixmap(str(logo_path))
            logo.setPixmap(pixmap.scaled(86, 56, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        header.addWidget(logo)
        title = QtWidgets.QLabel("Personal Training Protocol")
        title.setStyleSheet("font-size: 26px; font-weight: 700; color: #1f2937;")
        header.addWidget(title, stretch=1)
        layout.addLayout(header)

        user_row = QtWidgets.QHBoxLayout()
        user_row.addWidget(QtWidgets.QLabel("User name"))
        self.user_name = QtWidgets.QLineEdit()
        self.user_name.setPlaceholderText("example: tal")
        user_row.addWidget(self.user_name, stretch=1)
        self.lock_user_btn = QtWidgets.QPushButton("Next")
        self.lock_user_btn.clicked.connect(self.lock_user)
        self.back_btn = QtWidgets.QPushButton("Back")
        self.back_btn.clicked.connect(self.unlock_user)
        self.back_btn.setEnabled(False)
        user_row.addWidget(self.lock_user_btn)
        user_row.addWidget(self.back_btn)
        layout.addLayout(user_row)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._intro_page())
        self.stack.addWidget(self._protocol_page())
        layout.addWidget(self.stack, stretch=1)

        buttons = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start general gesture recording")
        self.start_btn.clicked.connect(lambda: self.start_protocol("general"))
        self.start_btn.setEnabled(False)
        self.next_btn = QtWidgets.QPushButton("Skip step")
        self.next_btn.clicked.connect(self._advance_stage)
        self.pause_recording_btn = QtWidgets.QPushButton("Pause recording")
        self.pause_recording_btn.clicked.connect(self.toggle_recording_pause)
        self.back_stage_btn = QtWidgets.QPushButton("Back one step")
        self.back_stage_btn.clicked.connect(self.back_one_stage)
        self.back_stage_btn.setEnabled(False)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_protocol)
        self.restart_btn = QtWidgets.QPushButton("New recording")
        self.restart_btn.clicked.connect(self.reset_recording)
        self.full_grid_btn = QtWidgets.QPushButton("Full grid")
        self.full_grid_btn.clicked.connect(lambda: self._request_train("full_grid"))
        self.full_grid_btn.setEnabled(False)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.next_btn)
        buttons.addWidget(self.pause_recording_btn)
        buttons.addWidget(self.back_stage_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.restart_btn)
        buttons.addStretch()
        buttons.addWidget(self.full_grid_btn)
        layout.addLayout(buttons)

    def _imu_transition_page(self) -> QtWidgets.QWidget:
        """Perform the imu transition page operation used by the CalibrationDialog workflow."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addStretch()
        title = QtWidgets.QLabel("Gesture recording complete")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 34px; font-weight: 700; color: #1f2937;")
        layout.addWidget(title)
        description = QtWidgets.QLabel(
            "The next stage records the inertial movements used by the computer-control functions.\n"
            "Follow the guided instructions for precise pointer movements, quick roll sequences, pitch, and yaw.\n"
            "This recording is stored separately and is not added to EMG gesture-classifier training."
        )
        description.setAlignment(QtCore.Qt.AlignCenter)
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 16px; color: #334155;")
        layout.addWidget(description)
        self.start_imu_btn = QtWidgets.QPushButton("Start guided IMU function recording")
        self.start_imu_btn.setMinimumHeight(44)
        self.start_imu_btn.clicked.connect(self.start_imu_protocol)
        layout.addWidget(self.start_imu_btn, alignment=QtCore.Qt.AlignCenter)
        layout.addStretch()
        return page

    def _intro_page(self) -> QtWidgets.QWidget:
        """Perform the intro page operation used by the CalibrationDialog workflow."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        text = QtWidgets.QLabel(
            "Before calibration:\n"
            "1. Verify the three sensors are placed on the correct forearm locations.\n"
            "2. Keep the strap about four fingers below the elbow.\n"
            "3. Tighten the strap enough to keep contact, but not so much that it is uncomfortable.\n"
            "4. Move away from strong electronics, chargers, power supplies, large metal surfaces, and laptops when possible.\n"
            "5. Sit comfortably and keep the forearm position consistent during the protocol.\n"
            "6. Perform gestures naturally. Strong or forceful muscle contraction is not required."
        )
        text.setWordWrap(True)
        text.setStyleSheet("font-size: 14px;")
        layout.addWidget(text)
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(10)
        gestures_box = QtWidgets.QGroupBox("Gesture set for this calibration")
        gestures_grid = QtWidgets.QGridLayout(gestures_box)
        gestures_grid.setContentsMargins(8, 6, 8, 6)
        gestures_grid.setHorizontalSpacing(8)
        gestures_grid.setVerticalSpacing(4)
        for index, gesture in enumerate(DISPLAY_GESTURES):
            tile = QtWidgets.QWidget()
            tile.setMinimumWidth(170)
            tile_layout = QtWidgets.QVBoxLayout(tile)
            tile_layout.setContentsMargins(2, 2, 2, 2)
            tile_layout.setSpacing(2)
            image_label = QtWidgets.QLabel()
            image_label.setAlignment(QtCore.Qt.AlignCenter)
            image_label.setFixedSize(166, 92)
            image_path = PICTURES_DIR / GESTURE_IMAGES.get(gesture, "")
            if image_path.exists():
                pixmap = QtGui.QPixmap(str(image_path))
                image_label.setPixmap(pixmap.scaled(162, 88, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            name_label = QtWidgets.QLabel(
                f"{GESTURE_DISPLAY_NAMES.get(gesture, gesture.replace('_', ' ').title())}\n"
                f"{GESTURE_HEBREW_NAMES.get(gesture, '')}"
            )
            name_label.setAlignment(QtCore.Qt.AlignCenter)
            name_label.setWordWrap(True)
            name_label.setMinimumHeight(34)
            name_label.setStyleSheet("font-size: 11px; font-weight: 600;")
            tile_layout.addWidget(image_label)
            tile_layout.addWidget(name_label)
            gestures_grid.addWidget(tile, index // 2, index % 2)
        body.addWidget(gestures_box, 3)

        placement_box = QtWidgets.QGroupBox("Sensor placement")
        placement_layout = QtWidgets.QVBoxLayout(placement_box)
        placement_layout.setContentsMargins(6, 6, 6, 6)
        placement_layout.setSpacing(6)
        for image_name in PLACEMENT_IMAGES:
            label = QtWidgets.QLabel()
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMinimumHeight(205)
            path = PICTURES_DIR / image_name
            if path.exists():
                pixmap = QtGui.QPixmap(str(path))
                label.setPixmap(pixmap.scaled(390, 215, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            label.setStyleSheet("border: 1px solid #d1d5db; background: #f9fafb;")
            placement_layout.addWidget(label)
        body.addWidget(placement_box, 2)
        layout.addLayout(body, 1)

        self.sensor_status_label = QtWidgets.QLabel("Sensors: waiting")
        self.sensor_status_label.setWordWrap(True)
        self.sensor_status_label.setStyleSheet("font-weight: 700; color: #991b1b;")
        layout.addWidget(self.sensor_status_label)
        self.noise_status_label = QtWidgets.QLabel("Noise: waiting for baseline")
        self.noise_status_label.setWordWrap(True)
        self.noise_status_label.setStyleSheet("font-weight: 700; color: #64748b;")
        layout.addWidget(self.noise_status_label)
        layout.addStretch()
        return page

    def _protocol_page(self) -> QtWidgets.QWidget:
        """Perform the protocol page operation used by the CalibrationDialog workflow."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.stage_title = QtWidgets.QLabel("Ready")
        self.stage_title.setAlignment(QtCore.Qt.AlignCenter)
        self.stage_title.setStyleSheet("font-size: 40px; font-weight: 700; color: #111827;")
        layout.addWidget(self.stage_title)
        self.countdown = QtWidgets.QLabel("00.0")
        self.countdown.setAlignment(QtCore.Qt.AlignCenter)
        self.countdown.setStyleSheet("font-size: 54px; font-weight: 700; color: #0f766e;")
        layout.addWidget(self.countdown)
        self.cue_lane = GestureCueLane()
        layout.addWidget(self.cue_lane)
        self.gesture_image = QtWidgets.QLabel()
        self.gesture_image.setAlignment(QtCore.Qt.AlignCenter)
        self.gesture_image.setMinimumHeight(260)
        self.gesture_image.setStyleSheet("border: 1px solid #d1d5db; background: #f9fafb;")
        layout.addWidget(self.gesture_image)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, len(self.protocol))
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.protocol_note = QtWidgets.QLabel(self.protocol_summary())
        self.protocol_note.setWordWrap(True)
        layout.addWidget(self.protocol_note)
        self.quality_gate_label = QtWidgets.QLabel("Recording quality: waiting for first completed stage")
        self.quality_gate_label.setWordWrap(True)
        self.quality_gate_label.setStyleSheet("font-weight: 700; color: #64748b;")
        layout.addWidget(self.quality_gate_label)
        return page

    def protocol_summary(self) -> str:
        """Perform the protocol summary operation used by the CalibrationDialog workflow."""
        total = sum(float(stage["duration_s"]) for stage in self.protocol)
        if self.protocol_mode == "imu_functions":
            return (
                f"Guided IMU function stage: {len(self.protocol)} stages, approximately {total / 60.0:.1f} minutes. "
                "This recording is stored separately from gesture-classifier data."
            )
        return (
            f"Protocol: {len(ACTIVE_GESTURES)} gestures, no side_flex. "
            f"Includes rest-to-gesture, short holds, long holds, and all gesture-to-gesture transitions with rest between. "
            f"Estimated active duration: {total / 60.0:.1f} minutes."
        )

    def lock_user(self) -> None:
        """Perform the lock user operation used by the CalibrationDialog workflow."""
        if not self.user_name.text().strip():
            QtWidgets.QMessageBox.warning(self, "User name required", "Enter a user name before continuing.")
            return
        self.user_locked = True
        self.user_name.setEnabled(False)
        self.lock_user_btn.setEnabled(False)
        self.back_btn.setEnabled(True)
        self.stack.setCurrentIndex(0)
        self._refresh_sensor_status()

    def unlock_user(self) -> None:
        """Perform the unlock user operation used by the CalibrationDialog workflow."""
        if self.running:
            QtWidgets.QMessageBox.warning(self, "Recording active", "Stop the current recording before changing user.")
            return
        self.user_locked = False
        self.user_name.setEnabled(True)
        self.lock_user_btn.setEnabled(True)
        self.back_btn.setEnabled(False)
        self.start_btn.setEnabled(False)

    def all_sensors_ready(self) -> bool:
        """Perform the all sensors ready operation used by the CalibrationDialog workflow."""
        now = time.time()
        return all((now - self.latest_sensor_seen.get(sensor_id, 0.0)) <= SENSOR_READY_MAX_AGE_S for sensor_id in SENSOR_ORDER)

    def _refresh_sensor_status(self) -> None:
        """Refresh sensor status for the current CalibrationDialog workflow."""
        now = time.time()
        parts = []
        for sensor_id in SENSOR_ORDER:
            age = now - self.latest_sensor_seen.get(sensor_id, 0.0)
            ok = age <= SENSOR_READY_MAX_AGE_S
            marker = "OK" if ok else "missing"
            parts.append(f"{SENSOR_LOCATIONS[sensor_id]}: {marker}")
        ready = self.all_sensors_ready()
        self.sensor_status_label.setText("Sensors: " + " | ".join(parts))
        self.sensor_status_label.setStyleSheet(
            "font-weight: 700; color: #166534;" if ready else "font-weight: 700; color: #991b1b;"
        )
        self.start_btn.setEnabled(self.user_locked and ready and not self.running)

    def start_protocol(self, mode: str = "general") -> None:
        """Start protocol for the current CalibrationDialog workflow."""
        if not self.user_locked:
            QtWidgets.QMessageBox.warning(self, "User not locked", "Enter a user name and press Next first.")
            return
        if not self.all_sensors_ready():
            QtWidgets.QMessageBox.warning(self, "Sensors not ready", "All three uMyo sensors must be connected and streaming before recording.")
            return
        user = safe_user_name(self.user_name.text())
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.protocol_mode = "general"
        self.protocol = self.general_protocol
        self.session_dir = CALIBRATION_DIR / f"{user}_{stamp}"
        self.recorder.start(self.session_dir, self.protocol)
        self.current_index = -1
        self.running = True
        self.paused = False
        self.finished = False
        self.imu_finished = False
        self.full_grid_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.progress.setRange(0, len(self.protocol))
        self.progress.setValue(0)
        self.protocol_note.setText(self.protocol_summary())
        self.stack.setCurrentIndex(1)
        self._advance_stage()
        self.timer.start(100)
        self.pause_recording_btn.setText("Pause recording")
        self.pause_recording_btn.setEnabled(True)
        self.back_stage_btn.setEnabled(False)

    def start_imu_protocol(self) -> None:
        """Start imu protocol for the current CalibrationDialog workflow."""
        if self.session_dir is None:
            return
        if not self.all_sensors_ready():
            QtWidgets.QMessageBox.warning(self, "Sensors not ready", "All three uMyo sensors must be connected and streaming before recording.")
            return
        self.protocol_mode = "imu_functions"
        self.protocol = self.imu_function_protocol
        self.recorder.start(self.session_dir / "imu_functions", self.protocol)
        self.current_index = -1
        self.running = True
        self.paused = False
        self.finished = False
        self.imu_finished = False
        self.full_grid_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.progress.setRange(0, len(self.protocol))
        self.progress.setValue(0)
        self.protocol_note.setText(self.protocol_summary())
        self.stack.setCurrentIndex(1)
        self._advance_stage()
        self.timer.start(100)
        self.pause_recording_btn.setText("Pause recording")
        self.pause_recording_btn.setEnabled(True)
        self.back_stage_btn.setEnabled(False)

    def toggle_recording_pause(self) -> None:
        """Perform the toggle recording pause operation used by the CalibrationDialog workflow."""
        if not self.running:
            return
        if not self.paused:
            self.paused = True
            self.timer.stop()
            self.recorder.pause()
            self.pause_recording_btn.setText("Resume current step")
            self.back_stage_btn.setEnabled(self.current_index > 0)
            self.stage_title.setText("Recording paused")
            self.cue_lane.set_cues(None, None, 0.0)
            return
        self.recorder.discard_from_stage(self.current_index)
        self.paused = False
        self.pause_recording_btn.setText("Pause recording")
        self.back_stage_btn.setEnabled(False)
        self._restart_current_stage()
        self.timer.start(100)

    def back_one_stage(self) -> None:
        """Perform the back one stage operation used by the CalibrationDialog workflow."""
        if not self.running or not self.paused or self.current_index <= 0:
            return
        self.current_index -= 1
        self.recorder.discard_from_stage(self.current_index)
        self.progress.setValue(self.current_index + 1)
        stage = self.protocol[self.current_index]
        gesture = str(stage["gesture_label"])
        title = str(stage["title"])
        for raw_name, display_name in GESTURE_DISPLAY_NAMES.items():
            title = title.replace(raw_name, display_name)
        self.stage_title.setText(f"Paused at: {title}")
        self.countdown.setText(f"{float(stage['duration_s']):04.1f}")
        self._set_image(gesture)
        self.back_stage_btn.setEnabled(self.current_index > 0)

    def _restart_current_stage(self) -> None:
        """Perform the restart current stage operation used by the CalibrationDialog workflow."""
        stage = self.protocol[self.current_index]
        self.stage_started_at = time.time()
        self.recorder.set_stage(self.current_index, stage)
        self.progress.setValue(self.current_index + 1)
        gesture = str(stage["gesture_label"])
        title = str(stage["title"])
        for raw_name, display_name in GESTURE_DISPLAY_NAMES.items():
            title = title.replace(raw_name, display_name)
        hebrew = GESTURE_HEBREW_NAMES.get(gesture, "")
        self.stage_title.setText(f"{title}\n{hebrew}" if hebrew else title)
        self.countdown.setText(f"{float(stage['duration_s']):04.1f}")
        self._set_image(gesture)
        next_stage = self.protocol[self.current_index + 1] if self.current_index + 1 < len(self.protocol) else None
        self.cue_lane.set_cues(stage, next_stage, float(stage["duration_s"]))

    def stop_protocol(self) -> None:
        """Stop protocol for the current CalibrationDialog workflow."""
        stopped_mode = self.protocol_mode
        self.running = False
        self.paused = False
        self.timer.stop()
        self.recorder.stop()
        self.stage_title.setText("Stopped")
        self.cue_lane.set_cues(None, None, 0.0)
        self.pause_recording_btn.setText("Pause recording")
        self.pause_recording_btn.setEnabled(False)
        self.back_stage_btn.setEnabled(False)
        can_train = self._gesture_recording_available()
        self.full_grid_btn.setEnabled(can_train)
        self._refresh_sensor_status()

    def reset_recording(self) -> None:
        """Reset recording for the current CalibrationDialog workflow."""
        self.stop_protocol()
        self.session_dir = None
        self.current_index = -1
        self.finished = False
        self.paused = False
        self.imu_finished = False
        self.progress.setValue(0)
        self.stage_title.setText("Ready")
        self.countdown.setText("00.0")
        self.gesture_image.clear()
        self.quality_gate_label.setText("Recording quality: waiting for first completed stage")
        self.quality_gate_label.setStyleSheet("font-weight: 700; color: #64748b;")
        self.full_grid_btn.setEnabled(False)
        self.stack.setCurrentIndex(0)
        self._refresh_sensor_status()

    def _gesture_recording_available(self) -> bool:
        """Perform the gesture recording available operation used by the CalibrationDialog workflow."""
        return bool(self.session_dir and (self.session_dir / "raw_recordings" / "calibration_recording.csv").exists())

    def receive_snapshots(self, snapshots: list[DeviceSnapshot]) -> None:
        """Perform the receive snapshots operation used by the CalibrationDialog workflow."""
        now = time.time()
        for snap in snapshots:
            sensor_id = f"{snap.unit_id:08X}"
            if sensor_id in SENSOR_ORDER and snap.emg.size > 0:
                self.latest_sensor_seen[sensor_id] = now
        self._refresh_sensor_status()
        self.recorder.add_snapshots(snapshots)

    def _tick(self) -> None:
        """Perform the tick operation used by the CalibrationDialog workflow."""
        if not self.running or self.current_index < 0 or self.current_index >= len(self.protocol):
            return
        stage = self.protocol[self.current_index]
        elapsed = time.time() - self.stage_started_at
        remaining = float(stage["duration_s"]) - elapsed
        self.countdown.setText(f"{max(0.0, remaining):04.1f}")
        next_stage = self.protocol[self.current_index + 1] if self.current_index + 1 < len(self.protocol) else None
        self.cue_lane.set_cues(stage, next_stage, remaining)
        if remaining <= 0:
            self._advance_stage()

    def _complete_stage_quality_check(self) -> bool:
        """Perform the complete stage quality check operation used by the CalibrationDialog workflow."""
        if self.current_index < 0 or self.current_index >= len(self.protocol):
            return True
        stage = self.protocol[self.current_index]
        result = self.recorder.evaluate_current_stage(float(stage["duration_s"]))
        if result is None:
            return True
        failures = [
            *result.get("hard_failures", []),
            *result.get("exclusion_reasons", []),
            *result.get("review_reasons", []),
        ]
        warnings = list(result.get("warnings", []))
        status = str(result.get("status", "PASS"))
        if result.get("hard_failures"):
            details = "\n".join(f"- {item}" for item in result["hard_failures"])
            QtWidgets.QMessageBox.critical(
                self,
                "Recording integrity failure",
                f"This stage must be repeated before recording can continue.\n\n{details}",
            )
            self.recorder.record_quality_decision(result, "repeat")
            self.recorder.discard_from_stage(self.current_index)
            self.quality_gate_label.setText(f"Recording quality: FAIL - repeating stage {self.current_index + 1}")
            self.quality_gate_label.setStyleSheet("font-weight: 700; color: #991b1b;")
            self._restart_current_stage()
            return False
        if result.get("repeat_recommended"):
            dialog = QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Recording quality check")
            dialog.setIcon(QtWidgets.QMessageBox.Warning)
            details = "\n".join(f"- {item}" for item in failures)
            dialog.setText("This stage did not pass the recording quality gate.")
            dialog.setInformativeText(f"{details}\n\nRepeat the complete stage for reliable training data.")
            repeat_button = dialog.addButton("Repeat stage", QtWidgets.QMessageBox.AcceptRole)
            dialog.addButton("Continue anyway", QtWidgets.QMessageBox.DestructiveRole)
            dialog.exec()
            if dialog.clickedButton() is repeat_button:
                self.recorder.record_quality_decision(result, "repeat")
                self.recorder.discard_from_stage(self.current_index)
                self.quality_gate_label.setText(f"Recording quality: {status} - repeating stage {self.current_index + 1}")
                self.quality_gate_label.setStyleSheet("font-weight: 700; color: #991b1b;")
                self._restart_current_stage()
                return False
            decision = "override"
        else:
            decision = "accepted"
        self.recorder.record_quality_decision(result, decision)
        if status == "PASS":
            text = f"Recording quality: PASS - stage {self.current_index + 1} accepted"
            color = "#166534"
        else:
            detail = (failures + warnings)[0] if failures or warnings else "review recommended"
            text = f"Recording quality: {status} - {detail}"
            color = "#c2410c" if status == "EXCLUDE" else "#a16207"
        self.quality_gate_label.setText(text)
        self.quality_gate_label.setStyleSheet(f"font-weight: 700; color: {color};")
        return True

    def _advance_stage(self) -> None:
        """Perform the advance stage operation used by the CalibrationDialog workflow."""
        if not self.running:
            return
        if not self._complete_stage_quality_check():
            return
        self.current_index += 1
        if self.current_index >= len(self.protocol):
            completed_mode = self.protocol_mode
            self.running = False
            self.paused = False
            self.finished = True
            self.timer.stop()
            self.recorder.stop()
            quality = self.recorder.quality_summary()
            actions = quality["actions"]
            self.quality_gate_label.setText(
                f"Recording quality summary: keep {actions.get('keep', 0)} | review {actions.get('review', 0)} | "
                f"exclude {actions.get('exclude_stage', 0)} | minimum retained per gesture "
                f"{quality.get('minimum_retained_per_gesture', 0)}"
            )
            self.quality_gate_label.setStyleSheet("font-weight: 700; color: #334155;")
            self.stage_title.setText("Calibration complete")
            self.countdown.setText("Done")
            self.cue_lane.set_cues(None, None, 0.0)
            self.pause_recording_btn.setEnabled(False)
            self.back_stage_btn.setEnabled(False)
            self.progress.setValue(len(self.protocol))
            can_train = self._gesture_recording_available()
            self.full_grid_btn.setEnabled(can_train)
            self.start_btn.setEnabled(self.all_sensors_ready())
            return
        stage = self.protocol[self.current_index]
        self.stage_started_at = time.time()
        self.recorder.set_stage(self.current_index, stage)
        self.progress.setValue(self.current_index + 1)
        gesture = str(stage["gesture_label"])
        title = str(stage["title"])
        for raw_name, display_name in GESTURE_DISPLAY_NAMES.items():
            title = title.replace(raw_name, display_name)
        hebrew = GESTURE_HEBREW_NAMES.get(gesture, "")
        self.stage_title.setText(f"{title}\n{hebrew}" if hebrew else title)
        self._set_image(str(stage["gesture_label"]))
        next_stage = self.protocol[self.current_index + 1] if self.current_index + 1 < len(self.protocol) else None
        self.cue_lane.set_cues(stage, next_stage, float(stage["duration_s"]))

    def _set_image(self, gesture: str) -> None:
        """Set image for the current CalibrationDialog workflow."""
        image_name = GESTURE_IMAGES.get(gesture)
        path = PICTURES_DIR / image_name if image_name else None
        if not path or not path.exists():
            self.gesture_image.clear()
            return
        pixmap = QtGui.QPixmap(str(path))
        self.gesture_image.setPixmap(
            pixmap.scaled(520, 280, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    def _request_train(self, mode: str) -> None:
        """Perform the request train operation used by the CalibrationDialog workflow."""
        if not self.session_dir:
            return
        audit = audit_session_for_training(self.session_dir, SENSOR_ORDER)
        if not audit["passed"]:
            details = "\n".join(f"- {item}" for item in audit["blockers"][:12])
            QtWidgets.QMessageBox.critical(
                self,
                "Training blocked by recording quality",
                f"Training cannot start until the recording is corrected.\n\n{details}",
            )
            self.quality_gate_label.setText("Recording quality: training blocked - review failed stages")
            self.quality_gate_label.setStyleSheet("font-weight: 700; color: #991b1b;")
            return
        if audit["warnings"]:
            details = "\n".join(f"- {item}" for item in audit["warnings"])
            answer = QtWidgets.QMessageBox.question(
                self,
                "Recording quality warnings",
                f"The recording passed hard checks with warnings:\n\n{details}\n\nContinue to training?",
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        self.full_grid_btn.setEnabled(False)
        self.stage_title.setText("Full grid training model...")
        self.train_requested.emit(str(self.session_dir), "full_grid")
        self.hide()

    def set_training_done(self, message: str) -> None:
        """Set training done for the current CalibrationDialog workflow."""
        self.stage_title.setText(message)
        can_train = self._gesture_recording_available()
        self.full_grid_btn.setEnabled(can_train)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Perform the showEvent operation used by the CalibrationDialog workflow."""
        super().showEvent(event)
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        if screen is None or self.isMaximized():
            return
        available = screen.availableGeometry()
        width = min(self.width(), max(720, available.width() - 40))
        height = min(self.height(), max(580, available.height() - 40))
        self.resize(width, height)
        frame = self.frameGeometry()
        if not available.contains(frame):
            self.move(
                max(available.left(), min(self.x(), available.right() - frame.width())),
                max(available.top(), min(self.y(), available.bottom() - frame.height())),
            )

    def set_noise_status(self, snapshot: dict[str, object]) -> None:
        """Set noise status for the current CalibrationDialog workflow."""
        if not hasattr(self, "noise_status_label"):
            return
        sensors = snapshot.get("sensors", [])
        if not sensors:
            self.noise_status_label.setText("Noise: waiting")
            return
        details = " | ".join(f"{row['location']}: {row['rms']:.0f} {row['label']}" for row in sensors)
        self.noise_status_label.setText(f"Noise: {snapshot.get('label', 'OK')} | {details}")
        self.noise_status_label.setStyleSheet(f"font-weight: 700; color: {snapshot.get('color', '#64748b')};")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window closure and release application resources safely."""
        self.stop_protocol()
        super().closeEvent(event)


class MetricComparisonChart(QtWidgets.QWidget):
    """Represent the MetricComparisonChart component and keep its related state and behavior together."""
    def __init__(self, original: dict[str, float] | None, current: dict[str, float], parent=None):
        """Initialize the MetricComparisonChart instance and its runtime state."""
        super().__init__(parent)
        self.original = original
        self.current = current
        self.setMinimumHeight(210)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Render the widget using its current state."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(45, 24, -20, -35)
        painter.setPen(QtGui.QPen(QtGui.QColor("#94a3b8"), 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        metrics = [("BA", "balanced_accuracy"), ("Recall", "macro_recall"), ("F1", "macro_f1")]
        group_width = rect.width() / len(metrics)
        for index, (label, key) in enumerate(metrics):
            center = rect.left() + group_width * (index + 0.5)
            values = []
            if self.original is not None:
                values.append((float(self.original.get(key, 0.0)), QtGui.QColor("#94a3b8"), "Original replay"))
            values.append((float(self.current.get(key, 0.0)), QtGui.QColor("#0f766e"), "Current test"))
            bar_width = min(38.0, group_width / 3.0)
            total_width = len(values) * bar_width + max(0, len(values) - 1) * 8
            x = center - total_width / 2
            for value, color, _name in values:
                height = rect.height() * max(0.0, min(1.0, value))
                bar = QtCore.QRectF(x, rect.bottom() - height, bar_width, height)
                painter.fillRect(bar, color)
                painter.setPen(QtGui.QColor("#0f172a"))
                painter.drawText(QtCore.QRectF(x - 8, bar.top() - 20, bar_width + 16, 18), QtCore.Qt.AlignCenter, f"{value:.2f}")
                x += bar_width + 8
            painter.drawText(
                QtCore.QRectF(center - group_width / 2, rect.bottom() + 8, group_width, 22),
                QtCore.Qt.AlignCenter,
                label,
            )
        painter.setPen(QtGui.QColor("#475569"))
        legend = "Teal: current test" if self.original is None else "Gray: original replay     Teal: current test"
        painter.drawText(QtCore.QRectF(rect.left(), 0, rect.width(), 20), QtCore.Qt.AlignCenter, legend)


class ModelTestResultsDialog(QtWidgets.QDialog):
    """Represent the ModelTestResultsDialog component and keep its related state and behavior together."""
    def __init__(
        self,
        labels: list[str],
        matrix: np.ndarray,
        current: dict[str, float],
        per_class: list[dict[str, object]],
        original: dict[str, float] | None,
        parent=None,
    ):
        """Initialize the ModelTestResultsDialog instance and its runtime state."""
        super().__init__(parent)
        self.setWindowTitle("Model Test Results")
        self.resize(1050, 760)
        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel(
            f"Current test: BA {current['balanced_accuracy']:.3f}   |   "
            f"Macro recall {current['macro_recall']:.3f}   |   Macro F1 {current['macro_f1']:.3f}"
        )
        heading.setStyleSheet("font-size: 18px; font-weight: 700; color: #0f172a;")
        layout.addWidget(heading)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        matrix_table = QtWidgets.QTableWidget(len(labels), len(labels))
        matrix_table.setHorizontalHeaderLabels([GESTURE_DISPLAY_NAMES.get(label, label) for label in labels])
        matrix_table.setVerticalHeaderLabels([GESTURE_DISPLAY_NAMES.get(label, label) for label in labels])
        matrix_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for row_index, row in enumerate(matrix):
            row_total = max(1, int(np.sum(row)))
            for column_index, count in enumerate(row):
                fraction = float(count) / row_total
                item = QtWidgets.QTableWidgetItem(f"{int(count)}\n{fraction:.0%}")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                if row_index == column_index:
                    color = QtGui.QColor(220 - int(85 * fraction), 245 - int(35 * fraction), 232 - int(70 * fraction))
                else:
                    color = QtGui.QColor(255, 247 - int(100 * fraction), 237 - int(120 * fraction))
                item.setBackground(color)
                matrix_table.setItem(row_index, column_index, item)
        matrix_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        matrix_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        splitter.addWidget(matrix_table)

        class_table = QtWidgets.QTableWidget(len(per_class), 3)
        class_table.setHorizontalHeaderLabels(["Gesture", "Recall", "F1"])
        class_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for row_index, values in enumerate(per_class):
            class_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(str(values["gesture"])))
            class_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(f"{float(values['recall']):.3f}"))
            class_table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(f"{float(values['f1']):.3f}"))
        class_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        splitter.addWidget(class_table)
        splitter.setSizes([730, 300])
        layout.addWidget(splitter, stretch=3)
        layout.addWidget(MetricComparisonChart(original, current), stretch=1)
        if original is None:
            note = QtWidgets.QLabel("Original replay comparison is unavailable for this model.")
        else:
            note = QtWidgets.QLabel(
                f"Original replay: BA {original['balanced_accuracy']:.3f}, "
                f"macro recall {original['macro_recall']:.3f}, macro F1 {original['macro_f1']:.3f}."
            )
        note.setStyleSheet("color: #475569;")
        layout.addWidget(note)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)


class ModelTestDialog(QtWidgets.QDialog):
    """Represent the ModelTestDialog component and keep its related state and behavior together."""
    test_requested = QtCore.Signal(str, str)

    def __init__(self, current_model_path: str = "", parent=None):
        """Initialize the ModelTestDialog instance and its runtime state."""
        super().__init__(parent)
        self.current_model_path = current_model_path
        self.protocol = build_model_test_protocol()
        self.recorder = CalibrationRecorder()
        self.session_dir: Path | None = None
        self.current_index = -1
        self.stage_started_at = 0.0
        self.running = False
        self.latest_sensor_seen: dict[str, float] = {}
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.setWindowTitle("Test the Model")
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowMinMaxButtonsHint | QtCore.Qt.WindowCloseButtonHint)
        self.resize(940, 720)
        self._build_ui()

    def _build_ui(self) -> None:
        """Create and configure ui for the current ModelTestDialog workflow."""
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Guided Model Test")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #0f172a;")
        layout.addWidget(title)
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(QtWidgets.QLabel("Model"))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItem("Use currently loaded model", "")
        for path in self._saved_models():
            self.model_combo.addItem(f"{path.parent.parent.name} | {path.name}", str(path))
        model_row.addWidget(self.model_combo, stretch=1)
        layout.addLayout(model_row)
        self.sensor_status = QtWidgets.QLabel("Sensors: waiting")
        self.sensor_status.setStyleSheet("font-weight: 700; color: #991b1b;")
        layout.addWidget(self.sensor_status)
        self.quality_status = QtWidgets.QLabel("Recording quality: waiting")
        self.quality_status.setWordWrap(True)
        self.quality_status.setStyleSheet("font-weight: 700; color: #64748b;")
        layout.addWidget(self.quality_status)
        self.stage_title = QtWidgets.QLabel("Ready to test")
        self.stage_title.setAlignment(QtCore.Qt.AlignCenter)
        self.stage_title.setStyleSheet("font-size: 30px; font-weight: 700;")
        layout.addWidget(self.stage_title)
        self.countdown = QtWidgets.QLabel("00.0")
        self.countdown.setAlignment(QtCore.Qt.AlignCenter)
        self.countdown.setStyleSheet("font-size: 48px; font-weight: 700; color: #0f766e;")
        layout.addWidget(self.countdown)
        self.cue_lane = GestureCueLane()
        layout.addWidget(self.cue_lane)
        self.gesture_image = QtWidgets.QLabel()
        self.gesture_image.setAlignment(QtCore.Qt.AlignCenter)
        self.gesture_image.setMinimumHeight(220)
        self.gesture_image.setStyleSheet("border: 1px solid #d1d5db; background: #f9fafb;")
        layout.addWidget(self.gesture_image)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, len(self.protocol))
        layout.addWidget(self.progress)
        controls = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start test recording")
        self.start_btn.clicked.connect(self.start_test)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_test)
        self.evaluate_btn = QtWidgets.QPushButton("Test recorded data")
        self.evaluate_btn.clicked.connect(self.request_evaluation)
        self.evaluate_btn.setEnabled(False)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch()
        controls.addWidget(self.evaluate_btn)
        layout.addLayout(controls)
        duration = sum(float(stage["duration_s"]) for stage in self.protocol)
        self.duration_note = QtWidgets.QLabel(
            f"Test duration: approximately {duration:.0f} seconds. The recording does not update the model."
        )
        layout.addWidget(self.duration_note)

    def _saved_models(self) -> list[Path]:
        """Perform the saved models operation used by the ModelTestDialog workflow."""
        paths = list(CALIBRATION_DIR.glob("*/trained_model/personal_model.pkl"))
        paths.extend(CALIBRATION_DIR.glob("*/trained_model/personal_fast_model.pkl"))
        paths.extend(CALIBRATION_DIR.glob("*/trained_model/personal_model_update_*.pkl"))
        return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)

    def selected_model_path(self) -> str:
        """Perform the selected model path operation used by the ModelTestDialog workflow."""
        return str(self.model_combo.currentData() or self.current_model_path or "")

    def set_current_model(self, model_path: str) -> None:
        """Set current model for the current ModelTestDialog workflow."""
        self.current_model_path = model_path

    def all_sensors_ready(self) -> bool:
        """Perform the all sensors ready operation used by the ModelTestDialog workflow."""
        now = time.time()
        return all(now - self.latest_sensor_seen.get(sensor_id, 0.0) <= SENSOR_READY_MAX_AGE_S for sensor_id in SENSOR_ORDER)

    def receive_snapshots(self, snapshots: list[DeviceSnapshot]) -> None:
        """Perform the receive snapshots operation used by the ModelTestDialog workflow."""
        now = time.time()
        for snapshot in snapshots:
            sensor_id = f"{snapshot.unit_id:08X}"
            if sensor_id in SENSOR_ORDER and snapshot.emg.size > 0:
                self.latest_sensor_seen[sensor_id] = now
        ready = self.all_sensors_ready()
        states = [f"{SENSOR_LOCATIONS[sensor]}: {'OK' if now - self.latest_sensor_seen.get(sensor, 0.0) <= SENSOR_READY_MAX_AGE_S else 'missing'}" for sensor in SENSOR_ORDER]
        self.sensor_status.setText("Sensors: " + " | ".join(states))
        self.sensor_status.setStyleSheet("font-weight: 700; color: #166534;" if ready else "font-weight: 700; color: #991b1b;")
        self.recorder.add_snapshots(snapshots)

    def start_test(self) -> None:
        """Start test for the current ModelTestDialog workflow."""
        model_path = self.selected_model_path()
        if not model_path:
            QtWidgets.QMessageBox.warning(self, "Model required", "Select a model or load one in the main window first.")
            return
        if not self.all_sensors_ready():
            QtWidgets.QMessageBox.warning(self, "Sensors not ready", "All three uMyo sensors must be connected and streaming.")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = MODEL_TEST_DIR / f"model_test_{stamp}"
        self.recorder.start(self.session_dir, self.protocol)
        (self.session_dir / "selected_model.txt").write_text(model_path, encoding="utf-8")
        self.current_index = -1
        self.running = True
        self.quality_status.setText("Recording quality: waiting for first completed stage")
        self.evaluate_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self._advance_stage()
        self.timer.start(100)

    def stop_test(self) -> None:
        """Stop test for the current ModelTestDialog workflow."""
        self.running = False
        self.timer.stop()
        self.recorder.stop()
        self.stage_title.setText("Test recording stopped")
        self.start_btn.setEnabled(True)
        self.evaluate_btn.setEnabled(bool(self.session_dir))

    def _tick(self) -> None:
        """Perform the tick operation used by the ModelTestDialog workflow."""
        if not self.running or self.current_index < 0:
            return
        stage = self.protocol[self.current_index]
        remaining = float(stage["duration_s"]) - (time.time() - self.stage_started_at)
        self.countdown.setText(f"{max(0.0, remaining):04.1f}")
        next_stage = self.protocol[self.current_index + 1] if self.current_index + 1 < len(self.protocol) else None
        self.cue_lane.set_cues(stage, next_stage, remaining)
        if remaining <= 0:
            self._advance_stage()

    def _complete_stage_quality_check(self) -> bool:
        """Perform the complete stage quality check operation used by the ModelTestDialog workflow."""
        if self.current_index < 0 or self.current_index >= len(self.protocol):
            return True
        stage = self.protocol[self.current_index]
        result = self.recorder.evaluate_current_stage(float(stage["duration_s"]))
        if result is None:
            return True
        failures = [
            *result.get("hard_failures", []),
            *result.get("exclusion_reasons", []),
            *result.get("review_reasons", []),
        ]
        if result.get("hard_failures"):
            details = "\n".join(f"- {item}" for item in result["hard_failures"])
            QtWidgets.QMessageBox.critical(
                self,
                "Recording integrity failure",
                f"This stage must be repeated before recording can continue.\n\n{details}",
            )
            self.recorder.record_quality_decision(result, "repeat")
            self.recorder.discard_from_stage(self.current_index)
            self.quality_status.setText(f"Recording quality: FAIL - repeating stage {self.current_index + 1}")
            self.quality_status.setStyleSheet("font-weight: 700; color: #991b1b;")
            self._start_current_stage()
            return False
        if result.get("repeat_recommended"):
            dialog = QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Recording quality check")
            dialog.setIcon(QtWidgets.QMessageBox.Warning)
            dialog.setText("This test/recalibration stage did not pass the quality gate.")
            dialog.setInformativeText("\n".join(f"- {item}" for item in failures) + "\n\nRepeat this complete stage.")
            repeat_button = dialog.addButton("Repeat stage", QtWidgets.QMessageBox.AcceptRole)
            dialog.addButton("Continue anyway", QtWidgets.QMessageBox.DestructiveRole)
            dialog.exec()
            if dialog.clickedButton() is repeat_button:
                self.recorder.record_quality_decision(result, "repeat")
                self.recorder.discard_from_stage(self.current_index)
                self.quality_status.setText(f"Recording quality: FAIL - repeating stage {self.current_index + 1}")
                self.quality_status.setStyleSheet("font-weight: 700; color: #991b1b;")
                self._start_current_stage()
                return False
            decision = "override"
        else:
            decision = "accepted"
        self.recorder.record_quality_decision(result, decision)
        status = str(result.get("status", "PASS"))
        color = "#166534" if status == "PASS" else "#c2410c" if status == "EXCLUDE" else "#a16207"
        detail = failures[0] if failures else "accepted"
        self.quality_status.setText(f"Recording quality: {status} - stage {self.current_index + 1} | {detail}")
        self.quality_status.setStyleSheet(f"font-weight: 700; color: {color};")
        return True

    def _advance_stage(self) -> None:
        """Perform the advance stage operation used by the ModelTestDialog workflow."""
        if not self._complete_stage_quality_check():
            return
        self.current_index += 1
        if self.current_index >= len(self.protocol):
            self.running = False
            self.timer.stop()
            self.recorder.stop()
            quality = self.recorder.quality_summary()
            actions = quality["actions"]
            self.quality_status.setText(
                f"Quality summary: keep {actions.get('keep', 0)} | review {actions.get('review', 0)} | "
                f"exclude {actions.get('exclude_stage', 0)} | minimum retained/class "
                f"{quality.get('minimum_retained_per_gesture', 0)}"
            )
            self.quality_status.setStyleSheet("font-weight: 700; color: #334155;")
            self.stage_title.setText("Test recording complete")
            self.countdown.setText("Done")
            self.cue_lane.set_cues(None, None, 0.0)
            self.progress.setValue(len(self.protocol))
            self.start_btn.setEnabled(True)
            self.evaluate_btn.setEnabled(True)
            return
        self._start_current_stage()

    def _start_current_stage(self) -> None:
        """Start current stage for the current ModelTestDialog workflow."""
        stage = self.protocol[self.current_index]
        self.stage_started_at = time.time()
        self.recorder.set_stage(self.current_index, stage)
        self.progress.setValue(self.current_index + 1)
        gesture = str(stage["gesture_label"])
        self.stage_title.setText(str(stage["title"]).replace("_", " "))
        image_path = PICTURES_DIR / GESTURE_IMAGES.get(gesture, "")
        if image_path.exists():
            pixmap = QtGui.QPixmap(str(image_path))
            self.gesture_image.setPixmap(pixmap.scaled(480, 240, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        next_stage = self.protocol[self.current_index + 1] if self.current_index + 1 < len(self.protocol) else None
        self.cue_lane.set_cues(stage, next_stage, float(stage["duration_s"]))

    def request_evaluation(self) -> None:
        """Perform the request evaluation operation used by the ModelTestDialog workflow."""
        if self.session_dir is None:
            return
        model_path = self.selected_model_path()
        if not model_path:
            QtWidgets.QMessageBox.warning(self, "Model required", "The selected model is no longer available.")
            return
        self.evaluate_btn.setEnabled(False)
        self.stage_title.setText("Evaluating recorded test...")
        self.test_requested.emit(str(self.session_dir), model_path)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window closure and release application resources safely."""
        if self.running:
            self.stop_test()
        super().closeEvent(event)


class ModelUpdateDialog(ModelTestDialog):
    """Represent the ModelUpdateDialog component and keep its related state and behavior together."""
    def __init__(self, current_model_path: str = "", parent=None):
        """Initialize the ModelUpdateDialog instance and its runtime state."""
        super().__init__(current_model_path, parent)
        self.protocol = build_short_model_update_protocol()
        self.progress.setRange(0, len(self.protocol))
        duration = sum(float(stage["duration_s"]) for stage in self.protocol)
        self.setWindowTitle("Short Model Recalibration")
        self.stage_title.setText("Ready for short recalibration")
        self.start_btn.setText("Start short recalibration")
        self.evaluate_btn.setText("Train update candidate")
        self.duration_note.setText(
            f"Duration: approximately {duration:.0f} seconds. The original model is preserved; a versioned update is created."
        )

    def start_test(self) -> None:
        """Start test for the current ModelUpdateDialog workflow."""
        model_path = self.selected_model_path()
        if not model_path:
            QtWidgets.QMessageBox.warning(self, "Model required", "Select the existing personal model to update.")
            return
        if not self.all_sensors_ready():
            QtWidgets.QMessageBox.warning(self, "Sensors not ready", "All three uMyo sensors must be connected and streaming.")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_session = Path(model_path).parent.parent.name
        self.session_dir = MODEL_UPDATE_DIR / f"{base_session}_update_{stamp}"
        self.recorder.start(self.session_dir, self.protocol)
        (self.session_dir / "base_model.txt").write_text(model_path, encoding="utf-8")
        self.current_index = -1
        self.running = True
        self.quality_status.setText("Recording quality: waiting for first completed stage")
        self.evaluate_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self._advance_stage()
        self.timer.start(100)


class RightPanel(QtWidgets.QWidget):
    """Represent the RightPanel component and keep its related state and behavior together."""
    settings_changed = QtCore.Signal()
    model_load_requested = QtCore.Signal(str)
    calibration_requested = QtCore.Signal()
    model_test_requested = QtCore.Signal()
    model_update_requested = QtCore.Signal()
    mouse_control_requested = QtCore.Signal()
    decision_changed = QtCore.Signal()

    def __init__(self, config: dict, parent=None):
        """Initialize the RightPanel instance and its runtime state."""
        super().__init__(parent)
        self.probability_bars: dict[str, QtWidgets.QProgressBar] = {}
        self._build_ui(config)

    def _build_ui(self, config: dict) -> None:
        """Create and configure ui for the current RightPanel workflow."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 3, 10, 6)

        header = QtWidgets.QHBoxLayout()
        logo = QtWidgets.QLabel()
        logo.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        logo_path = PICTURES_DIR / LOGO_IMAGE
        if logo_path.exists():
            pixmap = QtGui.QPixmap(str(logo_path))
            logo.setPixmap(pixmap.scaled(128, 70, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        header.addWidget(logo)

        self.gesture_label = QtWidgets.QLabel("No model")
        self.gesture_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gesture_label.setStyleSheet("font-size: 38px; font-weight: 700; color: #1f2937;")
        self.gesture_label.setFixedHeight(54)
        header.addWidget(self.gesture_label, stretch=1)
        header.addSpacing(128)
        layout.addLayout(header)

        self.confidence_bar = QtWidgets.QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setFormat("Confidence: %p%")
        layout.addWidget(self.confidence_bar)

        gesture_box = QtWidgets.QFrame()
        gesture_box.setFixedHeight(188)
        gesture_box.setStyleSheet("border: 1px solid #d1d5db; background: #f9fafb;")
        gesture_box_layout = QtWidgets.QVBoxLayout(gesture_box)
        gesture_box_layout.setContentsMargins(6, 6, 6, 5)
        gesture_box_layout.setSpacing(4)
        self.gesture_image = QtWidgets.QLabel()
        self.gesture_image.setAlignment(QtCore.Qt.AlignCenter)
        self.gesture_image.setFixedHeight(120)
        self.gesture_image.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.gesture_image.setStyleSheet("border: 0; background: transparent;")
        gesture_box_layout.addWidget(self.gesture_image)

        self.status_label = QtWidgets.QLabel("Status: starting")
        self.model_info_label = QtWidgets.QLabel("Model: none")
        self.warning_label = QtWidgets.QLabel("Warnings: none")
        self.warning_label.setStyleSheet("color: #8a4b00;")
        self.training_result_label = QtWidgets.QLabel("Training: idle")
        self.training_result_label.setStyleSheet("color: #0f766e; font-weight: 600;")
        self.noise_label = QtWidgets.QLabel("Noise: waiting")
        self.noise_label.setStyleSheet("color: #64748b; font-weight: 600;")
        info_labels = [self.status_label, self.model_info_label, self.warning_label, self.training_result_label, self.noise_label]
        for label in info_labels:
            label.setWordWrap(False)
            label.setMinimumWidth(0)
            label.setStyleSheet(label.styleSheet() + " font-size: 10px; border: 0; background: transparent;")
            label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        info_grid = QtWidgets.QGridLayout()
        info_grid.setContentsMargins(0, 0, 0, 0)
        info_grid.setHorizontalSpacing(8)
        info_grid.setVerticalSpacing(1)
        info_grid.addWidget(self.status_label, 0, 0)
        info_grid.addWidget(self.model_info_label, 0, 1)
        info_grid.addWidget(self.warning_label, 1, 0)
        info_grid.addWidget(self.training_result_label, 1, 1)
        info_grid.addWidget(self.noise_label, 2, 0, 1, 2)
        info_grid.setColumnStretch(0, 1)
        info_grid.setColumnStretch(1, 1)
        gesture_box_layout.addLayout(info_grid)
        layout.addWidget(gesture_box)

        probabilities = QtWidgets.QGroupBox("Gesture probabilities")
        probabilities_layout = QtWidgets.QGridLayout(probabilities)
        probabilities.setStyleSheet("QGroupBox { font-size: 11px; } QLabel { font-size: 10px; }")
        probabilities_layout.setContentsMargins(8, 8, 8, 6)
        probabilities_layout.setHorizontalSpacing(10)
        probabilities_layout.setVerticalSpacing(4)
        for index, gesture in enumerate(DISPLAY_GESTURES):
            row = index // 2
            col = (index % 2) * 2
            label = QtWidgets.QLabel(gesture)
            label.setMinimumWidth(100)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("%p%")
            bar.setValue(0)
            bar.setMinimumWidth(190)
            bar.setFixedHeight(18)
            self.probability_bars[gesture] = bar
            probabilities_layout.addWidget(label, row, col)
            probabilities_layout.addWidget(bar, row, col + 1)
        probabilities.setFixedHeight(112)
        layout.addWidget(probabilities)

        decision = QtWidgets.QGroupBox("Decision mechanism")
        decision.setStyleSheet("QGroupBox { font-size: 10px; } QLabel { font-size: 9px; }")
        decision_layout = QtWidgets.QGridLayout(decision)
        decision_layout.setContentsMargins(7, 7, 7, 5)
        decision_layout.setHorizontalSpacing(6)
        decision_layout.setVerticalSpacing(3)
        self.decision_mode = QtWidgets.QComboBox()
        self.decision_mode.addItem("Model winner", "model")
        self.decision_mode.addItem("Custom", "custom")
        self.decision_mode.setFixedHeight(22)
        self.model_winner_label = QtWidgets.QLabel("Winner: none")
        self.model_winner_label.setWordWrap(True)
        self.model_winner_label.setStyleSheet("font-size: 9px; color: #334155;")
        self.custom_method = QtWidgets.QComboBox()
        for label, value in [
            ("Raw", "raw"),
            ("Threshold", "threshold"),
            ("Majority vote", "majority"),
            ("Consecutive", "consecutive"),
            ("Hysteresis", "hysteresis"),
        ]:
            self.custom_method.addItem(label, value)
        self.custom_method.setFixedHeight(22)
        self.custom_threshold = QtWidgets.QDoubleSpinBox()
        self.custom_threshold.setRange(0.0, 1.0)
        self.custom_threshold.setSingleStep(0.05)
        self.custom_threshold.setValue(0.55)
        self.custom_threshold.setFixedHeight(22)
        self.custom_enter = QtWidgets.QDoubleSpinBox()
        self.custom_enter.setRange(0.0, 1.0)
        self.custom_enter.setSingleStep(0.05)
        self.custom_enter.setValue(0.55)
        self.custom_enter.setFixedHeight(22)
        self.custom_stay = QtWidgets.QDoubleSpinBox()
        self.custom_stay.setRange(0.0, 1.0)
        self.custom_stay.setSingleStep(0.05)
        self.custom_stay.setValue(0.40)
        self.custom_stay.setFixedHeight(22)
        self.custom_window = QtWidgets.QSpinBox()
        self.custom_window.setRange(1, 9)
        self.custom_window.setValue(3)
        self.custom_window.setFixedHeight(22)
        self.custom_confusion_threshold = QtWidgets.QDoubleSpinBox()
        self.custom_confusion_threshold.setRange(0.0, 1.0)
        self.custom_confusion_threshold.setSingleStep(0.05)
        self.custom_confusion_threshold.setValue(DEFAULT_CONFUSION_THRESHOLD)
        self.custom_confusion_threshold.setFixedHeight(22)
        for widget in [
            self.decision_mode,
            self.custom_method,
            self.custom_threshold,
            self.custom_enter,
            self.custom_stay,
            self.custom_window,
            self.custom_confusion_threshold,
        ]:
            widget.setStyleSheet("font-size: 9px;")
        decision_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        decision_layout.addWidget(self.decision_mode, 0, 1)
        decision_layout.addWidget(self.model_winner_label, 0, 2, 1, 2)
        decision_layout.addWidget(QtWidgets.QLabel("Method"), 1, 0)
        decision_layout.addWidget(self.custom_method, 1, 1)
        decision_layout.addWidget(QtWidgets.QLabel("Threshold"), 1, 2)
        decision_layout.addWidget(self.custom_threshold, 1, 3)
        decision_layout.addWidget(QtWidgets.QLabel("Enter"), 2, 0)
        decision_layout.addWidget(self.custom_enter, 2, 1)
        decision_layout.addWidget(QtWidgets.QLabel("Stay"), 2, 2)
        decision_layout.addWidget(self.custom_stay, 2, 3)
        decision_layout.addWidget(QtWidgets.QLabel("Windows"), 3, 0)
        decision_layout.addWidget(self.custom_window, 3, 1)
        decision_layout.addWidget(QtWidgets.QLabel("Confusing pair"), 3, 2)
        decision_layout.addWidget(self.custom_confusion_threshold, 3, 3)
        self._custom_decision_widgets = [
            self.custom_method,
            self.custom_threshold,
            self.custom_enter,
            self.custom_stay,
            self.custom_window,
            self.custom_confusion_threshold,
        ]
        self.model_winner_strategy = "raw_no_gate"
        for widget in [
            self.decision_mode,
            self.custom_method,
            self.custom_threshold,
            self.custom_enter,
            self.custom_stay,
            self.custom_window,
            self.custom_confusion_threshold,
        ]:
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._decision_ui_changed)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._decision_ui_changed)
        self._update_custom_decision_enabled()
        decision.setFixedHeight(142)
        layout.addWidget(decision)

        mid_row = QtWidgets.QHBoxLayout()
        calibration = QtWidgets.QGroupBox("Personal calibration")
        calibration.setStyleSheet("QGroupBox { font-size: 11px; } QLabel { font-size: 10px; } QPushButton, QComboBox, QCheckBox { font-size: 10px; }")
        calibration_layout = QtWidgets.QGridLayout(calibration)
        calibration_layout.setContentsMargins(8, 8, 8, 6)
        calibration_layout.setHorizontalSpacing(8)
        calibration_layout.setVerticalSpacing(4)
        self.user_model_combo = QtWidgets.QComboBox()
        self.user_model_combo.setMinimumWidth(220)
        self.refresh_users_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_users_btn.clicked.connect(self.refresh_user_models)
        self.load_user_model_btn = QtWidgets.QPushButton("Load user model")
        self.load_user_model_btn.clicked.connect(self._load_selected_user_model)
        self.open_calibration_btn = QtWidgets.QPushButton("Open training window")
        self.open_calibration_btn.clicked.connect(self.calibration_requested.emit)
        self.update_model_btn = QtWidgets.QPushButton("Short recalibration")
        self.update_model_btn.clicked.connect(self.model_update_requested.emit)
        self.test_model_btn = QtWidgets.QPushButton("Test the model")
        self.test_model_btn.clicked.connect(self.model_test_requested.emit)
        self.open_mouse_demo_btn = QtWidgets.QPushButton("Open mouse control, demo, and training")
        self.open_mouse_demo_btn.clicked.connect(self.mouse_control_requested.emit)
        for widget in [self.user_model_combo, self.refresh_users_btn, self.load_user_model_btn, self.open_calibration_btn, self.update_model_btn, self.test_model_btn, self.open_mouse_demo_btn]:
            widget.setFixedHeight(24)
        calibration_layout.addWidget(QtWidgets.QLabel("Known user"), 0, 0)
        calibration_layout.addWidget(self.user_model_combo, 0, 1)
        calibration_layout.addWidget(self.refresh_users_btn, 0, 2)
        training_buttons = QtWidgets.QHBoxLayout()
        training_buttons.setContentsMargins(0, 0, 0, 0)
        training_buttons.setSpacing(8)
        training_buttons.addWidget(self.load_user_model_btn)
        training_buttons.addWidget(self.open_calibration_btn)
        calibration_layout.addLayout(training_buttons, 1, 0, 1, 3)
        model_action_buttons = QtWidgets.QHBoxLayout()
        model_action_buttons.setContentsMargins(0, 0, 0, 0)
        model_action_buttons.setSpacing(8)
        model_action_buttons.addWidget(self.update_model_btn)
        model_action_buttons.addWidget(self.test_model_btn)
        model_action_buttons.addWidget(self.open_mouse_demo_btn)
        calibration_layout.addLayout(model_action_buttons, 2, 0, 1, 3)
        self.pause_plots = QtWidgets.QCheckBox("Pause plots")
        calibration_layout.addWidget(self.pause_plots, 3, 0, 1, 3)
        calibration.setFixedHeight(146)
        mid_row.addWidget(calibration, stretch=3)
        layout.addLayout(mid_row)
        layout.addStretch()
        self.refresh_user_models()

    def _browse(self, target: QtWidgets.QLineEdit) -> None:
        """Perform the browse operation used by the RightPanel workflow."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", str(PROJECT_ROOT), "Model files (*.pkl *.joblib *.sav);;All files (*)")
        if path:
            target.setText(path)

    def selected_sensors(self) -> list[str]:
        """Perform the selected sensors operation used by the RightPanel workflow."""
        return list(SENSOR_ORDER)

    def refresh_user_models(self) -> None:
        """Refresh user models for the current RightPanel workflow."""
        self.user_model_combo.clear()
        self.user_model_combo.addItem("Select saved user model", "")
        if not CALIBRATION_DIR.exists():
            return
        model_paths = list(CALIBRATION_DIR.glob("*/trained_model/personal_model.pkl"))
        model_paths.extend(CALIBRATION_DIR.glob("*/trained_model/personal_fast_model.pkl"))
        model_paths.extend(CALIBRATION_DIR.glob("*/trained_model/personal_model_update_*.pkl"))
        model_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in model_paths:
            self.user_model_combo.addItem(self._saved_model_label(path), str(path))

    def _saved_model_label(self, model_path: Path) -> str:
        """Perform the saved model label operation used by the RightPanel workflow."""
        session_dir = model_path.parent.parent
        if model_path.name.startswith("personal_model_update_"):
            try:
                with model_path.open("rb") as handle:
                    artifact = pickle.load(handle)
                adaptation = artifact.get("adaptation", {}) if isinstance(artifact, dict) else {}
                candidate = adaptation.get("candidate_validation", {})
                ba = candidate.get("balanced_accuracy")
                ba_text = f" BA {float(ba):.3f}" if ba is not None else ""
                return (
                    f"{session_dir.name} | Short update | {artifact.get('model_type', 'model')} | "
                    f"{artifact.get('window_ms', '?')}ms{ba_text}"
                )
            except Exception:
                return f"{session_dir.name} | Short update"
        best = self._best_result_from_session(session_dir, model_path.name)
        if not best:
            return session_dir.name
        model_type = best.get("model_type", "model")
        window_ms = best.get("window_ms", "?")
        margin_ms = best.get("trim_edge_ms", "?")
        ba = best.get("validation_balanced_accuracy")
        test_ba = best.get("test_balanced_accuracy")
        ba_text = f" test BA {float(test_ba):.3f}" if test_ba is not None else f" val BA {float(ba):.3f}" if ba is not None else ""
        training_mode = "Fast Train" if model_path.name == "personal_fast_model.pkl" else "Full Grid"
        return f"{session_dir.name} | {training_mode} | {model_type} | {window_ms}ms m{margin_ms}{ba_text}"

    @staticmethod
    def _best_result_from_session(session_dir: Path, model_name: str = "") -> dict[str, object]:
        """Perform the best result from session operation used by the RightPanel workflow."""
        if model_name == "personal_fast_model.pkl":
            summary_names = ["personal_training_summary.json"]
        elif model_name == "personal_model.pkl":
            summary_names = ["personal_grid_summary.json"]
        else:
            summary_names = ["personal_grid_summary.json", "personal_training_summary.json"]
        for name in summary_names:
            path = session_dir / "trained_model" / name
            if not path.exists():
                continue
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                best = summary.get("best_result") or {}
                if best:
                    return best
            except Exception:
                continue
        return {}

    def _load_selected_user_model(self) -> None:
        """Load and validate selected user model for the current RightPanel workflow."""
        path = self.user_model_combo.currentData()
        if path:
            self.model_load_requested.emit(str(path))

    def set_model_window(self, window_ms: int | None) -> None:
        """Set model window for the current RightPanel workflow."""
        return

    def threshold_value(self) -> float:
        """Perform the threshold value operation used by the RightPanel workflow."""
        config = self.decision_config()
        if config["method"] == "hysteresis":
            return float(config.get("enter", 0.55))
        return float(config.get("threshold", 0.55))

    def set_model_decision_strategy(self, strategy: str) -> None:
        """Set model decision strategy for the current RightPanel workflow."""
        self.model_winner_strategy = strategy or "raw_no_gate"
        self.model_winner_label.setText(f"Winner: {self.model_winner_strategy}")
        self._decision_ui_changed()

    def decision_config(self) -> dict[str, object]:
        """Perform the decision config operation used by the RightPanel workflow."""
        if self.decision_mode.currentData() == "model":
            return parse_decision_strategy(self.model_winner_strategy)
        method = str(self.custom_method.currentData())
        return {
            "method": method,
            "threshold": float(self.custom_threshold.value()),
            "enter": float(self.custom_enter.value()),
            "stay": float(self.custom_stay.value()),
            "window": int(self.custom_window.value()),
            "confusion_threshold": float(self.custom_confusion_threshold.value()),
        }

    def _decision_ui_changed(self, *_args) -> None:
        """Perform the decision ui changed operation used by the RightPanel workflow."""
        self._update_custom_decision_enabled()
        self.decision_changed.emit()

    def _update_custom_decision_enabled(self) -> None:
        """Refresh custom decision enabled for the current RightPanel workflow."""
        custom = self.decision_mode.currentData() == "custom"
        method = str(self.custom_method.currentData())
        for widget in self._custom_decision_widgets:
            widget.setEnabled(custom)
        self.custom_threshold.setEnabled(custom and method in {"threshold", "majority", "consecutive"})
        self.custom_window.setEnabled(custom and method in {"majority", "consecutive"})
        self.custom_enter.setEnabled(custom and method == "hysteresis")
        self.custom_stay.setEnabled(custom and method == "hysteresis")
        self.custom_confusion_threshold.setEnabled(custom)

    def set_prediction(self, result: PredictionResult) -> None:
        """Set prediction for the current RightPanel workflow."""
        shown_gesture = result.stable_gesture or result.gesture
        hidden_removed_gesture = shown_gesture == "side_flex"
        if hidden_removed_gesture:
            shown_gesture = "Uncertain"
        self.gesture_label.setText(shown_gesture)
        self.confidence_bar.setValue(int(max(0.0, min(1.0, result.confidence)) * 100))
        self._set_probabilities(result.probabilities or {})
        self._set_gesture_image(shown_gesture)
        if result.error:
            self.warning_label.setText(f"Warnings: {result.error}")
        elif hidden_removed_gesture:
            self.warning_label.setText("Warnings: side_flex is disabled in V2")
        elif result.is_uncertain:
            self.warning_label.setText("Warnings: prediction below threshold")
        elif result.debug_info:
            self.warning_label.setText(result.debug_info)
        else:
            self.warning_label.setText("Warnings: none")

    def set_noise_status(self, snapshot: dict[str, object]) -> None:
        """Set noise status for the current RightPanel workflow."""
        sensors = snapshot.get("sensors", [])
        if not sensors:
            self.noise_label.setText("Noise: waiting")
            return
        worst = snapshot.get("label", "OK")
        max_rms = max(float(row["rms"]) for row in sensors)
        self.noise_label.setText(f"Noise: {worst} | max rest RMS {max_rms:.0f}")
        self.noise_label.setStyleSheet(f"color: {snapshot.get('color', '#64748b')}; font-weight: 700; font-size: 10px; border: 0; background: transparent;")

    def _set_probabilities(self, probabilities: dict[str, float]) -> None:
        """Set probabilities for the current RightPanel workflow."""
        for gesture, bar in self.probability_bars.items():
            value = float(probabilities.get(gesture, 0.0))
            bar.setValue(int(max(0.0, min(1.0, value)) * 100))

    def _set_gesture_image(self, gesture: str) -> None:
        """Set gesture image for the current RightPanel workflow."""
        key = gesture.strip()
        if key in {"Uncertain", "Error", "No model"}:
            self.gesture_image.clear()
            return
        image_name = GESTURE_IMAGES.get(key)
        image_path = PICTURES_DIR / image_name if image_name else None
        if not image_path or not image_path.exists():
            self.gesture_image.clear()
            return
        pixmap = QtGui.QPixmap(str(image_path))
        if pixmap.isNull():
            self.gesture_image.clear()
            return
        self.gesture_image.setPixmap(
            pixmap.scaled(
                max(120, self.gesture_image.width() - 14),
                116,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )


class RealtimeGestureGui(QtWidgets.QMainWindow):
    """Represent the RealtimeGestureGui component and keep its related state and behavior together."""
    def __init__(self, config: dict):
        """Initialize the RealtimeGestureGui instance and its runtime state."""
        super().__init__()
        self.config = config
        self.fs = float(config.get("sampling_rate_hz", 1100.0))
        self.last_prediction_time = 0.0
        self.last_plot_refresh_time = 0.0
        self.last_metrics_update_time = 0.0
        self.last_update_loop_time = time.perf_counter()
        self.update_loop_intervals_ms: deque[float] = deque(maxlen=300)
        self.realtime_priority = configure_realtime_priority()
        self.packet_count = 0
        self.calibration_dialog: CalibrationDialog | None = None
        self.model_test_dialog: ModelTestDialog | None = None
        self.model_update_dialog: ModelUpdateDialog | None = None
        self.model_test_results_dialog: ModelTestResultsDialog | None = None
        self.mouse_control_window: MouseControlWindow | None = None
        self.training_process: QtCore.QProcess | None = None
        self.replay_process: QtCore.QProcess | None = None
        self.decision_process: QtCore.QProcess | None = None
        self.model_test_process: QtCore.QProcess | None = None
        self.model_update_process: QtCore.QProcess | None = None
        self.training_started_at = 0.0
        self.training_mode = ""
        self.decision_smoother = DecisionSmoother()
        self.last_stable_gesture = "No model"
        self.last_gesture_confidence = 0.0
        self.last_raw_gesture = "No model"
        self.last_raw_confidence = 0.0
        self.noise_profile = load_latest_noise_profile()
        self.signal_safety_gate = RealtimeSignalSafetyGate(SENSOR_ORDER)
        self.signal_safety = {"safe": False, "reason": "waiting for all sensors", "changed": False}
        self.last_signal_safety_update = 0.0
        self.latest_noise_snapshot: dict[str, object] = {"label": "OK", "sensors": []}
        self.training_timer = QtCore.QTimer(self)
        self.training_timer.timeout.connect(self._update_training_status)
        self.reader = UmyoSerialReader(baudrate=int(config.get("baudrate", 921600)))
        self.classifier = GestureClassifierAdapter(
            fs=self.fs,
            selected_channels=list(range(len(SENSOR_ORDER))),
            confidence_threshold=float(config.get("confidence_threshold", 0.55)),
        )
        self._build_ui()
        self.right_panel.status_label.setText("Status: streaming, no classifier loaded")
        self.reader.start()
        keep_display_awake(True)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(5)

    def _build_ui(self) -> None:
        """Create and configure ui for the current RealtimeGestureGui workflow."""
        self.setWindowTitle("uMyo Realtime Gesture Recognition")
        self.resize(1500, 850)
        pg.setConfigOptions(antialias=False)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        self.emg_panel = ThreeSensorEmgPanel(samples=int(self.config.get("emg_plot_samples", 1800)), fs=self.fs)
        self.right_panel = RightPanel(self.config)
        self.emg_panel.setMinimumWidth(700)
        self.right_panel.setMinimumWidth(700)
        self.right_panel.settings_changed.connect(self.apply_settings)
        self.right_panel.model_load_requested.connect(self.load_model)
        self.right_panel.calibration_requested.connect(self.open_calibration)
        self.right_panel.model_test_requested.connect(self.open_model_test)
        self.right_panel.model_update_requested.connect(self.open_model_update)
        self.right_panel.mouse_control_requested.connect(self.open_mouse_control)
        self.right_panel.decision_changed.connect(self.apply_decision_settings)
        layout.addWidget(self.emg_panel, stretch=3)
        layout.addWidget(self.right_panel, stretch=2)
        self.apply_settings()
        self.apply_decision_settings()

    def load_model(self, model_path: str) -> None:
        """Load and validate model for the current RealtimeGestureGui workflow."""
        ok = self.classifier.load(model_path, None)
        if ok:
            window_ms = self.classifier.window_ms(int(self.config.get("window_ms", 500)))
            model_fs = self.classifier.sampling_rate_hz(self.fs)
            strategy, decision_text = self._decision_strategy_for_model(Path(model_path))
            self.right_panel.set_model_decision_strategy(strategy)
            self.apply_decision_settings()
            self.right_panel.set_model_window(window_ms)
            self.right_panel.model_info_label.setText(
                f"Model: {Path(model_path).name} | window {window_ms} ms | fs {model_fs:.1f} Hz{decision_text}"
            )
            self.right_panel.status_label.setText(
                f"Status: loaded {Path(model_path).name} | window {window_ms} ms | model fs {model_fs:.1f} Hz"
            )
            if isinstance(self.classifier.model, dict) and "side_flex" in list(self.classifier.model.get("classes", [])):
                self.right_panel.warning_label.setText("Warnings: this older model includes disabled side_flex")
        else:
            self.right_panel.status_label.setText("Status: model load failed")
            self.right_panel.warning_label.setText(f"Warnings: {self.classifier.last_error}")

    def _decision_strategy_for_model(self, model_path: Path) -> tuple[str, str]:
        """Perform the decision strategy for model operation used by the RealtimeGestureGui workflow."""
        session_dir = model_path.parent.parent
        replay_root = session_dir / "replay_report"
        if not replay_root.exists():
            return "raw_no_gate", ""
        summaries = sorted(
            replay_root.glob("*/decision_strategy_comparison/decision_strategy_summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not summaries:
            return "raw_no_gate", ""
        try:
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            best = summary.get("best_strategy") or {}
            strategy = best.get("strategy")
            ba = best.get("balanced_accuracy_unknown_as_class")
            if strategy and ba is not None:
                return str(strategy), f" | decision {strategy} BA {float(ba):.3f}"
        except Exception:
            return "raw_no_gate", ""
        return "raw_no_gate", ""

    def open_calibration(self) -> None:
        """Open or display calibration for the current RealtimeGestureGui workflow."""
        if self.calibration_dialog is None:
            self.calibration_dialog = CalibrationDialog(self)
            self.calibration_dialog.train_requested.connect(self.handle_train_requested)
        self.calibration_dialog.show()
        self.calibration_dialog.raise_()
        self.calibration_dialog.activateWindow()

    def open_model_test(self) -> None:
        """Open or display model test for the current RealtimeGestureGui workflow."""
        current_model = str(self.classifier.model_path or "")
        if self.model_test_dialog is None:
            self.model_test_dialog = ModelTestDialog(current_model, self)
            self.model_test_dialog.test_requested.connect(self.handle_model_test_requested)
        else:
            self.model_test_dialog.set_current_model(current_model)
        self.model_test_dialog.show()
        self.model_test_dialog.raise_()
        self.model_test_dialog.activateWindow()

    def open_model_update(self) -> None:
        """Open or display model update for the current RealtimeGestureGui workflow."""
        current_model = str(self.classifier.model_path or "")
        if self.model_update_dialog is None:
            self.model_update_dialog = ModelUpdateDialog(current_model, self)
            self.model_update_dialog.test_requested.connect(self.handle_model_update_requested)
        else:
            self.model_update_dialog.set_current_model(current_model)
        self.model_update_dialog.show()
        self.model_update_dialog.raise_()
        self.model_update_dialog.activateWindow()

    def handle_model_update_requested(self, session_dir: str, base_model_path: str) -> None:
        """Handle model update requested for the current RealtimeGestureGui workflow."""
        if self.model_update_process is not None and self.model_update_process.state() != QtCore.QProcess.NotRunning:
            self.right_panel.warning_label.setText("Warnings: short recalibration is already running")
            return
        self.right_panel.training_result_label.setText("Training: short recalibration running...")
        self.model_update_process = QtCore.QProcess(self)
        self.model_update_process.setProgram(sys.executable)
        self.model_update_process.setArguments(
            [
                str(APP_DIR / "update_personal_model.py"),
                "--base-model",
                base_model_path,
                "--update-session",
                session_dir,
            ]
        )
        self.model_update_process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.model_update_process.finished.connect(
            lambda _code, _status: self._model_update_finished(Path(session_dir))
        )
        self.model_update_process.start()

    def _model_update_finished(self, session_dir: Path) -> None:
        """Perform the model update finished operation used by the RealtimeGestureGui workflow."""
        output = ""
        if self.model_update_process is not None:
            output = bytes(self.model_update_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        summary_path = session_dir / "model_update_summary.json"
        if not summary_path.exists():
            self.right_panel.warning_label.setText(
                f"Warnings: short recalibration failed | {output[-400:] if output else 'no output'}"
            )
            if self.model_update_dialog is not None:
                self.model_update_dialog.stage_title.setText("Short recalibration failed")
                self.model_update_dialog.evaluate_btn.setEnabled(True)
            return
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            base_ba = float(summary["base_validation"]["balanced_accuracy"])
            candidate_ba = float(summary["candidate_validation"]["balanced_accuracy"])
            improvement = float(summary["balanced_accuracy_improvement"])
            recommended = bool(summary.get("recommended", False))
            candidate_path = Path(summary["candidate_model"])
            status = "PROMOTED" if recommended else "saved for review"
            text = (
                f"Short recalibration complete | old BA {base_ba:.3f} | candidate BA {candidate_ba:.3f} | "
                f"change {improvement:+.3f} | {status}"
            )
            self.right_panel.training_result_label.setText(text)
            self.right_panel.refresh_user_models()
            if recommended and candidate_path.exists():
                self.load_model(str(candidate_path))
            if self.model_update_dialog is not None:
                self.model_update_dialog.stage_title.setText(text)
                self.model_update_dialog.evaluate_btn.setEnabled(True)
            QtWidgets.QMessageBox.information(
                self,
                "Short recalibration complete",
                text + f"\n\nOriginal model preserved.\nCandidate: {candidate_path}",
            )
        except Exception as exc:
            self.right_panel.warning_label.setText(f"Warnings: update summary could not be read ({exc})")

    def handle_model_test_requested(self, session_dir: str, model_path: str) -> None:
        """Handle model test requested for the current RealtimeGestureGui workflow."""
        if self.model_test_process is not None and self.model_test_process.state() != QtCore.QProcess.NotRunning:
            self.right_panel.warning_label.setText("Warnings: a model test is already running")
            return
        session_path = Path(session_dir)
        output_root = session_path / "replay_report"
        self.model_test_process = QtCore.QProcess(self)
        self.model_test_process.setProgram(sys.executable)
        self.model_test_process.setArguments(
            [
                str(APP_DIR / "replay_evaluation.py"),
                "replay",
                "--data-dir",
                str(session_path / "raw_recordings"),
                "--model-path",
                model_path,
                "--output-dir",
                str(output_root),
                "--name",
                "current_test",
                "--confidence-threshold",
                f"{self.right_panel.threshold_value():.2f}",
                "--majority-windows",
                "1",
            ]
        )
        self.model_test_process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.model_test_process.finished.connect(
            lambda _code, _status: self._model_test_finished(session_path, Path(model_path))
        )
        self.model_test_process.start()

    @staticmethod
    def _prediction_metrics(predictions_path: Path) -> tuple[dict[str, float], list[str], np.ndarray, list[dict[str, object]]]:
        """Perform the prediction metrics operation used by the RealtimeGestureGui workflow."""
        import pandas as pd

        frame = pd.read_csv(predictions_path)
        true = frame["true_label"].astype(str).to_numpy()
        predicted = frame["raw_prediction"].astype(str).to_numpy()
        labels = sorted(set(true) | set(predicted))
        recall_values = recall_score(true, predicted, labels=labels, average=None, zero_division=0)
        f1_values = f1_score(true, predicted, labels=labels, average=None, zero_division=0)
        metrics = {
            "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
            "macro_recall": float(recall_score(true, predicted, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        }
        per_class = [
            {
                "gesture": GESTURE_DISPLAY_NAMES.get(label, label),
                "recall": float(recall_values[index]),
                "f1": float(f1_values[index]),
            }
            for index, label in enumerate(labels)
        ]
        return metrics, labels, confusion_matrix(true, predicted, labels=labels), per_class

    def _original_replay_metrics(self, model_path: Path) -> dict[str, float] | None:
        """Perform the original replay metrics operation used by the RealtimeGestureGui workflow."""
        replay_root = model_path.parent.parent / "replay_report"
        if not replay_root.exists():
            return None
        candidates = sorted(
            replay_root.rglob("replay_predictions.csv"),
            key=lambda path: ("auto_replay" not in path.parts, -path.stat().st_mtime),
        )
        for path in candidates:
            try:
                metrics, _labels, _matrix, _per_class = self._prediction_metrics(path)
                return metrics
            except Exception:
                continue
        return None

    def _model_test_finished(self, session_dir: Path, model_path: Path) -> None:
        """Perform the model test finished operation used by the RealtimeGestureGui workflow."""
        output = ""
        if self.model_test_process is not None:
            output = bytes(self.model_test_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        predictions_path = session_dir / "replay_report" / "current_test" / "replay_predictions.csv"
        if not predictions_path.exists():
            self.right_panel.warning_label.setText(
                f"Warnings: model test failed | {output[-350:] if output else 'no evaluator output'}"
            )
            if self.model_test_dialog is not None:
                self.model_test_dialog.stage_title.setText("Model test failed")
                self.model_test_dialog.evaluate_btn.setEnabled(True)
            return
        try:
            current, labels, matrix, per_class = self._prediction_metrics(predictions_path)
            original = self._original_replay_metrics(model_path)
            result_payload = {
                "model_path": str(model_path),
                "test_session": str(session_dir),
                "current_test": current,
                "original_replay": original,
                "labels": labels,
                "confusion_matrix": matrix.tolist(),
                "per_class": per_class,
            }
            (session_dir / "model_test_summary.json").write_text(
                json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.model_test_results_dialog = ModelTestResultsDialog(
                labels, matrix, current, per_class, original, self
            )
            self.model_test_results_dialog.show()
            self.model_test_results_dialog.raise_()
            if self.model_test_dialog is not None:
                self.model_test_dialog.stage_title.setText(
                    f"Test complete | BA {current['balanced_accuracy']:.3f} | F1 {current['macro_f1']:.3f}"
                )
                self.model_test_dialog.evaluate_btn.setEnabled(True)
        except Exception as exc:
            self.right_panel.warning_label.setText(f"Warnings: model-test results failed ({exc})")

    def open_mouse_control(self) -> None:
        """Open or display mouse control for the current RealtimeGestureGui workflow."""
        if self.mouse_control_window is None:
            self.mouse_control_window = MouseControlWindow(self)
        self.mouse_control_window.show()
        self.mouse_control_window.raise_()
        self.mouse_control_window.activateWindow()

    def handle_train_requested(self, session_dir: str, mode: str) -> None:
        """Handle train requested for the current RealtimeGestureGui workflow."""
        if self.training_process is not None and self.training_process.state() != QtCore.QProcess.NotRunning:
            self.right_panel.warning_label.setText("Warnings: training is already running")
            return
        audit = audit_session_for_training(Path(session_dir), SENSOR_ORDER)
        if not audit["passed"]:
            self.right_panel.warning_label.setText(
                "Warnings: training blocked by recording quality | " + "; ".join(audit["blockers"][:3])
            )
            return
        mode_name = "Fast train" if mode == "fast" else "Full grid"
        self.training_mode = mode
        self.training_started_at = time.time()
        self.right_panel.training_result_label.setText(f"Training: {mode_name} started...")
        self.right_panel.warning_label.setText("Warnings: training is running; realtime streaming may be slower")
        self.training_process = QtCore.QProcess(self)
        self.training_process.setProgram(sys.executable)
        self.training_process.setArguments(
            [
                str(APP_DIR / "personal_stage_training.py"),
                "--session-dir",
                session_dir,
                "--mode",
                mode,
            ]
        )
        self.training_process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.training_process.finished.connect(lambda _code, _status: self._training_finished(session_dir, mode))
        self.training_process.start()
        self.training_timer.start(1000)

    def _update_training_status(self) -> None:
        """Refresh training status for the current RealtimeGestureGui workflow."""
        if self.training_started_at <= 0:
            return
        elapsed = int(time.time() - self.training_started_at)
        mode_name = "Fast train" if self.training_mode == "fast" else "Full grid"
        self.right_panel.training_result_label.setText(f"Training: {mode_name} running | elapsed {elapsed // 60:02d}:{elapsed % 60:02d}")

    def _training_finished(self, session_dir: str, mode: str) -> None:
        """Perform the training finished operation used by the RealtimeGestureGui workflow."""
        self.training_timer.stop()
        output = ""
        if self.training_process is not None:
            output = bytes(self.training_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        model_name = "personal_fast_model.pkl" if mode == "fast" else "personal_model.pkl"
        model_path = Path(session_dir) / "trained_model" / model_name
        if model_path.exists():
            summary = self._read_training_summary(Path(session_dir), mode)
            self.right_panel.training_result_label.setText(summary)
            self.load_model(str(model_path))
            self.right_panel.refresh_user_models()
            if self.calibration_dialog is not None:
                self.calibration_dialog.set_training_done("Training complete")
            self._start_auto_replay(session_dir, model_path)
        else:
            self.right_panel.training_result_label.setText("Training: failed")
            self.right_panel.warning_label.setText(f"Warnings: {output[-400:] if output else 'no training output'}")
            if self.calibration_dialog is not None:
                self.calibration_dialog.set_training_done("Training failed")

    def _read_training_summary(self, session_dir: Path, mode: str) -> str:
        """Read and parse training summary for the current RealtimeGestureGui workflow."""
        name = "personal_training_summary.json" if mode == "fast" else "personal_grid_summary.json"
        path = session_dir / "trained_model" / name
        if not path.exists():
            return "Training complete: model saved, summary file missing"
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            best = summary.get("best_result") or {}
            elapsed = float(summary.get("elapsed_seconds", 0.0))
            parts = [
                f"Training complete: {best.get('model_type', 'model')}",
                f"window {best.get('window_ms', '?')} ms",
                f"margin {best.get('trim_edge_ms', '?')} ms",
                f"validation BA {float(best.get('validation_balanced_accuracy', 0.0)):.3f}",
                f"held-out test BA {float(best.get('test_balanced_accuracy', 0.0)):.3f}",
                "deployment model retrained on all stages",
            ]
            params = best.get("model_params")
            if params:
                parts.append(str(params))
            if elapsed > 0:
                parts.append(f"{elapsed / 60.0:.1f} min")
            return " | ".join(parts)
        except Exception as exc:
            return f"Training complete: model saved, summary read failed ({exc})"

    def _start_auto_replay(self, session_dir: str, model_path: Path) -> None:
        """Start auto replay for the current RealtimeGestureGui workflow."""
        if self.replay_process is not None and self.replay_process.state() != QtCore.QProcess.NotRunning:
            return
        session_path = Path(session_dir)
        raw_dir = session_path / "raw_recordings"
        if not raw_dir.exists():
            self.right_panel.warning_label.setText("Warnings: replay skipped, raw recordings folder is missing")
            return
        self.right_panel.training_result_label.setText(self.right_panel.training_result_label.text() + " | replay running...")
        self.replay_process = QtCore.QProcess(self)
        self.replay_process.setProgram(sys.executable)
        self.replay_process.setArguments(
            [
                str(APP_DIR / "replay_evaluation.py"),
                "replay",
                "--data-dir",
                str(raw_dir),
                "--model-path",
                str(model_path),
                "--output-dir",
                str(session_path / "replay_report"),
                "--name",
                "auto_replay",
                "--confidence-threshold",
                f"{self.right_panel.threshold_value():.2f}",
                "--majority-windows",
                "3",
            ]
        )
        self.replay_process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.replay_process.finished.connect(lambda _code, _status: self._auto_replay_finished(session_path))
        self.replay_process.start()

    def _auto_replay_finished(self, session_dir: Path) -> None:
        """Perform the auto replay finished operation used by the RealtimeGestureGui workflow."""
        output = ""
        if self.replay_process is not None:
            output = bytes(self.replay_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        summary_path = session_dir / "replay_report" / "auto_replay" / "replay_summary.json"
        if not summary_path.exists():
            self.right_panel.warning_label.setText(f"Warnings: replay failed | {output[-300:] if output else 'no output'}")
            return
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            replay_text = (
                f"Calibration replay: raw BA {summary.get('raw_balanced_accuracy', 0.0):.3f} | "
                f"displayed acc {summary.get('displayed_accuracy_unknown_wrong', 0.0):.3f} | "
                f"majority acc {summary.get('majority_accuracy_unknown_wrong', 0.0):.3f} | "
                f"unknown {summary.get('unknown_rate', 0.0):.1%}"
            )
            self.right_panel.training_result_label.setText(self.right_panel.training_result_label.text().replace(" | replay running...", "") + " | " + replay_text)
            self._start_decision_comparison(summary_path.parent)
            QtWidgets.QMessageBox.information(self, "Replay complete", replay_text)
        except Exception as exc:
            self.right_panel.warning_label.setText(f"Warnings: replay summary read failed ({exc})")

    def _start_decision_comparison(self, replay_dir: Path) -> None:
        """Start decision comparison for the current RealtimeGestureGui workflow."""
        if self.decision_process is not None and self.decision_process.state() != QtCore.QProcess.NotRunning:
            return
        self.right_panel.training_result_label.setText(self.right_panel.training_result_label.text() + " | decision comparison running...")
        self.decision_process = QtCore.QProcess(self)
        self.decision_process.setProgram(sys.executable)
        self.decision_process.setArguments(
            [
                str(APP_DIR / "replay_evaluation.py"),
                "compare",
                "--replay-dir",
                str(replay_dir),
            ]
        )
        self.decision_process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.decision_process.finished.connect(lambda _code, _status: self._decision_comparison_finished(replay_dir))
        self.decision_process.start()

    def _decision_comparison_finished(self, replay_dir: Path) -> None:
        """Perform the decision comparison finished operation used by the RealtimeGestureGui workflow."""
        output = ""
        if self.decision_process is not None:
            output = bytes(self.decision_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        summary_path = replay_dir / "decision_strategy_comparison" / "decision_strategy_summary.json"
        if not summary_path.exists():
            self.right_panel.warning_label.setText(f"Warnings: decision comparison failed | {output[-250:] if output else 'no output'}")
            return
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best = summary.get("best_strategy") or {}
            best_text = (
                f"Decision: {best.get('strategy', 'unknown')} | "
                f"BA {float(best.get('balanced_accuracy_unknown_as_class', 0.0)):.3f} | "
                f"lag {float(best.get('mean_first_correct_lag_s', 0.0)):.2f}s"
            )
            if best.get("strategy"):
                self.right_panel.set_model_decision_strategy(str(best["strategy"]))
                self.apply_decision_settings()
            self.right_panel.training_result_label.setText(
                self.right_panel.training_result_label.text().replace(" | decision comparison running...", "") + " | " + best_text
            )
            current_model = self.right_panel.model_info_label.text()
            if " | decision " not in current_model:
                self.right_panel.model_info_label.setText(current_model + " | " + best_text.replace("Decision: ", "decision "))
        except Exception as exc:
            self.right_panel.warning_label.setText(f"Warnings: decision comparison summary read failed ({exc})")

    def apply_settings(self) -> None:
        """Apply settings for the current RealtimeGestureGui workflow."""
        selected = self.right_panel.selected_sensors()
        feature_indexes = [SENSOR_ORDER.index(sensor_id) for sensor_id in selected if sensor_id in SENSOR_ORDER]
        self.classifier.update_settings(feature_indexes, 0.0)

    def apply_decision_settings(self) -> None:
        """Apply decision settings for the current RealtimeGestureGui workflow."""
        self.decision_smoother.configure(self.right_panel.decision_config())

    def update_loop(self) -> None:
        """Refresh loop for the current RealtimeGestureGui workflow."""
        loop_now = time.perf_counter()
        self.update_loop_intervals_ms.append((loop_now - self.last_update_loop_time) * 1000.0)
        self.last_update_loop_time = loop_now
        snapshots = self.reader.snapshots()
        status = self.reader.status()
        if snapshots:
            updated = self.emg_panel.update_from_snapshots(snapshots)
            if self.calibration_dialog is not None:
                self.calibration_dialog.receive_snapshots(snapshots)
            if self.model_test_dialog is not None and self.model_test_dialog.isVisible():
                self.model_test_dialog.receive_snapshots(snapshots)
            if self.model_update_dialog is not None and self.model_update_dialog.isVisible():
                self.model_update_dialog.receive_snapshots(snapshots)
            if self.mouse_control_window is not None and self.mouse_control_window.isVisible():
                self.mouse_control_window.receive_snapshots(snapshots)
            self.packet_count += updated
            now = time.time()
            if now - self.last_signal_safety_update >= 0.05:
                self.latest_noise_snapshot = self.emg_panel.noise_snapshot(self.noise_profile)
                rest_expected = self.last_stable_gesture == "at_rest" and self.last_raw_gesture == "at_rest"
                self.signal_safety = self.signal_safety_gate.update(
                    snapshots, self.latest_noise_snapshot, rest_expected
                )
                if self.signal_safety.get("changed"):
                    self.decision_smoother.reset()
                self.last_signal_safety_update = now
            plot_refresh_hz = float(self.config.get("plot_refresh_hz", 25.0))
            if (
                not self.right_panel.pause_plots.isChecked()
                and now - self.last_plot_refresh_time >= 1.0 / max(1.0, plot_refresh_hz)
            ):
                self.emg_panel.refresh_plots()
                self.last_plot_refresh_time = now
            self._predict_if_due()
            if self.mouse_control_window is not None and self.mouse_control_window.isVisible():
                self.mouse_control_window.set_signal_safety(
                    bool(self.signal_safety["safe"]), str(self.signal_safety["reason"])
                )
                self.mouse_control_window.update_control(
                    self.last_stable_gesture,
                    self.last_gesture_confidence,
                    self.last_raw_gesture,
                    self.last_raw_confidence,
                )
            if now - self.last_metrics_update_time >= 0.25:
                known = [s for s in snapshots if f"{s.unit_id:08X}" in SENSOR_LOCATIONS]
                freshest_age = min((s.age_ms for s in snapshots), default=999999.0)
                seen_count = len({f"{s.unit_id:08X}" for s in known})
                stream_state = "streaming" if freshest_age < 1000 else "waiting for fresh packets"
                loop_p95 = float(np.percentile(self.update_loop_intervals_ms, 95)) if self.update_loop_intervals_ms else 0.0
                scheduling = "OK" if loop_p95 < 35.0 else "DELAYED"
                self.right_panel.status_label.setText(
                    f"Status: {stream_state} | sensors {seen_count}/3 | port {status['port']} | "
                    f"scheduler {scheduling} p95 {loop_p95:.0f} ms | "
                    f"occlusion throttle {self.realtime_priority.get('power_throttling', 'unknown')} | "
                    f"timer {self.realtime_priority.get('timer_resolution', 'default')} | "
                    f"signal {'SAFE' if self.signal_safety['safe'] else 'BLOCKED'}"
                )
                self.right_panel.set_noise_status(self.latest_noise_snapshot)
                if self.calibration_dialog is not None:
                    self.calibration_dialog.set_noise_status(self.latest_noise_snapshot)
                self.last_metrics_update_time = now
        else:
            self.signal_safety = self.signal_safety_gate.update([], {}, False)
            if self.mouse_control_window is not None:
                self.mouse_control_window.set_signal_safety(False, str(self.signal_safety["reason"]))
            err = status.get("last_error") or "waiting for uMyo device"
            self.right_panel.status_label.setText(f"Status: {err}")

    def _predict_if_due(self) -> None:
        """Predict if due for the current RealtimeGestureGui workflow."""
        interval_s = float(self.config.get("prediction_interval_ms", 120)) / 1000.0
        if time.time() - self.last_prediction_time < interval_s:
            return
        self.last_prediction_time = time.time()
        if not self.signal_safety.get("safe", False):
            reason = str(self.signal_safety.get("reason", "unsafe signal"))
            result = PredictionResult(
                gesture="Uncertain",
                confidence=0.0,
                stable_gesture="Uncertain",
                is_uncertain=True,
                error=f"signal safety gate: {reason}",
                probabilities={gesture: 0.0 for gesture in DISPLAY_GESTURES},
            )
            self.last_raw_gesture = "Uncertain"
            self.last_raw_confidence = 0.0
            self.last_stable_gesture = "Uncertain"
            self.last_gesture_confidence = 0.0
            self.right_panel.set_prediction(result)
            return
        model_window_ms = self.classifier.window_ms(int(self.config.get("window_ms", 500)))
        model_fs = self.classifier.sampling_rate_hz(self.fs)
        window_samples = max(2, int(model_fs * model_window_ms / 1000.0))
        selected = self.right_panel.selected_sensors()
        window = self.emg_panel.classifier_window(selected, min(window_samples, self.emg_panel.samples))
        result = self.classifier.predict(window)
        if not result.error:
            raw_label = result.gesture
            self.last_raw_gesture = raw_label
            self.last_raw_confidence = float(result.confidence)
            smoothed, uncertain = self.decision_smoother.apply(raw_label, result.confidence)
            result = replace(
                result,
                stable_gesture=smoothed,
                is_uncertain=uncertain,
                debug_info=(result.debug_info + f" | Decision: {self.right_panel.decision_config().get('method')}" if result.debug_info else f"Decision: {self.right_panel.decision_config().get('method')}"),
            )
        else:
            self.last_raw_gesture = result.gesture
            self.last_raw_confidence = float(result.confidence)
        self.last_stable_gesture = result.stable_gesture or result.gesture
        self.last_gesture_confidence = float(result.confidence)
        self.right_panel.set_prediction(result)

    @staticmethod
    def _rssi_quality(rssi: int) -> float:
        """Perform the rssi quality operation used by the RealtimeGestureGui workflow."""
        if rssi <= 0:
            return 0.0
        return max(0.0, min(100.0, (90.0 - rssi) * 1.6))

    @staticmethod
    def _battery_percent(batt_mv: int) -> float:
        """Perform the battery percent operation used by the RealtimeGestureGui workflow."""
        if batt_mv <= 0:
            return 0.0
        return max(0.0, min(100.0, (batt_mv - 3100.0) / 10.0))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window closure and release application resources safely."""
        self.reader.stop()
        self.reader.join(1.0)
        keep_display_awake(False)
        release_realtime_priority()
        super().closeEvent(event)


def load_config() -> dict:
    """Load and validate config for the current realtime gesture gui workflow."""
    cfg_path = APP_DIR / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    """Run the module's command-line or graphical application entry point."""
    app = QtWidgets.QApplication(sys.argv)
    window = RealtimeGestureGui(load_config())
    window.show()
    return app.exec()
