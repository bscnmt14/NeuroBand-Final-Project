"""Translate recognized gestures and uMyo IMU motion into mouse commands.

The module contains the safe cursor-control demonstration, Windows mouse output,
IMU calibration and filtering, gesture gates, click/drag handling, scrolling, and
diagnostic recording tools. EMG gestures express discrete intent while yaw and
pitch provide continuous cursor motion. Safety state and re-anchoring logic reduce
unintended movement during transitions between rest and active gestures.

"""

from __future__ import annotations

import csv
import ctypes
import json
import math
import multiprocessing
import random
import sys
import threading
import time
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from umyo_stream import DeviceSnapshot


REFERENCE_SENSOR_ID = "ED7A78C8"
REFERENCE_SENSOR_NAME = "Dorsal forearm"
REQUIRED_SENSOR_IDS = {"B0DAC7E9", "ED7A78C8", "37ED348F"}
FAST_ROLL_THRESHOLD = 750.0
ROLL_SEQUENCE_RIGHT_THRESHOLD = 450.0
ROLL_SEQUENCE_LEFT_THRESHOLD = 450.0
ROLL_SEQUENCE_TIMEOUT_S = 1.5
FAST_PITCH_THRESHOLD = 800.0
FAST_YAW_THRESHOLD = 800.0
ROLL_DOMINANCE_RATIO = 1.25
PITCH_DOMINANCE_RATIO = 1.25
YAW_DOMINANCE_RATIO = 1.10
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MOUSE_RECORDING_DIR = PROJECT_ROOT / "Data" / "mouse_control_recordings"
MOUSE_TRAINING_DIR = PROJECT_ROOT / "Data" / "mouse_calibration_sessions"

VK_SPACE = 0x20
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_MOVE_NOCOALESCE = 0x2000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
KEYEVENTF_KEYUP = 0x0002
VK_MENU = 0x12
VK_LEFT = 0x25
VK_RIGHT = 0x27
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

MOUSE_DIAGNOSTIC_PROTOCOL = [
    {"instruction": "Natural rest: remain relaxed", "motion_label": "normal_rest", "duration_s": 8.0},
    {"instruction": "Move the pointer slowly left and right", "motion_label": "pointer_horizontal_precise", "duration_s": 8.0},
    {"instruction": "Move the pointer slowly up and down", "motion_label": "pointer_vertical_precise", "duration_s": 8.0},
    {"instruction": "Move the pointer diagonally with small precise movements", "motion_label": "pointer_diagonal_precise", "duration_s": 8.0},
    {"instruction": "Return to neutral", "motion_label": "neutral", "duration_s": 3.0},
    {"instruction": "FAST roll right then left sequence while relaxed", "motion_label": "rest_roll_right_left_toggle", "duration_s": 5.0},
    {"instruction": "Return to neutral", "motion_label": "neutral", "duration_s": 3.0},
    {"instruction": "Hold fist and perform FAST roll right then left sequence", "motion_label": "fist_roll_right_left_drag_toggle", "duration_s": 5.0},
    {"instruction": "Return to neutral", "motion_label": "neutral", "duration_s": 3.0},
    {"instruction": "FAST pitch upward flick for scroll/navigation", "motion_label": "fast_pitch_positive", "duration_s": 3.0},
    {"instruction": "Return to neutral", "motion_label": "neutral", "duration_s": 3.0},
    {"instruction": "FAST pitch downward flick for scroll/navigation", "motion_label": "fast_pitch_negative", "duration_s": 3.0},
    {"instruction": "Return to neutral", "motion_label": "neutral", "duration_s": 3.0},
    {"instruction": "FAST yaw right flick for navigation", "motion_label": "fast_yaw_positive", "duration_s": 3.0},
    {"instruction": "Return to neutral", "motion_label": "neutral", "duration_s": 3.0},
    {"instruction": "FAST yaw left flick for navigation", "motion_label": "fast_yaw_negative", "duration_s": 3.0},
    {"instruction": "Normal everyday arm movement without commands", "motion_label": "normal_non_command_motion", "duration_s": 8.0},
    {"instruction": "Final natural rest", "motion_label": "normal_rest", "duration_s": 5.0},
]


def build_mouse_training_protocol() -> list[dict[str, object]]:
    """Create and configure mouse training protocol for the current mouse game control workflow."""
    protocol: list[dict[str, object]] = []

    def add(kind: str, label: str, duration_s: float, instruction: str) -> None:
        """Perform the add operation used by the build mouse training protocol workflow."""
        protocol.append({"kind": kind, "gesture_label": label, "duration_s": duration_s, "instruction": instruction})

    add("rest", "at_rest", 12.0, "Relax naturally; small arm movements are allowed")
    for gesture in ["open_hand", "pointing"]:
        add("prepare", "at_rest", 2.0, f"Prepare {gesture}")
        add("movement_hold", gesture, 5.0, f"Hold {gesture} still")
        for direction in ["left and right", "up and down", "diagonally", "with small precise movements"]:
            add("movement_hold", gesture, 6.0, f"Keep {gesture}; move {direction}")
        add("rest", "at_rest", 3.0, "Relax")

    for gesture, action in [("fist", "left click"), ("pinch", "left click"), ("like", "right click")]:
        for repetition in range(8):
            add("rest", "at_rest", 1.2, f"Prepare short {action}")
            add("short_action", gesture, 0.8, f"Short {gesture}: {action}")
        add("rest", "at_rest", 3.0, "Relax")

    for gesture, action in [("wrist_extension", "scroll up"), ("wrist_flexion", "scroll down")]:
        for _ in range(4):
            add("rest", "at_rest", 1.5, f"Prepare {action}")
            add("short_scroll", gesture, 1.2, f"Short {action}")
        add("long_scroll", gesture, 4.0, f"Hold for continuous {action}")
        add("rest", "at_rest", 3.0, "Relax")

    transitions = [
        ("open_hand", "fist"),
        ("open_hand", "pinch"),
        ("pointing", "fist"),
        ("pointing", "pinch"),
        ("fist", "like"),
        ("pointing", "wrist_extension"),
        ("open_hand", "wrist_flexion"),
    ]
    for first, second in transitions:
        add("transition", first, 2.0, f"Hold {first}")
        add("transition", second, 1.5, f"Switch directly to {second}")
        add("rest", "at_rest", 2.0, "Relax")
    add("rest", "at_rest", 8.0, "Final natural rest")
    return protocol


class MouseTrainingRecorder:
    """Represent the MouseTrainingRecorder component and keep its related state and behavior together."""
    def __init__(self) -> None:
        """Initialize the MouseTrainingRecorder instance and its runtime state."""
        self.active = False
        self.writer: csv.DictWriter | None = None
        self.handle = None
        self.started_at = 0.0
        self.stage_index = -1
        self.stage: dict[str, object] | None = None
        self.rows_written = 0
        self.session_dir: Path | None = None

    def start(self, protocol: list[dict[str, object]]) -> Path:
        """Perform the start operation used by the MouseTrainingRecorder workflow."""
        self.stop()
        self.session_dir = MOUSE_TRAINING_DIR / datetime.now().strftime("mouse_training_%Y%m%d_%H%M%S")
        raw_dir = self.session_dir / "raw_recordings"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "session_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        fields = [
            "timestamp", "trial_index", "protocol_step", "stage_kind", "gesture_label", "device_id", "unit_id",
            *[f"emg_{idx}" for idx in range(8)], "sp0", "sp1", "sp2", "sp3", "rssi", "battery_mv",
            "ax", "ay", "az", "yaw", "pitch", "roll",
        ]
        self.handle = (raw_dir / "mouse_training_recording.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        self.writer.writeheader()
        self.started_at = time.time()
        self.rows_written = 0
        self.active = True
        return self.session_dir

    def set_stage(self, index: int, stage: dict[str, object]) -> None:
        """Set stage for the current MouseTrainingRecorder workflow."""
        self.stage_index = index
        self.stage = stage

    def add_snapshots(self, snapshots: list[DeviceSnapshot]) -> None:
        """Add snapshots for the current MouseTrainingRecorder workflow."""
        if not self.active or self.writer is None or self.stage is None:
            return
        for snap in snapshots:
            sensor_id = f"{snap.unit_id:08X}"
            emg = np.asarray(snap.emg, dtype=float).reshape(-1)
            spectra = np.asarray(snap.spectrum, dtype=float)
            if sensor_id not in {"B0DAC7E9", "ED7A78C8", "37ED348F"} or emg.size == 0:
                continue
            if spectra.ndim == 1:
                spectra = spectra.reshape(1, -1)
            for packet_index in range(int(math.ceil(emg.size / 8.0))):
                packet = emg[packet_index * 8 : (packet_index + 1) * 8]
                packet = np.pad(packet, (0, max(0, 8 - packet.size)), constant_values=np.nan)[:8]
                spectrum = spectra[min(packet_index, len(spectra) - 1)] if len(spectra) else np.zeros(4)
                spectrum = np.pad(spectrum, (0, max(0, 4 - len(spectrum))))[:4]
                row = {
                    "timestamp": time.time() - self.started_at,
                    "trial_index": self.stage_index,
                    "protocol_step": self.stage_index,
                    "stage_kind": self.stage["kind"],
                    "gesture_label": self.stage["gesture_label"],
                    "device_id": sensor_id,
                    "unit_id": sensor_id,
                    "sp0": spectrum[0], "sp1": spectrum[1], "sp2": spectrum[2], "sp3": spectrum[3],
                    "rssi": snap.rssi, "battery_mv": snap.battery_mv,
                    "ax": snap.ax, "ay": snap.ay, "az": snap.az,
                    "yaw": snap.yaw, "pitch": snap.pitch, "roll": snap.roll,
                }
                row.update({f"emg_{idx}": packet[idx] for idx in range(8)})
                self.writer.writerow(row)
                self.rows_written += 1

    def stop(self) -> None:
        """Perform the stop operation used by the MouseTrainingRecorder workflow."""
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
        self.handle = None
        self.writer = None
        self.active = False


class WindowsMouse:
    """Represent the WindowsMouse component and keep its related state and behavior together."""
    @staticmethod
    def available() -> bool:
        """Perform the available operation used by the WindowsMouse workflow."""
        return sys.platform == "win32"

    @staticmethod
    def move_to(x: float, y: float) -> None:
        """Perform the move to operation used by the WindowsMouse workflow."""
        if WindowsMouse.available():
            ctypes.windll.user32.SetCursorPos(int(round(x)), int(round(y)))

    @staticmethod
    def move_relative(dx: int, dy: int) -> None:
        """Perform the move relative operation used by the WindowsMouse workflow."""
        if WindowsMouse.available() and (dx or dy):
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_MOVE_NOCOALESCE, int(dx), int(dy), 0, 0)

    @staticmethod
    def foreground_context() -> tuple[int, int, str]:
        """Perform the foreground context operation used by the WindowsMouse workflow."""
        if not WindowsMouse.available():
            return 0, 0, ""
        user32 = ctypes.windll.user32
        hwnd = int(user32.GetForegroundWindow())
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        length = int(user32.GetWindowTextLengthW(hwnd))
        buffer = ctypes.create_unicode_buffer(max(1, length + 1))
        user32.GetWindowTextW(hwnd, buffer, len(buffer))
        return hwnd, int(process_id.value), buffer.value

    @staticmethod
    def position() -> tuple[float, float]:
        """Perform the position operation used by the WindowsMouse workflow."""
        if WindowsMouse.available():
            point = wintypes.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
                return float(point.x), float(point.y)
        point = QtGui.QCursor.pos()
        return float(point.x()), float(point.y())

    @staticmethod
    def virtual_geometry() -> tuple[float, float, float, float]:
        """Perform the virtual geometry operation used by the WindowsMouse workflow."""
        if WindowsMouse.available():
            user32 = ctypes.windll.user32
            return (
                float(user32.GetSystemMetrics(SM_XVIRTUALSCREEN)),
                float(user32.GetSystemMetrics(SM_YVIRTUALSCREEN)),
                float(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)),
                float(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)),
            )
        geometry = QtGui.QGuiApplication.primaryScreen().virtualGeometry()
        return float(geometry.left()), float(geometry.top()), float(geometry.width()), float(geometry.height())

    @staticmethod
    def left_click() -> None:
        """Perform the left click operation used by the WindowsMouse workflow."""
        if WindowsMouse.available():
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    @staticmethod
    def double_click_window_s() -> float:
        """Perform the double click window s operation used by the WindowsMouse workflow."""
        if WindowsMouse.available():
            return max(0.25, min(0.75, float(ctypes.windll.user32.GetDoubleClickTime()) / 1000.0))
        return 0.50

    @staticmethod
    def left_down() -> None:
        """Perform the left down operation used by the WindowsMouse workflow."""
        if WindowsMouse.available():
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    @staticmethod
    def left_up() -> None:
        """Perform the left up operation used by the WindowsMouse workflow."""
        if WindowsMouse.available():
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    @staticmethod
    def right_click() -> None:
        """Perform the right click operation used by the WindowsMouse workflow."""
        if WindowsMouse.available():
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    @staticmethod
    def scroll(steps: int) -> None:
        """Perform the scroll operation used by the WindowsMouse workflow."""
        if WindowsMouse.available():
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(steps * 120), 0)

    @staticmethod
    def space_pressed() -> bool:
        """Perform the space pressed operation used by the WindowsMouse workflow."""
        return bool(WindowsMouse.available() and (ctypes.windll.user32.GetAsyncKeyState(VK_SPACE) & 0x8000))

    @staticmethod
    def browser_navigation(back: bool) -> None:
        """Perform the browser navigation operation used by the WindowsMouse workflow."""
        if not WindowsMouse.available():
            return
        arrow = VK_LEFT if back else VK_RIGHT
        ctypes.windll.user32.keybd_event(VK_MENU, 0, 0, 0)
        ctypes.windll.user32.keybd_event(arrow, 0, 0, 0)
        ctypes.windll.user32.keybd_event(arrow, 0, KEYEVENTF_KEYUP, 0)
        ctypes.windll.user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)


def _windows_cursor_process(
    state: object,
    enabled: object,
    stop_event: object,
    interval_s: float,
) -> None:
    """Run cursor interpolation outside the GUI process and its Python GIL."""
    if WindowsMouse.available():
        ctypes.windll.winmm.timeBeginPeriod(1)
        # This worker is a separate process, so raising only its priority
        # cannot starve the GUI/serial Python threads.
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00000080)
    previous_time = time.perf_counter()
    try:
        while not stop_event.wait(interval_s):
            now = time.perf_counter()
            dt = max(0.001, min(0.015, now - previous_time))
            previous_time = now
            if not enabled.value:
                continue
            with state.get_lock():
                target = (state[0], state[1])
                current = (state[2], state[3])
                generation = state[4]
            dx = target[0] - current[0]
            dy = target[1] - current[1]
            distance = math.hypot(dx, dy)
            if distance <= 0.75:
                with state.get_lock():
                    state[2], state[3] = target
                continue
            # First-order interpolation has no stored velocity, cannot overshoot,
            # and settles rapidly between sparse GUI/IMU updates.
            # A slightly longer follower time constant keeps sparse IMU target
            # updates moving continuously instead of producing stop-go bursts.
            alpha = 1.0 - math.exp(-dt / 0.018)
            maximum_step = 5000.0 * dt
            step = min(distance, maximum_step, max(1.0, distance * alpha))
            scale = step / distance
            next_position = (current[0] + dx * scale, current[1] + dy * scale)
            with state.get_lock():
                # Synchronization/freeze increments the generation. Do not let a
                # stale pre-freeze step move the real cursor afterwards.
                if not enabled.value or state[4] != generation:
                    continue
                state[2], state[3] = next_position
                WindowsMouse.move_to(*next_position)
    finally:
        if WindowsMouse.available():
            ctypes.windll.winmm.timeEndPeriod(1)


class WindowsCursorWorker:
    """Represent the WindowsCursorWorker component and keep its related state and behavior together."""
    def __init__(self, interval_s: float = 0.0025) -> None:
        """Initialize the WindowsCursorWorker instance and its runtime state."""
        self.interval_s = interval_s
        context = multiprocessing.get_context("spawn")
        x, y = WindowsMouse.position()
        # target x/y, current x/y, synchronization generation, reserved
        self.state = context.Array("d", [x, y, x, y, 0.0, 0.0], lock=True)
        self.enabled = context.Value("b", False, lock=False)
        self.stop_event = context.Event()
        self.process = context.Process(
            target=_windows_cursor_process,
            args=(self.state, self.enabled, self.stop_event, self.interval_s),
            name="windows-cursor-worker",
            daemon=True,
        )
        self.process.start()

    def set_enabled(self, enabled: bool) -> None:
        """Set enabled for the current WindowsCursorWorker workflow."""
        if enabled:
            x, y = WindowsMouse.position()
            with self.state.get_lock():
                generation = self.state[4] + 1.0
                self.state[:] = [x, y, x, y, generation, 0.0]
        else:
            with self.state.get_lock():
                self.state[4] += 1.0
        self.enabled.value = enabled

    def submit_target(self, x: float, y: float) -> None:
        """Perform the submit target operation used by the WindowsCursorWorker workflow."""
        submitted = (float(x), float(y))
        with self.state.get_lock():
            if math.hypot(submitted[0] - self.state[0], submitted[1] - self.state[1]) >= 1.5:
                self.state[0], self.state[1] = submitted

    def sync_position(self, x: float, y: float) -> None:
        """Perform the sync position operation used by the WindowsCursorWorker workflow."""
        with self.state.get_lock():
            generation = self.state[4] + 1.0
            self.state[:] = [float(x), float(y), float(x), float(y), generation, 0.0]

    def current_position(self) -> tuple[float, float]:
        """Perform the current position operation used by the WindowsCursorWorker workflow."""
        with self.state.get_lock():
            return self.state[2], self.state[3]

    def stop(self) -> None:
        """Perform the stop operation used by the WindowsCursorWorker workflow."""
        self.stop_event.set()
        self.process.join(timeout=0.5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.25)


class DirectCursorDispatcher:
    """Deliver GUI-computed cursor targets to Windows at a steady high rate."""

    def __init__(self) -> None:
        """Initialize the DirectCursorDispatcher instance and its runtime state."""
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._enabled = False
        self.position = WindowsMouse.position()
        self.target = self.position
        self._thread = threading.Thread(target=self._run, name="windows-cursor-dispatcher", daemon=True)
        self._thread.start()

    def set_enabled(self, enabled: bool) -> None:
        """Set enabled for the current DirectCursorDispatcher workflow."""
        with self._lock:
            self._enabled = bool(enabled)

    def submit_target(self, x: float, y: float) -> None:
        """Perform the submit target operation used by the DirectCursorDispatcher workflow."""
        with self._lock:
            self.target = (float(x), float(y))

    def sync_position(self, x: float, y: float) -> None:
        """Perform the sync position operation used by the DirectCursorDispatcher workflow."""
        with self._lock:
            self.position = (float(x), float(y))
            self.target = self.position

    def current_position(self) -> tuple[float, float]:
        """Perform the current position operation used by the DirectCursorDispatcher workflow."""
        with self._lock:
            return self.position

    def stop(self) -> None:
        """Perform the stop operation used by the DirectCursorDispatcher workflow."""
        self._stop_event.set()
        self._thread.join(timeout=0.5)

    def _run(self) -> None:
        """Perform the run operation used by the DirectCursorDispatcher workflow."""
        interval_s = 1.0 / 120.0
        while not self._stop_event.wait(interval_s):
            with self._lock:
                if not self._enabled:
                    continue
                current_x, current_y = self.position
                target_x, target_y = self.target
                dx = target_x - current_x
                dy = target_y - current_y
                if abs(dx) < 0.15 and abs(dy) < 0.15:
                    next_x, next_y = target_x, target_y
                else:
                    # Fast enough to avoid noticeable lag while filling gaps
                    # caused by plotting or model inference on the GUI thread.
                    next_x = current_x + dx * 0.72
                    next_y = current_y + dy * 0.72
                self.position = (next_x, next_y)
            WindowsMouse.move_to(next_x, next_y)


def _angle_delta(current: float, previous: float) -> float:
    """Perform the angle delta operation used by the mouse game control workflow."""
    delta = current - previous
    if delta > 4000.0:
        delta -= 8192.0
    elif delta < -4000.0:
        delta += 8192.0
    return delta


@dataclass
class MouseDemoState:
    """Represent the MouseDemoState component and keep its related state and behavior together."""
    cursor: QtCore.QPointF
    target: QtCore.QPointF
    target_radius: float = 30.0
    score: int = 0
    misses: int = 0
    right_clicks: int = 0
    scroll_value: int = 0
    drag_active: bool = False
    navigation_value: int = 0
    last_action: str = "Idle"
    enabled: bool = False
    calibrated: bool = False
    draggable: QtCore.QPointF = field(default_factory=lambda: QtCore.QPointF(180.0, 180.0))
    draggable_selected: bool = False
    game_mode: str = "click"
    required_gesture: str = "fist"
    gesture_hits: int = 0


class TargetClickGame(QtWidgets.QWidget):
    """Represent the TargetClickGame component and keep its related state and behavior together."""
    def __init__(self, parent=None):
        """Initialize the TargetClickGame instance and its runtime state."""
        super().__init__(parent)
        self.setMinimumSize(560, 360)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.state = MouseDemoState(
            cursor=QtCore.QPointF(280.0, 180.0),
            target=QtCore.QPointF(420.0, 180.0),
        )
        self.gesture_challenges = ["fist", "like", "pinch", "wrist_extension", "wrist_flexion"]
        self.last_attempted_gesture = ""

    def reset_game(self) -> None:
        """Reset game for the current TargetClickGame workflow."""
        rect = self._play_rect()
        self.state.cursor = rect.center()
        self.state.score = 0
        self.state.misses = 0
        self.state.right_clicks = 0
        self.state.scroll_value = 0
        self.state.drag_active = False
        self.state.draggable = QtCore.QPointF(rect.left() + 120.0, rect.center().y())
        self.state.draggable_selected = False
        self.state.navigation_value = 0
        self.state.gesture_hits = 0
        self.last_attempted_gesture = ""
        self.state.last_action = "Game reset"
        self._new_target()
        self.update()

    def set_game_mode(self, mode: str) -> None:
        """Set game mode for the current TargetClickGame workflow."""
        self.state.game_mode = mode
        self.state.target_radius = 55.0 if mode == "gesture" else 30.0
        self.state.last_action = "Gesture challenge ready" if mode == "gesture" else "Target clicking ready"
        self._new_target()
        self.update()

    def register_gesture(self, gesture: str) -> None:
        """Perform the register gesture operation used by the TargetClickGame workflow."""
        if self.state.game_mode != "gesture" or gesture == self.last_attempted_gesture:
            return
        self.last_attempted_gesture = gesture
        if gesture in {"Uncertain", "No model", "Error", "at_rest"}:
            return
        on_target = self._distance(self.state.cursor, self.state.target) <= self.state.target_radius
        if on_target and gesture == self.state.required_gesture:
            self.state.score += 1
            self.state.gesture_hits += 1
            self.state.last_action = f"{gesture}: challenge hit"
            self._new_target()
        elif on_target:
            self.state.misses += 1
            self.state.last_action = f"{gesture}: wrong gesture"
        self.update()

    def set_enabled(self, enabled: bool) -> None:
        """Set enabled for the current TargetClickGame workflow."""
        self.state.enabled = enabled
        self.update()

    def set_calibrated(self, calibrated: bool) -> None:
        """Set calibrated for the current TargetClickGame workflow."""
        self.state.calibrated = calibrated
        self.update()

    def move_cursor(self, dx: float, dy: float) -> None:
        """Perform the move cursor operation used by the TargetClickGame workflow."""
        rect = self._play_rect()
        x = max(rect.left(), min(rect.right(), self.state.cursor.x() + dx))
        y = max(rect.top(), min(rect.bottom(), self.state.cursor.y() + dy))
        self.state.cursor = QtCore.QPointF(x, y)
        self.state.last_action = "Move cursor"
        self.update()

    def set_cursor(self, point: QtCore.QPointF, action: str = "Point cursor") -> None:
        """Set cursor for the current TargetClickGame workflow."""
        rect = self._play_rect()
        x = max(rect.left(), min(rect.right(), point.x()))
        y = max(rect.top(), min(rect.bottom(), point.y()))
        self.state.cursor = QtCore.QPointF(x, y)
        if self.state.drag_active and self.state.draggable_selected:
            self.state.draggable = QtCore.QPointF(x, y)
        self.state.last_action = action
        self.update()

    def left_click(self) -> bool:
        """Perform the left click operation used by the TargetClickGame workflow."""
        if self.state.game_mode == "gesture":
            return False
        hit = self._distance(self.state.cursor, self.state.target) <= self.state.target_radius
        if hit:
            self.state.score += 1
            self.state.last_action = "Left click: target hit"
            self._new_target()
        else:
            self.state.misses += 1
            self.state.last_action = "Left click: miss"
        self.update()
        return hit

    def right_click(self) -> None:
        """Perform the right click operation used by the TargetClickGame workflow."""
        self.state.right_clicks += 1
        self.state.last_action = "Right click"
        self.update()

    def scroll(self, steps: int) -> None:
        """Perform the scroll operation used by the TargetClickGame workflow."""
        self.state.scroll_value += steps
        direction = "up" if steps > 0 else "down"
        self.state.last_action = f"Scroll {direction}"
        self.update()

    def set_drag(self, active: bool) -> None:
        """Set drag for the current TargetClickGame workflow."""
        self.state.drag_active = active
        if active:
            self.state.draggable_selected = self._distance(self.state.cursor, self.state.draggable) <= 42.0
        else:
            self.state.draggable_selected = False
        self.state.last_action = f"Drag {'ON' if active else 'OFF'}"
        self.update()

    def navigate(self, back: bool) -> None:
        """Perform the navigate operation used by the TargetClickGame workflow."""
        self.state.navigation_value += -1 if back else 1
        self.state.last_action = "Browser back" if back else "Browser forward"
        self.update()

    def _new_target(self) -> None:
        """Perform the new target operation used by the TargetClickGame workflow."""
        rect = self._play_rect().adjusted(50, 50, -50, -50)
        self.state.target = QtCore.QPointF(
            random.uniform(rect.left(), rect.right()),
            random.uniform(rect.top(), rect.bottom()),
        )
        if self.state.game_mode == "gesture":
            self.state.required_gesture = random.choice(self.gesture_challenges)

    def _play_rect(self) -> QtCore.QRectF:
        """Perform the play rect operation used by the TargetClickGame workflow."""
        return QtCore.QRectF(self.rect()).adjusted(18, 18, -18, -56)

    @staticmethod
    def _distance(first: QtCore.QPointF, second: QtCore.QPointF) -> float:
        """Perform the distance operation used by the TargetClickGame workflow."""
        return math.hypot(first.x() - second.x(), first.y() - second.y())

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        """Render the widget using its current state."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#f8fafc"))

        rect = self._play_rect()
        painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1"), 1))
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.drawRoundedRect(rect, 8, 8)

        object_rect = QtCore.QRectF(self.state.draggable.x() - 30, self.state.draggable.y() - 24, 60, 48)
        painter.setPen(QtGui.QPen(QtGui.QColor("#9a3412"), 3 if self.state.draggable_selected else 2))
        painter.setBrush(QtGui.QColor("#fb923c") if self.state.draggable_selected else QtGui.QColor("#fdba74"))
        painter.drawRoundedRect(object_rect, 6, 6)
        painter.setPen(QtGui.QColor("#7c2d12"))
        painter.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
        painter.drawText(object_rect, QtCore.Qt.AlignCenter, "DRAG")

        target_color = QtGui.QColor("#0ea5e9") if self.state.enabled and self.state.calibrated else QtGui.QColor("#94a3b8")
        painter.setPen(QtGui.QPen(QtGui.QColor("#075985"), 2))
        painter.setBrush(target_color)
        painter.drawEllipse(self.state.target, self.state.target_radius, self.state.target_radius)
        painter.setPen(QtGui.QColor("#ffffff"))
        painter.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        painter.drawText(
            QtCore.QRectF(
                self.state.target.x() - self.state.target_radius,
                self.state.target.y() - 10,
                self.state.target_radius * 2,
                20,
            ),
            QtCore.Qt.AlignCenter,
            self.state.required_gesture.replace("_", " ").title() if self.state.game_mode == "gesture" else "TARGET",
        )

        painter.setPen(QtGui.QPen(QtGui.QColor("#111827"), 2))
        painter.setBrush(QtGui.QColor("#f97316"))
        painter.drawEllipse(self.state.cursor, 8, 8)
        painter.drawLine(QtCore.QPointF(self.state.cursor.x() - 14, self.state.cursor.y()), QtCore.QPointF(self.state.cursor.x() + 14, self.state.cursor.y()))
        painter.drawLine(QtCore.QPointF(self.state.cursor.x(), self.state.cursor.y() - 14), QtCore.QPointF(self.state.cursor.x(), self.state.cursor.y() + 14))

        footer = (
            f"Score {self.state.score} | Misses {self.state.misses} | "
            f"Right clicks {self.state.right_clicks} | Scroll {self.state.scroll_value} | "
            f"Drag {'ON' if self.state.drag_active else 'OFF'} | Navigation {self.state.navigation_value} | "
            f"Gesture hits {self.state.gesture_hits} | {self.state.last_action}"
        )
        painter.setPen(QtGui.QColor("#1f2937"))
        painter.setFont(QtGui.QFont("Segoe UI", 10))
        painter.drawText(QtCore.QRectF(18, self.height() - 40, self.width() - 36, 24), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, footer)


class FullScreenPointerCalibration(QtWidgets.QDialog):
    """Represent the FullScreenPointerCalibration component and keep its related state and behavior together."""
    calibration_ready = QtCore.Signal(dict)

    TARGETS = [
        ("Center", 0.50, 0.50),
        ("Upper right", 0.90, 0.10),
        ("Center", 0.50, 0.50),
        ("Upper left", 0.10, 0.10),
        ("Center", 0.50, 0.50),
        ("Lower left", 0.10, 0.90),
        ("Center", 0.50, 0.50),
        ("Lower right", 0.90, 0.90),
        ("Center validation", 0.50, 0.50),
    ]

    def __init__(self, parent=None):
        """Initialize the FullScreenPointerCalibration instance and its runtime state."""
        super().__init__(parent)
        self.setWindowTitle("Automatic Mouse Calibration")
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.setStyleSheet("background: #f8fafc; color: #0f172a;")
        self.latest_angles = (0.0, 0.0, 0.0)
        self.index = 0
        self.samples: list[dict[str, object]] = []
        self.target_started_at = 0.0
        self.target_duration_s = 2.0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    def start(self) -> None:
        """Perform the start operation used by the FullScreenPointerCalibration workflow."""
        self.index = 0
        self.samples = []
        self.target_started_at = time.time()
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.update()

    def update_angles(self, angles: tuple[float, float, float]) -> None:
        """Refresh angles for the current FullScreenPointerCalibration workflow."""
        self.latest_angles = angles

    def _tick(self) -> None:
        """Perform the tick operation used by the FullScreenPointerCalibration workflow."""
        if not self.isVisible() or self.index >= len(self.TARGETS):
            return
        if time.time() - self.target_started_at >= self.target_duration_s:
            self.capture_target()
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle a keyboard command delivered to the widget."""
        if event.key() == QtCore.Qt.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, _event: QtGui.QMouseEvent) -> None:
        """Handle a mouse-button press delivered to the widget."""
        pass

    def capture_target(self) -> None:
        """Perform the capture target operation used by the FullScreenPointerCalibration workflow."""
        if self.index >= len(self.TARGETS):
            return
        name, x_ratio, y_ratio = self.TARGETS[self.index]
        self.samples.append(
            {
                "name": name,
                "x_ratio": x_ratio,
                "y_ratio": y_ratio,
                "yaw": self.latest_angles[0],
                "pitch": self.latest_angles[1],
                "roll": self.latest_angles[2],
            }
        )
        self.index += 1
        if self.index >= len(self.TARGETS):
            self._finish()
            return
        self.target_started_at = time.time()
        self.update()

    def _finish(self) -> None:
        """Perform the finish operation used by the FullScreenPointerCalibration workflow."""
        non_center = [row for row in self.samples if not str(row["name"]).startswith("Center")]
        center = [row for row in self.samples if str(row["name"]).startswith("Center")]
        if len(non_center) < 4 or not center:
            self.reject()
            return
        center_yaw = float(np.median([row["yaw"] for row in center]))
        center_pitch = float(np.median([row["pitch"] for row in center]))
        yaw_offsets = [_angle_delta(float(row["yaw"]), center_yaw) for row in non_center]
        pitch_offsets = [_angle_delta(float(row["pitch"]), center_pitch) for row in non_center]
        x_offsets = [-value for value in yaw_offsets]
        y_offsets = [-value for value in pitch_offsets]
        calibration = {
            "center_yaw": center_yaw,
            "center_pitch": center_pitch,
            "yaw_span": max(abs(value) for value in yaw_offsets),
            "pitch_span": max(abs(value) for value in pitch_offsets),
            "x_positive_span": max([value for value in x_offsets if value > 0.0] or [1.0]),
            "x_negative_span": abs(min([value for value in x_offsets if value < 0.0] or [-1.0])),
            "y_positive_span": max([value for value in y_offsets if value > 0.0] or [1.0]),
            "y_negative_span": abs(min([value for value in y_offsets if value < 0.0] or [-1.0])),
            "samples": self.samples,
        }
        self.calibration_ready.emit(calibration)
        self.accept()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        """Render the widget using its current state."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#f8fafc"))
        if self.index >= len(self.TARGETS):
            return
        name, x_ratio, y_ratio = self.TARGETS[self.index]
        point = QtCore.QPointF(self.width() * x_ratio, self.height() * y_ratio)
        previous_name, previous_x_ratio, previous_y_ratio = self.TARGETS[max(0, self.index - 1)]
        previous_point = QtCore.QPointF(self.width() * previous_x_ratio, self.height() * previous_y_ratio)
        if self.index > 0:
            painter.setPen(QtGui.QPen(QtGui.QColor("#0f766e"), 8))
            painter.drawLine(previous_point, point)
            direction = point - previous_point
            length = max(1.0, math.hypot(direction.x(), direction.y()))
            unit = QtCore.QPointF(direction.x() / length, direction.y() / length)
            normal = QtCore.QPointF(-unit.y(), unit.x())
            tip = point
            left = tip - unit * 34 + normal * 18
            right = tip - unit * 34 - normal * 18
            painter.setBrush(QtGui.QColor("#0f766e"))
            painter.drawPolygon(QtGui.QPolygonF([tip, left, right]))
        painter.setPen(QtGui.QPen(QtGui.QColor("#075985"), 4))
        painter.setBrush(QtGui.QColor("#0ea5e9"))
        painter.drawEllipse(point, 32, 32)
        painter.setPen(QtGui.QColor("#0f172a"))
        painter.setFont(QtGui.QFont("Segoe UI", 24, QtGui.QFont.Bold))
        painter.drawText(QtCore.QRectF(0, 30, self.width(), 60), QtCore.Qt.AlignCenter, f"Point to: {name}")
        painter.setFont(QtGui.QFont("Segoe UI", 14))
        remaining = max(0.0, self.target_duration_s - (time.time() - self.target_started_at))
        painter.drawText(
            QtCore.QRectF(0, self.height() - 90, self.width(), 50),
            QtCore.Qt.AlignCenter,
            f"Move in the arrow direction and hold the target. No click required. Automatic capture in {remaining:.1f}s. Esc cancels.",
        )


class WindowsCursorIndicator(QtWidgets.QWidget):
    """Click-through blue ring that makes NeuroBand Windows control obvious."""

    def __init__(self):
        """Initialize the WindowsCursorIndicator instance and its runtime state."""
        flags = (
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.WindowTransparentForInput
        )
        super().__init__(None, flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.setFixedSize(52, 52)
        self.follow_timer = QtCore.QTimer(self)
        self.follow_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.follow_timer.timeout.connect(self._follow_cursor)
        self.follow_timer.start(16)

    def _follow_cursor(self) -> None:
        """Perform the follow cursor operation used by the WindowsCursorIndicator workflow."""
        if not self.isVisible():
            return
        point = QtGui.QCursor.pos()
        self.move(point.x() - self.width() // 2, point.y() - self.height() // 2)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        """Render the widget using its current state."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        center = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        painter.setPen(QtGui.QPen(QtGui.QColor("#075985"), 4))
        painter.setBrush(QtGui.QColor(14, 165, 233, 105))
        painter.drawEllipse(center, 19, 19)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.drawEllipse(center, 3, 3)


class MouseControlWindow(QtWidgets.QDialog):
    """Represent the MouseControlWindow component and keep its related state and behavior together."""
    def __init__(self, parent=None):
        """Initialize the MouseControlWindow instance and its runtime state."""
        super().__init__(parent)
        self.setWindowTitle("Safe Mouse / Target Clicking Demo")
        self.setWindowFlag(QtCore.Qt.WindowMinMaxButtonsHint, True)
        self.resize(880, 720)
        self.setMinimumSize(760, 620)
        self.reference_sensor_id = REFERENCE_SENSOR_ID
        self.latest_reference: DeviceSnapshot | None = None
        self.latest_snapshots_by_id: dict[str, DeviceSnapshot] = {}
        self.connected_sensor_ids: set[str] = set()
        self.neutral_angles: tuple[float, float, float] | None = None
        self.angle_history: deque[tuple[float, float, float]] = deque(maxlen=7)
        self.pointer_last_raw_angles: tuple[float, float, float] | None = None
        self.pointer_unwrapped_angles: tuple[float, float, float] | None = None
        self.pointer_filtered_angles: tuple[float, float, float] | None = None
        self.current_gesture = "No model"
        self.current_confidence = 0.0
        self.raw_gesture = "No model"
        self.raw_confidence = 0.0
        self.signal_safe = False
        self.signal_safety_reason = "waiting for signal safety gate"
        self.mouse_gate_candidate = ""
        self.mouse_gate_count = 0
        self.last_gesture = "No model"
        self.last_movement_gesture = "open_hand"
        self.last_movement_seen = 0.0
        self.last_click_time = 0.0
        self.fist_first_click_time = 0.0
        self.fist_click_armed = True
        self.fist_release_started = 0.0
        self.fist_clutch_active = False
        self.fist_clutch_started = 0.0
        self.fist_stable_started = 0.0
        self.fist_click_sent = False
        self.fist_clutch_demo_position: QtCore.QPointF | None = None
        self.fist_clutch_os_position: QtCore.QPointF | None = None
        self.fist_first_click_demo_position: QtCore.QPointF | None = None
        self.fist_first_click_os_position: QtCore.QPointF | None = None
        self.last_scroll_time = 0.0
        self.movement_active = False
        self.movement_anchor_angles: tuple[float, float, float] | None = None
        self.movement_anchor_cursor: QtCore.QPointF | None = None
        self.movement_anchor_os_cursor: QtCore.QPointF | None = None
        self.cursor_clutch_just_engaged = False
        self.imu_movement_toggle = False
        self.previous_combo_angles: tuple[float, float, float] | None = None
        self.combo_angle_history: deque[tuple[float, tuple[float, float, float]]] = deque(maxlen=40)
        self.roll_sequence_gesture = ""
        self.roll_sequence_started = 0.0
        self.last_combo_time = 0.0
        self.suppress_fist_action_until = 0.0
        self.drag_active = False
        self.drag_mode: str | None = None
        self.pinch_started_at = 0.0
        self.pinch_last_seen_at = 0.0
        self.pinch_click_pending = False
        self.combo_gesture_started = time.time()
        self.previous_combo_gesture = ""
        self.last_control_tick = time.time()
        self.last_control_dt_ms = 0.0
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.filtered_cursor: QtCore.QPointF | None = None
        self.filtered_os_cursor: QtCore.QPointF | None = None
        self.last_sent_os_cursor: QtCore.QPointF | None = None
        self.last_demo_point_for_os: QtCore.QPointF | None = None
        self.os_relative_remainder = QtCore.QPointF(0.0, 0.0)
        self.os_screen_geometry = WindowsMouse.virtual_geometry()
        self.windows_cursor_worker = DirectCursorDispatcher()
        self.windows_cursor_indicator = WindowsCursorIndicator()
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.windows_cursor_worker.stop)
            application.aboutToQuit.connect(self.windows_cursor_indicator.close)
        self.recording_active = False
        self.recording_started_at = 0.0
        self.recording_stage_started_at = 0.0
        self.recording_stage_index = 0
        self.recording_rows: list[dict[str, object]] = []
        self.recording_dir: Path | None = None
        self.cursor_debug_active = False
        self.cursor_debug_started_at = 0.0
        self.cursor_debug_rows: list[dict[str, object]] = []
        self.cursor_debug_dir: Path | None = None
        self.cursor_debug_previous_raw: tuple[float, float, float] | None = None
        self.cursor_debug_previous_data_id: int | None = None
        self.os_injection_count = 0
        self.last_os_injection_perf = 0.0
        self.last_os_debug: dict[str, object] = {}
        self.last_movement_debug: dict[str, object] = {}
        self.mouse_training_protocol = build_mouse_training_protocol()
        self.mouse_training_recorder = MouseTrainingRecorder()
        self.mouse_training_index = -1
        self.mouse_training_stage_started = 0.0
        self.screen_calibration: dict[str, object] | None = None
        self.fullscreen_calibration = FullScreenPointerCalibration(self)
        self.fullscreen_calibration.calibration_ready.connect(self._accept_screen_calibration)
        self.space_was_down = False
        self.space_emergency_enabled = True
        self._build_ui()
        self._load_screen_calibration()
        QtGui.QShortcut(QtGui.QKeySequence("F11"), self, activated=self.toggle_full_screen)
        QtGui.QShortcut(QtGui.QKeySequence("Escape"), self, activated=self.exit_full_screen)
        self.control_timer = QtCore.QTimer(self)
        self.control_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.control_timer.timeout.connect(self._control_tick)
        self.control_timer.start(12)
        self.recording_timer = QtCore.QTimer(self)
        self.recording_timer.timeout.connect(self._recording_tick)
        self.recording_timer.start(100)
        self.emergency_timer = QtCore.QTimer(self)
        self.emergency_timer.timeout.connect(self._poll_emergency_space)
        self.emergency_timer.start(30)
        self.mouse_training_timer = QtCore.QTimer(self)
        self.mouse_training_timer.timeout.connect(self._mouse_training_tick)
        self.mouse_training_timer.start(100)

    def _build_ui(self) -> None:
        """Create and configure ui for the current MouseControlWindow workflow."""
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.content_scroll = scroll
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Safe target-clicking demo")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #111827;")
        header.addWidget(title)
        header.addStretch()
        self.safe_mode_label = QtWidgets.QLabel("Safe mode: OS mouse is not controlled")
        self.safe_mode_label.setStyleSheet("font-weight: 700; color: #166534;")
        header.addWidget(self.safe_mode_label)
        self.full_screen_btn = QtWidgets.QPushButton("Full screen")
        self.full_screen_btn.clicked.connect(self.toggle_full_screen)
        header.addWidget(self.full_screen_btn)
        layout.addLayout(header)

        self.game = TargetClickGame()
        layout.addWidget(self.game, stretch=1)

        controls = QtWidgets.QGroupBox("Mouse/game control")
        grid = QtWidgets.QGridLayout(controls)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        self.enable_control = QtWidgets.QCheckBox("Enable demo control")
        self.enable_os_control = QtWidgets.QCheckBox("Enable real Windows mouse")
        self.enable_os_control.setEnabled(True)
        self.enable_os_control.setStyleSheet("font-weight: 700; color: #991b1b;")
        self.ignore_gesture_gate = QtWidgets.QCheckBox("IMU tuning mode: move without gesture")
        self.calibrate_btn = QtWidgets.QPushButton("Calibrate neutral pose")
        self.fullscreen_calibrate_btn = QtWidgets.QPushButton("Optional full-screen range recording")
        self.reset_btn = QtWidgets.QPushButton("Reset game")
        self.control_style = QtWidgets.QComboBox()
        self.control_style.addItem("Laser pointer", "laser")
        self.control_style.addItem("Velocity", "velocity")
        self.gesture_source = QtWidgets.QComboBox()
        self.gesture_source.addItem("Fast raw", "raw")
        self.gesture_source.addItem("Stable display", "stable")
        self.gesture_source.addItem("Mouse responsive", "mouse_responsive")
        self.gesture_source.addItem("Mouse balanced", "mouse_balanced")
        self.gesture_source.addItem("Mouse safe", "mouse_safe")
        self.gesture_source.setCurrentIndex(self.gesture_source.findData("mouse_safe"))
        self.game_mode = QtWidgets.QComboBox()
        self.game_mode.addItem("Target clicking", "click")
        self.game_mode.addItem("Gesture-at-target challenge", "gesture")
        self.game_mode.currentIndexChanged.connect(lambda: self.game.set_game_mode(str(self.game_mode.currentData())))
        self.enable_imu_combos = QtWidgets.QCheckBox("Enable gesture + IMU combinations")
        self.enable_imu_combos.setChecked(True)
        self.horizontal_sensitivity = QtWidgets.QDoubleSpinBox()
        self.horizontal_sensitivity.setRange(0.01, 20.0)
        self.horizontal_sensitivity.setSingleStep(0.05)
        self.horizontal_sensitivity.setValue(0.70)
        self.horizontal_sensitivity.setSuffix(" x")
        self.horizontal_sensitivity.setToolTip("Horizontal sensitivity. 0.70 is the calibrated 1x baseline.")
        self.vertical_sensitivity = QtWidgets.QDoubleSpinBox()
        self.vertical_sensitivity.setRange(0.01, 20.0)
        self.vertical_sensitivity.setSingleStep(0.05)
        self.vertical_sensitivity.setValue(0.70)
        self.vertical_sensitivity.setSuffix(" x")
        self.vertical_sensitivity.setToolTip("Vertical sensitivity. 0.70 is the calibrated 1x baseline.")
        self.windows_dpi = QtWidgets.QDoubleSpinBox()
        self.windows_dpi.setRange(0.20, 3.00)
        self.windows_dpi.setSingleStep(0.10)
        self.windows_dpi.setValue(1.00)
        self.windows_dpi.setSuffix(" x")
        self.windows_dpi.setToolTip(
            "Overall Windows pointer gain. Horizontal and vertical sensitivity remain separate axis controls."
        )
        self.dead_zone = QtWidgets.QDoubleSpinBox()
        self.dead_zone.setRange(0.0, 300.0)
        self.dead_zone.setSingleStep(5.0)
        self.dead_zone.setDecimals(0)
        self.dead_zone.setValue(12.0)
        self.dead_zone.setSuffix(" units")
        self.axis_mode = QtWidgets.QComboBox()
        self.axis_mode.addItem("Yaw + pitch", "yaw_pitch")
        self.axis_mode.addItem("Roll + pitch", "roll_pitch")
        self.axis_mode.addItem("Yaw + roll", "yaw_roll")
        self.scroll_step = QtWidgets.QSpinBox()
        self.scroll_step.setRange(1, 20)
        self.scroll_step.setValue(2)
        self.reference_label = QtWidgets.QLabel(f"Reference IMU: {REFERENCE_SENSOR_NAME} | {REFERENCE_SENSOR_ID}")
        self.gesture_label = QtWidgets.QLabel("Gesture: No model")
        self.imu_label = QtWidgets.QLabel("IMU: waiting")
        self.status_label = QtWidgets.QLabel("Status: calibrate neutral pose, then enable demo control")
        self.control_guide_btn = QtWidgets.QPushButton("Control guide")
        self.control_guide_btn.clicked.connect(self.show_control_guide)
        self.combo_status_label = QtWidgets.QLabel(
            "Combination status: ready | rest+right-left roll = movement toggle | fist+right-left roll = drag toggle | "
            "held pinch = precision drag | like+pitch = scroll | rest+yaw flick = navigation"
        )
        self.combo_status_label.setWordWrap(True)
        self.combo_status_label.setStyleSheet("font-weight: 600; color: #075985;")

        self.calibrate_btn.clicked.connect(self.calibrate_neutral_pose)
        self.fullscreen_calibrate_btn.clicked.connect(self.start_fullscreen_calibration)
        self.reset_btn.clicked.connect(self.reset_game)
        self.enable_control.toggled.connect(self.game.set_enabled)
        self.enable_os_control.toggled.connect(self._os_control_toggled)

        grid.addWidget(self.enable_control, 0, 0)
        grid.addWidget(self.ignore_gesture_gate, 0, 1)
        grid.addWidget(self.calibrate_btn, 0, 2)
        grid.addWidget(self.reset_btn, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Demo"), 1, 0)
        grid.addWidget(self.game_mode, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Gesture gate"), 1, 2)
        grid.addWidget(self.gesture_source, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Control style"), 2, 0)
        grid.addWidget(self.control_style, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Windows DPI / gain"), 2, 2)
        grid.addWidget(self.windows_dpi, 2, 3)
        grid.addWidget(QtWidgets.QLabel("Horizontal sensitivity"), 3, 0)
        grid.addWidget(self.horizontal_sensitivity, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Vertical sensitivity"), 3, 2)
        grid.addWidget(self.vertical_sensitivity, 3, 3)
        grid.addWidget(QtWidgets.QLabel("Dead zone"), 4, 0)
        grid.addWidget(self.dead_zone, 4, 1)
        grid.addWidget(QtWidgets.QLabel("Axis mode"), 4, 2)
        grid.addWidget(self.axis_mode, 4, 3)
        grid.addWidget(QtWidgets.QLabel("Scroll step"), 5, 0)
        grid.addWidget(self.scroll_step, 5, 1)
        grid.addWidget(self.reference_label, 6, 0, 1, 2)
        grid.addWidget(self.gesture_label, 6, 2, 1, 2)
        grid.addWidget(self.imu_label, 7, 0, 1, 2)
        grid.addWidget(self.status_label, 7, 2, 1, 2)
        grid.addWidget(self.enable_imu_combos, 8, 0, 1, 2)
        grid.addWidget(self.fullscreen_calibrate_btn, 8, 2)
        grid.addWidget(self.enable_os_control, 8, 3)
        grid.addWidget(self.control_guide_btn, 9, 0, 1, 4)
        grid.addWidget(self.combo_status_label, 10, 0, 1, 4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addWidget(controls)

        cursor_debug = QtWidgets.QGroupBox("Windows cursor jump diagnostic")
        cursor_debug_layout = QtWidgets.QGridLayout(cursor_debug)
        self.start_cursor_debug_btn = QtWidgets.QPushButton("Start cursor diagnostic")
        self.stop_cursor_debug_btn = QtWidgets.QPushButton("Stop and save diagnostic")
        self.stop_cursor_debug_btn.setEnabled(False)
        self.cursor_debug_status = QtWidgets.QLabel(
            "Diagnostic status: idle | Records raw IMU, filtered IMU, calculated target, worker position, and Windows cursor."
        )
        self.cursor_debug_status.setWordWrap(True)
        self.start_cursor_debug_btn.clicked.connect(self.start_cursor_debug_recording)
        self.stop_cursor_debug_btn.clicked.connect(self.stop_cursor_debug_recording)
        cursor_debug_layout.addWidget(self.start_cursor_debug_btn, 0, 0)
        cursor_debug_layout.addWidget(self.stop_cursor_debug_btn, 0, 1)
        cursor_debug_layout.addWidget(self.cursor_debug_status, 1, 0, 1, 2)
        layout.addWidget(cursor_debug)

    def show_control_guide(self) -> None:
        """Open or display control guide for the current MouseControlWindow workflow."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Mouse Control Guide")
        dialog.setMinimumWidth(560)
        layout = QtWidgets.QVBoxLayout(dialog)
        table = QtWidgets.QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Gesture or combination", "Action"])
        rows = [
            ("open_hand or pointing", "Move cursor"),
            ("fist or short pinch", "Left click"),
            ("like", "Right click"),
            ("wrist_extension / wrist_flexion", "Scroll up / down"),
            ("at_rest", "Freeze cursor"),
            ("at_rest + fast right-left roll", "Toggle movement without a movement gesture"),
            ("fist + fast right-left roll", "Toggle drag"),
            ("held pinch", "Drag while held; release ends drag"),
            ("like + fast pitch", "Scroll"),
            ("at_rest + fast yaw", "Browser back / forward"),
            ("Spacebar", "Emergency stop for Windows mouse control"),
        ]
        table.setRowCount(len(rows))
        for row, (control, action) in enumerate(rows):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(control))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(action))
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(table)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()

    def toggle_full_screen(self) -> None:
        """Perform the toggle full screen operation used by the MouseControlWindow workflow."""
        if self.isFullScreen():
            self.showNormal()
            self.full_screen_btn.setText("Full screen")
        else:
            self.showFullScreen()
            self.full_screen_btn.setText("Exit full screen")

    def exit_full_screen(self) -> None:
        """Perform the exit full screen operation used by the MouseControlWindow workflow."""
        if self.isFullScreen():
            self.showNormal()
            self.full_screen_btn.setText("Full screen")

    def reset_game(self) -> None:
        """Reset game for the current MouseControlWindow workflow."""
        self.game.reset_game()
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.filtered_cursor = self.game.state.cursor
        self.status_label.setText("Status: game reset")

    def calibrate_neutral_pose(self) -> None:
        """Perform the calibrate neutral pose operation used by the MouseControlWindow workflow."""
        if self.latest_reference is None:
            self.status_label.setText("Status: waiting for dorsal forearm IMU before calibration")
            return
        self.neutral_angles = self._reset_pointer_filter_to_latest()
        self._reset_motion()
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.last_control_tick = time.time()
        self.game.set_calibrated(True)
        self.game.state.cursor = self.game._play_rect().center()
        self.filtered_cursor = QtCore.QPointF(self.game.state.cursor)
        self.movement_anchor_angles = self.neutral_angles
        self.movement_anchor_cursor = QtCore.QPointF(self.game.state.cursor)
        self.last_demo_point_for_os = QtCore.QPointF(self.game.state.cursor)
        self.status_label.setText("Status: neutral pose calibrated")
        self.game.update()

    def start_fullscreen_calibration(self) -> None:
        """Start fullscreen calibration for the current MouseControlWindow workflow."""
        if self.latest_reference is None:
            self.status_label.setText("Status: waiting for dorsal forearm IMU before full-screen calibration")
            return
        self.enable_os_control.setChecked(False)
        self.fullscreen_calibration.update_angles(self._stable_angles())
        self.status_label.setText("Status: fullscreen calibration running; follow each target for 2 seconds, no click required")
        self.fullscreen_calibration.start()

    def _load_screen_calibration(self) -> None:
        """Load and validate screen calibration for the current MouseControlWindow workflow."""
        path = MOUSE_RECORDING_DIR / "screen_calibration" / "latest_screen_calibration.json"
        if not path.exists():
            return
        try:
            calibration = json.loads(path.read_text(encoding="utf-8"))
            if calibration.get("center_yaw") is None or calibration.get("center_pitch") is None:
                return
            self.screen_calibration = calibration
            self.enable_os_control.setEnabled(True)
            self.status_label.setText("Status: saved range recording loaded; Windows control mirrors the demo directly")
        except Exception:
            return

    def _accept_screen_calibration(self, calibration: dict) -> None:
        """Perform the accept screen calibration operation used by the MouseControlWindow workflow."""
        self.screen_calibration = calibration
        self.enable_os_control.setEnabled(True)
        self.neutral_angles = (
            float(calibration["center_yaw"]),
            float(calibration["center_pitch"]),
            self._stable_angles()[2],
        )
        calibration_dir = MOUSE_RECORDING_DIR / "screen_calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        path = calibration_dir / "latest_screen_calibration.json"
        path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        self._hold_current_pose_as_anchor()
        self.status_label.setText(
            "Status: full-screen calibration applied | "
            f"horizontal span {float(calibration['yaw_span']):.0f} | vertical span {float(calibration['pitch_span']):.0f}"
        )

    def _os_control_toggled(self, enabled: bool) -> None:
        """Perform the os control toggled operation used by the MouseControlWindow workflow."""
        if enabled:
            self.enable_control.setChecked(True)
            cursor_x, cursor_y = WindowsMouse.position()
            self.filtered_os_cursor = QtCore.QPointF(cursor_x, cursor_y)
            self.last_sent_os_cursor = QtCore.QPointF(cursor_x, cursor_y)
            self.last_demo_point_for_os = QtCore.QPointF(self.filtered_cursor or self.game.state.cursor)
            self.os_relative_remainder = QtCore.QPointF(0.0, 0.0)
            self.windows_cursor_worker.set_enabled(False)
            self._hold_current_pose_as_anchor()
            self.windows_cursor_indicator.show()
            self.windows_cursor_indicator.raise_()
            self.status_label.setText("Status: REAL WINDOWS MOUSE ACTIVE | mirrors demo movement | Spacebar disables immediately")
        else:
            self.windows_cursor_worker.set_enabled(False)
            self.windows_cursor_indicator.hide()
            self.filtered_os_cursor = None
            self.last_sent_os_cursor = None
            self.last_demo_point_for_os = None
            self.os_relative_remainder = QtCore.QPointF(0.0, 0.0)
            self.status_label.setText("Status: real Windows mouse disabled")

    def _mirror_demo_movement_to_windows(self, demo_point: QtCore.QPointF) -> None:
        """Perform the mirror demo movement to windows operation used by the MouseControlWindow workflow."""
        point = QtCore.QPointF(demo_point)
        if not self.enable_os_control.isChecked():
            self.last_demo_point_for_os = point
            return
        if self.last_demo_point_for_os is None:
            self.last_demo_point_for_os = point
            return
        dpi_scale = float(self.windows_dpi.value())
        dx = (point.x() - self.last_demo_point_for_os.x()) * dpi_scale
        dy = (point.y() - self.last_demo_point_for_os.y()) * dpi_scale
        self.last_demo_point_for_os = point
        accumulated_x = self.os_relative_remainder.x() + dx
        accumulated_y = self.os_relative_remainder.y() + dy
        send_dx = math.trunc(accumulated_x)
        send_dy = math.trunc(accumulated_y)
        self.os_relative_remainder = QtCore.QPointF(accumulated_x - send_dx, accumulated_y - send_dy)
        if send_dx == 0 and send_dy == 0:
            return
        WindowsMouse.move_relative(send_dx, send_dy)
        self.os_injection_count += 1
        self.last_os_injection_perf = time.perf_counter()
        current_x, current_y = WindowsMouse.position()
        self.filtered_os_cursor = QtCore.QPointF(current_x, current_y)
        self.last_sent_os_cursor = QtCore.QPointF(current_x, current_y)
        self.last_os_debug.update(
            {
                "relative_dx": send_dx,
                "relative_dy": send_dy,
                "relative_remainder_x": self.os_relative_remainder.x(),
                "relative_remainder_y": self.os_relative_remainder.y(),
                "target_x": current_x,
                "target_y": current_y,
                "worker_x": current_x,
                "worker_y": current_y,
                "actual_cursor_x": current_x,
                "actual_cursor_y": current_y,
            }
        )

    def _poll_emergency_space(self) -> None:
        """Perform the poll emergency space operation used by the MouseControlWindow workflow."""
        if not self.space_emergency_enabled:
            self.space_was_down = False
            return
        down = WindowsMouse.space_pressed()
        if down and not self.space_was_down:
            self._emergency_stop_all_control()
        self.space_was_down = down

    def _emergency_stop_all_control(self) -> None:
        """Perform the emergency stop all control operation used by the MouseControlWindow workflow."""
        if self.drag_active:
            WindowsMouse.left_up()
        self.drag_active = False
        self.drag_mode = None
        self.game.set_drag(False)
        self.imu_movement_toggle = False
        self.roll_sequence_gesture = ""
        self.mouse_gate_candidate = ""
        self.mouse_gate_count = 0
        self.pinch_click_pending = False
        self._cancel_fist_click_clutch()
        self.fist_click_armed = True
        self.enable_os_control.setChecked(False)
        self.enable_control.setChecked(False)
        self.ignore_gesture_gate.setChecked(False)
        self.windows_cursor_worker.set_enabled(False)
        self.windows_cursor_indicator.hide()
        self._reset_motion()
        self.status_label.setText("Status: EMERGENCY STOP - all mouse control disabled")

    def _move_os_cursor_from_angles(self, angles: tuple[float, float, float]) -> None:
        """Perform the move os cursor from angles operation used by the MouseControlWindow workflow."""
        if not self.enable_os_control.isChecked() or self.screen_calibration is None:
            return
        calibration = self.screen_calibration
        yaw_span = float(calibration.get("yaw_span", 0.0))
        pitch_span = float(calibration.get("pitch_span", 0.0))
        if yaw_span < 1.0:
            center_yaw = float(calibration["center_yaw"])
            yaw_span = max(abs(_angle_delta(float(calibration["yaw_min"]), center_yaw)), abs(_angle_delta(float(calibration["yaw_max"]), center_yaw)))
        if pitch_span < 1.0:
            center_pitch = float(calibration["center_pitch"])
            pitch_span = max(abs(_angle_delta(float(calibration["pitch_min"]), center_pitch)), abs(_angle_delta(float(calibration["pitch_max"]), center_pitch)))
        if yaw_span < 1.0 or pitch_span < 1.0:
            return
        x_angle, y_angle = self._movement_offsets(angles, anchored=True)
        x_angle = self._apply_dead_zone(x_angle)
        y_angle = self._apply_dead_zone(y_angle)
        x_span = float(calibration.get("x_positive_span" if x_angle >= 0.0 else "x_negative_span", yaw_span))
        y_span = float(calibration.get("y_positive_span" if y_angle >= 0.0 else "y_negative_span", pitch_span))
        x_span = max(1.0, x_span)
        y_span = max(1.0, y_span)
        x_angle = self._precision_curve(x_angle, x_span)
        y_angle = self._precision_curve(y_angle, y_span)
        horizontal_scale = float(self.horizontal_sensitivity.value()) / 0.70
        vertical_scale = float(self.vertical_sensitivity.value()) / 0.70
        dpi_scale = float(self.windows_dpi.value())
        screen_left, screen_top, screen_width, screen_height = WindowsMouse.virtual_geometry()
        if self.movement_anchor_os_cursor is None:
            cursor_x, cursor_y = WindowsMouse.position()
            self.movement_anchor_os_cursor = QtCore.QPointF(cursor_x, cursor_y)
        raw_target = QtCore.QPointF(
            self.movement_anchor_os_cursor.x() + (x_angle / x_span) * screen_width * horizontal_scale * dpi_scale,
            self.movement_anchor_os_cursor.y() + (y_angle / y_span) * screen_height * vertical_scale * dpi_scale,
        )
        target = QtCore.QPointF(raw_target)
        target.setX(max(screen_left, min(screen_left + screen_width - 1.0, target.x())))
        target.setY(max(screen_top, min(screen_top + screen_height - 1.0, target.y())))
        if self.filtered_os_cursor is None:
            cursor_x, cursor_y = WindowsMouse.position()
            self.filtered_os_cursor = QtCore.QPointF(cursor_x, cursor_y)
        worker_x, worker_y = self.windows_cursor_worker.current_position()
        actual_x, actual_y = WindowsMouse.position()
        self.last_os_debug = {
            "x_angle": x_angle,
            "y_angle": y_angle,
            "x_span": x_span,
            "y_span": y_span,
            "raw_target_x": raw_target.x(),
            "raw_target_y": raw_target.y(),
            "target_x": target.x(),
            "target_y": target.y(),
            "worker_x": worker_x,
            "worker_y": worker_y,
            "actual_cursor_x": actual_x,
            "actual_cursor_y": actual_y,
            "hit_horizontal_edge": False,
            "hit_vertical_edge": False,
        }
        self.filtered_os_cursor = QtCore.QPointF(target)
        self.windows_cursor_worker.submit_target(target.x(), target.y())
        hit_horizontal_edge = (
            (raw_target.x() < screen_left and worker_x <= screen_left + 1.0)
            or (raw_target.x() > screen_left + screen_width - 1.0 and worker_x >= screen_left + screen_width - 2.0)
        )
        hit_vertical_edge = (
            (raw_target.y() < screen_top and worker_y <= screen_top + 1.0)
            or (raw_target.y() > screen_top + screen_height - 1.0 and worker_y >= screen_top + screen_height - 2.0)
        )
        if hit_horizontal_edge or hit_vertical_edge:
            self.movement_anchor_angles = angles
            self.movement_anchor_os_cursor = QtCore.QPointF(worker_x, worker_y)
        self.last_os_debug["hit_horizontal_edge"] = hit_horizontal_edge
        self.last_os_debug["hit_vertical_edge"] = hit_vertical_edge

    def start_mouse_training(self) -> None:
        """Start mouse training for the current MouseControlWindow workflow."""
        missing = sorted(REQUIRED_SENSOR_IDS - self.connected_sensor_ids)
        if missing:
            self.mouse_training_status.setText(f"Training recording: connect all three sensors | missing {', '.join(missing)}")
            return
        session_dir = self.mouse_training_recorder.start(self.mouse_training_protocol)
        self.mouse_training_index = 0
        self.mouse_training_stage_started = time.time()
        self.mouse_training_recorder.set_stage(0, self.mouse_training_protocol[0])
        self.start_mouse_training_btn.setEnabled(False)
        self.stop_mouse_training_btn.setEnabled(True)
        self.mouse_training_status.setText(f"Training recording active: {session_dir}")
        self._update_mouse_training_ui()

    def stop_mouse_training(self) -> None:
        """Stop mouse training for the current MouseControlWindow workflow."""
        self.mouse_training_recorder.stop()
        self.start_mouse_training_btn.setEnabled(True)
        self.stop_mouse_training_btn.setEnabled(False)
        self.mouse_training_status.setText(
            f"Training recording stopped | rows {self.mouse_training_recorder.rows_written} | "
            f"saved {self.mouse_training_recorder.session_dir}"
        )

    def _mouse_training_tick(self) -> None:
        """Perform the mouse training tick operation used by the MouseControlWindow workflow."""
        if not self.mouse_training_recorder.active or self.mouse_training_index < 0:
            return
        stage = self.mouse_training_protocol[self.mouse_training_index]
        if time.time() - self.mouse_training_stage_started >= float(stage["duration_s"]):
            self.mouse_training_index += 1
            if self.mouse_training_index >= len(self.mouse_training_protocol):
                self.stop_mouse_training()
                self.mouse_training_instruction.setText("Training instruction: complete")
                self.mouse_training_progress.setValue(len(self.mouse_training_protocol))
                return
            self.mouse_training_stage_started = time.time()
            self.mouse_training_recorder.set_stage(self.mouse_training_index, self.mouse_training_protocol[self.mouse_training_index])
        self._update_mouse_training_ui()

    def _update_mouse_training_ui(self) -> None:
        """Refresh mouse training ui for the current MouseControlWindow workflow."""
        if self.mouse_training_index < 0 or self.mouse_training_index >= len(self.mouse_training_protocol):
            return
        stage = self.mouse_training_protocol[self.mouse_training_index]
        remaining = max(0.0, float(stage["duration_s"]) - (time.time() - self.mouse_training_stage_started))
        self.mouse_training_instruction.setText(f"Training instruction: {stage['instruction']} | {remaining:.1f}s")
        self.mouse_training_progress.setValue(self.mouse_training_index + 1)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window closure and release application resources safely."""
        self.stop_cursor_debug_recording()
        if self.drag_active:
            WindowsMouse.left_up()
            self.drag_active = False
            self.drag_mode = None
            self.game.set_drag(False)
        self.enable_os_control.setChecked(False)
        self.enable_control.setChecked(False)
        self.windows_cursor_worker.set_enabled(False)
        self.windows_cursor_indicator.hide()
        self.mouse_training_recorder.stop()
        self.hide()
        event.ignore()

    def start_cursor_debug_recording(self) -> None:
        """Start cursor debug recording for the current MouseControlWindow workflow."""
        if self.cursor_debug_active:
            return
        if self.latest_reference is None:
            self.cursor_debug_status.setText("Diagnostic status: waiting for the dorsal forearm IMU.")
            return
        stamp = datetime.now().strftime("cursor_debug_%Y%m%d_%H%M%S")
        self.cursor_debug_dir = MOUSE_RECORDING_DIR / stamp
        self.cursor_debug_dir.mkdir(parents=True, exist_ok=True)
        self.cursor_debug_rows = []
        self.cursor_debug_previous_raw = None
        self.cursor_debug_previous_data_id = None
        self.cursor_debug_started_at = time.perf_counter()
        self.cursor_debug_active = True
        self.start_cursor_debug_btn.setEnabled(False)
        self.stop_cursor_debug_btn.setEnabled(True)
        self.cursor_debug_status.setText(
            "Diagnostic status: RECORDING | Reproduce the stutter and unwanted jump, then press Stop and save diagnostic."
        )

    def stop_cursor_debug_recording(self) -> None:
        """Stop cursor debug recording for the current MouseControlWindow workflow."""
        if not self.cursor_debug_active:
            return
        self.cursor_debug_active = False
        self.start_cursor_debug_btn.setEnabled(True)
        self.stop_cursor_debug_btn.setEnabled(False)
        if not self.cursor_debug_rows or self.cursor_debug_dir is None:
            self.cursor_debug_status.setText("Diagnostic status: stopped without samples.")
            return
        csv_path = self.cursor_debug_dir / "cursor_pipeline.csv"
        fields = list(self.cursor_debug_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.cursor_debug_rows)
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rows": len(self.cursor_debug_rows),
            "reference_sensor_id": self.reference_sensor_id,
            "control_style": self.control_style.currentData(),
            "axis_mode": self.axis_mode.currentData(),
            "horizontal_sensitivity": float(self.horizontal_sensitivity.value()),
            "vertical_sensitivity": float(self.vertical_sensitivity.value()),
            "windows_dpi_gain": float(self.windows_dpi.value()),
            "dead_zone": float(self.dead_zone.value()),
            "gesture_gate": self.gesture_source.currentData(),
            "windows_mouse_enabled_at_end": self.enable_os_control.isChecked(),
        }
        (self.cursor_debug_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        diagnosis = ""
        try:
            from analyze_cursor_pipeline import analyze

            result = analyze(self.cursor_debug_dir)
            diagnosis = str(result.get("movement_diagnosis", ""))
        except Exception as exc:
            diagnosis = f"automatic analysis failed: {exc}"
        self.cursor_debug_status.setText(
            f"Diagnostic status: saved {len(self.cursor_debug_rows)} samples | {diagnosis} | {csv_path}"
        )

    def _record_cursor_debug_tick(
        self, now: float, gesture: str, confidence: float, movement_requested: bool
    ) -> None:
        """Record cursor debug tick for the current MouseControlWindow workflow."""
        if not self.cursor_debug_active or self.latest_reference is None:
            return
        raw = self._latest_angles()
        filtered = self._pointer_angles()
        anchor = self.movement_anchor_angles or (math.nan, math.nan, math.nan)
        raw_delta = (
            tuple(_angle_delta(raw[index], self.cursor_debug_previous_raw[index]) for index in range(3))
            if self.cursor_debug_previous_raw is not None
            else (0.0, 0.0, 0.0)
        )
        self.cursor_debug_previous_raw = raw
        sample_perf = time.perf_counter()
        worker_x, worker_y = self.windows_cursor_worker.current_position()
        actual_x, actual_y = WindowsMouse.position()
        foreground_hwnd, foreground_pid, foreground_title = WindowsMouse.foreground_context()
        data_id = int(self.latest_reference.data_id)
        data_id_delta = (
            data_id - self.cursor_debug_previous_data_id
            if self.cursor_debug_previous_data_id is not None
            else 0
        )
        self.cursor_debug_previous_data_id = data_id
        row: dict[str, object] = {
            "elapsed_s": sample_perf - self.cursor_debug_started_at,
            "wall_time_s": now,
            "data_id": data_id,
            "data_id_delta": data_id_delta,
            "snapshot_age_ms": self.latest_reference.age_ms,
            "control_dt_ms": self.last_control_dt_ms,
            "foreground_hwnd": foreground_hwnd,
            "foreground_pid": foreground_pid,
            "foreground_title": foreground_title,
            "os_injection_count": self.os_injection_count,
            "ms_since_os_injection": (
                (sample_perf - self.last_os_injection_perf) * 1000.0
                if self.last_os_injection_perf > 0.0
                else math.nan
            ),
            "gesture": gesture,
            "confidence": confidence,
            "movement_requested": movement_requested,
            "movement_active": self.movement_active,
            "enable_demo_control": self.enable_control.isChecked(),
            "imu_tuning_mode": self.ignore_gesture_gate.isChecked(),
            "neutral_calibrated": self.neutral_angles is not None,
            "clutch_just_engaged": self.cursor_clutch_just_engaged,
            "windows_mouse_enabled": self.enable_os_control.isChecked(),
            "raw_yaw": raw[0],
            "raw_pitch": raw[1],
            "raw_roll": raw[2],
            "reference_ax": self.latest_reference.ax,
            "reference_ay": self.latest_reference.ay,
            "reference_az": self.latest_reference.az,
            "reference_q0": self.latest_reference.quaternion[0],
            "reference_q1": self.latest_reference.quaternion[1],
            "reference_q2": self.latest_reference.quaternion[2],
            "reference_q3": self.latest_reference.quaternion[3],
            "raw_yaw_delta": raw_delta[0],
            "raw_pitch_delta": raw_delta[1],
            "raw_roll_delta": raw_delta[2],
            "filtered_yaw": filtered[0],
            "filtered_pitch": filtered[1],
            "filtered_roll": filtered[2],
            "anchor_yaw": anchor[0],
            "anchor_pitch": anchor[1],
            "anchor_roll": anchor[2],
            "worker_x": worker_x,
            "worker_y": worker_y,
            "actual_cursor_x": actual_x,
            "actual_cursor_y": actual_y,
            "demo_cursor_x": self.game.state.cursor.x(),
            "demo_cursor_y": self.game.state.cursor.y(),
        }
        for sensor_id, snapshot in self.latest_snapshots_by_id.items():
            prefix = sensor_id.lower()
            row[f"{prefix}_yaw"] = snapshot.yaw
            row[f"{prefix}_pitch"] = snapshot.pitch
            row[f"{prefix}_roll"] = snapshot.roll
            row[f"{prefix}_ax"] = snapshot.ax
            row[f"{prefix}_ay"] = snapshot.ay
            row[f"{prefix}_az"] = snapshot.az
            for index, value in enumerate(snapshot.quaternion):
                row[f"{prefix}_q{index}"] = value
        for key in (
            "block_reason",
            "movement_x_angle",
            "movement_y_angle",
            "deadzone_x",
            "deadzone_y",
            "curved_x",
            "curved_y",
            "demo_target_x",
            "demo_target_y",
        ):
            row[key] = self.last_movement_debug.get(key, "")
        for key in (
            "x_angle",
            "y_angle",
            "x_span",
            "y_span",
            "raw_target_x",
            "raw_target_y",
            "target_x",
            "target_y",
            "hit_horizontal_edge",
            "hit_vertical_edge",
        ):
            row[key] = self.last_os_debug.get(key, math.nan)
        self.cursor_debug_rows.append(row)

    def receive_snapshots(self, snapshots: list[DeviceSnapshot]) -> None:
        """Perform the receive snapshots operation used by the MouseControlWindow workflow."""
        for snap in snapshots:
            sensor_id = f"{snap.unit_id:08X}"
            if sensor_id in REQUIRED_SENSOR_IDS:
                self.latest_snapshots_by_id[sensor_id] = snap
        self.connected_sensor_ids = {
            f"{snap.unit_id:08X}" for snap in snapshots if snap.age_ms < 1000 and f"{snap.unit_id:08X}" in REQUIRED_SENSOR_IDS
        }
        for snap in snapshots:
            if f"{snap.unit_id:08X}" == self.reference_sensor_id:
                self.latest_reference = snap
                raw_angles = self._angles(snap)
                self.angle_history.append(raw_angles)
                self._update_pointer_angles(raw_angles)
                self.fullscreen_calibration.update_angles(raw_angles)
                self.imu_label.setText(f"IMU yaw {snap.yaw:.2f} | pitch {snap.pitch:.2f} | roll {snap.roll:.2f}")
                break
        if self.recording_active:
            self._record_snapshots(snapshots)
        self.mouse_training_recorder.add_snapshots(snapshots)

    def update_control(self, stable_gesture: str, confidence: float, raw_gesture: str | None = None, raw_confidence: float | None = None) -> None:
        """Refresh control for the current MouseControlWindow workflow."""
        self.current_gesture = stable_gesture or "Uncertain"
        self.current_confidence = float(confidence)
        self.raw_gesture = raw_gesture or self.current_gesture
        self.raw_confidence = float(raw_confidence if raw_confidence is not None else self.current_confidence)
        self.gesture_label.setText(
            f"Gesture: stable {self.current_gesture} {self.current_confidence:.0%} | raw {self.raw_gesture} {self.raw_confidence:.0%}"
        )

    def set_signal_safety(self, safe: bool, reason: str) -> None:
        """Set signal safety for the current MouseControlWindow workflow."""
        safe = bool(safe)
        reason = str(reason)
        if safe == self.signal_safe and reason == self.signal_safety_reason:
            return
        was_safe = self.signal_safe
        self.signal_safe = safe
        self.signal_safety_reason = reason
        if not safe:
            if self.drag_active:
                WindowsMouse.left_up()
            self.drag_active = False
            self.drag_mode = None
            self.game.set_drag(False)
            self.pinch_click_pending = False
            self.fist_click_armed = False
            self._cancel_fist_click_clutch()
            self.imu_movement_toggle = False
            self.roll_sequence_gesture = ""
            self._reset_motion()
            if self.latest_reference is not None:
                self._hold_current_pose_as_anchor()
            self.status_label.setText(f"Status: SIGNAL SAFETY FREEZE | {reason}")
        elif not was_safe:
            self.fist_click_armed = True
            if self.latest_reference is not None:
                self._reset_pointer_filter_to_latest()
                self._hold_current_pose_as_anchor()
            self.status_label.setText("Status: signal recovered; mouse control re-anchored")

    def _control_tick(self) -> None:
        """Perform the control tick operation used by the MouseControlWindow workflow."""
        now = time.time()
        self.last_os_debug = {}
        self.last_movement_debug = {"block_reason": ""}
        self.cursor_clutch_just_engaged = False
        raw_control_dt = now - self.last_control_tick
        self.last_control_dt_ms = raw_control_dt * 1000.0
        dt = max(0.001, min(0.05, raw_control_dt))
        self.last_control_tick = now
        if not self.signal_safe:
            self.last_movement_debug["block_reason"] = "signal_safety_gate"
            self.status_label.setText(f"Status: SIGNAL SAFETY FREEZE | {self.signal_safety_reason}")
            return
        gesture, confidence = self._mouse_gate_output()
        self.game.register_gesture(gesture)
        if gesture in {"open_hand", "pointing"}:
            self.last_movement_gesture = gesture
            self.last_movement_seen = now
        elif gesture == "Uncertain" and now - self.last_movement_seen <= 0.40:
            gesture = self.last_movement_gesture
            confidence = max(confidence, 0.45)
        if self.latest_reference is None:
            self.last_movement_debug["block_reason"] = "no_reference_imu"
            self.status_label.setText("Status: waiting for dorsal forearm IMU")
            return
        if not self.enable_control.isChecked():
            self.last_movement_debug["block_reason"] = "demo_control_disabled"
            self._cancel_fist_click_clutch()
            self._hold_current_pose_as_anchor()
            self._record_cursor_debug_tick(now, gesture, confidence, False)
            return
        if self.neutral_angles is None:
            self.last_movement_debug["block_reason"] = "neutral_not_calibrated"
            self.status_label.setText("Status: calibrate neutral pose before enabling movement")
            self._record_cursor_debug_tick(now, gesture, confidence, False)
            return
        self._apply_imu_combinations(gesture, angles=self._stable_angles(), now=now)
        self._handle_pinch_lifecycle(gesture, now)
        self._update_fist_click_clutch(gesture, confidence, now)
        tuning_mode = self.ignore_gesture_gate.isChecked()
        persistent_movement = self.imu_movement_toggle
        action_gesture = gesture if gesture not in {"open_hand", "pointing", "at_rest", "Uncertain", "No model", "Error"} else ""
        if action_gesture == "fist" and now < self.suppress_fist_action_until:
            action_gesture = ""
        dragging_with_gesture = self.drag_active and self.drag_mode in {"pinch", "toggle"}
        persistent_can_move = persistent_movement and (not action_gesture or dragging_with_gesture) and gesture not in {"Uncertain", "No model", "Error"}
        movement_requested = tuning_mode or persistent_can_move or dragging_with_gesture or gesture in {"open_hand", "pointing"}
        if self.fist_clutch_active:
            movement_requested = False
            self._enforce_fist_click_clutch()
        if not movement_requested and (confidence < 0.45 or gesture in {"Uncertain", "No model", "Error", "at_rest"}):
            self.last_movement_debug["block_reason"] = f"movement_gate_closed:{gesture}"
            self._hold_current_pose_as_anchor()
            self.status_label.setText("Status: frozen")
            self.last_gesture = gesture
            self._record_cursor_debug_tick(now, gesture, confidence, False)
            return

        angles = self._pointer_angles()
        if movement_requested:
            if not self.movement_active:
                angles = self._reset_pointer_filter_to_latest()
                self._start_movement_anchor(angles)
                self.cursor_clutch_just_engaged = True
            x_angle, y_angle = self._movement_offsets(angles, anchored=True)
            x_offset = self._apply_dead_zone(x_angle)
            y_offset = self._apply_dead_zone(y_angle)
            curved_x = self._precision_curve(x_offset, 500.0)
            curved_y = self._precision_curve(y_offset, 500.0)
            self.last_movement_debug.update(
                {
                    "block_reason": "movement_active",
                    "movement_x_angle": x_angle,
                    "movement_y_angle": y_angle,
                    "deadzone_x": x_offset,
                    "deadzone_y": y_offset,
                    "curved_x": curved_x,
                    "curved_y": curved_y,
                }
            )
            x_offset = curved_x
            y_offset = curved_y
            horizontal_gain = float(self.horizontal_sensitivity.value())
            vertical_gain = float(self.vertical_sensitivity.value())
            if self.control_style.currentData() == "laser":
                origin = self.movement_anchor_cursor or self.game.state.cursor
                target = QtCore.QPointF(origin.x() + x_offset * horizontal_gain, origin.y() + y_offset * vertical_gain)
                self.last_movement_debug["demo_target_x"] = target.x()
                self.last_movement_debug["demo_target_y"] = target.y()
                if self.filtered_cursor is None:
                    self.filtered_cursor = self.game.state.cursor
                alpha = 0.22
                self.filtered_cursor = QtCore.QPointF(
                    (1.0 - alpha) * self.filtered_cursor.x() + alpha * target.x(),
                    (1.0 - alpha) * self.filtered_cursor.y() + alpha * target.y(),
                )
                self.game.set_cursor(self.filtered_cursor, "Point cursor")
                self._mirror_demo_movement_to_windows(self.filtered_cursor)
                self.status_label.setText("Status: persistent cursor movement" if persistent_movement else "Status: laser pointer tuning" if tuning_mode else "Status: laser pointer")
            else:
                target_vx = x_offset * horizontal_gain
                target_vy = y_offset * vertical_gain
                max_speed = 1000.0
                target_vx = max(-max_speed, min(max_speed, target_vx))
                target_vy = max(-max_speed, min(max_speed, target_vy))
                alpha = 0.16
                self.filtered_vx = (1.0 - alpha) * self.filtered_vx + alpha * target_vx
                self.filtered_vy = (1.0 - alpha) * self.filtered_vy + alpha * target_vy
                if abs(self.filtered_vx) < 2.0:
                    self.filtered_vx = 0.0
                if abs(self.filtered_vy) < 2.0:
                    self.filtered_vy = 0.0
                self.game.move_cursor(self.filtered_vx * dt, self.filtered_vy * dt)
                self._mirror_demo_movement_to_windows(self.game.state.cursor)
                self.status_label.setText("Status: persistent cursor movement" if persistent_movement else "Status: tuning movement" if tuning_mode else "Status: moving cursor")

        if action_gesture:
            if persistent_movement and not dragging_with_gesture:
                self._hold_current_pose_as_anchor()
            self._apply_gesture_action(action_gesture, now, preserve_movement=persistent_movement or dragging_with_gesture)
        elif not movement_requested and not self.fist_clutch_active:
            self._reset_motion()
            self.status_label.setText("Status: gesture has no mouse action")

        self.last_gesture = gesture
        self._record_cursor_debug_tick(now, gesture, confidence, movement_requested)

    def _apply_gesture_action(self, gesture: str, now: float, preserve_movement: bool) -> None:
        """Apply gesture action for the current MouseControlWindow workflow."""
        if not preserve_movement:
            self._reset_motion()
        if gesture == "pinch":
            self.status_label.setText("Status: pinch held for drag" if self.drag_active else "Status: pinch pending; release to click")
        elif gesture == "fist":
            if self.drag_active:
                self.status_label.setText("Status: dragging")
            elif self.fist_clutch_active and not self.fist_click_sent:
                self.status_label.setText("Status: fist click stabilizing")
        elif gesture == "like":
            if self.last_gesture != "like" or now - self.last_click_time > 0.65:
                self.game.right_click()
                if self.enable_os_control.isChecked():
                    WindowsMouse.right_click()
                self.last_click_time = now
            self.status_label.setText("Status: right click")
        elif gesture in {"wrist_extension", "wrist_flexion"}:
            if now - self.last_scroll_time > 0.20:
                sign = 1 if gesture == "wrist_extension" else -1
                self.game.scroll(sign * int(self.scroll_step.value()))
                if self.enable_os_control.isChecked():
                    WindowsMouse.scroll(sign * int(self.scroll_step.value()))
                self.last_scroll_time = now
            self.status_label.setText("Status: scrolling")

    def _update_fist_click_clutch(self, gesture: str, confidence: float, now: float) -> None:
        """Refresh fist click clutch for the current MouseControlWindow workflow."""
        if self.fist_first_click_time and now - self.fist_first_click_time > WindowsMouse.double_click_window_s():
            self.fist_first_click_time = 0.0
            self.fist_first_click_demo_position = None
            self.fist_first_click_os_position = None

        raw_fist = self.raw_gesture == "fist" and self.raw_confidence >= 0.55
        stable_fist = gesture == "fist" and confidence >= 0.50
        fist_present = raw_fist or stable_fist
        combo_in_progress = self.roll_sequence_gesture == "fist"

        if (
            fist_present
            and not self.fist_clutch_active
            and self.fist_click_armed
            and not self.drag_active
            and now >= self.suppress_fist_action_until
        ):
            self._begin_fist_click_clutch(now)

        if not self.fist_clutch_active:
            return

        self._enforce_fist_click_clutch()
        if combo_in_progress or now < self.suppress_fist_action_until:
            self.fist_stable_started = 0.0
        elif stable_fist and not self.fist_click_sent:
            if not self.fist_stable_started:
                self.fist_stable_started = now
            elif now - self.fist_stable_started >= 0.12:
                self._perform_fist_clutch_click(now)

        if fist_present:
            self.fist_release_started = 0.0
            return
        if not self.fist_release_started:
            self.fist_release_started = now
        elif now - self.fist_release_started >= 0.15:
            self.fist_click_armed = True
            self._finish_fist_click_clutch()

    def _begin_fist_click_clutch(self, now: float) -> None:
        """Perform the begin fist click clutch operation used by the MouseControlWindow workflow."""
        self.fist_clutch_active = True
        self.fist_clutch_started = now
        self.fist_stable_started = 0.0
        self.fist_release_started = 0.0
        self.fist_click_sent = False
        if self.fist_first_click_time and now - self.fist_first_click_time <= WindowsMouse.double_click_window_s():
            self.fist_clutch_demo_position = (
                QtCore.QPointF(self.fist_first_click_demo_position)
                if self.fist_first_click_demo_position is not None
                else QtCore.QPointF(self.game.state.cursor)
            )
            self.fist_clutch_os_position = (
                QtCore.QPointF(self.fist_first_click_os_position)
                if self.fist_first_click_os_position is not None
                else QtCore.QPointF(*WindowsMouse.position())
            )
        else:
            self.fist_clutch_demo_position = QtCore.QPointF(self.game.state.cursor)
            self.fist_clutch_os_position = QtCore.QPointF(*WindowsMouse.position())
        self._reset_motion()
        self._enforce_fist_click_clutch()
        self.status_label.setText("Status: fist click clutch")

    def _enforce_fist_click_clutch(self) -> None:
        """Perform the enforce fist click clutch operation used by the MouseControlWindow workflow."""
        if not self.fist_clutch_active:
            return
        if self.fist_clutch_demo_position is not None:
            self.game.set_cursor(self.fist_clutch_demo_position, "Click clutch")
        if self.enable_os_control.isChecked() and self.fist_clutch_os_position is not None:
            x, y = self.fist_clutch_os_position.x(), self.fist_clutch_os_position.y()
            WindowsMouse.move_to(x, y)
            self.windows_cursor_worker.sync_position(x, y)
            self.filtered_os_cursor = QtCore.QPointF(x, y)
            self.last_sent_os_cursor = QtCore.QPointF(x, y)

    def _perform_fist_clutch_click(self, now: float) -> None:
        """Perform the perform fist clutch click operation used by the MouseControlWindow workflow."""
        self._enforce_fist_click_clutch()
        self.game.left_click()
        if self.enable_os_control.isChecked():
            WindowsMouse.left_click()
        double_click = (
            self.fist_first_click_time > 0.0
            and now - self.fist_first_click_time <= WindowsMouse.double_click_window_s()
        )
        if double_click:
            self.fist_first_click_time = 0.0
            self.fist_first_click_demo_position = None
            self.fist_first_click_os_position = None
        else:
            self.fist_first_click_time = now
            self.fist_first_click_demo_position = (
                QtCore.QPointF(self.fist_clutch_demo_position)
                if self.fist_clutch_demo_position is not None
                else None
            )
            self.fist_first_click_os_position = (
                QtCore.QPointF(self.fist_clutch_os_position)
                if self.fist_clutch_os_position is not None
                else None
            )
        self.fist_click_sent = True
        self.fist_click_armed = False
        self.last_click_time = now
        self.status_label.setText("Status: fist double-click" if double_click else "Status: fist click 1/2")

    def _finish_fist_click_clutch(self) -> None:
        """Perform the finish fist click clutch operation used by the MouseControlWindow workflow."""
        self.fist_clutch_active = False
        self.fist_clutch_started = 0.0
        self.fist_stable_started = 0.0
        self.fist_release_started = 0.0
        self.fist_click_sent = False
        self.fist_clutch_demo_position = None
        self.fist_clutch_os_position = None
        if self.latest_reference is not None:
            self._reset_pointer_filter_to_latest()
            self._hold_current_pose_as_anchor()
        self.status_label.setText("Status: click released; cursor re-anchored")

    def _cancel_fist_click_clutch(self) -> None:
        """Perform the cancel fist click clutch operation used by the MouseControlWindow workflow."""
        self.fist_clutch_active = False
        self.fist_clutch_started = 0.0
        self.fist_stable_started = 0.0
        self.fist_release_started = 0.0
        self.fist_click_sent = False
        self.fist_clutch_demo_position = None
        self.fist_clutch_os_position = None
        self.fist_first_click_time = 0.0
        self.fist_first_click_demo_position = None
        self.fist_first_click_os_position = None

    def _handle_pinch_lifecycle(self, gesture: str, now: float) -> None:
        """Handle pinch lifecycle for the current MouseControlWindow workflow."""
        if gesture == "pinch":
            self.pinch_last_seen_at = now
            if not self.pinch_click_pending and not (self.drag_active and self.drag_mode == "pinch"):
                self.pinch_started_at = now
                self.pinch_click_pending = True
            if self.pinch_click_pending and now - self.pinch_started_at >= 0.70 and not self.drag_active:
                self.drag_active = True
                self.drag_mode = "pinch"
                self.pinch_click_pending = False
                self.game.set_drag(True)
                if self.enable_os_control.isChecked():
                    WindowsMouse.left_down()
                self.status_label.setText("Status: held pinch drag ON")
            return
        if self.drag_active and self.drag_mode == "pinch":
            if now - self.pinch_last_seen_at < 0.35:
                return
            if self.enable_os_control.isChecked():
                WindowsMouse.left_up()
            self.drag_active = False
            self.drag_mode = None
            self.game.set_drag(False)
            self.status_label.setText("Status: pinch drag released")
            return
        if self.pinch_click_pending:
            duration = now - self.pinch_started_at
            self.pinch_click_pending = False
            self.pinch_started_at = 0.0
            if duration < 0.70 and not (self.drag_active and self.drag_mode == "pinch"):
                self.game.left_click()
                if self.enable_os_control.isChecked():
                    WindowsMouse.left_click()
                self.last_click_time = now
                self.status_label.setText("Status: short pinch click")

    def _mouse_gate_output(self) -> tuple[str, float]:
        """Perform the mouse gate output operation used by the MouseControlWindow workflow."""
        mode = str(self.gesture_source.currentData())
        if mode == "raw":
            return self.raw_gesture or "Uncertain", self.raw_confidence
        if mode == "stable":
            return self.current_gesture or "Uncertain", self.current_confidence

        label = self.raw_gesture or "Uncertain"
        confidence = self.raw_confidence
        movement = label in {"open_hand", "pointing"}
        click = label in {"fist", "pinch", "like"}
        if self.drag_active and self.drag_mode == "pinch" and label == "pinch":
            return "pinch", confidence
        if mode == "mouse_responsive":
            threshold = 0.35 if movement else 0.65 if click else 0.50
            required = 1
        elif mode == "mouse_safe":
            threshold = 0.60 if movement else 0.80 if click else 0.70
            required = 2 if movement else 3
        else:
            threshold = 0.45 if movement else 0.72 if click else 0.60
            required = 1 if movement else 2

        candidate = label if confidence >= threshold else "Uncertain"
        if candidate == self.mouse_gate_candidate:
            self.mouse_gate_count += 1
        else:
            self.mouse_gate_candidate = candidate
            self.mouse_gate_count = 1
        if candidate == "Uncertain" or self.mouse_gate_count < required:
            return "Uncertain", confidence
        return candidate, confidence

    def _apply_imu_combinations(self, gesture: str, angles: tuple[float, float, float], now: float) -> None:
        """Apply imu combinations for the current MouseControlWindow workflow."""
        if not self.enable_imu_combos.isChecked():
            if self.drag_active:
                WindowsMouse.left_up()
                self.drag_active = False
                self.game.set_drag(False)
            self.previous_combo_angles = angles
            self.combo_angle_history.clear()
            self.combo_status_label.setText("Combination status: disabled")
            return

        self.combo_angle_history.append((now, angles))
        while self.combo_angle_history and now - self.combo_angle_history[0][0] > 0.35:
            self.combo_angle_history.popleft()
        if len(self.combo_angle_history) < 2:
            return
        if gesture != self.previous_combo_gesture:
            self.combo_gesture_started = now
            self.previous_combo_gesture = gesture
        reference_angles = self.combo_angle_history[0][1]
        roll_delta = _angle_delta(angles[2], reference_angles[2])
        yaw_delta = _angle_delta(angles[0], reference_angles[0])
        pitch_delta = _angle_delta(angles[1], reference_angles[1])
        roll_right = (
            roll_delta >= ROLL_SEQUENCE_RIGHT_THRESHOLD
            and abs(roll_delta) >= 0.65 * max(abs(yaw_delta), abs(pitch_delta), 1.0)
        )
        roll_left = (
            roll_delta <= -ROLL_SEQUENCE_LEFT_THRESHOLD
            and abs(roll_delta) >= 0.90 * max(abs(yaw_delta), abs(pitch_delta), 1.0)
        )
        pitch_flick = (
            abs(pitch_delta) >= FAST_PITCH_THRESHOLD
            and abs(pitch_delta) >= PITCH_DOMINANCE_RATIO * max(abs(yaw_delta), abs(roll_delta), 1.0)
        )
        yaw_flick = (
            abs(yaw_delta) >= FAST_YAW_THRESHOLD
            and abs(yaw_delta) >= YAW_DOMINANCE_RATIO * max(abs(pitch_delta), abs(roll_delta), 1.0)
        )
        event = ""
        if self.roll_sequence_started and now - self.roll_sequence_started > ROLL_SEQUENCE_TIMEOUT_S:
            self.roll_sequence_started = 0.0
            self.roll_sequence_gesture = ""
            self.combo_status_label.setText("Combination status: ready")
        if gesture in {"at_rest", "fist"} and roll_right and now - self.last_combo_time > 0.5:
            self.roll_sequence_started = now
            self.roll_sequence_gesture = gesture
            self.combo_status_label.setText("Combination status: roll sequence detected")
            self.combo_angle_history.clear()
            self.combo_angle_history.append((now, angles))
            self.previous_combo_angles = angles
            return
        roll_sequence_complete = (
            self.roll_sequence_started > 0.0
            and now - self.roll_sequence_started <= ROLL_SEQUENCE_TIMEOUT_S
            and roll_left
        )
        if roll_sequence_complete and self.roll_sequence_gesture == "at_rest":
            self.imu_movement_toggle = not self.imu_movement_toggle
            self.last_combo_time = now
            event = f"rest + right-left roll sequence: movement toggle {'ON' if self.imu_movement_toggle else 'OFF'}"
        if roll_sequence_complete and self.roll_sequence_gesture == "fist":
            releasing_drag = self.drag_active and self.drag_mode == "toggle"
            self.drag_active = not releasing_drag
            self.drag_mode = "toggle" if self.drag_active else None
            self.game.set_drag(self.drag_active)
            if self.drag_active and self.enable_os_control.isChecked():
                WindowsMouse.left_down()
            elif releasing_drag:
                WindowsMouse.left_up()
            self.last_combo_time = now
            self.suppress_fist_action_until = now + 0.85
            event = f"fist + right-left roll sequence: drag {'ON' if self.drag_active else 'OFF'}"
        if roll_sequence_complete:
            self.roll_sequence_started = 0.0
            self.roll_sequence_gesture = ""
        if gesture == "like" and pitch_flick and now - self.last_combo_time > 0.35:
            direction = -1 if pitch_delta > 0 else 1
            self.game.scroll(direction)
            if self.enable_os_control.isChecked():
                WindowsMouse.scroll(direction)
            self.last_combo_time = now
            event = f"like + pitch: scroll {'up' if direction > 0 else 'down'}"
        if gesture == "at_rest" and yaw_flick and now - self.last_combo_time > 0.8:
            back = yaw_delta > 0
            self.game.navigate(back=back)
            if self.enable_os_control.isChecked():
                WindowsMouse.browser_navigation(back=back)
            self.last_combo_time = now
            event = "rest + yaw flick: browser back" if back else "rest + yaw flick: browser forward"
        if event:
            self.status_label.setText(f"Status: {event}")
            self.combo_status_label.setText(f"Combination status: {event}")
            self.combo_angle_history.clear()
            self.combo_angle_history.append((now, angles))
        self.previous_combo_angles = angles

    def _apply_dead_zone(self, value: float) -> float:
        """Apply dead zone for the current MouseControlWindow workflow."""
        dead_zone = float(self.dead_zone.value())
        if abs(value) <= dead_zone:
            return 0.0
        return math.copysign(abs(value) - dead_zone, value)

    @staticmethod
    def _precision_curve(value: float, reference_span: float) -> float:
        """Perform the precision curve operation used by the MouseControlWindow workflow."""
        if value == 0.0:
            return 0.0
        span = max(1.0, abs(reference_span))
        normalized = abs(value) / span
        curved = normalized ** 1.45
        return math.copysign(curved * span, value)

    @staticmethod
    def _point_distance(first: QtCore.QPointF, second: QtCore.QPointF) -> float:
        """Perform the point distance operation used by the MouseControlWindow workflow."""
        return math.hypot(first.x() - second.x(), first.y() - second.y())

    @staticmethod
    def _adaptive_pointer_response(distance: float) -> tuple[float, float]:
        """Perform the adaptive pointer response operation used by the MouseControlWindow workflow."""
        if distance < 12.0:
            return 0.24, 6.0
        if distance < 50.0:
            return 0.38, 24.0
        if distance < 180.0:
            return 0.55, 90.0
        if distance < 500.0:
            return 0.72, 240.0
        return 0.85, 500.0

    @staticmethod
    def _limit_pointer_step(current: QtCore.QPointF, target: QtCore.QPointF, maximum_step: float) -> QtCore.QPointF:
        """Perform the limit pointer step operation used by the MouseControlWindow workflow."""
        dx = target.x() - current.x()
        dy = target.y() - current.y()
        distance = math.hypot(dx, dy)
        if distance <= maximum_step or distance <= 0.0:
            return target
        scale = maximum_step / distance
        return QtCore.QPointF(current.x() + dx * scale, current.y() + dy * scale)

    def _reset_motion(self) -> None:
        """Reset motion for the current MouseControlWindow workflow."""
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.filtered_cursor = None
        self.movement_active = False
        self.movement_anchor_angles = None
        self.movement_anchor_cursor = None
        self.movement_anchor_os_cursor = None
        self.last_demo_point_for_os = None

    def _start_movement_anchor(self, angles: tuple[float, float, float]) -> None:
        """Start movement anchor for the current MouseControlWindow workflow."""
        self.angle_history.clear()
        self.angle_history.append(angles)
        self.movement_active = True
        self.movement_anchor_angles = angles
        self.movement_anchor_cursor = self.game.state.cursor
        cursor_x, cursor_y = WindowsMouse.position()
        self.movement_anchor_os_cursor = QtCore.QPointF(cursor_x, cursor_y)
        self.filtered_cursor = self.game.state.cursor
        self.last_demo_point_for_os = QtCore.QPointF(self.filtered_cursor)
        self.filtered_os_cursor = self.movement_anchor_os_cursor
        self.last_sent_os_cursor = QtCore.QPointF(cursor_x, cursor_y)
        self.windows_cursor_worker.sync_position(cursor_x, cursor_y)

    def _reset_pointer_filter_to_latest(self) -> tuple[float, float, float]:
        """Remove filter lag before engaging movement from a frozen state."""
        raw = self._latest_angles()
        self.pointer_last_raw_angles = raw
        self.pointer_unwrapped_angles = raw
        self.pointer_filtered_angles = raw
        return raw

    def _hold_current_pose_as_anchor(self) -> None:
        """Perform the hold current pose as anchor operation used by the MouseControlWindow workflow."""
        angles = self._pointer_angles()
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.filtered_cursor = self.game.state.cursor
        self.last_demo_point_for_os = QtCore.QPointF(self.filtered_cursor)
        self.movement_active = False
        self.movement_anchor_angles = angles
        self.movement_anchor_cursor = self.game.state.cursor
        cursor_x, cursor_y = WindowsMouse.position()
        self.movement_anchor_os_cursor = QtCore.QPointF(cursor_x, cursor_y)
        self.filtered_os_cursor = self.movement_anchor_os_cursor
        self.last_sent_os_cursor = QtCore.QPointF(cursor_x, cursor_y)
        self.windows_cursor_worker.sync_position(cursor_x, cursor_y)
        self.angle_history.clear()
        self.angle_history.append(angles)

    def start_movement_recording(self) -> None:
        """Start movement recording for the current MouseControlWindow workflow."""
        if self.recording_active:
            return
        if self.latest_reference is None:
            self.recording_status_label.setText("Recording status: waiting for dorsal forearm IMU before recording")
            return
        self.neutral_angles = self._stable_angles()
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.filtered_cursor = self.game.state.cursor
        self.game.set_calibrated(True)
        MOUSE_RECORDING_DIR.mkdir(parents=True, exist_ok=True)
        session_name = datetime.now().strftime("mouse_diag_%Y%m%d_%H%M%S")
        self.recording_dir = MOUSE_RECORDING_DIR / session_name
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.recording_rows = []
        self.recording_active = True
        self.recording_started_at = time.time()
        self.recording_stage_started_at = self.recording_started_at
        self.recording_stage_index = 0
        self.start_recording_btn.setEnabled(False)
        self.stop_recording_btn.setEnabled(True)
        self.recording_status_label.setText(f"Recording status: active | saving to {self.recording_dir}")
        self._update_recording_labels()

    def finish_movement_recording(self, manual: bool = False) -> None:
        """Perform the finish movement recording operation used by the MouseControlWindow workflow."""
        if not self.recording_active and not self.recording_rows:
            return
        self.recording_active = False
        self.start_recording_btn.setEnabled(True)
        self.stop_recording_btn.setEnabled(False)
        saved_path = self._save_recording(manual=manual)
        self.recording_instruction_label.setText("Instruction: recording complete")
        self.recording_timer_label.setText("Done")
        analysis_path = saved_path.parent / "imu_function_analysis.json"
        suffix = f" | thresholds: {analysis_path}" if analysis_path.exists() else ""
        self.recording_status_label.setText(f"Recording status: saved {len(self.recording_rows)} rows to {saved_path}{suffix}")

    def _recording_tick(self) -> None:
        """Perform the recording tick operation used by the MouseControlWindow workflow."""
        if not self.recording_active:
            return
        now = time.time()
        if self.recording_stage_index >= len(MOUSE_DIAGNOSTIC_PROTOCOL):
            self.finish_movement_recording(manual=False)
            return
        stage = MOUSE_DIAGNOSTIC_PROTOCOL[self.recording_stage_index]
        elapsed = now - self.recording_stage_started_at
        if elapsed >= float(stage["duration_s"]):
            self.recording_stage_index += 1
            self.recording_stage_started_at = now
            if self.recording_stage_index >= len(MOUSE_DIAGNOSTIC_PROTOCOL):
                self.finish_movement_recording(manual=False)
                return
        self._update_recording_labels()

    def _update_recording_labels(self) -> None:
        """Refresh recording labels for the current MouseControlWindow workflow."""
        if self.recording_stage_index >= len(MOUSE_DIAGNOSTIC_PROTOCOL):
            return
        now = time.time()
        stage = MOUSE_DIAGNOSTIC_PROTOCOL[self.recording_stage_index]
        remaining = max(0.0, float(stage["duration_s"]) - (now - self.recording_stage_started_at))
        self.recording_instruction_label.setText(f"Instruction: {stage['instruction']}")
        self.recording_timer_label.setText(f"{remaining:04.1f} s")

    def _record_snapshots(self, snapshots: list[DeviceSnapshot]) -> None:
        """Record snapshots for the current MouseControlWindow workflow."""
        if self.recording_stage_index >= len(MOUSE_DIAGNOSTIC_PROTOCOL):
            return
        now = time.time()
        stage = MOUSE_DIAGNOSTIC_PROTOCOL[self.recording_stage_index]
        stable_angles = self._stable_angles()
        x_offset, y_offset = self._movement_offsets(stable_angles)
        for snap in snapshots:
            sensor_id = f"{snap.unit_id:08X}"
            q0, q1, q2, q3 = snap.quaternion
            self.recording_rows.append(
                {
                    "timestamp_s": now,
                    "elapsed_s": now - self.recording_started_at,
                    "stage_index": self.recording_stage_index,
                    "instruction": stage["instruction"],
                    "motion_label": stage.get("motion_label", "unknown"),
                    "stage_elapsed_s": now - self.recording_stage_started_at,
                    "sensor_id": sensor_id,
                    "is_reference_sensor": sensor_id == self.reference_sensor_id,
                    "gesture": self.current_gesture,
                    "confidence": self.current_confidence,
                    "axis_mode": self.axis_mode.currentData(),
                    "yaw": snap.yaw,
                    "pitch": snap.pitch,
                    "roll": snap.roll,
                    "stable_yaw": stable_angles[0],
                    "stable_pitch": stable_angles[1],
                    "stable_roll": stable_angles[2],
                    "x_offset": x_offset,
                    "y_offset": y_offset,
                    "q_w": q0,
                    "q_x": q1,
                    "q_y": q2,
                    "q_z": q3,
                    "rssi": snap.rssi,
                    "battery_mv": snap.battery_mv,
                    "age_ms": snap.age_ms,
                }
            )

    def _save_recording(self, manual: bool) -> Path:
        """Save recording for the current MouseControlWindow workflow."""
        if self.recording_dir is None:
            MOUSE_RECORDING_DIR.mkdir(parents=True, exist_ok=True)
            self.recording_dir = MOUSE_RECORDING_DIR / datetime.now().strftime("mouse_diag_%Y%m%d_%H%M%S")
            self.recording_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.recording_dir / "mouse_movement_recording.csv"
        meta_path = self.recording_dir / "mouse_movement_recording_metadata.json"
        fieldnames = [
            "timestamp_s",
            "elapsed_s",
            "stage_index",
            "instruction",
            "motion_label",
            "stage_elapsed_s",
            "sensor_id",
            "is_reference_sensor",
            "gesture",
            "confidence",
            "axis_mode",
            "yaw",
            "pitch",
            "roll",
            "stable_yaw",
            "stable_pitch",
            "stable_roll",
            "x_offset",
            "y_offset",
            "q_w",
            "q_x",
            "q_y",
            "q_z",
            "rssi",
            "battery_mv",
            "age_ms",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.recording_rows)
        metadata = {
            "manual_stop": manual,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "reference_sensor_id": self.reference_sensor_id,
            "reference_sensor_name": REFERENCE_SENSOR_NAME,
            "axis_mode_at_end": self.axis_mode.currentData(),
            "row_count": len(self.recording_rows),
            "protocol": MOUSE_DIAGNOSTIC_PROTOCOL,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            from analyze_mouse_movement_recording import load_recording, summarize

            analysis = summarize(load_recording(csv_path))
            (self.recording_dir / "imu_function_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        except Exception as exc:
            metadata["analysis_error"] = str(exc)
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return csv_path

    def _movement_offsets(self, angles: tuple[float, float, float], anchored: bool = False) -> tuple[float, float]:
        """Perform the movement offsets operation used by the MouseControlWindow workflow."""
        baseline = self.movement_anchor_angles if anchored and self.movement_anchor_angles is not None else self.neutral_angles
        if baseline is None:
            return 0.0, 0.0
        yaw = _angle_delta(angles[0], baseline[0])
        pitch = _angle_delta(angles[1], baseline[1])
        roll = _angle_delta(angles[2], baseline[2])
        mode = self.axis_mode.currentData()
        if mode == "yaw_pitch":
            return -yaw, -pitch
        if mode == "yaw_roll":
            return -yaw, -roll
        return roll, -pitch

    def _stable_angles(self) -> tuple[float, float, float]:
        """Perform the stable angles operation used by the MouseControlWindow workflow."""
        if not self.angle_history:
            return self._angles(self.latest_reference) if self.latest_reference is not None else (0.0, 0.0, 0.0)
        values = list(self.angle_history)
        yaw = self._circular_mean([row[0] for row in values])
        pitch = self._median([row[1] for row in values])
        roll = self._median([row[2] for row in values])
        return yaw, pitch, roll

    def _update_pointer_angles(self, raw: tuple[float, float, float]) -> None:
        """Track continuous, wrap-aware angles without median-filter stair steps."""
        if self.pointer_last_raw_angles is None or self.pointer_unwrapped_angles is None:
            self.pointer_last_raw_angles = raw
            self.pointer_unwrapped_angles = raw
            self.pointer_filtered_angles = raw
            return
        deltas = tuple(
            _angle_delta(current, last)
            for current, last in zip(raw, self.pointer_last_raw_angles)
        )
        # Ignore isolated orientation glitches; real hand movement cannot create
        # a jump this large in one incoming IMU update.
        deltas = tuple(delta if abs(delta) <= 900.0 else 0.0 for delta in deltas)
        unwrapped = tuple(
            previous + delta for previous, delta in zip(self.pointer_unwrapped_angles, deltas)
        )
        filtered = self.pointer_filtered_angles or unwrapped
        largest_delta = max(abs(delta) for delta in deltas)
        # Keep enough filtering for sensor noise without allowing the pointer
        # pose to keep catching up after the user's hand has stopped.
        alpha = 0.65 if largest_delta < 80.0 else 0.85
        self.pointer_filtered_angles = tuple(
            previous + alpha * (current - previous)
            for previous, current in zip(filtered, unwrapped)
        )
        self.pointer_unwrapped_angles = unwrapped
        self.pointer_last_raw_angles = raw

    def _pointer_angles(self) -> tuple[float, float, float]:
        """Perform the pointer angles operation used by the MouseControlWindow workflow."""
        return self.pointer_filtered_angles or self._latest_angles()

    def _latest_angles(self) -> tuple[float, float, float]:
        """Perform the latest angles operation used by the MouseControlWindow workflow."""
        return self._angles(self.latest_reference) if self.latest_reference is not None else (0.0, 0.0, 0.0)

    @staticmethod
    def _median(values: list[float]) -> float:
        """Perform the median operation used by the MouseControlWindow workflow."""
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    @staticmethod
    def _circular_mean(values: list[float]) -> float:
        """Perform the circular mean operation used by the MouseControlWindow workflow."""
        if not values:
            return 0.0
        radians = np.asarray(values, dtype=float) * (2.0 * math.pi / 8192.0)
        angle = math.atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean()))
        return (angle % (2.0 * math.pi)) * (8192.0 / (2.0 * math.pi))

    @staticmethod
    def _angles(snapshot: DeviceSnapshot) -> tuple[float, float, float]:
        """Perform the angles operation used by the MouseControlWindow workflow."""
        return snapshot.yaw, snapshot.pitch, snapshot.roll
