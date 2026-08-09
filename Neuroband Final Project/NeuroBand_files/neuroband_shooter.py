"""NeuroBand target-shooting game and model-selection launcher.

This end application demonstrates the complete acquisition and classification
pipeline. It loads a personal gesture model, receives live uMyo measurements,
uses IMU orientation for aiming, and maps gestures to shooting, reloading, taking
cover, and collecting bonuses. Game state, health, ammunition, targets, and safety
feedback are managed independently from the underlying classifier.

"""

from __future__ import annotations

import json
import math
import pickle
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets
from scipy import ndimage, signal

from classifier_adapter import GestureClassifierAdapter, PredictionResult
from mouse_game_control import MouseControlWindow
from realtime_gesture_gui import (
    DecisionSmoother,
    classify_noise_rms,
    configure_realtime_priority,
    keep_display_awake,
    load_latest_noise_profile,
    parse_decision_strategy,
    release_realtime_priority,
)
from recording_quality_gate import RealtimeSignalSafetyGate
from umyo_stream import DeviceSnapshot, UmyoSerialReader


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
PICTURES_DIR = PROJECT_ROOT / "pictures"
CALIBRATION_DIR = PROJECT_ROOT / "Data" / "calibration_sessions"
SENSOR_ORDER = ["B0DAC7E9", "ED7A78C8", "37ED348F"]
DISPLAY_GESTURES = [
    "at_rest",
    "fist",
    "like",
    "open_hand",
    "pinch",
    "pointing",
    "wrist_extension",
    "wrist_flexion",
]
DEFAULT_FS = 620.0


def install_application_font(app: QtWidgets.QApplication) -> None:
    """Perform the install application font operation used by the neuroband shooter workflow."""
    candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        font_id = QtGui.QFontDatabase.addApplicationFont(str(path))
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QtGui.QFont(families[0], 10))
            return


def load_cutout_pixmap(path: Path) -> QtGui.QPixmap:
    """Load and validate cutout pixmap for the current neuroband shooter workflow."""
    image = Image.open(path).convert("RGBA")
    pixels = np.asarray(image).copy()
    if np.min(pixels[..., 3]) == 255:
        rgb = pixels[..., :3].astype(np.int16)
        bright_neutral = (np.max(rgb, axis=2) - np.min(rgb, axis=2) < 24) & (np.min(rgb, axis=2) > 165)
        edge_seed = np.zeros(bright_neutral.shape, dtype=bool)
        edge_seed[0, :] = bright_neutral[0, :]
        edge_seed[-1, :] = bright_neutral[-1, :]
        edge_seed[:, 0] |= bright_neutral[:, 0]
        edge_seed[:, -1] |= bright_neutral[:, -1]
        connected_background = ndimage.binary_propagation(edge_seed, mask=bright_neutral)
        pixels[connected_background, 3] = 0
    height, width = pixels.shape[:2]
    qimage = QtGui.QImage(
        pixels.data,
        width,
        height,
        width * 4,
        QtGui.QImage.Format_RGBA8888,
    ).copy()
    return QtGui.QPixmap.fromImage(qimage)


@dataclass
class TargetEntity:
    """Represent the TargetEntity component and keep its related state and behavior together."""
    kind: str
    pixmap: QtGui.QPixmap
    rect: QtCore.QRectF
    deadline: float
    shots_fired: int = 0


class SensorWindowStore:
    """Represent the SensorWindowStore component and keep its related state and behavior together."""
    def __init__(self, capacity: int = 7000) -> None:
        """Initialize the SensorWindowStore instance and its runtime state."""
        self.capacity = capacity
        self.emg = {sensor_id: deque(maxlen=capacity) for sensor_id in SENSOR_ORDER}
        self.spectrum = {sensor_id: deque(maxlen=max(64, capacity // 8)) for sensor_id in SENSOR_ORDER}

    def append(self, snapshots: list[DeviceSnapshot]) -> None:
        """Perform the append operation used by the SensorWindowStore workflow."""
        for snapshot in snapshots:
            sensor_id = f"{snapshot.unit_id:08X}"
            if sensor_id not in self.emg:
                continue
            self.emg[sensor_id].extend(np.asarray(snapshot.emg, dtype=float).reshape(-1).tolist())
            spectra = np.asarray(snapshot.spectrum, dtype=float)
            if spectra.ndim == 1 and spectra.size:
                spectra = spectra.reshape(1, -1)
            for row in spectra:
                padded = np.pad(row[:4], (0, max(0, 4 - len(row))), mode="constant")
                self.spectrum[sensor_id].append(np.asarray(padded[:4], dtype=float))

    def ready(self, samples: int) -> bool:
        """Perform the ready operation used by the SensorWindowStore workflow."""
        return all(len(self.emg[sensor_id]) >= samples for sensor_id in SENSOR_ORDER)

    def classifier_window(self, samples: int) -> dict[str, dict[str, np.ndarray]]:
        """Perform the classifier window operation used by the SensorWindowStore workflow."""
        packets = max(1, int(math.ceil(samples / 8.0)))
        return {
            sensor_id: {
                "emg": np.asarray(list(self.emg[sensor_id])[-samples:], dtype=float),
                "spectrum": np.asarray(list(self.spectrum[sensor_id])[-packets:], dtype=float),
            }
            for sensor_id in SENSOR_ORDER
        }

    def noise_snapshot(self, fs: float, profile: dict[str, object]) -> dict[str, object]:
        """Perform the noise snapshot operation used by the SensorWindowStore workflow."""
        sensors_profile = profile.get("sensors", {}) if isinstance(profile, dict) else {}
        rows: list[dict[str, object]] = []
        worst = (0, "OK", "#22c55e")
        sample_count = max(96, int(fs * 0.6))
        for sensor_id in SENSOR_ORDER:
            values = np.asarray(list(self.emg[sensor_id])[-sample_count:], dtype=float)
            rms = self._filtered_rms(values, fs)
            limits = sensors_profile.get(sensor_id, {}) if isinstance(sensors_profile, dict) else {}
            label, color = classify_noise_rms(rms, limits)
            rank = {"OK": 0, "Elevated": 1, "High noise": 2}.get(label, 0)
            if rank >= worst[0]:
                worst = (rank, label, color)
            rows.append({"sensor_id": sensor_id, "rms": rms, "label": label, "color": color})
        return {"label": worst[1], "color": worst[2], "sensors": rows}

    @staticmethod
    def _filtered_rms(values: np.ndarray, fs: float) -> float:
        """Perform the filtered rms operation used by the SensorWindowStore workflow."""
        finite = values[np.isfinite(values)]
        if finite.size < 48:
            return 0.0
        finite = finite - float(np.median(finite))
        nyquist = fs / 2.0
        high = min(500.0, nyquist * 0.92)
        try:
            if high > 40.0:
                sos = signal.butter(4, [35.0, high], btype="bandpass", fs=fs, output="sos")
                finite = signal.sosfiltfilt(sos, finite)
            if nyquist > 55.0:
                b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs)
                finite = signal.filtfilt(b_notch, a_notch, finite)
        except ValueError:
            pass
        return float(np.sqrt(np.mean(np.square(finite))))


class ImageBackgroundWidget(QtWidgets.QWidget):
    """Represent the ImageBackgroundWidget component and keep its related state and behavior together."""
    def __init__(self, image_path: Path, parent=None) -> None:
        """Initialize the ImageBackgroundWidget instance and its runtime state."""
        super().__init__(parent)
        self.background = QtGui.QPixmap(str(image_path))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Render the widget using its current state."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        if not self.background.isNull():
            painter.drawPixmap(self.rect(), self.background)
        painter.fillRect(self.rect(), QtGui.QColor(4, 12, 22, 72))
        super().paintEvent(event)


class MainMenuPage(ImageBackgroundWidget):
    """Represent the MainMenuPage component and keep its related state and behavior together."""
    load_model_requested = QtCore.Signal(str)
    calibration_requested = QtCore.Signal()
    start_requested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        """Initialize the MainMenuPage instance and its runtime state."""
        super().__init__(PICTURES_DIR / "LOGO.png", parent)
        self._build_ui()
        self.refresh_models()

    def _build_ui(self) -> None:
        """Create and configure ui for the current MainMenuPage workflow."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(46, 38, 46, 38)
        layout.addStretch(3)
        panel = QtWidgets.QFrame()
        panel.setObjectName("menuPanel")
        panel.setMaximumWidth(540)
        panel.setMinimumWidth(430)
        panel.setStyleSheet(
            "QFrame#menuPanel {background: rgba(7, 17, 28, 225); border: 1px solid #365268;}"
            "QLabel {color: #eef6fb;}"
            "QComboBox {min-height: 38px; padding: 4px 10px; background: #f8fafc; color: #0f172a;}"
            "QPushButton {min-height: 42px; background: #e2e8f0; color: #0f172a; border: 1px solid #94a3b8; font-weight: 700;}"
            "QPushButton:hover {background: #ffffff;}"
            "QPushButton#startGame {background: #dc2626; color: white; border-color: #f87171; font-size: 18px;}"
            "QPushButton#startGame:hover {background: #ef4444;}"
        )
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(28, 28, 28, 28)
        panel_layout.setSpacing(14)

        title = QtWidgets.QLabel("NEUROBAND: STRIKE RESPONSE")
        title.setStyleSheet("font-size: 29px; font-weight: 800; color: white;")
        title.setWordWrap(True)
        panel_layout.addWidget(title)
        subtitle = QtWidgets.QLabel("Gesture-controlled target engagement")
        subtitle.setStyleSheet("font-size: 15px; color: #a8c5d8;")
        panel_layout.addWidget(subtitle)
        panel_layout.addSpacing(12)

        panel_layout.addWidget(QtWidgets.QLabel("Gesture model"))
        self.model_combo = QtWidgets.QComboBox()
        panel_layout.addWidget(self.model_combo)
        model_row = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_models)
        self.load_button = QtWidgets.QPushButton("Load selected model")
        self.load_button.clicked.connect(self._load_selected)
        model_row.addWidget(self.refresh_button)
        model_row.addWidget(self.load_button, stretch=2)
        panel_layout.addLayout(model_row)

        self.model_status = QtWidgets.QLabel("Model: none")
        self.model_status.setWordWrap(True)
        panel_layout.addWidget(self.model_status)
        self.sensor_status = QtWidgets.QLabel("Sensors: waiting for uMyo devices")
        self.sensor_status.setWordWrap(True)
        panel_layout.addWidget(self.sensor_status)
        self.noise_status = QtWidgets.QLabel("Signal noise: waiting")
        panel_layout.addWidget(self.noise_status)
        panel_layout.addStretch()

        self.calibration_button = QtWidgets.QPushButton("Open mouse control calibration")
        self.calibration_button.clicked.connect(self.calibration_requested.emit)
        panel_layout.addWidget(self.calibration_button)
        self.start_button = QtWidgets.QPushButton("START GAME")
        self.start_button.setObjectName("startGame")
        self.start_button.clicked.connect(self.start_requested.emit)
        panel_layout.addWidget(self.start_button)
        self.exit_button = QtWidgets.QPushButton("Exit")
        self.exit_button.clicked.connect(QtWidgets.QApplication.quit)
        panel_layout.addWidget(self.exit_button)
        layout.addWidget(panel, stretch=2)

    def refresh_models(self) -> None:
        """Refresh models for the current MainMenuPage workflow."""
        current = str(self.model_combo.currentData() or "")
        self.model_combo.clear()
        self.model_combo.addItem("Select a trained personal model", "")
        patterns = [
            "*/trained_model/personal_model.pkl",
            "*/trained_model/personal_fast_model.pkl",
            "*/trained_model/personal_model_update_*.pkl",
        ]
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(CALIBRATION_DIR.glob(pattern))
        unique_paths = sorted(set(paths), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in unique_paths:
            self.model_combo.addItem(self._model_label(path), str(path))
        if current:
            index = self.model_combo.findData(current)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

    @staticmethod
    def _model_label(path: Path) -> str:
        """Perform the model label operation used by the MainMenuPage workflow."""
        session = path.parent.parent.name
        if path.name.startswith("personal_model_update_"):
            try:
                with path.open("rb") as handle:
                    artifact = pickle.load(handle)
                adaptation = artifact.get("adaptation", {}) if isinstance(artifact, dict) else {}
                candidate = adaptation.get("candidate_validation", {})
                ba = candidate.get("balanced_accuracy")
                recommended = adaptation.get("recommended")
                status = "accepted" if recommended else "review"
                return f"{session} | update {status} | {artifact.get('model_type', 'model')} | BA {float(ba):.3f}" if ba is not None else f"{session} | update {status}"
            except Exception:
                return f"{session} | short update"
        summary_name = "personal_training_summary.json" if path.name == "personal_fast_model.pkl" else "personal_grid_summary.json"
        summary_path = path.parent / summary_name
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best = summary.get("best_result") or {}
            return (
                f"{session} | {best.get('model_type', 'model')} | {best.get('window_ms', '?')} ms | "
                f"test BA {float(best.get('test_balanced_accuracy', 0.0)):.3f}"
            )
        except Exception:
            return session

    def _load_selected(self) -> None:
        """Load and validate selected for the current MainMenuPage workflow."""
        path = str(self.model_combo.currentData() or "")
        if path:
            self.load_model_requested.emit(path)

    def set_model_status(self, text: str, ok: bool) -> None:
        """Set model status for the current MainMenuPage workflow."""
        self.model_status.setText(text)
        self.model_status.setStyleSheet(f"color: {'#86efac' if ok else '#fca5a5'}; font-weight: 700;")

    def set_stream_status(self, sensors: int, port: str, noise_label: str, noise_color: str) -> None:
        """Set stream status for the current MainMenuPage workflow."""
        self.sensor_status.setText(f"Sensors: {sensors}/3 | port {port}")
        self.sensor_status.setStyleSheet(f"color: {'#86efac' if sensors == 3 else '#fbbf24'};")
        self.noise_status.setText(f"Signal noise: {noise_label}")
        self.noise_status.setStyleSheet(f"color: {noise_color}; font-weight: 700;")


class PauseOverlay(QtWidgets.QFrame):
    """Represent the PauseOverlay component and keep its related state and behavior together."""
    resume_requested = QtCore.Signal()
    restart_requested = QtCore.Signal()
    menu_requested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        """Initialize the PauseOverlay instance and its runtime state."""
        super().__init__(parent)
        self.setObjectName("pauseOverlay")
        self.setStyleSheet(
            "QFrame#pauseOverlay {background: rgba(3, 10, 18, 235); border: 1px solid #64748b;}"
            "QLabel {color: white;}"
            "QPushButton {min-height: 42px; background: #e2e8f0; color: #111827; font-weight: 700;}"
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(34, 30, 34, 30)
        layout.setSpacing(12)
        self.title = QtWidgets.QLabel("PAUSED")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 34px; font-weight: 800;")
        layout.addWidget(self.title)
        self.detail = QtWidgets.QLabel("")
        self.detail.setAlignment(QtCore.Qt.AlignCenter)
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.resume_button = QtWidgets.QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume_requested.emit)
        layout.addWidget(self.resume_button)
        restart = QtWidgets.QPushButton("New game")
        restart.clicked.connect(self.restart_requested.emit)
        layout.addWidget(restart)
        menu = QtWidgets.QPushButton("Main menu")
        menu.clicked.connect(self.menu_requested.emit)
        layout.addWidget(menu)

    def show_pause(self) -> None:
        """Open or display pause for the current PauseOverlay workflow."""
        self.title.setText("PAUSED")
        self.detail.setText("Gesture input is frozen.")
        self.resume_button.setVisible(True)
        self.show()
        self.raise_()

    def show_game_over(self, score: int) -> None:
        """Open or display game over for the current PauseOverlay workflow."""
        self.title.setText("MISSION FAILED")
        self.detail.setText(f"Final score: {score}")
        self.resume_button.setVisible(False)
        self.show()
        self.raise_()


class ShooterGamePage(QtWidgets.QWidget):
    """Represent the ShooterGamePage component and keep its related state and behavior together."""
    menu_requested = QtCore.Signal()
    free_aim_changed = QtCore.Signal(bool)

    def __init__(self, parent=None) -> None:
        """Initialize the ShooterGamePage instance and its runtime state."""
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.background = QtGui.QPixmap(str(PICTURES_DIR / "background.jpg"))
        self.bullet = load_cutout_pixmap(PICTURES_DIR / "bullet.png")
        self.cover_crate = QtGui.QPixmap(str(PICTURES_DIR / "Box2.png"))
        self.hostile_images = self._load_images(["terrorist1.jpeg", "terrorist2.jpeg", "terrorist3.jpg", "terrorist4.png", "terrorist5.png", "terrorist6.png"])
        self.hostage_images = self._load_images(["hostage1.jpeg", "hostage2.png", "hostage3.png"])
        self.powerup_images = {
            "heart": load_cutout_pixmap(PICTURES_DIR / "heart.png"),
            "armor": load_cutout_pixmap(PICTURES_DIR / "armor.png"),
            "star": load_cutout_pixmap(PICTURES_DIR / "star.jpeg"),
        }
        self.random = random.Random()
        self.entities: list[TargetEntity] = []
        self.crosshair = QtCore.QPointF(0.5, 0.55)
        self.manual_mouse_until = 0.0
        self.lives = 5
        self.ammo = 5
        self.score = 0
        self.running = False
        self.paused = False
        self.paused_at = 0.0
        self.game_over = False
        self.reloading = False
        self.reload_until = 0.0
        self.next_spawn_at = 0.0
        self.next_powerup_at = 0.0
        self.shield_until = 0.0
        self.in_cover = False
        self.cover_progress = 0.0
        self.free_aim = False
        self.last_shot_at = 0.0
        self.last_command_gesture = "at_rest"
        self.gesture = "No model"
        self.gesture_confidence = 0.0
        self.noise_label = "Waiting"
        self.noise_color = QtGui.QColor("#fbbf24")
        self.signal_safe = False
        self.message = "Load a model and calibrate the pointer"
        self.message_until = 0.0
        self.damage_flash_until = 0.0

        self.pause_button = QtWidgets.QPushButton("Pause", self)
        self.pause_button.setFixedSize(92, 36)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setStyleSheet("background: rgba(15,23,42,220); color: white; font-weight: 700; border: 1px solid #94a3b8;")
        self.free_aim_button = QtWidgets.QPushButton("Free aim: OFF", self)
        self.free_aim_button.setCheckable(True)
        self.free_aim_button.setFixedSize(132, 36)
        self.free_aim_button.toggled.connect(self.set_free_aim)
        self.free_aim_button.setStyleSheet(
            "QPushButton {background: rgba(15,23,42,220); color: white; font-weight: 700; border: 1px solid #94a3b8;}"
            "QPushButton:checked {background: #0369a1; border-color: #38bdf8;}"
        )
        self.overlay = PauseOverlay(self)
        self.overlay.setFixedSize(390, 300)
        self.overlay.hide()
        self.overlay.resume_requested.connect(self.resume_game)
        self.overlay.restart_requested.connect(self.start_new_game)
        self.overlay.menu_requested.connect(self.menu_requested.emit)

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

    @staticmethod
    def _load_images(names: list[str]) -> list[QtGui.QPixmap]:
        """Load and validate images for the current ShooterGamePage workflow."""
        pixmaps = []
        for name in names:
            path = PICTURES_DIR / name
            if not path.exists():
                continue
            try:
                pixmap = load_cutout_pixmap(path)
            except Exception:
                pixmap = QtGui.QPixmap(str(path))
            if not pixmap.isNull():
                pixmaps.append(pixmap)
        return pixmaps

    def start_new_game(self) -> None:
        """Start new game for the current ShooterGamePage workflow."""
        self.lives = 5
        self.ammo = 5
        self.score = 0
        self.entities.clear()
        self.running = True
        self.paused = False
        self.paused_at = 0.0
        self.game_over = False
        self.reloading = False
        self.next_spawn_at = time.monotonic() + 0.7
        self.next_powerup_at = time.monotonic() + self.random.uniform(8.0, 12.0)
        self.shield_until = 0.0
        self.in_cover = False
        self.cover_progress = 0.0
        self.free_aim_button.setChecked(False)
        self.last_command_gesture = "at_rest"
        self.crosshair = QtCore.QPointF(0.5, 0.58)
        self.message = "Mission started"
        self.message_until = time.monotonic() + 1.3
        self.overlay.hide()
        self.pause_button.setText("Pause")
        self.setFocus()
        self.update()

    def stop_game(self) -> None:
        """Stop game for the current ShooterGamePage workflow."""
        self.running = False
        self.paused = False
        self.in_cover = False
        self.cover_progress = 0.0
        self.free_aim_button.setChecked(False)
        self.overlay.hide()

    def set_live_status(self, gesture: str, confidence: float, noise_label: str, noise_color: str, safe: bool) -> None:
        """Set live status for the current ShooterGamePage workflow."""
        self.gesture = gesture
        self.gesture_confidence = confidence
        self.noise_label = noise_label
        self.noise_color = QtGui.QColor(noise_color)
        self.signal_safe = safe
        if not self.running or self.paused or self.game_over:
            self.last_command_gesture = gesture
            self.update()
            return
        entering_cover = gesture == "wrist_flexion" and not self.in_cover
        leaving_cover = gesture != "wrist_flexion" and self.in_cover
        self.in_cover = gesture == "wrist_flexion"
        if entering_cover:
            self._show_message("Taking cover", 0.7)
        elif leaving_cover:
            self._show_message("Leaving cover", 0.5)
        if not safe:
            self.last_command_gesture = gesture
            self.update()
            return
        if gesture == "fist" and self.last_command_gesture != "fist":
            self.shoot()
        elif gesture == "wrist_extension" and self.last_command_gesture != "wrist_extension":
            self.reload_weapon()
        elif gesture == "pinch" and self.last_command_gesture != "pinch":
            self.collect_powerup()
        self.last_command_gesture = gesture
        self.update()

    def set_aim_from_demo(self, normalized_x: float, normalized_y: float) -> None:
        """Set aim from demo for the current ShooterGamePage workflow."""
        if time.monotonic() < self.manual_mouse_until:
            return
        self.crosshair = QtCore.QPointF(
            max(0.0, min(1.0, normalized_x)),
            max(0.0, min(1.0, normalized_y)),
        )
        self.update()

    def shoot(self) -> None:
        """Perform the shoot operation used by the ShooterGamePage workflow."""
        now = time.monotonic()
        if self.in_cover:
            self._show_message("Leave cover before firing", 0.7)
            return
        if not self.running or self.paused or self.game_over or self.reloading or now - self.last_shot_at < 0.30:
            return
        if self.ammo <= 0:
            self._show_message("Magazine empty - wrist extension to reload", 1.4)
            return
        self.last_shot_at = now
        self.ammo -= 1
        point = QtCore.QPointF(self.crosshair.x() * self.width(), self.crosshair.y() * self.height())
        hit: TargetEntity | None = None
        for entity in reversed(self.entities):
            if entity.rect.contains(point):
                hit = entity
                break
        if hit is None:
            self._show_message("Miss", 0.45)
        elif hit.kind == "hostile":
            remaining = max(0.0, hit.deadline - now)
            self.score += 100 + int(remaining * 25)
            self.entities.remove(hit)
            self.entities = [entity for entity in self.entities if entity.kind != "hostage"]
            self.next_spawn_at = now + self.random.uniform(0.45, 0.95)
            self._show_message("Threat neutralized", 0.65)
        else:
            self.entities.remove(hit)
            self._lose_life("Civilian hit")
        if self.ammo == 0 and not self.game_over:
            self._show_message("Magazine empty - wrist extension to reload", 1.4)
        self.update()

    def reload_weapon(self) -> None:
        """Perform the reload weapon operation used by the ShooterGamePage workflow."""
        if not self.running or self.paused or self.game_over or self.reloading or self.ammo == 5:
            return
        self.reloading = True
        self.reload_until = time.monotonic() + 1.2
        self._show_message("Reloading...", 1.2)

    def collect_powerup(self) -> None:
        """Perform the collect powerup operation used by the ShooterGamePage workflow."""
        if not self.running or self.paused or self.game_over or self.in_cover:
            return
        point = QtCore.QPointF(self.crosshair.x() * self.width(), self.crosshair.y() * self.height())
        powerup = next(
            (entity for entity in reversed(self.entities) if entity.kind in self.powerup_images and entity.rect.contains(point)),
            None,
        )
        if powerup is None:
            self._show_message("Pinch missed the power-up", 0.45)
            return
        self.entities.remove(powerup)
        if powerup.kind == "heart":
            if self.lives < 5:
                self.lives += 1
                self._show_message("Extra life", 0.9)
            else:
                self.score += 150
                self._show_message("Lives full: +150 points", 0.9)
        elif powerup.kind == "armor":
            self.shield_until = time.monotonic() + 30.0
            self._show_message("Shield active for 30 seconds", 1.1)
        else:
            self.score += 500
            self._show_message("Star collected: +500 points", 0.9)
        self.next_powerup_at = time.monotonic() + self.random.uniform(9.0, 15.0)

    def set_free_aim(self, enabled: bool) -> None:
        """Set free aim for the current ShooterGamePage workflow."""
        self.free_aim = bool(enabled)
        self.free_aim_button.setText(f"Free aim: {'ON' if enabled else 'OFF'}")
        self.free_aim_changed.emit(self.free_aim)
        self._show_message("Free aiming enabled" if enabled else "Gesture-gated aiming enabled", 0.8)

    def toggle_pause(self) -> None:
        """Perform the toggle pause operation used by the ShooterGamePage workflow."""
        if not self.running or self.game_over:
            return
        if self.paused:
            self.resume_game()
        else:
            self.paused = True
            self.paused_at = time.monotonic()
            self.pause_button.setText("Resume")
            self.overlay.show_pause()

    def resume_game(self) -> None:
        """Perform the resume game operation used by the ShooterGamePage workflow."""
        if not self.running or self.game_over:
            return
        pause_duration = max(0.0, time.monotonic() - self.paused_at)
        self.paused = False
        self.paused_at = 0.0
        self.pause_button.setText("Pause")
        self.overlay.hide()
        self.next_spawn_at += pause_duration
        if self.reloading:
            self.reload_until += pause_duration
        if self.message_until:
            self.message_until += pause_duration
        if self.damage_flash_until:
            self.damage_flash_until += pause_duration
        for entity in self.entities:
            if math.isfinite(entity.deadline):
                entity.deadline += pause_duration
        if self.shield_until:
            self.shield_until += pause_duration
        if self.next_powerup_at:
            self.next_powerup_at += pause_duration
        self.last_command_gesture = self.gesture
        self.setFocus()

    def _tick(self) -> None:
        """Perform the tick operation used by the ShooterGamePage workflow."""
        now = time.monotonic()
        target_cover = 1.0 if self.in_cover and self.running and not self.game_over else 0.0
        step = 0.075
        self.cover_progress += max(-step, min(step, target_cover - self.cover_progress))
        if not self.running or self.paused or self.game_over:
            self.update()
            return
        if self.reloading and now >= self.reload_until:
            self.reloading = False
            self.ammo = 5
            self._show_message("Reload complete", 0.7)
        hostile = next((entity for entity in self.entities if entity.kind == "hostile"), None)
        if hostile is not None and now >= hostile.deadline:
            self._resolve_enemy_shot(hostile, now)
        elif hostile is None and now >= self.next_spawn_at:
            self._spawn_wave(now)
        expired_powerups = [
            entity for entity in self.entities
            if entity.kind in self.powerup_images and now >= entity.deadline
        ]
        for entity in expired_powerups:
            self.entities.remove(entity)
        if not any(entity.kind in self.powerup_images for entity in self.entities) and now >= self.next_powerup_at:
            self._spawn_powerup(now)
        self.update()

    def _resolve_enemy_shot(self, hostile: TargetEntity, now: float) -> None:
        """Perform the resolve enemy shot operation used by the ShooterGamePage workflow."""
        protected = self.in_cover or now < self.shield_until
        if protected:
            protection = "Cover blocked enemy fire" if self.in_cover else "Shield absorbed enemy fire"
            self._show_message(protection, 0.9)
        else:
            self._lose_life("Enemy fire")
        if self.game_over:
            return
        if hostile.shots_fired == 0:
            hostile.shots_fired = 1
            hostile.deadline = now + 3.0
            self._show_message("Enemy fired - second shot incoming", 1.1)
        else:
            self.entities = [entity for entity in self.entities if entity.kind not in {"hostile", "hostage"}]
            self.next_spawn_at = now + self.random.uniform(0.55, 0.95)

    def _spawn_wave(self, now: float) -> None:
        """Perform the spawn wave operation used by the ShooterGamePage workflow."""
        self.entities = [entity for entity in self.entities if entity.kind in self.powerup_images]
        zone = self.random.choice(self._spawn_zones())
        self.entities.append(self._make_entity("hostile", self.random.choice(self.hostile_images), zone, now + 6.0))
        if self.hostage_images and self.random.random() < 0.38:
            hostage_zone = self.random.choice([candidate for candidate in self._spawn_zones() if candidate != zone])
            self.entities.append(self._make_entity("hostage", self.random.choice(self.hostage_images), hostage_zone, float("inf")))

    def _spawn_powerup(self, now: float) -> None:
        """Perform the spawn powerup operation used by the ShooterGamePage workflow."""
        kind = self.random.choice(list(self.powerup_images))
        pixmap = self.powerup_images[kind]
        size = self.height() * self.random.uniform(0.075, 0.105)
        ratio = pixmap.width() / max(1.0, float(pixmap.height()))
        width = size * ratio
        for _attempt in range(12):
            center = QtCore.QPointF(
                self.random.uniform(self.width() * 0.16, self.width() * 0.88),
                self.random.uniform(self.height() * 0.22, self.height() * 0.78),
            )
            rect = QtCore.QRectF(center.x() - width / 2.0, center.y() - size / 2.0, width, size)
            if not any(rect.intersects(entity.rect.adjusted(-18, -18, 18, 18)) for entity in self.entities):
                self.entities.append(TargetEntity(kind, pixmap, rect, now + 7.0))
                return
        self.next_powerup_at = now + 2.0

    @staticmethod
    def _spawn_zones() -> list[tuple[float, float, float]]:
        """Perform the spawn zones operation used by the ShooterGamePage workflow."""
        return [
            (0.08, 0.39, 0.19),
            (0.30, 0.48, 0.13),
            (0.43, 0.38, 0.12),
            (0.56, 0.35, 0.15),
            (0.70, 0.43, 0.18),
            (0.82, 0.52, 0.24),
        ]

    def _make_entity(self, kind: str, pixmap: QtGui.QPixmap, zone: tuple[float, float, float], deadline: float) -> TargetEntity:
        """Create and configure entity for the current ShooterGamePage workflow."""
        x, baseline_y, height_fraction = zone
        height = self.height() * height_fraction * self.random.uniform(0.90, 1.12)
        ratio = pixmap.width() / max(1.0, float(pixmap.height()))
        width = height * ratio
        center_x = self.width() * (x + self.random.uniform(-0.025, 0.025))
        bottom = self.height() * (baseline_y + self.random.uniform(-0.018, 0.018))
        rect = QtCore.QRectF(center_x - width / 2.0, bottom - height, width, height)
        return TargetEntity(kind, pixmap, rect, deadline)

    def _lose_life(self, reason: str) -> None:
        """Perform the lose life operation used by the ShooterGamePage workflow."""
        self.lives = max(0, self.lives - 1)
        self.damage_flash_until = time.monotonic() + 0.25
        self._show_message(f"{reason} - life lost", 1.1)
        if self.lives <= 0:
            self.game_over = True
            self.overlay.show_game_over(self.score)

    def _show_message(self, text: str, duration: float) -> None:
        """Open or display message for the current ShooterGamePage workflow."""
        self.message = text
        self.message_until = time.monotonic() + duration

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Render the widget using its current state."""
        painter = QtGui.QPainter(self)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        painter.drawPixmap(self.rect(), self.background)
        painter.fillRect(self.rect(), QtGui.QColor(3, 8, 14, 18))
        now = time.monotonic()
        for entity in self.entities:
            if entity.kind in self.powerup_images:
                glow = entity.rect.adjusted(-10, -10, 10, 10)
                painter.setBrush(QtGui.QColor(14, 165, 233, 80))
                painter.setPen(QtGui.QPen(QtGui.QColor("#7dd3fc"), 2))
                painter.drawEllipse(glow)
            painter.drawPixmap(entity.rect, entity.pixmap, QtCore.QRectF(entity.pixmap.rect()))
            if entity.kind == "hostile":
                remaining = max(0.0, entity.deadline - now)
                text_rect = QtCore.QRectF(entity.rect.left() - 12, entity.rect.top() - 42, entity.rect.width() + 24, 34)
                painter.setBrush(QtGui.QColor(127, 29, 29, 220))
                painter.setPen(QtGui.QPen(QtGui.QColor("#fecaca"), 1))
                painter.drawRect(text_rect)
                painter.setPen(QtGui.QColor("white"))
                painter.setFont(QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold))
                prefix = "SHOT 2  " if entity.shots_fired else ""
                painter.drawText(text_rect, QtCore.Qt.AlignCenter, f"{prefix}{remaining:0.1f}s")
            elif entity.kind == "hostage":
                painter.setPen(QtGui.QColor("#93c5fd"))
                painter.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
                painter.drawText(entity.rect.adjusted(0, -26, 0, 0), QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter, "CIVILIAN")
        self._draw_hud(painter, now)
        if self.cover_progress < 0.95:
            self._draw_crosshair(painter)
        if self.cover_progress > 0.0:
            self._draw_cover(painter)
        if now < self.damage_flash_until:
            painter.fillRect(self.rect(), QtGui.QColor(220, 38, 38, 90))

    def _draw_hud(self, painter: QtGui.QPainter, now: float) -> None:
        """Perform the draw hud operation used by the ShooterGamePage workflow."""
        hud = QtCore.QRectF(18, 16, self.width() - 36, 84)
        painter.setBrush(QtGui.QColor(3, 10, 18, 210))
        painter.setPen(QtGui.QPen(QtGui.QColor("#64748b"), 1))
        painter.drawRect(hud)
        painter.setFont(QtGui.QFont("Segoe UI", 13, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor("white"))
        painter.drawText(QtCore.QRectF(34, 30, 240, 28), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, f"SCORE  {self.score:05d}")
        self._draw_hearts(painter, 34, 67)
        self._draw_ammo(painter, self.width() - 370, 38)
        gesture_text = f"GESTURE  {self.gesture}  {self.gesture_confidence:.0%}"
        painter.drawText(QtCore.QRectF(self.width() * 0.35, 28, 370, 26), QtCore.Qt.AlignCenter, gesture_text)
        painter.setPen(self.noise_color)
        safety = "SAFE" if self.signal_safe else "BLOCKED"
        painter.drawText(QtCore.QRectF(self.width() * 0.35, 59, 370, 24), QtCore.Qt.AlignCenter, f"NOISE  {self.noise_label} | SIGNAL {safety}")
        if now < self.shield_until:
            painter.setPen(QtGui.QColor("#60a5fa"))
            painter.drawText(QtCore.QRectF(250, 61, 220, 24), QtCore.Qt.AlignLeft, f"SHIELD  {self.shield_until - now:0.0f}s")
        if now < self.message_until:
            message_rect = QtCore.QRectF(self.width() / 2.0 - 245, self.height() - 74, 490, 46)
            painter.setBrush(QtGui.QColor(3, 10, 18, 225))
            painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1"), 1))
            painter.drawRect(message_rect)
            painter.setPen(QtGui.QColor("white"))
            painter.setFont(QtGui.QFont("Segoe UI", 15, QtGui.QFont.Bold))
            painter.drawText(message_rect, QtCore.Qt.AlignCenter, self.message)

    def _draw_hearts(self, painter: QtGui.QPainter, x: float, y: float) -> None:
        """Perform the draw hearts operation used by the ShooterGamePage workflow."""
        for index in range(5):
            left = x + index * 30
            path = QtGui.QPainterPath()
            path.moveTo(left + 12, y + 18)
            path.cubicTo(left - 3, y + 7, left + 2, y - 3, left + 12, y + 5)
            path.cubicTo(left + 22, y - 3, left + 27, y + 7, left + 12, y + 18)
            painter.setBrush(QtGui.QColor("#ef4444") if index < self.lives else QtGui.QColor("#475569"))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawPath(path)

    def _draw_ammo(self, painter: QtGui.QPainter, x: float, y: float) -> None:
        """Perform the draw ammo operation used by the ShooterGamePage workflow."""
        painter.setPen(QtGui.QColor("white"))
        painter.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
        painter.drawText(QtCore.QRectF(x, y - 12, 100, 24), QtCore.Qt.AlignLeft, "AMMO")
        for index in range(5):
            rect = QtCore.QRectF(x + 72 + index * 34, y - 12, 26, 42)
            if index < self.ammo:
                painter.drawPixmap(rect, self.bullet, QtCore.QRectF(self.bullet.rect()))
            else:
                painter.setPen(QtGui.QPen(QtGui.QColor("#64748b"), 1))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawRect(rect)
        if self.reloading:
            painter.setPen(QtGui.QColor("#fbbf24"))
            painter.drawText(QtCore.QRectF(x, y + 30, 250, 24), QtCore.Qt.AlignLeft, "RELOADING")

    def _draw_crosshair(self, painter: QtGui.QPainter) -> None:
        """Perform the draw crosshair operation used by the ShooterGamePage workflow."""
        point = QtCore.QPointF(self.crosshair.x() * self.width(), self.crosshair.y() * self.height())
        color = QtGui.QColor("#22d3ee") if self.signal_safe else QtGui.QColor("#f59e0b")
        painter.setPen(QtGui.QPen(color, 3))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(point, 17, 17)
        painter.drawLine(point + QtCore.QPointF(-28, 0), point + QtCore.QPointF(-8, 0))
        painter.drawLine(point + QtCore.QPointF(8, 0), point + QtCore.QPointF(28, 0))
        painter.drawLine(point + QtCore.QPointF(0, -28), point + QtCore.QPointF(0, -8))
        painter.drawLine(point + QtCore.QPointF(0, 8), point + QtCore.QPointF(0, 28))

    def _draw_cover(self, painter: QtGui.QPainter) -> None:
        """Perform the draw cover operation used by the ShooterGamePage workflow."""
        top = self.height() * (1.0 - 0.80 * self.cover_progress)
        cover_rect = QtCore.QRectF(0, top, self.width(), self.height() - top)
        painter.fillRect(cover_rect, QtGui.QColor(42, 28, 19, 245))
        crate_width = self.width() / 3.0
        for index in range(3):
            rect = QtCore.QRectF(index * crate_width - 6, top - 4, crate_width + 12, self.height() - top + 8)
            painter.drawPixmap(rect, self.cover_crate, QtCore.QRectF(self.cover_crate.rect()))
        painter.setPen(QtGui.QColor("white"))
        painter.setFont(QtGui.QFont("Segoe UI", 22, QtGui.QFont.Bold))
        painter.drawText(QtCore.QRectF(0, top + 28, self.width(), 42), QtCore.Qt.AlignCenter, "IN COVER")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Update layout-dependent state after the widget is resized."""
        self.pause_button.move(self.width() - self.pause_button.width() - 20, 112)
        self.free_aim_button.move(self.width() - self.free_aim_button.width() - 124, 112)
        self.overlay.move((self.width() - self.overlay.width()) // 2, (self.height() - self.overlay.height()) // 2)
        super().resizeEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Perform the mouseMoveEvent operation used by the ShooterGamePage workflow."""
        self.manual_mouse_until = time.monotonic() + 0.45
        self.crosshair = QtCore.QPointF(event.position().x() / max(1, self.width()), event.position().y() / max(1, self.height()))
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle a mouse-button press delivered to the widget."""
        if event.button() == QtCore.Qt.LeftButton:
            self.shoot()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle a keyboard command delivered to the widget."""
        if event.key() == QtCore.Qt.Key_Escape:
            self.toggle_pause()
        elif event.key() == QtCore.Qt.Key_P:
            self.toggle_pause()
        elif event.key() == QtCore.Qt.Key_R:
            self.reload_weapon()
        elif event.key() == QtCore.Qt.Key_Space:
            self.free_aim_button.toggle()
        else:
            super().keyPressEvent(event)


class NeuroBandShooter(QtWidgets.QMainWindow):
    """Represent the NeuroBandShooter component and keep its related state and behavior together."""
    def __init__(self, config: dict[str, object] | None = None) -> None:
        """Initialize the NeuroBandShooter instance and its runtime state."""
        super().__init__()
        self.config = config or {}
        self.setWindowTitle("NeuroBand: Strike Response")
        self.resize(1500, 850)
        self.setMinimumSize(1080, 680)
        self.fs = float(self.config.get("sampling_rate_hz", DEFAULT_FS))
        self.reader = UmyoSerialReader(baudrate=int(self.config.get("baudrate", 921600)))
        self.classifier = GestureClassifierAdapter(
            fs=self.fs,
            selected_channels=list(range(3)),
            confidence_threshold=float(self.config.get("confidence_threshold", 0.55)),
        )
        self.buffers = SensorWindowStore()
        self.smoother = DecisionSmoother()
        self.safety_gate = RealtimeSignalSafetyGate(SENSOR_ORDER)
        self.signal_safety = {"safe": False, "reason": "waiting for sensors", "changed": False}
        self.noise_profile = load_latest_noise_profile()
        self.noise_snapshot = {"label": "Waiting", "color": "#fbbf24", "sensors": []}
        self.latest_snapshots: list[DeviceSnapshot] = []
        self.mouse_window: MouseControlWindow | None = None
        self.last_prediction_at = 0.0
        self.last_noise_at = 0.0
        self.last_safety_at = 0.0
        self.last_stable_gesture = "No model"
        self.last_raw_gesture = "No model"
        self.last_confidence = 0.0
        self.last_raw_confidence = 0.0

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)
        self.menu = MainMenuPage()
        self.game = ShooterGamePage()
        self.stack.addWidget(self.menu)
        self.stack.addWidget(self.game)
        self.menu.load_model_requested.connect(self.load_model)
        self.menu.calibration_requested.connect(self.open_mouse_calibration)
        self.menu.start_requested.connect(self.start_game)
        self.game.menu_requested.connect(self.show_menu)
        self.game.free_aim_changed.connect(self.set_free_aim)

        self.reader.start()
        keep_display_awake(True)
        configure_realtime_priority()
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_realtime)
        self.timer.start(8)

    def load_model(self, path: str) -> None:
        """Load and validate model for the current NeuroBandShooter workflow."""
        if not self.classifier.load(path):
            self.menu.set_model_status(f"Model load failed: {self.classifier.last_error}", False)
            return
        strategy = self._decision_strategy_for_model(Path(path))
        self.smoother.configure(parse_decision_strategy(strategy))
        self.fs = self.classifier.sampling_rate_hz(self.fs)
        self.menu.set_model_status(
            f"Model: {Path(path).parent.parent.name} | {self.classifier.window_ms()} ms | decision {strategy}",
            True,
        )

    @staticmethod
    def _decision_strategy_for_model(model_path: Path) -> str:
        """Perform the decision strategy for model operation used by the NeuroBandShooter workflow."""
        replay_root = model_path.parent.parent / "replay_report"
        summaries = sorted(
            replay_root.glob("*/decision_strategy_comparison/decision_strategy_summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if replay_root.exists() else []
        for summary_path in summaries:
            try:
                best = (json.loads(summary_path.read_text(encoding="utf-8")).get("best_strategy") or {})
                if best.get("strategy"):
                    return str(best["strategy"])
            except Exception:
                continue
        return "raw_no_gate"

    def _ensure_mouse_window(self) -> MouseControlWindow:
        """Perform the ensure mouse window operation used by the NeuroBandShooter workflow."""
        if self.mouse_window is None:
            self.mouse_window = MouseControlWindow(self)
            self.mouse_window.enable_os_control.setChecked(False)
            self.mouse_window.enable_imu_combos.setChecked(False)
            self.mouse_window.enable_imu_combos.setEnabled(False)
            self.mouse_window.imu_movement_toggle = False
            self.mouse_window.space_emergency_enabled = False
        return self.mouse_window

    def set_free_aim(self, enabled: bool) -> None:
        """Set free aim for the current NeuroBandShooter workflow."""
        mouse = self._ensure_mouse_window()
        mouse.imu_movement_toggle = False
        mouse.ignore_gesture_gate.setChecked(bool(enabled))

    def open_mouse_calibration(self) -> None:
        """Open or display mouse calibration for the current NeuroBandShooter workflow."""
        window = self._ensure_mouse_window()
        window.show()
        window.raise_()
        window.activateWindow()

    def start_game(self) -> None:
        """Start game for the current NeuroBandShooter workflow."""
        if not self.classifier.is_loaded:
            QtWidgets.QMessageBox.warning(self, "Model required", "Select and load a trained gesture model before starting.")
            return
        mouse = self._ensure_mouse_window()
        mouse.enable_os_control.setChecked(False)
        mouse.enable_control.setChecked(True)
        mouse.enable_imu_combos.setChecked(False)
        mouse.imu_movement_toggle = False
        if self.latest_snapshots:
            mouse.receive_snapshots(self.latest_snapshots)
        if mouse.latest_reference is not None and mouse.neutral_angles is None:
            mouse.calibrate_neutral_pose()
        mouse.hide()
        self.stack.setCurrentWidget(self.game)
        self.game.start_new_game()

    def show_menu(self) -> None:
        """Open or display menu for the current NeuroBandShooter workflow."""
        self.game.stop_game()
        if self.mouse_window is not None:
            self.mouse_window.enable_control.setChecked(False)
        self.stack.setCurrentWidget(self.menu)
        self.menu.refresh_models()

    def update_realtime(self) -> None:
        """Refresh realtime for the current NeuroBandShooter workflow."""
        now = time.monotonic()
        snapshots = self.reader.snapshots()
        status = self.reader.status()
        if snapshots:
            self.latest_snapshots = snapshots
            self.buffers.append(snapshots)
            if self.mouse_window is not None:
                self.mouse_window.receive_snapshots(snapshots)
            if now - self.last_noise_at >= 0.20:
                self.noise_snapshot = self.buffers.noise_snapshot(self.fs, self.noise_profile)
                self.last_noise_at = now
            if now - self.last_safety_at >= 0.05:
                rest_expected = self.last_stable_gesture == "at_rest" and self.last_raw_gesture == "at_rest"
                self.signal_safety = self.safety_gate.update(snapshots, self.noise_snapshot, rest_expected)
                if self.signal_safety.get("changed"):
                    self.smoother.reset()
                self.last_safety_at = now
            self._predict_if_due(now)

        sensors = len({f"{snapshot.unit_id:08X}" for snapshot in snapshots if f"{snapshot.unit_id:08X}" in SENSOR_ORDER and snapshot.age_ms < 1000})
        self.menu.set_stream_status(
            sensors,
            str(status.get("port", "auto")),
            str(self.noise_snapshot.get("label", "Waiting")),
            str(self.noise_snapshot.get("color", "#fbbf24")),
        )
        if self.mouse_window is not None:
            self.mouse_window.set_signal_safety(bool(self.signal_safety.get("safe", False)), str(self.signal_safety.get("reason", "waiting")))
            self.mouse_window.update_control(
                self.last_stable_gesture,
                self.last_confidence,
                self.last_raw_gesture,
                self.last_raw_confidence,
            )
            self._update_game_aim_from_mouse_demo()
        self.game.set_live_status(
            self.last_stable_gesture,
            self.last_confidence,
            str(self.noise_snapshot.get("label", "Waiting")),
            str(self.noise_snapshot.get("color", "#fbbf24")),
            bool(self.signal_safety.get("safe", False)),
        )

    def _predict_if_due(self, now: float) -> None:
        """Predict if due for the current NeuroBandShooter workflow."""
        if not self.classifier.is_loaded:
            return
        interval = float(self.config.get("prediction_interval_ms", 120.0)) / 1000.0
        if now - self.last_prediction_at < interval:
            return
        self.last_prediction_at = now
        if not self.signal_safety.get("safe", False):
            self.last_raw_gesture = "Uncertain"
            self.last_stable_gesture = "Uncertain"
            self.last_raw_confidence = 0.0
            self.last_confidence = 0.0
            return
        samples = max(8, int(self.classifier.sampling_rate_hz(self.fs) * self.classifier.window_ms() / 1000.0))
        if not self.buffers.ready(samples):
            return
        result = self.classifier.predict(self.buffers.classifier_window(samples))
        if result.error:
            self.last_raw_gesture = "Error"
            self.last_stable_gesture = "Error"
            self.last_raw_confidence = 0.0
            self.last_confidence = 0.0
            self.menu.set_model_status(f"Prediction error: {result.error}", False)
            return
        self.last_raw_gesture = result.gesture
        self.last_raw_confidence = float(result.confidence)
        stable, _uncertain = self.smoother.apply(result.gesture, result.confidence)
        self.last_stable_gesture = stable
        self.last_confidence = float(result.confidence)

    def _update_game_aim_from_mouse_demo(self) -> None:
        """Refresh game aim from mouse demo for the current NeuroBandShooter workflow."""
        if self.stack.currentWidget() is not self.game or self.mouse_window is None:
            return
        demo = self.mouse_window.game
        rect = demo._play_rect()
        if rect.width() <= 1 or rect.height() <= 1:
            return
        point = demo.state.cursor
        self.game.set_aim_from_demo(
            (point.x() - rect.left()) / rect.width(),
            (point.y() - rect.top()) / rect.height(),
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window closure and release application resources safely."""
        self.timer.stop()
        self.reader.stop()
        self.reader.join(1.0)
        if self.mouse_window is not None:
            self.mouse_window.windows_cursor_worker.stop()
            self.mouse_window.close()
        keep_display_awake(False)
        release_realtime_priority()
        super().closeEvent(event)


def load_config() -> dict[str, object]:
    """Load and validate config for the current neuroband shooter workflow."""
    path = APP_DIR / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def main() -> int:
    """Run the module's command-line or graphical application entry point."""
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    install_application_font(app)
    window = NeuroBandShooter(load_config())
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
