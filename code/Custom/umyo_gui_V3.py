import sys
import time
import threading
import math
import os
import csv
import json
from datetime import datetime

import numpy as np
import serial
from serial.tools import list_ports

import umyo_parser  # Must be in the same folder

# --- GUI and Plotting Libraries ---
from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# --- Signal Processing Library ---
from scipy import signal


# ---------- Serial Reader Thread ----------

class SerialReader(threading.Thread):
    def __init__(self, baudrate=921600):
        super().__init__(daemon=True)
        self.baudrate = baudrate
        self.running = True
        self.ser = None

    def find_port(self):
        ports = list(list_ports.comports())
        for p in ports:
            if "usbserial" in p.device or "COM4" in p.device or 'FTDI' in p.description or 'USB' in p.device:
                return p.device
        return None

    def open_serial(self):
        port = self.find_port()
        if port is None:
            print("No uMyo serial port found!")
            return False

        try:
            temp_ser = serial.Serial(port, timeout=1)
            temp_ser.close()
            time.sleep(0.5)

            self.ser = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=1,
                bytesize=8,
                timeout=0,
            )
            print(f"Opened serial on {port}")
            return True
        except serial.SerialException as e:
            print(f"Error opening serial port {port}: {e}")
            return False

    def run(self):
        if not self.open_serial():
            return
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    umyo_parser.umyo_parse_preprocessor(data)
                time.sleep(0.001)
            except Exception as e:
                print("Serial error:", e)
                break
        if self.ser is not None:
            self.ser.close()
            print("Serial closed")

    def stop(self):
        self.running = False


# ---------- Filter Settings Panel ----------

class FilterPanel(QtWidgets.QGroupBox):
    filter_changed = QtCore.Signal()  # Signal to notify DevicePanel to re-init filter

    def __init__(self, parent=None):
        super().__init__("Real-time Filter Settings", parent)
        self.FS = 1100.0  # EMG Sampling Rate in Hz
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Filter Type
        layout.addWidget(QtWidgets.QLabel("Filter Type:"))
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItem("None", userData="none")
        self.type_combo.addItem("High-Pass (HPF)", userData="hp")
        self.type_combo.addItem("Low-Pass (LPF)", userData="lp")
        self.type_combo.addItem("Band-Stop (BSF)", userData="bs")

        self.type_combo.setCurrentText("None")

        self.type_combo.currentTextChanged.connect(self._on_settings_changed)
        layout.addWidget(self.type_combo)

        # Filter Order
        layout.addWidget(QtWidgets.QLabel("Order (N):"))
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(4)
        self.order_spin.valueChanged.connect(self._on_settings_changed)
        layout.addWidget(self.order_spin)

        # Cutoff/Center Frequency (Wn)
        layout.addWidget(QtWidgets.QLabel("Cutoff Freq (Hz):"))
        self.center_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.center_cutoff_spin.setRange(0.1, self.FS / 2.0 - 1)
        self.center_cutoff_spin.setSingleStep(1.0)
        self.center_cutoff_spin.setValue(50.0)
        self.center_cutoff_spin.valueChanged.connect(self._on_settings_changed)
        layout.addWidget(self.center_cutoff_spin)

        # Band-Stop Delta Frequencies
        self.bs_label = QtWidgets.QLabel(r"Band-Stop Deltas (Left $\Delta$, Right $\Delta$ in Hz):")
        layout.addWidget(self.bs_label)

        self.bs_container = QtWidgets.QWidget()
        band_layout = QtWidgets.QHBoxLayout(self.bs_container)

        self.bs_left_delta_spin = QtWidgets.QDoubleSpinBox()
        self.bs_left_delta_spin.setRange(0.1, self.FS / 2.0 - 1)
        self.bs_left_delta_spin.setSingleStep(0.5)
        self.bs_left_delta_spin.setValue(5.0)
        self.bs_left_delta_spin.valueChanged.connect(self._on_settings_changed)
        band_layout.addWidget(self.bs_left_delta_spin)

        self.bs_right_delta_spin = QtWidgets.QDoubleSpinBox()
        self.bs_right_delta_spin.setRange(0.1, self.FS / 2.0 - 1)
        self.bs_right_delta_spin.setSingleStep(0.5)
        self.bs_right_delta_spin.setValue(5.0)
        self.bs_right_delta_spin.valueChanged.connect(self._on_settings_changed)
        band_layout.addWidget(self.bs_right_delta_spin)

        layout.addWidget(self.bs_container)

        self._update_bs_visibility(self.type_combo.currentData() == "bs")
        layout.addStretch()

    def _update_bs_visibility(self, is_bs: bool):
        if hasattr(self, 'bs_label'):
            self.bs_label.setVisible(is_bs)
        if hasattr(self, 'bs_container'):
            self.bs_container.setVisible(is_bs)

        label_text = "Center Freq (Hz):" if is_bs else "Cutoff Freq (Hz):"
        try:
            layout_obj = self.center_cutoff_spin.parent().layout()
            label = layout_obj.itemAt(
                layout_obj.indexOf(self.center_cutoff_spin) - 1
            ).widget()
            if isinstance(label, QtWidgets.QLabel):
                label.setText(label_text)
        except:
            pass

    def _on_settings_changed(self):
        self._update_bs_visibility(self.type_combo.currentData() == "bs")
        self.filter_changed.emit()

    def get_filter_params(self):
        """Returns (type, order, Wn/Wn_band)"""
        f_type = self.type_combo.currentData()
        order = self.order_spin.value()
        MIN_BANDWIDTH_HZ = 0.5

        if f_type in ["hp", "lp"]:
            cutoff = self.center_cutoff_spin.value()
            if cutoff <= 0.1 or cutoff >= self.FS / 2.0 - 0.1:
                return "none", 0, 0.0  # Fail safe
            return f_type, order, cutoff

        elif f_type == "bs":
            center_freq = self.center_cutoff_spin.value()
            left_delta = self.bs_left_delta_spin.value()
            right_delta = self.bs_right_delta_spin.value()

            low = max(0.1, center_freq - left_delta)
            high = min(self.FS / 2.0 - 0.1, center_freq + right_delta)

            if low >= high or (high - low) < MIN_BANDWIDTH_HZ:
                # Invalid BSF, fallback to none to prevent crash
                return "none", 0, 0.0

            w_band = [low, high]
            return f_type, order, w_band
        else:  # "none"
            return "none", 0, 0.0


# ---------- Device Display Panel ----------

class DevicePanel(QtWidgets.QWidget):
    filter_panel: FilterPanel = None

    def __init__(self, index: int, emg_len=2000, acc_len=200, parent=None):
        super().__init__(parent)

        self.index = index
        self.emg_len = emg_len
        self.acc_len = acc_len
        self.sp_len = 500  # History length for spectrum graph

        # Buffers
        self.emg_buffer = np.zeros(self.emg_len, dtype=float)
        self.acc_x = np.zeros(self.acc_len, dtype=float)
        self.acc_y = np.zeros(self.acc_len, dtype=float)
        self.acc_z = np.zeros(self.acc_len, dtype=float)

        # Spectrum buffers (4 bins)
        self.sp_hist = np.zeros((4, self.sp_len), dtype=float)

        # Filter State
        self.FS = 1100.0
        self.b, self.a = None, None
        self.zi = None
        self.filtered_emg_buffer = np.zeros(self.emg_len, dtype=float)

        if DevicePanel.filter_panel is not None:
            DevicePanel.filter_panel.filter_changed.connect(self._init_filter)
            self._init_filter()

        self._build_ui()

    def _init_filter(self):
        if DevicePanel.filter_panel is None:
            self.b, self.a, self.zi = None, None, None
            return

        f_type, order, w_freq = DevicePanel.filter_panel.get_filter_params()

        if f_type == "none":
            self.b, self.a, self.zi = None, None, None
            return

        if f_type in ["hp", "lp"]:
            Wn = w_freq / (self.FS / 2.0)
        elif f_type == "bs":
            Wn = np.array(w_freq) / (self.FS / 2.0)
        else:
            self.b, self.a, self.zi = None, None, None
            return

        try:
            self.b, self.a = signal.butter(order, Wn, f_type, analog=False)
            if len(self.a) > 1:
                self.zi = np.zeros(len(self.a) - 1, dtype=float)
            else:
                self.zi = None
        except Exception as e:
            print(f"Device {self.index}: Filter Error: {e}")
            self.b, self.a, self.zi = None, None, None

        self.filtered_emg_buffer[:] = 0.0
        if hasattr(self, 'emg_curve'):
            self.emg_curve.setData(self.filtered_emg_buffer)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(f"Device {self.index}")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # EMG - Static Axis
        self.emg_plot = pg.PlotWidget(title="EMG data")
        self.emg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.emg_plot.setYRange(-3000, 3000)
        self.emg_curve = self.emg_plot.plot(pen='g')
        layout.addWidget(self.emg_plot)

        # Spectrum - Time Domain (4 Lines) [UPDATED]
        self.spg_plot = pg.PlotWidget(title="Spectrum Bins (Time Domain)")
        self.spg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spg_plot.setYRange(0, 1)  # Normalized
        self.spg_plot.addLegend()
        # Colors: Green, Yellow, Magenta, Red
        self.spg_curves = []
        colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 0, 0)]
        names = ["sp0", "sp1", "sp2", "sp3"]
        for i in range(4):
            c = self.spg_plot.plot(pen=colors[i], name=names[i])
            self.spg_curves.append(c)
        layout.addWidget(self.spg_plot)

        # Accel - Static Axis
        self.acc_plot = pg.PlotWidget(title="Accelerometer")
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        self.acc_plot.setYRange(-20000, 20000)
        self.acc_curve_x = self.acc_plot.plot(pen='r', name='ax')
        self.acc_curve_y = self.acc_plot.plot(pen='y', name='ay')
        self.acc_curve_z = self.acc_plot.plot(pen='b', name='az')
        layout.addWidget(self.acc_plot)

        # Info
        info_layout = QtWidgets.QVBoxLayout()
        self.id_label = QtWidgets.QLabel("ID: N/A")
        info_layout.addWidget(self.id_label)

        rssi_layout = QtWidgets.QHBoxLayout()
        rssi_layout.addWidget(QtWidgets.QLabel("RSSI:"))
        self.rssi_bar = QtWidgets.QProgressBar()
        self.rssi_bar.setRange(0, 100)
        rssi_layout.addWidget(self.rssi_bar)
        info_layout.addLayout(rssi_layout)

        batt_layout = QtWidgets.QHBoxLayout()
        batt_layout.addWidget(QtWidgets.QLabel("Battery:"))
        self.batt_bar = QtWidgets.QProgressBar()
        self.batt_bar.setRange(0, 100)
        batt_layout.addWidget(self.batt_bar)
        info_layout.addLayout(batt_layout)

        self.compass_label = QtWidgets.QLabel("Compass: N/A")
        info_layout.addWidget(self.compass_label)
        layout.addLayout(info_layout)
        layout.addStretch()

    def update_from_device(self, dev):
        if hasattr(dev, "unit_id"):
            self.id_label.setText(f"ID: {dev.unit_id:08X}")

        # Update EMG
        if hasattr(dev, "data_array") and hasattr(dev, "data_count"):
            samples = dev.data_array[:dev.data_count]
            if len(samples) > 0:
                shift = len(samples)
                # Raw buffer
                if shift >= self.emg_len:
                    self.emg_buffer[:] = samples[-self.emg_len:]
                else:
                    self.emg_buffer = np.roll(self.emg_buffer, -shift)
                    self.emg_buffer[-shift:] = samples

                # Filtered
                y = samples
                if self.b is not None:
                    y, self.zi = signal.lfilter(self.b, self.a, samples, zi=self.zi)

                if shift >= self.emg_len:
                    self.filtered_emg_buffer[:] = y[-self.emg_len:]
                else:
                    self.filtered_emg_buffer = np.roll(self.filtered_emg_buffer, -shift)
                    self.filtered_emg_buffer[-shift:] = y

                self.emg_curve.setData(self.filtered_emg_buffer)

        # Update Spectrum (Local calc on Filtered data, displayed as Time Domain)
        try:
            current_vals = [0.0] * 4
            if len(self.filtered_emg_buffer) > 256:
                f, Pxx = signal.welch(self.filtered_emg_buffer, fs=self.FS, nperseg=256, scaling='spectrum')
                bins_hz = [0, 100, 200, 300, self.FS / 2]
                binned = []
                for i in range(4):
                    indices = np.where((f >= bins_hz[i]) & (f < bins_hz[i + 1]))
                    binned.append(np.sum(Pxx[indices]))

                vals = np.array(binned, dtype=float)
                max_v = np.max(vals)
                if max_v > 1e-9:
                    current_vals = vals / max_v  # Normalize

            # Roll and update history for 4 channels
            self.sp_hist = np.roll(self.sp_hist, -1, axis=1)
            self.sp_hist[:, -1] = current_vals

            for i in range(4):
                self.spg_curves[i].setData(self.sp_hist[i])

        except:
            pass

        # Update Accel
        for buf, attr in zip((self.acc_x, self.acc_y, self.acc_z), ("ax", "ay", "az")):
            val = getattr(dev, attr, None)
            if val is not None:
                buf[:] = np.roll(buf, -1);
                buf[-1] = val
        self.acc_curve_x.setData(self.acc_x)
        self.acc_curve_y.setData(self.acc_y)
        self.acc_curve_z.setData(self.acc_z)

        # Update Info
        rssi = getattr(dev, "rssi", 0)
        if rssi > 0: self.rssi_bar.setValue(int(max(0, min(100, (90 - rssi) * 1.6))))

        batt = getattr(dev, "batt", 0)
        if batt > 0: self.batt_bar.setValue(int(max(0, min(100, (batt - 3100) / 10.0))))

        mag = getattr(dev, "mag_angle", None)
        if mag is not None:
            self.compass_label.setText(f"Compass: {(math.degrees(mag) + 360) % 360:.1f}°")

    def clear_panel(self):
        self.id_label.setText("ID: N/A")


# ---------- Gesture Dialogs ----------

class GestureDialog(QtWidgets.QDialog):
    """Dialog to Add a new gesture."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("הוספת מחווה חדשה")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("שם תצוגה (עברית):"))
        self.display_name = QtWidgets.QLineEdit()
        layout.addWidget(self.display_name)
        layout.addWidget(QtWidgets.QLabel("שם מפתח (אנגלית):"))
        self.key_name = QtWidgets.QLineEdit()
        layout.addWidget(self.key_name)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        return self.display_name.text().strip(), self.key_name.text().strip().lower().replace(' ', '_')


class GestureManagerDialog(QtWidgets.QDialog):
    """Dialog to Edit/Delete existing gestures."""

    def __init__(self, gestures, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ניהול רשימת מחוות")
        self.resize(400, 300)
        self.gestures = gestures  # List of dicts
        self.init_ui()

    def init_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QtWidgets.QWidget()
        self.list_layout = QtWidgets.QVBoxLayout(self.container)

        self.refresh_list()

        self.scroll.setWidget(self.container)
        self.main_layout.addWidget(self.scroll)

        close_btn = QtWidgets.QPushButton("סגור")
        close_btn.clicked.connect(self.accept)
        self.main_layout.addWidget(close_btn)

    def refresh_list(self):
        # Clear
        while self.list_layout.count():
            child = self.list_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        for idx, g in enumerate(self.gestures):
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)

            lbl = QtWidgets.QLabel(f"{g['display']} ({g['key']})")
            edit_btn = QtWidgets.QPushButton("ערוך")
            del_btn = QtWidgets.QPushButton("מחק")
            del_btn.setStyleSheet("background-color: #ff4d4d; color: white;")

            # Using closures to capture index
            edit_btn.clicked.connect(lambda checked, i=idx: self.edit_gesture(i))
            del_btn.clicked.connect(lambda checked, i=idx: self.delete_gesture(i))

            row_layout.addWidget(lbl)
            row_layout.addStretch()
            row_layout.addWidget(edit_btn)
            row_layout.addWidget(del_btn)
            self.list_layout.addWidget(row)
        self.list_layout.addStretch()

    def edit_gesture(self, idx):
        g = self.gestures[idx]
        d, ok1 = QtWidgets.QInputDialog.getText(self, "עריכה", "שם תצוגה:", text=g['display'])
        if ok1 and d:
            k, ok2 = QtWidgets.QInputDialog.getText(self, "עריכה", "שם מפתח:", text=g['key'])
            if ok2 and k:
                self.gestures[idx] = {'display': d, 'key': k}
                self.refresh_list()

    def delete_gesture(self, idx):
        ret = QtWidgets.QMessageBox.question(self, "מחיקה", "למחוק מחווה זו?",
                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if ret == QtWidgets.QMessageBox.Yes:
            self.gestures.pop(idx)
            self.refresh_list()


# ---------- Main GUI Window ----------

class UmyoGui(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("uMyo Realtime Viewer - Per Device + Gesture Recording")
        self.resize(1600, 800)
        pg.setConfigOptions(antialias=True)

        self.gesture_file = "gestures.json"
        self.subject_name = ""
        self.save_directory = os.getcwd()  # Default save path

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Left Panel (Devices)
        self.max_devices_display = 3
        panels_container = QtWidgets.QHBoxLayout()
        self.filter_panel = FilterPanel()
        DevicePanel.filter_panel = self.filter_panel

        self.device_panels = []
        for i in range(self.max_devices_display):
            panel = DevicePanel(index=i)
            self.device_panels.append(panel)
            panels_container.addWidget(panel)
        main_layout.addLayout(panels_container, stretch=3)

        # Right Panel (Controls)
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(self.filter_panel)
        control_layout.addSpacing(20)

        control_layout.addWidget(QtWidgets.QLabel("הגדרת מחווה והקלטה:"))

        # Gestures
        control_layout.addWidget(QtWidgets.QLabel("סוג תנועה:"))
        gesture_select_layout = QtWidgets.QHBoxLayout()
        self.gesture_combo = QtWidgets.QComboBox()
        self.load_gestures()
        gesture_select_layout.addWidget(self.gesture_combo)

        btn_add = QtWidgets.QPushButton("+")
        btn_add.setFixedWidth(30)
        btn_add.clicked.connect(self.open_gesture_dialog)

        btn_manage = QtWidgets.QPushButton("⚙")
        btn_manage.setFixedWidth(30)
        btn_manage.clicked.connect(self.open_gesture_manager)

        gesture_select_layout.addWidget(btn_add)
        gesture_select_layout.addWidget(btn_manage)
        control_layout.addLayout(gesture_select_layout)

        # Settings Spins
        control_layout.addWidget(QtWidgets.QLabel("מספר חזרות:"))
        self.reps_spin = QtWidgets.QSpinBox();
        self.reps_spin.setRange(1, 100);
        self.reps_spin.setValue(5)
        control_layout.addWidget(self.reps_spin)

        control_layout.addWidget(QtWidgets.QLabel("זמן הכנה (שניות):"))
        self.pre_start_spin = QtWidgets.QDoubleSpinBox();
        self.pre_start_spin.setValue(3.0)
        control_layout.addWidget(self.pre_start_spin)

        control_layout.addWidget(QtWidgets.QLabel("מרווח בין חזרות (שניות) - מנוחה:"))
        self.interval_spin = QtWidgets.QDoubleSpinBox();
        self.interval_spin.setValue(2.0)
        control_layout.addWidget(self.interval_spin)

        control_layout.addWidget(QtWidgets.QLabel("זמן הקלטה (שניות) - מחווה:"))
        self.duration_spin = QtWidgets.QDoubleSpinBox();
        self.duration_spin.setValue(1.5)
        control_layout.addWidget(self.duration_spin)

        # --- NEW: Recording Parameter ---
        control_layout.addWidget(QtWidgets.QLabel("פרמטר בדיקה (לקובץ):"))
        self.param_input = QtWidgets.QLineEdit()
        self.param_input.setPlaceholderText("לדוגמה: sensor_loc_1")
        control_layout.addWidget(self.param_input)

        # --- NEW: Save Location ---
        control_layout.addWidget(QtWidgets.QLabel("תיקיית שמירה:"))
        loc_layout = QtWidgets.QHBoxLayout()
        self.save_loc_label = QtWidgets.QLineEdit(self.save_directory)
        self.save_loc_label.setReadOnly(True)
        loc_layout.addWidget(self.save_loc_label)
        btn_browse = QtWidgets.QPushButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self.browse_folder)
        loc_layout.addWidget(btn_browse)
        control_layout.addLayout(loc_layout)

        # Indicators
        control_layout.addWidget(QtWidgets.QLabel("חיווי הקלטה / טיימר:"))
        self.led_frame = QtWidgets.QFrame()
        self.led_frame.setFixedSize(40, 40)
        self.led_frame.setStyleSheet("background-color: gray; border-radius: 20px;")
        control_layout.addWidget(self.led_frame, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

        self.countdown_label = QtWidgets.QLabel("")
        self.countdown_label.setStyleSheet("font-size: 30px; font-weight: bold; color: red;")
        self.countdown_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.countdown_label)

        self.record_button = QtWidgets.QPushButton("התחל הקלטת מחווה")
        self.record_button.clicked.connect(self.on_record_button_clicked)
        control_layout.addWidget(self.record_button)

        self.status_label = QtWidgets.QLabel("סטטוס: מוכן")
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        main_layout.addLayout(control_layout, stretch=1)

        # Logic Variables
        self.session_active = False
        self.current_trial = 0
        self.total_trials = 0
        self.current_label = "at_rest"
        self.full_session_data = []
        self.session_start_time = 0.0

        # Timers
        self.trial_timer = QtCore.QTimer()  # End of phase
        self.trial_timer.setSingleShot(True)
        self.trial_timer.timeout.connect(self.end_current_phase)

        self.countdown_update_timer = QtCore.QTimer()  # Pre-start
        self.countdown_update_timer.timeout.connect(self.update_countdown_display)
        self.countdown_start_time = 0.0

        self.phase_countdown_timer = QtCore.QTimer()  # During Phase
        self.phase_countdown_timer.timeout.connect(self.update_phase_countdown_display)
        self.phase_end_time = 0.0

        self.gui_timer = QtCore.QTimer()  # Plotting
        self.gui_timer.timeout.connect(self.update_from_devices)
        self.gui_timer.start(30)

    # --- File System Logic ---
    def browse_folder(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, "בחר תיקייה לשמירה", self.save_directory)
        if dir_path:
            self.save_directory = dir_path
            self.save_loc_label.setText(dir_path)

    # --- Gesture Management ---
    def load_gestures(self):
        if os.path.exists(self.gesture_file):
            try:
                with open(self.gesture_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        self.gesture_combo.addItem(item['display'], userData=item['key'])
            except:
                self.populate_default_gestures()
        else:
            self.populate_default_gestures()

    def populate_default_gestures(self):
        self.gesture_combo.clear()
        defaults = [("אגרוף", "fist"), ("כפיפת כף יד", "wrist_flexion"),
                    ("יישור כף יד", "wrist_extension"), ("תנועה אקראית", "random")]
        for d, k in defaults:
            self.gesture_combo.addItem(d, userData=k)

    def save_gestures(self):
        data = []
        for i in range(self.gesture_combo.count()):
            data.append({
                'display': self.gesture_combo.itemText(i),
                'key': self.gesture_combo.itemData(i)
            })
        with open(self.gesture_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def open_gesture_dialog(self):
        dialog = GestureDialog(self)
        if dialog.exec():
            d, k = dialog.get_data()
            if d and k:
                self.gesture_combo.addItem(d, userData=k)
                self.save_gestures()
                self.status_label.setText(f"נוספה: {d}")

    def open_gesture_manager(self):
        current_gestures = []
        for i in range(self.gesture_combo.count()):
            current_gestures.append({
                'display': self.gesture_combo.itemText(i),
                'key': self.gesture_combo.itemData(i)
            })

        manager = GestureManagerDialog(current_gestures, self)
        if manager.exec():
            self.gesture_combo.clear()
            for g in current_gestures:
                self.gesture_combo.addItem(g['display'], userData=g['key'])
            self.save_gestures()
            self.status_label.setText("רשימת מחוות עודכנה")

    # --- Session Logic ---

    def set_led(self, state: str):
        if state == 'record':
            self.led_frame.setStyleSheet("background-color: red; border-radius: 20px;")
        elif state == 'rest':
            self.led_frame.setStyleSheet("background-color: yellow; border-radius: 20px;")
        else:
            self.led_frame.setStyleSheet("background-color: gray; border-radius: 20px;")

    def on_record_button_clicked(self):
        if not self.session_active:
            self.start_session()
        else:
            self.stop_session("נעצר ידנית", should_save=True)

    def start_session(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Subject", "Enter subject name:")
        if not ok or not text.strip():
            self.status_label.setText("בוטל: חסר שם נבדק")
            return
        self.subject_name = text.strip()

        self.total_trials = self.reps_spin.value()
        self.current_trial = 0
        self.session_active = True
        self.full_session_data = []
        self.session_start_time = 0.0
        self.record_button.setText("עצור הקלטה ושמור")

        # Disable inputs
        self.gesture_combo.setEnabled(False)
        self.reps_spin.setEnabled(False)
        self.interval_spin.setEnabled(False)
        self.duration_spin.setEnabled(False)
        self.pre_start_spin.setEnabled(False)
        self.param_input.setEnabled(False)  # Disable param input during recording

        # Pre-start Countdown
        self.countdown_start_time = time.time() + self.pre_start_spin.value()
        self.status_label.setText("מתחיל ספירה לאחור...")
        self.countdown_update_timer.start(100)

        if self.pre_start_spin.value() == 0.0:
            self.countdown_label.setText("START!")
            self.countdown_update_timer.stop()
            self._start_rest_phase()

    def update_countdown_display(self):
        left = self.countdown_start_time - time.time()
        if left > 0:
            self.countdown_label.setText(f"recording in: {int(math.ceil(left))}")
            self.set_led('rest')
        else:
            self.countdown_label.setText("START!")
            self.countdown_update_timer.stop()
            self._start_rest_phase()

    def _start_rest_phase(self):
        if not self.session_active: return
        if self.session_start_time == 0.0: self.session_start_time = time.time()

        if self.current_trial >= self.total_trials and self.current_trial > 0:
            self.stop_session("Session הושלם", should_save=True)
            return

        interval = self.interval_spin.value()
        self.current_label = "at_rest"
        self.status_label.setText(f"במנוחה - מחכה {interval} שניות...")
        self.set_led('rest')

        self.trial_timer.start(int(interval * 1000))
        self.phase_end_time = time.time() + interval
        self.phase_countdown_timer.start(100)

    def _start_recording_phase(self):
        if not self.session_active: return
        self.current_trial += 1

        dur = self.duration_spin.value()
        self.current_label = self.gesture_combo.currentData()
        name = self.gesture_combo.currentText()

        self.status_label.setText(f"מבצע מחווה: {name} ({self.current_trial}/{self.total_trials})")
        self.set_led('record')

        self.trial_timer.start(int(dur * 1000))
        self.phase_end_time = time.time() + dur
        self.phase_countdown_timer.start(100)

    def update_phase_countdown_display(self):
        left = self.phase_end_time - time.time()
        if left > 0:
            disp = int(math.ceil(left))
            if self.current_label == "at_rest":
                self.countdown_label.setText(f"resting... {disp}")
            else:
                g_name = self.gesture_combo.currentText()
                self.countdown_label.setText(f"{g_name}... {disp}")
        else:
            self.phase_countdown_timer.stop()
            self.countdown_label.setText(self.current_label.upper())

    def end_current_phase(self):
        self.phase_countdown_timer.stop()
        if self.current_label == self.gesture_combo.currentData():
            self._start_rest_phase()
        elif self.current_label == "at_rest":
            self._start_recording_phase()
        else:
            self.stop_session("Logic Error")

    def stop_session(self, reason, should_save=False):
        self.session_active = False
        self.trial_timer.stop()
        self.countdown_update_timer.stop()
        self.phase_countdown_timer.stop()
        self.set_led('off')
        self.countdown_label.setText("")
        self.record_button.setText("התחל הקלטת מחווה")

        # Enable inputs
        self.gesture_combo.setEnabled(True)
        self.reps_spin.setEnabled(True)
        self.interval_spin.setEnabled(True)
        self.duration_spin.setEnabled(True)
        self.pre_start_spin.setEnabled(True)
        self.param_input.setEnabled(True)

        if should_save:
            self._save_file()
            self.status_label.setText(f"נשמר. ({reason})")
        else:
            self.status_label.setText(f"{reason}")

    def update_from_devices(self):
        devs = umyo_parser.umyo_get_list()
        for i, p in enumerate(self.device_panels):
            if i < len(devs):
                p.update_from_device(devs[i])
            else:
                p.clear_panel()

        if self.session_active and len(devs) > 0:
            for i, d in enumerate(devs[:self.max_devices_display]):
                self.collect_data(i, d)

    def collect_data(self, idx, dev):
        if not hasattr(dev, "data_array") or not hasattr(dev, "data_count"): return
        samples = dev.data_array[:dev.data_count]
        if not len(samples): return

        # Metadata
        sp = list(getattr(dev, "device_spectr", [])) + [None] * 4
        ax, ay, az = getattr(dev, "ax", None), getattr(dev, "ay", None), getattr(dev, "az", None)
        uid = getattr(dev, "unit_id", 0)

        now = time.time()
        t_rel_start = now - self.session_start_time if self.session_start_time > 0 else 0
        dt = datetime.now()
        d_str, t_str = dt.strftime("%d-%m-%y"), dt.strftime("%H-%M-%S")

        t_per_sample = 0.03 / len(samples)  # Approx

        for i, s in enumerate(samples):
            t_sample = t_rel_start - (len(samples) - 1 - i) * t_per_sample
            trial = self.current_trial if self.current_label != "at_rest" or self.current_trial > 0 else 0

            self.full_session_data.append({
                "gesture_label": self.current_label, "trial_index": trial, "unit_id": uid,
                "date": d_str, "clock_time": t_str, "subject_name": self.subject_name,
                "t": t_sample, "emg": s, "sp0": sp[0], "sp1": sp[1], "sp2": sp[2], "sp3": sp[3],
                "ax": ax, "ay": ay, "az": az, "device_index": idx
            })

    def _save_file(self):
        if not self.full_session_data: return

        g_key = self.gesture_combo.currentData()
        subj = self.subject_name.replace(" ", "_")
        dt = datetime.now()

        # Get Parameter text
        param_txt = self.param_input.text().strip().replace(" ", "_")
        if param_txt:
            param_txt = f", {param_txt}"

        # Filename: Subject, Gesture, [Param], HH-MM-SS, DD-MM-YY.csv
        fname = f"{subj}, {g_key}{param_txt}, {dt.strftime('%H-%M-%S')}, {dt.strftime('%d-%m-%y')}.csv"

        # Combine with Save Directory
        full_path = os.path.join(self.save_directory, fname)

        # Duplicate check
        if os.path.exists(full_path):
            base, ext = os.path.splitext(full_path)
            idx = 1
            while os.path.exists(f"{base}_{idx}{ext}"): idx += 1
            full_path = f"{base}_{idx}{ext}"

        try:
            with open(full_path, "w", newline="", encoding="utf-8") as f:
                cols = ["gesture_label", "trial_index", "unit_id", "date", "time", "subject_name",
                        "t_rel_sec", "emg", "sp0", "sp1", "sp2", "sp3", "ax", "ay", "az"]
                writer = csv.writer(f)
                writer.writerow(cols)

                for r in self.full_session_data:
                    uid_str = f"{r['unit_id']:08X}" if isinstance(r['unit_id'], int) else str(r['unit_id'])
                    writer.writerow([
                        r["gesture_label"], r["trial_index"], uid_str, r["date"], r["clock_time"], r["subject_name"],
                        f"{r['t']:.6f}", r["emg"], r["sp0"], r["sp1"], r["sp2"], r["sp3"], r["ax"], r["ay"], r["az"]
                    ])
            self.status_label.setText(f"נשמר: {os.path.basename(full_path)}")
        except Exception as e:
            self.status_label.setText(f"שגיאה: {e}")

        self.full_session_data = []


def main():
    t = SerialReader()
    t.start()
    app = QtWidgets.QApplication(sys.argv)
    gui = UmyoGui()
    app.aboutToQuit.connect(lambda: (t.stop(), t.join(1.0)))
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()