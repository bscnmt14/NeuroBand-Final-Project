import sys
import time
import threading
import math
import os
import csv
import json # NEW: for saving gestures
from datetime import datetime

import numpy as np
import serial
from serial.tools import list_ports

import umyo_parser

# --- GUI and Plotting Libraries ---
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg

# --- Signal Processing Library ---
from scipy import signal


# ---------- Thread לקריאת UART מהדונגל / גשר ----------

class SerialReader(threading.Thread):
    def __init__(self, baudrate=921600):
        super().__init__(daemon=True)
        self.baudrate = baudrate
        self.running = True
        self.ser = None

    def find_port(self):
        ports = list(list_ports.comports())
        for p in ports:
            # Check for common port patterns
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


# ---------- פאנל להגדרות פילטרציה ----------
class FilterPanel(QtWidgets.QGroupBox):
    filter_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("Real-time Filter Settings", parent)
        self.FS = 1100.0
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("Filter Type:"))
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItem("None", userData="none")
        self.type_combo.addItem("High-Pass (HPF)", userData="hp")
        self.type_combo.addItem("Low-Pass (LPF)", userData="lp")
        self.type_combo.addItem("Band-Stop (BSF)", userData="bs")

        # --- IMPROVEMENT: Set default to None ---
        self.type_combo.setCurrentText("None")

        self.type_combo.currentTextChanged.connect(self._on_settings_changed)
        layout.addWidget(self.type_combo)

        layout.addWidget(QtWidgets.QLabel("Order (N):"))
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(4)
        self.order_spin.valueChanged.connect(self._on_settings_changed)
        layout.addWidget(self.order_spin)

        layout.addWidget(QtWidgets.QLabel("Cutoff Freq (Hz):"))
        self.center_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.center_cutoff_spin.setRange(0.1, self.FS / 2.0 - 1)
        self.center_cutoff_spin.setSingleStep(1.0)
        self.center_cutoff_spin.setValue(50.0)
        self.center_cutoff_spin.valueChanged.connect(self._on_settings_changed)
        layout.addWidget(self.center_cutoff_spin)

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
        if hasattr(self, 'bs_label'): self.bs_label.setVisible(is_bs)
        if hasattr(self, 'bs_container'): self.bs_container.setVisible(is_bs)
        label_text = "Center Freq (Hz):" if is_bs else "Cutoff Freq (Hz):"
        try:
             layout_obj = self.center_cutoff_spin.parent().layout()
             label = layout_obj.itemAt(layout_obj.indexOf(self.center_cutoff_spin) - 1).widget()
             if isinstance(label, QtWidgets.QLabel): label.setText(label_text)
        except: pass

    def _on_settings_changed(self):
        self._update_bs_visibility(self.type_combo.currentData() == "bs")
        self.filter_changed.emit()

    def get_filter_params(self):
        f_type = self.type_combo.currentData()
        order = self.order_spin.value()
        MIN_BANDWIDTH_HZ = 0.5

        if f_type in ["hp", "lp"]:
            cutoff = self.center_cutoff_spin.value()
            return f_type, order, cutoff
        elif f_type == "bs":
            center = self.center_cutoff_spin.value()
            low = max(0.1, center - self.bs_left_delta_spin.value())
            high = min(self.FS / 2.0 - 0.1, center + self.bs_right_delta_spin.value())
            if low >= high or (high - low) < MIN_BANDWIDTH_HZ:
                return "none", 0, 0.0
            return f_type, order, [low, high]
        return "none", 0, 0.0


# ---------- פאנל אחד עבור Device אחד ----------

class DevicePanel(QtWidgets.QWidget):
    filter_panel: FilterPanel = None

    def __init__(self, index: int, emg_len=2000, acc_len=200, parent=None):
        super().__init__(parent)
        self.index, self.emg_len, self.acc_len = index, emg_len, acc_len
        self.emg_buffer = np.zeros(self.emg_len, dtype=float)
        self.acc_x = np.zeros(self.acc_len, dtype=float)
        self.acc_y = np.zeros(self.acc_len, dtype=float)
        self.acc_z = np.zeros(self.acc_len, dtype=float)
        self.FS = 1100.0
        self.b, self.a, self.zi = None, None, None
        self.filtered_emg_buffer = np.zeros(self.emg_len, dtype=float)

        if DevicePanel.filter_panel is not None:
            DevicePanel.filter_panel.filter_changed.connect(self._init_filter)
            self._init_filter()
        self._build_ui()

    def _init_filter(self):
        if DevicePanel.filter_panel is None: return
        f_type, order, w_freq = DevicePanel.filter_panel.get_filter_params()
        if f_type == "none":
            self.b, self.a, self.zi = None, None, None
            return
        Wn = np.array(w_freq) / (self.FS / 2.0)
        try:
            self.b, self.a = signal.butter(order, Wn, f_type, analog=False)
            self.zi = np.zeros(len(self.a) - 1, dtype=float) if len(self.a) > 1 else None
        except: self.b, self.a, self.zi = None, None, None

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"Device {self.index}"))
        self.emg_plot = pg.PlotWidget(title="EMG data")
        self.emg_plot.setYRange(-3000, 3000)
        self.emg_curve = self.emg_plot.plot(pen='g')
        layout.addWidget(self.emg_plot)

        self.spg_plot = pg.PlotWidget(title="Spectrum")
        self.spg_plot.setYRange(0, 1)
        self.spg_bars = pg.BarGraphItem(x=[0, 1, 2, 3], height=[0]*4, width=0.6, brushes=['g', 'y', 'm', 'r'])
        self.spg_plot.addItem(self.spg_bars)
        layout.addWidget(self.spg_plot)

        self.acc_plot = pg.PlotWidget(title="Accel")
        self.acc_plot.setYRange(-20000, 20000)
        self.acc_curve_x = self.acc_plot.plot(pen='r')
        self.acc_curve_y = self.acc_plot.plot(pen='y')
        self.acc_curve_z = self.acc_plot.plot(pen='b')
        layout.addWidget(self.acc_plot)

    def update_from_device(self, dev):
        if hasattr(dev, "data_array") and hasattr(dev, "data_count"):
            samples = dev.data_array[:dev.data_count]
            if len(samples) > 0:
                y = samples
                if self.b is not None:
                     y, self.zi = signal.lfilter(self.b, self.a, samples, zi=self.zi)
                self.filtered_emg_buffer = np.roll(self.filtered_emg_buffer, -len(y))
                self.filtered_emg_buffer[-len(y):] = y
                self.emg_curve.setData(self.filtered_emg_buffer)

        # Simple PSD for visual
        if len(self.filtered_emg_buffer) > 256:
            f, Pxx = signal.welch(self.filtered_emg_buffer, fs=self.FS, nperseg=256)
            bins = [np.sum(Pxx[(f >= i*100) & (f < (i+1)*100)]) for i in range(4)]
            if np.sum(bins) > 1e-9: self.spg_bars.setOpts(height=bins/np.max(bins))

        for buf, curve, attr in [(self.acc_x, self.acc_curve_x, "ax"), (self.acc_y, self.acc_curve_y, "ay"), (self.acc_z, self.acc_curve_z, "az")]:
            val = getattr(dev, attr, 0)
            buf[:] = np.roll(buf, -1); buf[-1] = val
            curve.setData(buf)

# ---------- פופ-אפ להוספת מחוות ----------

class GestureDialog(QtWidgets.QDialog):
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
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

# ---------- GUI עיקרי ----------

class UmyoGui(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("uMyo Realtime Viewer")
        self.resize(1600, 800)
        self.gesture_file = "gestures.json" # File to store gestures

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        self.filter_panel = FilterPanel()
        DevicePanel.filter_panel = self.filter_panel
        self.device_panels = [DevicePanel(i) for i in range(3)]

        p_layout = QtWidgets.QHBoxLayout()
        for p in self.device_panels: p_layout.addWidget(p)
        main_layout.addLayout(p_layout, 3)

        ctrl = QtWidgets.QVBoxLayout()
        ctrl.addWidget(self.filter_panel)

        self.gesture_combo = QtWidgets.QComboBox()
        self.load_gestures() # LOAD FROM FILE

        g_layout = QtWidgets.QHBoxLayout()
        g_layout.addWidget(self.gesture_combo)
        btn_add = QtWidgets.QPushButton("+"); btn_add.clicked.connect(self.open_gesture_dialog)
        g_layout.addWidget(btn_add)
        ctrl.addLayout(g_layout)

        self.reps = QtWidgets.QSpinBox(); self.reps.setValue(5); ctrl.addWidget(QtWidgets.QLabel("חזרות:")); ctrl.addWidget(self.reps)
        self.prep = QtWidgets.QDoubleSpinBox(); self.prep.setValue(3); ctrl.addWidget(QtWidgets.QLabel("הכנה (ש):")); ctrl.addWidget(self.prep)
        self.rest = QtWidgets.QDoubleSpinBox(); self.rest.setValue(2); ctrl.addWidget(QtWidgets.QLabel("מנוחה (ש):")); ctrl.addWidget(self.rest)
        self.dur = QtWidgets.QDoubleSpinBox(); self.dur.setValue(1.5); ctrl.addWidget(QtWidgets.QLabel("מחווה (ש):")); ctrl.addWidget(self.dur)

        self.countdown_label = QtWidgets.QLabel("")
        self.countdown_label.setStyleSheet("font-size: 30px; color: red;")
        ctrl.addWidget(self.countdown_label)

        self.record_btn = QtWidgets.QPushButton("התחל הקלטה")
        self.record_btn.clicked.connect(self.start_session)
        ctrl.addWidget(self.record_btn)

        self.status = QtWidgets.QLabel("סטטוס: מוכן")
        ctrl.addWidget(self.status)
        ctrl.addStretch()
        main_layout.addLayout(ctrl, 1)

        self.session_active = False
        self.full_session_data = []
        self.gui_timer = QtCore.QTimer()
        self.gui_timer.timeout.connect(self.update_from_devices)
        self.gui_timer.start(30)

        self.trial_timer = QtCore.QTimer(); self.trial_timer.setSingleShot(True)
        self.trial_timer.timeout.connect(self.next_phase)
        self.cd_timer = QtCore.QTimer(); self.cd_timer.timeout.connect(self.update_cd)

    def load_gestures(self):
        """Loads gestures from JSON file or sets defaults."""
        if os.path.exists(self.gesture_file):
            with open(self.gesture_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    self.gesture_combo.addItem(item['display'], userData=item['key'])
        else:
            # Defaults
            defaults = [("אגרוף", "fist"), ("כפיפה", "flexion"), ("יישור", "extension")]
            for d, k in defaults: self.gesture_combo.addItem(d, userData=k)

    def save_gestures(self):
        """Saves current combobox gestures to JSON."""
        data = []
        for i in range(self.gesture_combo.count()):
            data.append({
                'display': self.gesture_combo.itemText(i),
                'key': self.gesture_combo.itemData(i)
            })
        with open(self.gesture_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def open_gesture_dialog(self):
        dlg = GestureDialog(self)
        if dlg.exec():
            d, k = dlg.display_name.text(), dlg.key_name.text()
            if d and k:
                self.gesture_combo.addItem(d, userData=k)
                self.save_gestures() # SAVE IMMEDIATELY

    def start_session(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Subject", "Name:")
        if not ok or not name: return
        self.subject = name
        self.session_active = True
        self.full_session_data = []
        self.current_rep = 0
        self.session_start = 0
        self.target_reps = self.reps.value()
        self.cd_end = time.time() + self.prep.value()
        self.state = "PREP"
        self.cd_timer.start(100)

    def update_cd(self):
        left = self.cd_end - time.time()
        if left > 0:
            txt = "recording in" if self.state == "PREP" else ("resting" if self.state=="REST" else self.gesture_combo.currentText())
            self.countdown_label.setText(f"{txt}: {int(math.ceil(left))}")
        else:
            self.cd_timer.stop()
            if self.state == "PREP": self.start_rest()

    def start_rest(self):
        if self.current_rep >= self.target_reps: self.finish(); return
        if self.session_start == 0: self.session_start = time.time()
        self.state = "REST"
        self.cd_end = time.time() + self.rest.value()
        self.trial_timer.start(self.rest.value()*1000)
        self.cd_timer.start(100)

    def start_gesture(self):
        self.current_rep += 1
        self.state = "GESTURE"
        self.cd_end = time.time() + self.dur.value()
        self.trial_timer.start(self.dur.value()*1000)
        self.cd_timer.start(100)

    def next_phase(self):
        if self.state == "REST": self.start_gesture()
        else: self.start_rest()

    def finish(self):
        self.session_active = False
        self.countdown_label.setText("DONE")
        dt = datetime.now()
        fn = f"{self.subject}, {self.gesture_combo.currentData()}, {dt.strftime('%H-%M-%S')}, {dt.strftime('%d-%m-%y')}.csv"
        with open(fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label","rep","id","date","time","t","emg","ax","ay","az"])
            w.writeheader(); w.writerows(self.full_session_data)
        self.status.setText(f"Saved: {fn}")

    def update_from_devices(self):
        devs = umyo_parser.umyo_get_list()
        for i, p in enumerate(self.device_panels):
            if i < len(devs): p.update_from_device(devs[i])
        if self.session_active and len(devs) > 0:
            dev = devs[0]
            if hasattr(dev, "data_array"):
                for s in dev.data_array[:dev.data_count]:
                    self.full_session_data.append({
                        "label": "rest" if self.state == "REST" else self.gesture_combo.currentData(),
                        "rep": self.current_rep, "id": f"{dev.unit_id:08X}",
                        "date": datetime.now().strftime("%d/%m/%y"), "time": datetime.now().strftime("%H:%M:%S"),
                        "t": time.time()-self.session_start, "emg": s, "ax": dev.ax, "ay": dev.ay, "az": dev.az
                    })

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = UmyoGui(); gui.show()
    sys.exit(app.exec())