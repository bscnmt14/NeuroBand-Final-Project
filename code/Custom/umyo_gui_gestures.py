import sys
import time
import threading
import math
import os
import csv

import numpy as np
import serial
from serial.tools import list_ports

import umyo_parser  # חייב להיות באותה תיקייה (Must be in the same folder)

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


# ---------- פאנל אחד עבור Device אחד ----------

class DevicePanel(QtWidgets.QWidget):
    def __init__(self, index: int, emg_len=2000, acc_len=200, parent=None):
        super().__init__(parent)

        self.index = index
        self.emg_len = emg_len
        self.acc_len = acc_len

        self.emg_buffer = np.zeros(self.emg_len, dtype=float)
        self.acc_x = np.zeros(self.acc_len, dtype=float)
        self.acc_y = np.zeros(self.acc_len, dtype=float)
        self.acc_z = np.zeros(self.acc_len, dtype=float)

        # --- Filter Setup (High-Pass) ---
        self.FS = 1100.0  # EMG Sampling Rate in Hz (167 Hz)
        self.CUTOFF_FREQ = 5.0  # 10 Hz High-Pass filter to remove low-freq drift
        self.ORDER = 4  # Butterworth filter order

        # Design the filter
        Wn = self.CUTOFF_FREQ / (self.FS / 2.0)
        self.b, self.a = signal.butter(self.ORDER, Wn, 'hp', analog=False)

        # Initialize filter state for continuous, streaming operation
        # FIX: Initialize self.zi as a zero array matching the filter order (len(a) - 1)
        # This prevents the initial state from causing instability.
        self.zi = np.zeros(len(self.a) - 1, dtype=float)
        self.filtered_emg_buffer = np.zeros(self.emg_len, dtype=float)
        # --------------------------------

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel(f"Device {self.index}")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # ----- EMG -----
        self.emg_plot = pg.PlotWidget(title="EMG (Filtered)")
        self.emg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.emg_curve = self.emg_plot.plot(pen='g')
        layout.addWidget(self.emg_plot)

        # ----- Spectrum (sp0–sp3) -----
        self.spg_plot = pg.PlotWidget(title="Spectrum bins (sp0–sp3)")
        self.spg_plot.setYRange(0, 1)
        self.spg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spg_plot.getAxis('bottom').setTicks(
            [[(0, "sp0"), (1, "sp1"), (2, "sp2"), (3, "sp3")]]
        )
        self.spg_bars = pg.BarGraphItem(
            x=[0, 1, 2, 3],
            height=[0, 0, 0, 0],
            width=0.6,
            brushes=['g', 'y', 'm', 'r']
        )
        self.spg_plot.addItem(self.spg_bars)
        layout.addWidget(self.spg_plot)

        # ----- Accelerometer -----
        self.acc_plot = pg.PlotWidget(title="Accelerometer (ax, ay, az)")
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        self.acc_curve_x = self.acc_plot.plot(pen='r', name='ax')
        self.acc_curve_y = self.acc_plot.plot(pen='y', name='ay')
        self.acc_curve_z = self.acc_plot.plot(pen='b', name='az')
        layout.addWidget(self.acc_plot)

        # ----- Info: ID, RSSI, Battery, Compass -----
        info_layout = QtWidgets.QVBoxLayout()

        self.id_label = QtWidgets.QLabel("ID: N/A")
        self.id_label.setStyleSheet("font-size: 11px;")
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

    # ------------ עדכון לפי אובייקט device מה-parser ------------

    def update_from_device(self, dev):
        # ID
        if hasattr(dev, "unit_id"):
            self.id_label.setText(f"ID: {dev.unit_id:08X}")
        else:
            self.id_label.setText("ID: N/A")

        # EMG (with HPF applied)
        if hasattr(dev, "data_array") and hasattr(dev, "data_count"):
            samples = dev.data_array[:dev.data_count]
            if len(samples) > 0:
                shift = len(samples)

                # 1. Update the Raw buffer (kept for completeness, though plot uses filtered data)
                if shift >= self.emg_len:
                    self.emg_buffer[:] = samples[-self.emg_len:]
                else:
                    self.emg_buffer = np.roll(self.emg_buffer, -shift)
                    self.emg_buffer[-shift:] = samples

                # 2. Apply the High-Pass Filter (HPF) to the new chunk
                # FIX: Pass self.zi directly. This maintains stability for streaming data.
                y, self.zi = signal.lfilter(self.b, self.a, samples, zi=self.zi)

                # 3. Update the Filtered buffer
                if shift >= self.emg_len:
                    self.filtered_emg_buffer[:] = y[-self.emg_len:]
                else:
                    self.filtered_emg_buffer = np.roll(self.filtered_emg_buffer, -shift)
                    self.filtered_emg_buffer[-shift:] = y

                # 4. Update the Plot with the FILTERED data
                self.emg_curve.setData(self.filtered_emg_buffer)

        # Spectrum bins
        sp = getattr(dev, "device_spectr", None)
        if sp is not None and len(sp) >= 4:
            vals = np.array(sp[:4], dtype=float)
            max_val = max(np.max(np.abs(vals)), 1.0)
            heights = np.clip(np.abs(vals) / max_val, 0, 1)
            self.spg_bars.setOpts(height=heights)

        # Accelerometer
        for buf, attr in zip(
                (self.acc_x, self.acc_y, self.acc_z),
                ("ax", "ay", "az"),
        ):
            val = getattr(dev, attr, None)
            if val is not None:
                buf[:] = np.roll(buf, -1)
                buf[-1] = val

        self.acc_curve_x.setData(self.acc_x)
        self.acc_curve_y.setData(self.acc_y)
        self.acc_curve_z.setData(self.acc_z)

        # RSSI
        rssi = getattr(dev, "rssi", None)
        if rssi is not None and rssi > 0:
            sig_level = (90 - rssi) * 1.6
            sig_level = max(0, min(100, sig_level))
            self.rssi_bar.setValue(int(sig_level))

        # Battery
        batt_mv = getattr(dev, "batt", None)
        if batt_mv is not None:
            batt_perc = (batt_mv - 3100) / 10.0
            batt_perc = max(0, min(100, batt_perc))
            self.batt_bar.setValue(int(batt_perc))

        # Compass
        mag_angle = getattr(dev, "mag_angle", None)
        if mag_angle is not None:
            ang_deg = (math.degrees(mag_angle) + 360.0) % 360.0
            self.compass_label.setText(f"Compass: {ang_deg:5.1f}°")
        else:
            self.compass_label.setText("Compass: N/A")

    def clear_panel(self):
        self.id_label.setText("ID: N/A")


# ---------- GUI עיקרי שמכיל כמה DevicePanel + פאנל הקלטה ----------

class UmyoGui(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("uMyo Realtime Viewer - Per Device + Gesture Recording")
        self.resize(1600, 800)

        pg.setConfigOptions(antialias=True)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)

        # כמה Devices נציג
        self.max_devices_display = 3

        # ----- צד שמאל: פאנלים לכל Device -----
        panels_container = QtWidgets.QHBoxLayout()
        self.device_panels = []
        for i in range(self.max_devices_display):
            panel = DevicePanel(index=i)
            self.device_panels.append(panel)
            panels_container.addWidget(panel)

        main_layout.addLayout(panels_container, stretch=3)

        # ----- צד ימין: פאנל הגדרת מחווה והקלטה -----
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(QtWidgets.QLabel("הגדרת מחווה והקלטה:"))

        # בחירת מחווה
        control_layout.addWidget(QtWidgets.QLabel("סוג תנועה:"))
        self.gesture_combo = QtWidgets.QComboBox()
        # userData באנגלית לשם הקובץ
        self.gesture_combo.addItem("אגרוף", userData="fist")
        self.gesture_combo.addItem("כפיפת כף יד", userData="wrist_flexion")
        control_layout.addWidget(self.gesture_combo)

        # מספר חזרות
        control_layout.addWidget(QtWidgets.QLabel("מספר חזרות:"))
        self.reps_spin = QtWidgets.QSpinBox()
        self.reps_spin.setRange(1, 100)
        self.reps_spin.setValue(5)
        control_layout.addWidget(self.reps_spin)

        # מרווח בין חזרות (שניות)
        control_layout.addWidget(QtWidgets.QLabel("מרווח בין חזרות (שניות):"))
        self.interval_spin = QtWidgets.QDoubleSpinBox()
        self.interval_spin.setRange(0.5, 30.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setValue(3.0)
        control_layout.addWidget(self.interval_spin)

        # זמן הקלטה לכל חזרה (שניות)
        control_layout.addWidget(QtWidgets.QLabel("זמן הקלטה לכל חזרה (שניות):"))
        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0.2, 10.0)
        self.duration_spin.setSingleStep(0.1)
        self.duration_spin.setValue(1.5)
        control_layout.addWidget(self.duration_spin)

        # LED ויזואלי
        control_layout.addWidget(QtWidgets.QLabel("חיווי הקלטה (נורה):"))
        self.led_frame = QtWidgets.QFrame()
        self.led_frame.setFixedSize(40, 40)
        self.led_frame.setStyleSheet("background-color: gray; border-radius: 20px;")
        control_layout.addWidget(self.led_frame, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

        # כפתור הקלטה
        self.record_button = QtWidgets.QPushButton("התחל הקלטת מחווה")
        self.record_button.clicked.connect(self.on_record_button_clicked)
        control_layout.addWidget(self.record_button)

        # סטטוס
        self.status_label = QtWidgets.QLabel("סטטוס: מוכן")
        control_layout.addWidget(self.status_label)

        control_layout.addStretch()
        main_layout.addLayout(control_layout, stretch=1)

        # ----- Timers ל-session הקלטה -----
        self.session_active = False
        self.current_trial = 0
        self.total_trials = 0
        self.recording_now = False

        self.record_timer = QtCore.QTimer()
        self.record_timer.setSingleShot(True)
        self.record_timer.timeout.connect(self.stop_current_trial)

        self.interval_timer = QtCore.QTimer()
        self.interval_timer.setSingleShot(True)
        self.interval_timer.timeout.connect(self.start_next_trial)

        # Buffer להקלטה
        self.trial_data = []
        self.trial_start_time = 0.0

        # Timer לעדכון מה-parser
        self.gui_timer = QtCore.QTimer()
        self.gui_timer.timeout.connect(self.update_from_devices)
        self.gui_timer.start(30)  # ~33Hz

    # ------------ עדכון תצוגת Devices (EMG/Spg/Acc/Info) ------------

    def update_from_devices(self):
        devices = umyo_parser.umyo_get_list()
        dev_count = len(devices)

        # עדכון פאנלים
        for idx, panel in enumerate(self.device_panels):
            if idx < dev_count:
                dev = devices[idx]
                panel.update_from_device(dev)
            else:
                panel.clear_panel()

        # אם יש הקלטה פעילה – נאסוף דאטא מכל ה-devices
        if self.recording_now and dev_count > 0:
            for dev_index, dev in enumerate(devices[:self.max_devices_display]):
                self.collect_recording_data(dev_index, dev)

    # ------------ לוגיקת LED ------------

    def set_led(self, on: bool):
        if on:
            self.led_frame.setStyleSheet(
                "background-color: red; border-radius: 20px;"
            )
        else:
            self.led_frame.setStyleSheet(
                "background-color: gray; border-radius: 20px;"
            )

    # ------------ לוגיקת Session הקלטת מחוות ------------

    def on_record_button_clicked(self):
        if not self.session_active:
            self.start_session()
        else:
            self.stop_session("נעצר ידנית על ידי המשתמש")

    def start_session(self):
        self.total_trials = self.reps_spin.value()
        self.current_trial = 0
        self.session_active = True
        self.status_label.setText("סטטוס: Session פעיל")
        self.record_button.setText("עצור הקלטה")

        # ננעל את השדות בזמן ריצה
        self.gesture_combo.setEnabled(False)
        self.reps_spin.setEnabled(False)
        self.interval_spin.setEnabled(False)
        self.duration_spin.setEnabled(False)

        # חזרה ראשונה
        self.start_next_trial()

    def stop_session(self, reason: str = ""):
        self.session_active = False
        self.recording_now = False
        self.record_timer.stop()
        self.interval_timer.stop()
        self.set_led(False)
        self.record_button.setText("התחל הקלטת מחווה")
        self.gesture_combo.setEnabled(True)
        self.reps_spin.setEnabled(True)
        self.interval_spin.setEnabled(True)
        self.duration_spin.setEnabled(True)

        if reason:
            self.status_label.setText(f"סטטוס: {reason}")
        else:
            self.status_label.setText("סטטוס: Session הסתיים")

    def start_next_trial(self):
        if not self.session_active:
            return

        if self.current_trial >= self.total_trials:
            self.stop_session("Session הושלם")
            return

        self.current_trial += 1
        self.status_label.setText(
            f"סטטוס: הקלטה חזרה {self.current_trial}/{self.total_trials}"
        )

        # הכנה לחזרה
        self.trial_data = []
        self.trial_start_time = time.time()

        # התחלת הקלטה + נורה
        self.recording_now = True
        self.set_led(True)

        duration_sec = self.duration_spin.value()
        self.record_timer.start(int(duration_sec * 1000))

    def stop_current_trial(self):
        # עצירת הקלטה + כיבוי נורה
        self.recording_now = False
        self.set_led(False)

        # שמירה לקובץ
        self.save_current_trial_to_file()

        # מעבר לחזרה הבאה או סיום
        if self.current_trial < self.total_trials and self.session_active:
            interval_sec = self.interval_spin.value()
            self.status_label.setText(
                f"סטטוס: מחכה {interval_sec:.1f} שניות לחזרה הבאה..."
            )
            self.interval_timer.start(int(interval_sec * 1000))
        else:
            self.stop_session("Session הושלם")

    # ------------ איסוף דאטא במהלך הקלטה ------------

    def collect_recording_data(self, dev_index, dev):
        """
        נאסוף דאטא עבור החזרה הנוכחית מכל אחד מה-devices.
        לכל sample נשמור device_index + unit_id.
        """
        if not (hasattr(dev, "data_array") and hasattr(dev, "data_count")):
            return

        samples = dev.data_array[:dev.data_count]
        if len(samples) == 0:
            return

        sp = getattr(dev, "device_spectr", [None, None, None, None])
        if len(sp) < 4:
            sp = list(sp) + [None] * (4 - len(sp))

        ax = getattr(dev, "ax", None)
        ay = getattr(dev, "ay", None)
        az = getattr(dev, "az", None)

        unit_id = getattr(dev, "unit_id", 0)

        now = time.time()
        t_rel = now - self.trial_start_time

        for s in samples:
            self.trial_data.append({
                "t": t_rel,
                "emg": s,
                "sp0": sp[0],
                "sp1": sp[1],
                "sp2": sp[2],
                "sp3": sp[3],
                "ax": ax,
                "ay": ay,
                "az": az,
                "unit_id": unit_id,
                "device_index": dev_index,
                "trial_index": self.current_trial,
            })

    # ------------ שמירת חזרה לקובץ ------------

    def save_current_trial_to_file(self):
        if not self.trial_data:
            self.status_label.setText("סטטוס: אין דאטא להקלטה בחזרה זו")
            return

        gesture_key = self.gesture_combo.currentData()  # באנגלית לקובץ
        gesture_name_he = self.gesture_combo.currentText()  # בעברית

        base_name = gesture_key
        idx = 1
        while True:
            filename = f"{base_name}_{idx:03d}.csv"
            if not os.path.exists(filename):
                break
            idx += 1

        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # כותרות
                writer.writerow([
                    "gesture_he",
                    "gesture_key",
                    "trial_index",
                    "device_index",
                    "t_rel_sec",
                    "emg",
                    "sp0",
                    "sp1",
                    "sp2",
                    "sp3",
                    "ax",
                    "ay",
                    "az",
                    "unit_id",
                ])
                # שורות
                for row in self.trial_data:
                    writer.writerow([
                        gesture_name_he,
                        gesture_key,
                        row["trial_index"],
                        row["device_index"],
                        row["t"],
                        row["emg"],
                        row["sp0"],
                        row["sp1"],
                        row["sp2"],
                        row["sp3"],
                        row["ax"],
                        row["ay"],
                        row["az"],
                        f"{row['unit_id']:08X}",
                    ])
            self.status_label.setText(
                f"סטטוס: נשמר קובץ {filename} (חזרה {self.current_trial})"
            )
        except Exception as e:
            self.status_label.setText(f"סטטוס: שגיאה בשמירת קובץ: {e}")


# ---------- main ----------

def main():
    serial_thread = SerialReader()
    serial_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    gui = UmyoGui()

    def cleanup():
        serial_thread.stop()
        serial_thread.join(timeout=1.0)

    app.aboutToQuit.connect(cleanup)

    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()