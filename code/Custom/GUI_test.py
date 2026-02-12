import sys
import time
import threading
import math

import numpy as np
import serial
from serial.tools import list_ports

import umyo_parser  # אותו parser שיש לך

from PyQt5  import QtWidgets, QtCore
import pyqtgraph as pg


# ---------- Thread לקריאת ה-BLE דרך Serial ----------

class SerialReader(threading.Thread):
    def __init__(self, baudrate=921600):
        super().__init__(daemon=True)
        self.baudrate = baudrate
        self.running = True
        self.ser = None

    def find_port(self):
        ports = list(list_ports.comports())
        for p in ports:
            desc = (p.device + " " + p.description + " " + p.hwid).lower()
            if "usb" in desc or "serial" in desc or "umyo" in desc or "cp210" in desc or "ch340" in desc:
                return p.device
        return None

    def open_serial(self):
        port = self.find_port()
        if port is None:
            print("No uMyo serial port found!")
            return False

        # כמו בקוד ה logger שלך
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


# ---------- GUI עיקרי ----------

class UmyoGui(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("uMyo Realtime EMG Viewer")
        self.resize(1200, 600)

        # הגדרות Plot
        pg.setConfigOptions(antialias=True)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)

        # ---- Raw EMG ----
        self.emg_plot = pg.PlotWidget(title="Raw EMG")
        self.emg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.emg_curve = self.emg_plot.plot(pen='g')
        layout.addWidget(self.emg_plot, 0, 0, 1, 2)

        # Buffer ל-EMG (ערוץ יחיד, 2000 דגימות)
        self.emg_len = 2000
        self.emg_buffer = np.zeros(self.emg_len, dtype=float)

        # ---- Spectrum bins (sp0–sp3) ----
        self.spg_plot = pg.PlotWidget(title="Spectrum bins (sp0–sp3)")
        self.spg_plot.setYRange(0, 1)
        self.spg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spg_plot.getAxis('bottom').setTicks(
            [[(0, "sp0"), (1, "sp1"), (2, "sp2"), (3, "sp3")]]
        )
        self.spg_bars = pg.BarGraphItem(x=[0, 1, 2, 3],
                                        height=[0, 0, 0, 0],
                                        width=0.6)
        self.spg_plot.addItem(self.spg_bars)
        layout.addWidget(self.spg_plot, 1, 0)

        # ---- Accelerometer ----
        self.acc_plot = pg.PlotWidget(title="3-axis Accelerometer")
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        self.acc_len = 200
        self.acc_x = np.zeros(self.acc_len)
        self.acc_y = np.zeros(self.acc_len)
        self.acc_z = np.zeros(self.acc_len)
        self.acc_curve_x = self.acc_plot.plot(pen='r', name='ax')
        self.acc_curve_y = self.acc_plot.plot(pen='y', name='ay')
        self.acc_curve_z = self.acc_plot.plot(pen='b', name='az')
        layout.addWidget(self.acc_plot, 1, 1)

        # ---- מידע טקסטואלי: RSSI / Battery / Compass ----
        info_layout = QtWidgets.QVBoxLayout()

        self.device_label = QtWidgets.QLabel("Device: N/A")
        self.device_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        info_layout.addWidget(self.device_label)

        # RSSI
        rssi_layout = QtWidgets.QHBoxLayout()
        rssi_layout.addWidget(QtWidgets.QLabel("RSSI:"))
        self.rssi_bar = QtWidgets.QProgressBar()
        self.rssi_bar.setRange(0, 100)
        rssi_layout.addWidget(self.rssi_bar)
        info_layout.addLayout(rssi_layout)

        # Battery
        batt_layout = QtWidgets.QHBoxLayout()
        batt_layout.addWidget(QtWidgets.QLabel("Battery:"))
        self.batt_bar = QtWidgets.QProgressBar()
        self.batt_bar.setRange(0, 100)
        batt_layout.addWidget(self.batt_bar)
        info_layout.addLayout(batt_layout)

        # Compass / Orientation
        self.compass_label = QtWidgets.QLabel("Compass angle: N/A")
        info_layout.addWidget(self.compass_label)

        info_layout.addStretch()
        layout.addLayout(info_layout, 0, 2, 2, 1)

        # Timer לעדכון GUI מה-uMyo
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_from_devices)
        self.timer.start(30)  # כל 30ms

    # ----- עיבוד נתונים מכל המכשירים -----

    def get_first_device(self):
        devices = umyo_parser.umyo_get_list()
        if not devices:
            return None
        return devices[0]  # כרגע: סנסור אחד

    def update_from_devices(self):
        dev = self.get_first_device()
        if dev is None:
            return

        # מזהה מכשיר
        if hasattr(dev, "unit_id"):
            self.device_label.setText(f"Device: {dev.unit_id:08X}")

        # ----- EMG -----
        if hasattr(dev, "data_array") and hasattr(dev, "data_count"):
            samples = dev.data_array[:dev.data_count]
            if len(samples) > 0:
                # הוספה ל-buffer מתגלגל
                self.emg_buffer = np.roll(self.emg_buffer, -len(samples))
                self.emg_buffer[-len(samples):] = samples
                self.emg_curve.setData(self.emg_buffer)

        # ----- Spectrum bins (4 ערכים) -----
        sp = getattr(dev, "device_spectr", None)
        if sp is not None and len(sp) >= 4:
            # ננרמל למשהו הגיוני כדי שייראה יפה (|val| / max)
            vals = np.array(sp[:4], dtype=float)
            max_val = max(np.max(np.abs(vals)), 1.0)
            heights = np.clip(np.abs(vals) / max_val, 0, 1)
            #heights = np.abs(vals)
            # pyqtgraph מצפה למערך numpy
            self.spg_plot.removeItem(self.spg_bars)
            self.spg_bars = pg.BarGraphItem(x=[0, 1, 2, 3],
                                            height=heights,
                                            width=0.6,
                                            brushes=['g', 'y', 'm', 'r'])
            self.spg_plot.addItem(self.spg_bars)

        # ----- Accelerometer -----
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

        # ----- RSSI -----
        rssi = getattr(dev, "rssi", None)
        if rssi is not None and rssi > 0:
            # אותו scaling כמו בקוד pygame: sig_level ~ 0-100
            sig_level = (90 - rssi) * 1.6
            sig_level = max(0, min(100, sig_level))
            self.rssi_bar.setValue(int(sig_level))

        # ----- Battery -----
        batt_mv = getattr(dev, "batt", None)
        if batt_mv is not None:
            batt_perc = (batt_mv - 3100) / 10  # כמו אצלך
            batt_perc = max(0, min(100, batt_perc))
            self.batt_bar.setValue(int(batt_perc))

        # ----- Compass -----
        mag_angle = getattr(dev, "mag_angle", None)
        if mag_angle is not None:
            # נהפוך לזווית במעלות, 0–360
            ang_deg = (math.degrees(mag_angle) + 360) % 360
            self.compass_label.setText(
                f"Compass angle: {ang_deg:5.1f}°"
            )


# ---------- main ----------

def main():
    # Thread שקורא מה-serial ומעדכן את umyo_parser
    serial_thread = SerialReader()
    serial_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    gui = UmyoGui()

    # כשסוגרים את האפליקציה – נעצור את ה-thread
    def cleanup():
        serial_thread.stop()
        serial_thread.join(timeout=1.0)

    app.aboutToQuit.connect(cleanup)

    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
