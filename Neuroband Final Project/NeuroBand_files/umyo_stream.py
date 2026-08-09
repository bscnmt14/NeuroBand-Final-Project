"""Threaded serial acquisition for the three uMyo forearm sensors.

The reader discovers the serial port, parses incoming Bluetooth gateway packets with
the supplied OEM parser, and stores the latest EMG, spectrum, IMU, RSSI, battery, and
timing state for each fixed sensor identifier. Thread-safe snapshots decouple high-
rate acquisition from GUI rendering and machine-learning inference.

"""

from __future__ import annotations

import sys
import threading
import time
import ctypes
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import serial
from serial.tools import list_ports

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OEM_DIR = PROJECT_ROOT / "OEM_files"
if str(OEM_DIR) not in sys.path:
    sys.path.insert(0, str(OEM_DIR))

import umyo_parser  # type: ignore  # OEM packet parser supplied with the uMyo devices.


@dataclass
class DeviceSnapshot:
    """Represent the DeviceSnapshot component and keep its related state and behavior together."""
    index: int
    unit_id: int
    data_id: int
    emg: np.ndarray
    spectrum: np.ndarray
    rssi: int
    battery_mv: int
    ax: float
    ay: float
    az: float
    yaw: float
    pitch: float
    roll: float
    quaternion: tuple[float, float, float, float]
    age_ms: float


class UmyoSerialReader(threading.Thread):
    """Represent the UmyoSerialReader component and keep its related state and behavior together."""
    def __init__(self, baudrate: int = 921600, port: str | None = None):
        """Initialize the UmyoSerialReader instance and its runtime state."""
        super().__init__(daemon=True)
        self.baudrate = baudrate
        self.requested_port = port
        self.ser: Optional[serial.Serial] = None
        self.running = threading.Event()
        self.running.set()
        self.lock = threading.Lock()
        self.last_error = ""
        self.port_name = ""
        self.last_packet_time = 0.0
        self.total_bytes = 0
        self.parse_backlog = 0
        self.last_data_ids: dict[int, int] = {}
        self.pending_emg: dict[int, list[float]] = defaultdict(list)
        self.pending_spectrum: dict[int, list[np.ndarray]] = defaultdict(list)
        self.latest_snapshot: dict[int, DeviceSnapshot] = {}
        self.realtime_priority_applied = False

    def _enable_realtime_priority(self) -> None:
        """Perform the enable realtime priority operation used by the UmyoSerialReader workflow."""
        self.realtime_priority_applied = False

    def find_port(self) -> str | None:
        """Locate port for the current UmyoSerialReader workflow."""
        if self.requested_port:
            return self.requested_port
        ports = list(list_ports.comports())
        for p in ports:
            desc = (p.description or "").lower()
            dev = (p.device or "").lower()
            if "usbserial" in dev or "usb" in desc or "serial" in desc or "uart" in desc:
                return p.device
        return ports[0].device if ports else None

    def run(self) -> None:
        """Perform the run operation used by the UmyoSerialReader workflow."""
        self._enable_realtime_priority()
        while self.running.is_set():
            if self.ser is None or not self.ser.is_open:
                port = self.find_port()
                if not port:
                    with self.lock:
                        self.last_error = "No serial port found; waiting for dongle"
                        self.port_name = ""
                    self._retry_delay(1.0)
                    continue
                try:
                    self.ser = serial.Serial(port=port, baudrate=self.baudrate, timeout=0)
                    with self.lock:
                        self.port_name = port
                        self.last_error = ""
                except Exception as exc:
                    with self.lock:
                        self.last_error = f"Could not open {port}: {exc}; retrying"
                        self.port_name = ""
                    self.close()
                    self._retry_delay(1.0)
                    continue

            try:
                waiting = self.ser.in_waiting if self.ser else 0
                if waiting > 0 and self.ser is not None:
                    data = self.ser.read(min(waiting, 256))
                    umyo_parser.umyo_parse_preprocessor(data)
                    backlog = len(getattr(umyo_parser, "parse_buf", b""))
                    self._collect_new_packets()
                    with self.lock:
                        self.total_bytes += len(data)
                        self.parse_backlog = backlog
                        self.last_packet_time = time.time()
                        self.last_error = ""
                time.sleep(0.0005)
            except Exception as exc:
                with self.lock:
                    self.last_error = f"Serial connection lost: {exc}; retrying"
                    self.port_name = ""
                self.close()
                self._retry_delay(0.75)
        self.close()

    def _retry_delay(self, seconds: float) -> None:
        """Perform the retry delay operation used by the UmyoSerialReader workflow."""
        deadline = time.time() + seconds
        while self.running.is_set() and time.time() < deadline:
            time.sleep(0.1)

    def stop(self) -> None:
        """Perform the stop operation used by the UmyoSerialReader workflow."""
        self.running.clear()

    def close(self) -> None:
        """Perform the close operation used by the UmyoSerialReader workflow."""
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def _collect_new_packets(self) -> None:
        """Perform the collect new packets operation used by the UmyoSerialReader workflow."""
        now = time.time()
        devices = list(umyo_parser.umyo_get_list())
        with self.lock:
            for idx, dev in enumerate(devices):
                uid = int(getattr(dev, "unit_id", 0))
                data_id = int(getattr(dev, "data_id", 0))
                if self.last_data_ids.get(uid) == data_id:
                    continue
                self.last_data_ids[uid] = data_id
                count = int(getattr(dev, "data_count", 0) or 0)
                raw = list(getattr(dev, "data_array", []))[:count]
                spectrum = np.array((list(getattr(dev, "device_spectr", [])) + [0] * 4)[:4], dtype=float)
                self.pending_emg[uid].extend(float(v) for v in raw)
                self.pending_spectrum[uid].append(spectrum)
                self.latest_snapshot[uid] = DeviceSnapshot(
                    index=idx,
                    unit_id=uid,
                    data_id=data_id,
                    emg=np.array([], dtype=float),
                    spectrum=spectrum,
                    rssi=int(getattr(dev, "rssi", 0) or 0),
                    battery_mv=int(getattr(dev, "batt", 0) or 0),
                    ax=float(getattr(dev, "ax", 0) or 0),
                    ay=float(getattr(dev, "ay", 0) or 0),
                    az=float(getattr(dev, "az", 0) or 0),
                    yaw=float(getattr(dev, "yaw", 0.0) or 0.0),
                    pitch=float(getattr(dev, "pitch", 0.0) or 0.0),
                    roll=float(getattr(dev, "roll", 0.0) or 0.0),
                    quaternion=tuple(float(v) for v in (list(getattr(dev, "Qsg", [])) + [0.0, 0.0, 0.0, 0.0])[:4]),
                    age_ms=(now - self.last_packet_time) * 1000 if self.last_packet_time else 0.0,
                )

    def snapshots(self) -> list[DeviceSnapshot]:
        """Perform the snapshots operation used by the UmyoSerialReader workflow."""
        now = time.time()
        result: list[DeviceSnapshot] = []
        with self.lock:
            for uid, base in self.latest_snapshot.items():
                pending = self.pending_emg.get(uid, [])
                spectra = self.pending_spectrum.get(uid, [])
                emg = np.array(pending, dtype=float)
                spectrum = np.vstack(spectra) if spectra else np.zeros((0, 4), dtype=float)
                pending.clear()
                spectra.clear()
                result.append(
                    DeviceSnapshot(
                        index=base.index,
                        unit_id=base.unit_id,
                        data_id=base.data_id,
                        emg=emg,
                        spectrum=spectrum,
                        rssi=base.rssi,
                        battery_mv=base.battery_mv,
                        ax=base.ax,
                        ay=base.ay,
                        az=base.az,
                        yaw=base.yaw,
                        pitch=base.pitch,
                        roll=base.roll,
                        quaternion=base.quaternion,
                        age_ms=(now - self.last_packet_time) * 1000 if self.last_packet_time else 999999.0,
                    )
                )
        return result

    def status(self) -> dict[str, object]:
        """Perform the status operation used by the UmyoSerialReader workflow."""
        with self.lock:
            age_ms = (time.time() - self.last_packet_time) * 1000 if self.last_packet_time else None
            return {
                "port": self.port_name or self.requested_port or "auto",
                "connected": self.ser is not None and self.ser.is_open,
                "last_error": self.last_error,
                "age_ms": age_ms,
                "total_bytes": self.total_bytes,
                "parse_backlog": self.parse_backlog,
                "serial_priority": "normal",
            }
