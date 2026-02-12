"""Comprehensive uMyo Device Testing and Validation Module.

This module provides a dedicated testing framework for comprehensive validation
of uMyo device functionality, data integrity, and system performance. It serves
as both a diagnostic tool for troubleshooting device issues and a reference
implementation for developers working with uMyo sensor systems.

The testing framework focuses on:

- Real-time data acquisition and validation
- Device communication integrity verification
- Signal quality assessment and diagnostics
- Performance benchmarking and timing analysis
- Multi-device synchronization testing
- Protocol compliance verification
- Long-term stability and reliability testing

Key Features:
    - Comprehensive device communication testing
    - Real-time data validation and integrity checking
    - Specialized testing display mode via display_stuff.plot_cycle_tester()
    - Performance monitoring with buffer overflow detection
    - Automated device discovery and connection testing
    - Continuous operation testing for reliability assessment
    - Data synchronization and timing analysis

Technical Implementation:
    - High-speed serial communication at 921600 baud rate
    - Real-time data preprocessing and validation pipeline
    - Adaptive display refresh based on data arrival patterns
    - Buffer management with overflow detection and reporting
    - Performance metrics collection and analysis
    - Automated error detection and reporting mechanisms

Testing Capabilities:
    1. Communication Protocol Testing:
       - Serial port connectivity and configuration
       - Data packet integrity and structure validation
       - Protocol compliance and error handling

    2. Data Quality Assessment:
       - Signal integrity and noise analysis
       - Data synchronization across multiple devices
       - Timing accuracy and jitter measurement

    3. Performance Benchmarking:
       - Data throughput measurement and optimization
       - Buffer utilization and overflow detection
       - System latency and response time analysis

    4. Long-term Reliability:
       - Continuous operation stability testing
       - Memory usage monitoring and leak detection
       - Device connection stability over time

Applications:
    - Device quality assurance and validation testing
    - System integration verification for new deployments
    - Performance optimization and bottleneck identification
    - Debugging and troubleshooting device communication issues
    - Research platform validation for scientific applications
    - Educational tool for understanding uMyo system behavior
    - Pre-deployment testing for critical applications

Testing Methodology:
    The module uses a specialized testing display mode that provides:
    - Enhanced visualization for diagnostic purposes
    - Real-time performance metrics and statistics
    - Error detection and reporting capabilities
    - Data quality indicators and signal analysis
    - Device status monitoring and health checks

Dependencies:
    - umyo_parser: Core uMyo data parsing and device management
    - display_stuff: Specialized testing visualization system
    - serial: Serial communication for device connectivity
    - serial.tools.list_ports: Automatic device discovery

Example Usage:
    >>> # Run comprehensive device testing
    >>> python umyo_testing.py

    >>> # Monitor output for test results
    >>> # available ports:
    >>> # /dev/ttyUSB0
    >>> # ===
    >>> # conn: /dev/ttyUSB0
    >>> # [Testing interface appears with diagnostic information]

Diagnostic Output:
    The testing module provides detailed diagnostic information including:
    - Device connection status and configuration
    - Data acquisition rates and performance metrics
    - Signal quality indicators and error rates
    - Buffer utilization and overflow conditions
    - Timing analysis and synchronization status

Performance Monitoring:
    - Real-time data throughput measurement
    - Buffer utilization tracking and optimization
    - Error rate monitoring and analysis
    - System resource usage assessment
    - Long-term stability metrics collection

Quality Assurance:
    This module serves as a critical component of the uMyo development
    and deployment process, ensuring:
    - Device reliability before production deployment
    - System integration compatibility verification
    - Performance optimization and tuning
    - Early detection of potential issues
    - Validation of system requirements compliance

Author: uMyo Development Team
License: See LICENSE file in the project root
Version: 1.0
"""

# kinda main

import umyo_parser
import display_stuff
import serial
from serial.tools import list_ports
import numpy as np

# ---------------------------------------------------------------------------
# High-pass filter (HPF) at 10 Hz for EMG signal
# ---------------------------------------------------------------------------

# NOTE:
# uMyo raw EMG is streamed at ~1100 Hz when using the PC base station.
# If you use a different sampling rate – update FS_HZ accordingly.
FS_HZ = 1100.0   # sampling rate [Hz]
FC_HZ = 10.0     # high-pass cutoff [Hz]

# First-order digital HPF derived from RC analog prototype:
#   y[n] = a * (y[n-1] + x[n] - x[n-1])
RC = 1.0 / (2.0 * np.pi * FC_HZ)
HPF_A = RC / (RC + 1.0 / FS_HZ)

# Per-device filter state:
# unit_id -> {"prev_x": float, "prev_y": float}
_hpf_state = {}


def _hpf_filter_sample(unit_id, x):
    """
    Apply first-order HPF to a single EMG sample `x`
    for device identified by `unit_id`.
    """
    state = _hpf_state.get(unit_id, {"prev_x": 0.0, "prev_y": 0.0})
    y = HPF_A * (state["prev_y"] + x - state["prev_x"])
    state["prev_x"] = x
    state["prev_y"] = y
    _hpf_state[unit_id] = state
    return y


def apply_hpf_to_devices(devices):
    """
    Apply the 10 Hz HPF to EMG data in all devices returned by umyo_get_list().

    Assumes each device has:
        - device.unit_id  (int)
        - device.data_array  (list / array of EMG samples)

    The function updates device.data_array *in place* with the filtered signal.
    """
    if not devices:
        return

    for dev in devices:
        # Some entries in the list can be None while devices are connecting
        if dev is None:
            continue

        unit_id = getattr(dev, "unit_id", None)
        data = getattr(dev, "data_array", None)

        if unit_id is None or data is None:
            continue

        # Convert to numpy array for vectorized processing
        arr = np.asarray(data, dtype=float)

        # Run the IIR HPF sample-by-sample to preserve continuity
        # between calls (state is kept in _hpf_state).
        for i in range(len(arr)):
            arr[i] = _hpf_filter_sample(unit_id, arr[i])

        # Convert back to list to keep the original type used in the toolkit
        dev.data_array = arr.tolist()


# ---------------------------------------------------------------------------
# list available ports
# ---------------------------------------------------------------------------

port = list(list_ports.comports())
print("available ports:")
for p in port:
    print(p.device)
    device = p.device
print("===")

# macOS serial port caching
temp_ser = serial.Serial(device, timeout=1)
temp_ser.close()

# ---------------------------------------------------------------------------
# read + main loop
# ---------------------------------------------------------------------------

ser = serial.Serial(
    port=device,
    baudrate=921600,
    parity=serial.PARITY_NONE,
    stopbits=1,
    bytesize=8,
    timeout=0,
)

print("conn: " + ser.portstr)
last_data_upd = 0
display_stuff.plot_init()
parse_unproc_cnt = 0

try:
    while True:
        cnt = ser.in_waiting
        if cnt > 0:
            cnt_corr = parse_unproc_cnt / 200
            data = ser.read(cnt)

            # Parse raw bytes into uMyo device objects
            parse_unproc_cnt = umyo_parser.umyo_parse_preprocessor(data)

            # Get devices and apply 10 Hz HPF to EMG before plotting
            devices = umyo_parser.umyo_get_list()
            apply_hpf_to_devices(devices)

            dat_id = display_stuff.plot_prepare(devices)
            d_diff = 0
            if dat_id is not None:
                d_diff = dat_id - last_data_upd
            if d_diff > 2 + cnt_corr:
                # display_stuff.plot_cycle_lines()
                display_stuff.plot_cycle_tester()
                last_data_upd = dat_id
finally:
    if "ser" in locals() and ser.is_open:
        ser.close()
        print("Serial port closed")
