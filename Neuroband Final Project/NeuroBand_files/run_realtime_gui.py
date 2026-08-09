"""Protected entry point for launching the NeuroBand realtime interface.

The wrapper configures Windows process behavior, prevents duplicate application
instances, and delegates construction of the Qt application to the main GUI module.

"""

import ctypes
import sys
from pathlib import Path

from PySide6 import QtCore

from realtime_gesture_gui import main


def configure_windows_interactive_scheduling() -> None:
    """Perform the configure windows interactive scheduling operation used by the run realtime gui workflow."""
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    process = kernel32.GetCurrentProcess()
    # Keep the app responsive when another program owns the foreground window.
    # ABOVE_NORMAL avoids the starvation risks of HIGH/REALTIME priority.
    kernel32.SetPriorityClass(process, 0x00008000)
    ctypes.windll.winmm.timeBeginPeriod(1)

    class ProcessPowerThrottlingState(ctypes.Structure):
        """Represent the ProcessPowerThrottlingState component and keep its related state and behavior together."""
        _fields_ = [
            ("Version", ctypes.c_ulong),
            ("ControlMask", ctypes.c_ulong),
            ("StateMask", ctypes.c_ulong),
        ]

    state = ProcessPowerThrottlingState(
        Version=1,
        ControlMask=0x1,
        StateMask=0x0,
    )
    # ProcessPowerThrottling = 4; disable execution-speed throttling/EcoQoS.
    kernel32.SetProcessInformation(process, 4, ctypes.byref(state), ctypes.sizeof(state))


if __name__ == "__main__":
    configure_windows_interactive_scheduling()
    lock_path = Path(QtCore.QDir.tempPath()) / "umyo_realtime_gesture_gui.lock"
    instance_lock = QtCore.QLockFile(str(lock_path))
    instance_lock.setStaleLockTime(3000)
    if not instance_lock.tryLock(100):
        print("uMyo Realtime Gesture Recognition is already running.")
        raise SystemExit(2)
    raise SystemExit(main())
