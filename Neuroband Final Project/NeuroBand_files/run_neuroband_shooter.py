"""Protected entry point for launching the NeuroBand shooter application.

The wrapper configures Windows process behavior and uses a lock file to prevent
multiple game instances from competing for the same serial stream and input state.

"""

import ctypes
import sys
from pathlib import Path

from PySide6 import QtCore

from neuroband_shooter import main


def configure_windows_scheduling() -> None:
    """Perform the configure windows scheduling operation used by the run neuroband shooter workflow."""
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    process = kernel32.GetCurrentProcess()
    kernel32.SetPriorityClass(process, 0x00008000)
    ctypes.windll.winmm.timeBeginPeriod(1)


if __name__ == "__main__":
    configure_windows_scheduling()
    lock_path = Path(QtCore.QDir.tempPath()) / "neuroband_shooter.lock"
    instance_lock = QtCore.QLockFile(str(lock_path))
    instance_lock.setStaleLockTime(3000)
    if not instance_lock.tryLock(100):
        print("NeuroBand shooter is already running.")
        raise SystemExit(2)
    raise SystemExit(main())
