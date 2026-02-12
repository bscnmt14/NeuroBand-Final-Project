import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# שם הקובץ שההקלטה יצרה
df = pd.read_csv("fist_007.csv")

# נתמקד לדוגמה ב-device הראשון (device_index == 0)
df0 = df[df["device_index"] == 0].copy()

# וקטור הזמן היחסי ו-EMG
t = df0["t_rel_sec"].to_numpy()
emg = df0["emg"].to_numpy()

# ננקה NaN אם יש
mask = ~np.isnan(emg)
t = t[mask]
emg = emg[mask]

# נחשב משך ו-fs משוער
duration = t.max() - t.min()
fs = len(emg) / duration
print("Estimated fs:", fs, "Hz")

# נסיר ממוצע ונחשב FFT
emg_detrend = emg - emg.mean()
Y = np.fft.rfft(emg_detrend)
f = np.fft.rfftfreq(len(emg_detrend), d=1/fs)

# נמצא את התדר הדומיננטי
idx_peak = np.argmax(np.abs(Y))
f_noise = f[idx_peak]
print("Dominant noise frequency:", f_noise, "Hz")

# אפשר גם להציג ספקטרום
plt.figure()
plt.plot(f, np.abs(Y))
plt.xlim(0, 200)  # לתחום EMG טיפוסי
plt.xlabel("Frequency [Hz]")
plt.ylabel("|Y(f)|")
plt.grid(True)
plt.show()