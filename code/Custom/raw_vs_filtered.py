import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, iirnotch

# --- CONFIG ---
FILE_NAME = "session_fist_001.csv"  # Put one of your files here
FS = 1100  # Your sampling rate

# --- FILTERS ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filter_emg(data, fs):
    # We will be AGGRESSIVE here.
    # Standard is 20Hz. Let's try 40Hz to kill ALL motion artifacts.
    b, a = butter_bandpass(40, 450, fs, order=4)
    y = lfilter(b, a, data)
    return y

# --- PLOTTING ---
df = pd.read_csv(FILE_NAME)

# Create the filtered version
df['emg_filtered'] = filter_emg(df['emg'], FS)

# Map labels for coloring
LABEL_MAP = {'none': 'at_rest', 'ID_REST_1': 'at_rest', 'ID_REST_2': 'at_rest'}
df['gesture_label'] = df['gesture_label'].replace(LABEL_MAP)
colors = {'at_rest': 'blue', 'fist': 'red'}
colors_mapped = [colors.get(l, 'gray') for l in df['gesture_label']]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot 1: Raw
ax1.set_title("Raw Signal (Input)")
ax1.plot(df['t_rel_sec'], df['emg'], color='black', alpha=0.3)
ax1.scatter(df['t_rel_sec'], df['emg'], c=colors_mapped, s=10, alpha=0.8)
ax1.set_ylabel("Amplitude")

# Plot 2: Filtered (What the ML sees)
ax2.set_title("Filtered Signal (Frequency > 40Hz)")
ax2.plot(df['t_rel_sec'], df['emg_filtered'], color='black', alpha=0.3)
ax2.scatter(df['t_rel_sec'], df['emg_filtered'], c=colors_mapped, s=10, alpha=0.8)
ax2.set_ylabel("Amplitude")
ax2.set_xlabel("Time (s)")

plt.show()