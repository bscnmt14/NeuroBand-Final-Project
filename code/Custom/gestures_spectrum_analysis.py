import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
CSV_FILE_PATH = 'session_fist_013.csv'    # <--- CHANGE THIS to your CSV file name
SIGNAL_COLUMN_NAME = 'emg'     # <--- CHANGE THIS to your EMG column name
TIMESTAMP_COLUMN_NAME = 't_rel_sec'      # <--- CHANGE THIS to your timestamp column name

FREQUENCY_BIN_COLUMNS = [
    'sp0',  # e.g., 'Alpha_Power'
    'sp1',  # e.g., 'Beta_Power'
    'sp2',  # e.g., 'Gamma_Power'
    'sp3'   # e.g., 'Low_Frequency'
]
if len(FREQUENCY_BIN_COLUMNS) != 4:
    raise ValueError("Please ensure exactly 4 column names are listed in FREQUENCY_BIN_COLUMNS.")

# --- 2. DATA LOADING ---
try:
    df = pd.read_csv(CSV_FILE_PATH)

    # Clean up NaNs across all required columns
    all_cols = [SIGNAL_COLUMN_NAME, TIMESTAMP_COLUMN_NAME] + FREQUENCY_BIN_COLUMNS
    df_cleaned = df[all_cols].apply(pd.to_numeric, errors='coerce').dropna()

    emg_signal = df_cleaned[SIGNAL_COLUMN_NAME].values
    time_vector = df_cleaned[TIMESTAMP_COLUMN_NAME].values
    bin_data = df_cleaned[FREQUENCY_BIN_COLUMNS]

    print(f"Data loaded successfully. Total Samples: {len(emg_signal)}")

except Exception as e:
    print(f"Error during data loading: {e}. Check file path and column names.")
    exit()

# ----------------------------------------------------
## FIGURE 1: TIME DOMAIN GRAPH
# ----------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(time_vector, emg_signal, color='darkblue', linewidth=0.7)
plt.title('Figure 1: Raw EMG Signal (Time Domain)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (Volts or mV)')
plt.grid(True, alpha=0.6)

# ----------------------------------------------------
## FIGURE 2: FREQUENCY BIN GRAPHS (4 Subplots)
# ----------------------------------------------------
# Create a 2x2 grid of subplots
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Figure 2: Frequency Bin Power Over Time (Individual Plots)', fontsize=16)

# Flatten the 2x2 array of axes to easily iterate through them
axes_flat = axes.flatten()

# Iterate through the 4 columns and plot each one in a separate subplot
for i, col in enumerate(FREQUENCY_BIN_COLUMNS):
    ax = axes_flat[i]

    # Plot the frequency bin power over time
    ax.plot(time_vector, bin_data[col], linewidth=1.5, color=f'C{i}')  # C{i} assigns a distinct color

    # Set titles and labels
    ax.set_title(f'Power of {col}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Power (a.u.)')
    ax.grid(True, alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle

# Show both figures
plt.show()

# --- 2. DATA LOADING & Fs CALCULATION (Reusing robust logic) ---
try:
    df = pd.read_csv(CSV_FILE_PATH)

    emg_signal = pd.to_numeric(df[SIGNAL_COLUMN_NAME], errors='coerce').values
    time_vector = pd.to_numeric(df[TIMESTAMP_COLUMN_NAME], errors='coerce').values

    valid_mask = ~np.isnan(emg_signal) & ~np.isnan(time_vector)
    emg_signal = emg_signal[valid_mask]
    time_vector = time_vector[valid_mask]

    N = len(emg_signal)
    total_duration = time_vector[-1] - time_vector[0]
    SAMPLING_RATE_HZ = (N - 1) / total_duration if total_duration > 0 else 433.2

    print(f"Fs calculated: {SAMPLING_RATE_HZ:.2f} Hz")

except Exception as e:
    print(f"Error: {e}")
    exit()

# ----------------------------------------------------
## FIGURE 3: SPECTROGRAM
# ----------------------------------------------------

# Parameters derived from Fs ~433Hz, aiming for ~6.8Hz resolution
WINDOW_SIZE = 64
OVERLAP = 32 # 50% overlap

plt.figure(figsize=(10, 6))
# The plt.specgram function calculates the FFT segments and plots the magnitude
power, freqs, t_time, im = plt.specgram(
    emg_signal,
    Fs=SAMPLING_RATE_HZ,
    NFFT=WINDOW_SIZE,          # Use NFFT to define the window size
    noverlap=OVERLAP,          # Overlap
    cmap='viridis',
)

# Configuration and Labeling
plt.title(f'Figure 3: Spectrogram (Time-Frequency Plot)')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')

# Use the actual time vector to set the extent of the X-axis
plt.xlim(time_vector[0], time_vector[-1])
plt.ylim(0, SAMPLING_RATE_HZ / 2)  # Plot only up to the Nyquist Frequency

# Add a color bar to show the magnitude scale
cbar = plt.colorbar(im)
cbar.set_label('Magnitude (dB or a.u.)')

plt.tight_layout()
plt.show()