import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch

# --- CONFIG ---
NEW_FILE_NAME = "session_fist_test.csv"  # <--- PUT YOUR NEW FILE NAME HERE
FS = 1100
WINDOW_SIZE_MS = 500
OVERLAP = 0.5

# --- LOAD BRAIN ---
print("Loading model...")
clf = joblib.load('my_model.pkl')
scaler = joblib.load('my_scaler.pkl')


# --- PREPROCESSING (Must match training exactly) ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filter_emg(data, fs):
    # Aggressive 40Hz filter (Same as training)
    low_cut = 40
    high_cut = 450
    if high_cut >= 0.5 * fs: high_cut = (0.5 * fs) - 10
    b, a = butter_bandpass(low_cut, high_cut, fs, order=4)
    y = lfilter(b, a, data)
    return y


def extract_features(window_data):
    # Same features as training
    rms = np.sqrt(np.mean(window_data ** 2))
    std = np.std(window_data)
    max_amp = np.max(np.abs(window_data))
    zero_crossings = ((window_data[:-1] * window_data[1:]) < 0).sum()
    waveform_length = np.sum(np.abs(np.diff(window_data)))
    return [rms, std, max_amp, zero_crossings, waveform_length]


# --- PROCESSING LOOP ---
print(f"Processing {NEW_FILE_NAME}...")
df = pd.read_csv(NEW_FILE_NAME)

# Filter
df['emg_filtered'] = filter_emg(df['emg'], FS)

# Windowing
win_samples = int(FS * (WINDOW_SIZE_MS / 1000))
step_samples = int(win_samples * (1 - OVERLAP))

X_new = []
time_steps = []

for start in range(0, len(df) - win_samples, step_samples):
    end = start + win_samples
    window = df.iloc[start:end]

    # Extract features
    features = extract_features(window['emg_filtered'].values)
    X_new.append(features)

    # Keep track of time for plotting (middle of the window)
    mid_time = window['t_rel_sec'].mean()
    time_steps.append(mid_time)

X_new = np.array(X_new)

# --- PREDICTION ---
# 1. Scale using the saved scaler (CRITICAL!)
X_scaled = scaler.transform(X_new)

# 2. Predict
predictions = clf.predict(X_scaled)

# --- VISUALIZATION ---
# Plot the raw signal with predictions overlaid
plt.figure(figsize=(15, 6))
plt.title(f"Model Performance on New Data: {NEW_FILE_NAME}")

# Plot raw signal (gray)
plt.plot(df['t_rel_sec'], df['emg'], color='gray', alpha=0.5, label='Raw EMG')

# Overlay predictions
# We create a scatter plot. Red = Fist, Blue = Rest
colors = ['red' if p == 'fist' else 'blue' for p in predictions]

# We plot the prediction markers at the timestamp of the window
plt.scatter(time_steps, [df['emg'].max()] * len(time_steps), c=colors, marker='s', s=50, label='Prediction')

# Custom Legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='gray', label='Signal'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', label='Pred: Rest'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', label='Pred: Fist')
]
plt.legend(handles=legend_elements)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()