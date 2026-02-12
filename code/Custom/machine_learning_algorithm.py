import pandas as pd
import numpy as np
import glob
import joblib  # For saving the model
from scipy.signal import butter, lfilter, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION ---
FS = 1100  # Sampling Rate (Hz)
WINDOW_SIZE_MS = 500  # Window size (500ms smooths over small gaps in signal)
OVERLAP = 0.5  # 50% overlap between windows
THRESHOLD = 0.35  # Sensitivity knob (Lower = easier to trigger 'fist')

# --- LABEL MAPPING ---
# Merges different labels into standard classes
LABEL_MAP = {
    'none': 'at_rest',
    'ID_REST_1': 'at_rest',
    'ID_REST_2': 'at_rest',
    # Add any other labels you want to fix here
}


# --- 2. SIGNAL PROCESSING FUNCTIONS ---

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def notch_filter(data, freq, fs, q=30):
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = iirnotch(freq, q)
    return lfilter(b, a, data)


def filter_emg(data, fs):
    # 1. Aggressive Bandpass (40Hz - 450Hz)
    # We use 40Hz to remove the "wavy" motion artifacts visible in your plots.
    low_cut = 40
    high_cut = 450

    # Safety check: Ensure high_cut is below Nyquist frequency
    if high_cut >= 0.5 * fs:
        high_cut = (0.5 * fs) - 10

    b, a = butter_bandpass(low_cut, high_cut, fs, order=4)
    y = lfilter(b, a, data)

    # 2. Notch filter 50Hz (Removes electrical hum)
    y = notch_filter(y, 50, fs)
    return y


def extract_features(window_data):
    """
    Extracts 5 key features from the signal window.
    """
    # 1. Root Mean Square (Loudness)
    rms = np.sqrt(np.mean(window_data ** 2))

    # 2. Standard Deviation
    std = np.std(window_data)

    # 3. Maximum Amplitude
    max_amp = np.max(np.abs(window_data))

    # 4. Zero Crossing Rate (Frequency / Texture)
    zero_crossings = ((window_data[:-1] * window_data[1:]) < 0).sum()

    # 5. Waveform Length (Complexity/Effort)
    waveform_length = np.sum(np.abs(np.diff(window_data)))

    return [rms, std, max_amp, zero_crossings, waveform_length]


def create_windows(df, window_size_samples, step_size_samples):
    X = []
    y = []

    for start in range(0, len(df) - window_size_samples, step_size_samples):
        end = start + window_size_samples
        window = df.iloc[start:end]

        # Get processed EMG
        sig_chunk = window['emg_filtered'].values

        # Extract features
        features = extract_features(sig_chunk)
        X.append(features)

        # Get Label (Most common label in this window)
        label = window['gesture_label'].mode()[0]
        y.append(label)

    return np.array(X), np.array(y)


# --- 3. MAIN DATA PIPELINE ---

all_X = []
all_y = []
files = glob.glob("session_fist_0**.csv")
print(f"Found {len(files)} CSV files.")
print(f"Sampling Rate: {FS} Hz | Window: {WINDOW_SIZE_MS} ms")

for file in files:
    print(f"Processing {file}...")
    try:
        df = pd.read_csv(file)

        # 1. Fix Labels
        if 'gesture_label' in df.columns:
            df['gesture_label'] = df['gesture_label'].replace(LABEL_MAP)
        else:
            print(f"Skipping {file}: No 'gesture_label' column found.")
            continue

        # 2. Filter Signal
        df['emg_filtered'] = filter_emg(df['emg'], FS)

        # 3. Calculate Window parameters
        win_samples = int(FS * (WINDOW_SIZE_MS / 1000))
        step_samples = int(win_samples * (1 - OVERLAP))

        # 4. Windowing
        X_chunk, y_chunk = create_windows(df, win_samples, step_samples)

        if len(X_chunk) > 0:
            all_X.append(X_chunk)
            all_y.append(y_chunk)

    except Exception as e:
        print(f"Error reading {file}: {e}")

# --- 4. TRAINING ---

if len(all_X) > 0:
    # Concatenate all files
    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"\nTotal Data Shape: {X.shape}")
    print(f"Classes found: {np.unique(y)}")

    # Normalization (Crucial for amplitude consistency)
    print("Normalizing data...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # --- 5. EVALUATION WITH TUNED THRESHOLD ---

    # Get probability scores
    y_probs = clf.predict_proba(X_test)

    # Find index of 'fist' class
    if 'fist' in clf.classes_:
        fist_index = list(clf.classes_).index('fist')

        # Apply custom threshold
        y_pred_tuned = np.where(y_probs[:, fist_index] > THRESHOLD, 'fist', 'at_rest')

        print(f"\n--- Model Evaluation (Threshold: {THRESHOLD}) ---")
        print(classification_report(y_test, y_pred_tuned))
    else:
        print("\n'fist' class not found in data. Standard evaluation:")
        print(classification_report(y_test, clf.predict(X_test)))

    # --- 6. SAVE MODEL ---
    print("Saving model and scaler to disk...")
    joblib.dump(clf, 'emg_model.pkl')
    joblib.dump(scaler, 'emg_scaler.pkl')
    print("Done! Files saved as 'emg_model.pkl' and 'emg_scaler.pkl'")

else:
    print("No valid data processed. Please check your CSV files.")