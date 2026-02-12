import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from scipy.signal import butter, iirnotch, filtfilt

# --- CONFIGURATION ---
frequency_data_path = Path(
    r"C:\Users\Nadav\OneDrive - Afeka College Of Engineering\uMyo Python Project - test\Frequency Response Experiment\Recordings From custom code")

#frequency_data_path = Path(r"C:\Users\Nadav\OneDrive - Afeka College Of Engineering\uMyo Python Project - test\Frequency Response Experiment\New folder\up to 100 Hz")

# Noise floor file (Optional)
noise_floor_filename = "ED7A78C8, base_pos, Frequency_NOISE, 15-25-23, 11-01-26.csv"

fs = 1200  # Sampling frequency


# --- 1. SETUP FILTERS ---
def get_filters(fs):
    nyq = 0.5 * fs
    b_high, a_high = butter(4, 20 / nyq, btype='high')
    b_notch, a_notch = iirnotch(50.0, 30.0, fs)
    b_notch2, a_notch2 = iirnotch(100.0, 30.0, fs)
    return (b_high, a_high), (b_notch, a_notch), (b_notch2, a_notch2)


(b_high, a_high), (b_notch, a_notch), (b_notch2, a_notch2) = get_filters(fs)


# --- HELPER: CALCULATE RMS ---
def calc_rms(signal_data):
    # Filter the reconstructed time-domain signal
    sig_filt = filtfilt(b_notch, a_notch, signal_data)
    # sig_filt = filtfilt(b_notch2, a_notch2, sig_filt)
    sig_filt = filtfilt(b_high, a_high, signal_data)

    # Trim start/end artifacts
    if len(sig_filt) > 200:
        clean_sig = sig_filt[100:-100]
    else:
        clean_sig = sig_filt

    return np.sqrt(np.mean(clean_sig ** 2))


# --- 2. MAIN LOOP ---
results = []
files_list = list(frequency_data_path.glob('*.csv'))

print(f"Processing {len(files_list)} files with fs={fs}Hz...")

for file in files_list:
    if file.name == noise_floor_filename: continue

    # --- UNIVERSAL FREQUENCY PARSING ---
    # Looks for any number followed by "Hz".
    # Works for: "50 Hz.csv", "Frequency_50_Hz.csv", "Frequency - 50 Hz.csv"
    match = re.search(r'(\d+\.?\d*)\s*_?Hz', file.name, re.IGNORECASE)

    if not match:
        print(f"Skipping {file.name}: No frequency found in filename.")
        continue

    freq = float(match.group(1))

    try:
        df = pd.read_csv(file)

        # --- A. DETECT & RECREATE TIME DOMAIN SIGNAL ---
        # 1. Try OEM format first (emg_ch_0), then Custom format (emg_0)
        if 'emg_ch_0' in df.columns:
            emg_cols = [f'emg_ch_{i}' for i in range(8)]
        else:
            emg_cols = [f'emg_{i}' for i in range(8)]

        valid_cols = [c for c in emg_cols if c in df.columns]

        if valid_cols:
            # 2. Extract as Matrix (Rows x 8)
            matrix_data = df[valid_cols].values

            # 3. FLATTEN into 1D Array (Time Domain Reconstruction)
            reconstructed_signal = matrix_data.flatten()

            rms_val = calc_rms(reconstructed_signal)
        else:
            print(f"No EMG columns found in {file.name}")
            rms_val = 1e-9

        # --- B. DETECT SPECTRUM BINS ---
        sp_vals = {}
        for i in range(4):
            # Check for OEM name (spectrum_0) or Custom name (sp0)
            col_oem = f'spectrum_{i}'
            col_custom = f'sp{i}'

            if col_oem in df.columns:
                val = df[col_oem].mean()
            elif col_custom in df.columns:
                val = df[col_custom].mean()
            else:
                val = 1e-9

            # Store with standardized keys for plotting
            sp_vals[f'sp{i}'] = val

        entry = {'freq': freq, 'emg': rms_val}
        entry.update(sp_vals)
        results.append(entry)

    except Exception as e:
        print(f"Error processing {file.name}: {e}")

# --- 3. PLOTTING ---
res_df = pd.DataFrame(results)

if not res_df.empty:
    res_df = res_df.sort_values(by='freq')


    def to_db(series):
        max_val = series.max()
        if max_val == 0: return series * 0
        return 20 * np.log10((series / max_val) + 1e-12)


    res_df['emg_db'] = to_db(res_df['emg'])

    # --- PLOT 1: Reconstructed Signal FR ---
    plt.figure(figsize=(10, 8))

    # Main Signal
    plt.subplot(2, 1, 1)
    plt.plot(res_df['freq'], res_df['emg_db'], 'b-o', linewidth=2, label='Reconstructed Signal (1200Hz)')

    plt.title(f'Frequency Response of Reconstructed Signal | fs={fs}Hz')
    plt.ylabel('Magnitude [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.axvline(x=50, color='gray', linestyle=':', label='Notch 50Hz')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()

    # --- PLOT 2: Spectrum Bins ---
    plt.subplot(2, 1, 2)
    # Calculate global max for spectrum normalization
    max_sp = max(res_df['sp0'].max(), res_df['sp1'].max(), res_df['sp2'].max(), res_df['sp3'].max())

    plt.plot(res_df['freq'], 20 * np.log10((res_df['sp0'] / max_sp) + 1e-12), 'o-', label='SP0')
    plt.plot(res_df['freq'], 20 * np.log10((res_df['sp1'] / max_sp) + 1e-12), 's-', label='SP1')
    plt.plot(res_df['freq'], 20 * np.log10((res_df['sp2'] / max_sp) + 1e-12), '^-', label='SP2')
    plt.plot(res_df['freq'], 20 * np.log10((res_df['sp3'] / max_sp) + 1e-12), 'x-', label='SP3')

    plt.title('Internal Spectrum Bins Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()

    plt.tight_layout()
    plt.show()

else:
    print("No valid data found.")