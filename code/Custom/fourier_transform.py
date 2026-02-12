import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal


def plot_emg_fft_limited(csv_file_path, sampling_rate=None, cutoff_freq=20, max_freq=600):
    """
    Reads a CSV, applies HPF, and plots FFT up to a specified maximum frequency.

    Parameters:
    - csv_file_path: str, path to the CSV file.
    - sampling_rate: float or None. If None, it is calculated from the 'timestamp' column.
    - cutoff_freq: float, High-Pass Filter cutoff in Hz (default 20).
    - max_freq: float, Maximum frequency to plot on x-axis (default 600).
    """
    try:
        df = pd.read_csv(csv_file_path)

        # 1. Determine Sampling Rate
        if sampling_rate is None:
            if 'timestamp' in df.columns:
                # Calculate median time difference to estimate fs
                dt = df['timestamp'].diff().median()
                sampling_rate = 1 / dt
                print(f"Calculated sampling rate: {sampling_rate:.2f} Hz")
            else:
                print("Error: 'timestamp' column not found and no sampling_rate provided.")
                return

        nyquist = 0.5 * sampling_rate

        # 2. Check Nyquist Limit
        if max_freq > nyquist:
            print(f"Warning: Requested max_freq ({max_freq} Hz) exceeds Nyquist limit ({nyquist:.1f} Hz).")
            print(f"Plotting up to {nyquist:.1f} Hz instead.")
            plot_limit = nyquist
        else:
            plot_limit = max_freq

        # 3. Process EMG Data
        emg_columns = [f'emg_{i}' for i in range(8)]
        if not set(emg_columns).issubset(df.columns):
            print("Error: EMG columns missing.")
            return

        avg_emg = df[emg_columns].mean(axis=1).to_numpy()

        # 4. Apply Filter
        if nyquist > cutoff_freq:
            norm_cutoff = cutoff_freq / nyquist
            b, a = signal.butter(4, norm_cutoff, btype='high', analog=False)
            filtered_signal = signal.filtfilt(b, a, avg_emg)
        else:
            print("Warning: Cutoff frequency is too high for this sampling rate. Skipping filter.")
            filtered_signal = avg_emg

        # 5. Compute FFT
        N = len(filtered_signal)
        yf = fft(filtered_signal)
        xf = fftfreq(N, 1 / sampling_rate)

        # 6. Plot
        plt.figure(figsize=(10, 6))

        # Plot positive frequencies only
        pos_mask = xf[:N // 2] >= 0
        freqs = xf[:N // 2][pos_mask]
        mags = 2.0 / N * np.abs(yf[:N // 2][pos_mask])

        plt.plot(freqs, mags)
        plt.xlim(0, plot_limit)

        plt.title(f'FFT of Average EMG (HPF {cutoff_freq}Hz, Fs {int(sampling_rate)}Hz)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage:
plot_emg_fft_limited('ED7A78C8, base_pos, Frequency_90_Hz, 12-36-47, 11-01-26.csv', max_freq=600)