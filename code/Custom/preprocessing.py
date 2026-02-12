import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import scipy as sci
from scipy import signal
import re
from scipy.signal import filtfilt, lfilter

#Reading the data from the .csv files:
folder_path = Path(r"C:\Users\Nadav\OneDrive - Afeka College Of Engineering\uMyo Python Project - test\Data\מיקום אלקטרודות\1")
files_list = list(folder_path.glob('*.csv'))

for file in files_list:
    print(f"Processing {file.name}...")
    # Read the specific file in the current iteration
    data = pd.read_csv(file)

    # Extract the signal
    emg = data['emg']
    sp0 = data['sp0']
    sp1 = data['sp1']
    sp2 = data['sp2']
    sp3 = data['sp3']

    sampling_frequency = 1100
    signal_length = len(emg)
    time_axis = np.linspace(0, signal_length / sampling_frequency, signal_length)
    frequency_axis = np.linspace(0, sampling_frequency, signal_length)

    # a.
    plt.plot(time_axis, emg)
    plt.title('EMG signal')
    plt.suptitle(f'EMG data for {file.name}', fontsize=16)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [μV]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plt.savefig(f"{file.stem}_EMG_Data.png")
    # plt.close()


target_columns = ['sp0', 'sp1', 'sp2', 'sp3']

for file in files_list:
    try:
        print(f"Processing {file.name}...")

        # Read the dataframe
        data = pd.read_csv(file)

        # Create a figure with 4 subplots (one for each sp channel)
        fig, axes = plt.subplots(nrows=len(target_columns), ncols=1, figsize=(10, 12), sharex=True)
        fig.suptitle(f'Spectrograms for {file.name}', fontsize=16)

        for i, col in enumerate(target_columns):
            if col in data.columns:
                # Extract the signal
                sig = data[col].values

                # Check if signal contains valid numbers (handling NaNs)
                if np.isnan(sig).any():
                    sig = np.nan_to_num(sig)

                # Compute Spectrogram
                # nperseg: Length of each segment.
                # Increase nperseg for better frequency resolution, decrease for better time resolution.
                f, t, Sxx = signal.spectrogram(sig, sampling_frequency, nperseg=256)

                # Plot on the specific subplot
                ax = axes[i]
                # Use Log scale for intensity (dB)
                img = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')

                ax.set_ylabel('Frequency [Hz]')
                ax.set_title(f'Channel: {col}')

                # Add colorbar
                cbar = fig.colorbar(img, ax=ax, orientation='vertical')
                cbar.set_label('Intensity [dB]')
            else:
                print(f"Warning: Column {col} not found in {file.name}")

        axes[-1].set_xlabel('Time [sec]')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
        plt.show()

        # plt.savefig(f"{file.stem}_spectrogram.png")
        # plt.close()

    except Exception as e:
        print(f"Error processing {file.name}: {e}")
################--Function declaration--#######################

################--End of Function declaration--################

################--Function declaration--#######################
def noise_detect(data):
    data_length = len(data)
    sampling_frequency = 1100
    frequency_axis = np.linspace(0, sampling_frequency, data_length)
    normalized_dft = np.fft.fft(data) / (data_length / 2)
    signal_dft = 20 * np.log10(np.abs(normalized_dft))

    plt.plot(frequency_axis, signal_dft)
    plt.title('Frequency spectrum of the signal')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    peaks, props = sci.signal.find_peaks(signal_dft, height=np.max(signal_dft) * 0.1)

    #peak_frequencies = freqs[peaks]
    #peak_values = signal_dft[peaks]

    return
################--End of Function declaration--################


################--Function declaration--#######################
def notch_filter(input, stop_freq, sampling_frequency = 1100, plot = False):
    """
    Notch filter difference equation.
    Stop frequency = 8.5 Hz

    Parameters
    ----------
    signal : array-like
        Input signal.

    Returns
    -------
    output : np.ndarray
        Filtered signal.
    """
    stop_frequency = 2*np.pi*stop_freq/1100
    N = len(input)
    a = np.cos(stop_frequency)
    b = 0.9**2

    output = np.zeros(N)

    #Initial conditions:
    output[0] = input[0]
    output[1] = input[1] -2*a*input[0]+1.8*a*output[0]

    for i in range(2, N):
        output[i] = input[i] - 2*a*input[i-1] + input[i-2] + 1.8*a*output[i-1] - b*output[i-2]

    if plot == True:
        plt.figure()
        plt.plot(time_axis, output)
        plt.title('Filtered signal - %f hz' % stop_freq)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return output
################--End of Function declaration--################


# --- 1. Corrected 'desired_filter' function ---
# I've added the scaling factors (cutoff/pi) so the gain is 1.0 (0 dB)
def desired_filter_corrected(filter_type, cutoff_frequency, pass_frequency=None, stop_frequency=None, order=100,
                             fs=None):
    N = order + 1
    M = order // 2
    n = np.arange(N)
    hd = np.zeros(N)

    # Helper: Convert Hz to normalized radians (0 to pi)
    def to_rad(f):
        if f is None: return None
        return 2 * np.pi * f / fs if fs else f

    fc = to_rad(cutoff_frequency)
    fp = to_rad(pass_frequency) if pass_frequency else None
    fs_ = to_rad(stop_frequency) if stop_frequency else None

    # Note: The scaling factor (omega / pi) is added before the sinc
    if filter_type.upper() == 'LPF':
        hd = (fc / np.pi) * np.sinc((n - M) * fc / np.pi)

    elif filter_type.upper() == 'HPF':
        hd = (fc / np.pi) * np.sinc((n - M) * fc / np.pi)
        hd = -hd
        hd[M] += 1

    elif filter_type.upper() == 'BPF':
        # BPF is LPF_high_cutoff - LPF_low_cutoff
        # Here: stop_frequency is the Upper Cutoff, pass_frequency is Lower Cutoff
        if fp is None or fs_ is None: raise ValueError("BPF requires two freqs")
        hd = (fs_ / np.pi) * np.sinc((n - M) * fs_ / np.pi) - \
             (fp / np.pi) * np.sinc((n - M) * fp / np.pi)

    elif filter_type.upper() == 'BSF':
        # BSF (Notch) is 1 - BPF
        if fp is None or fs_ is None: raise ValueError("BSF requires two freqs")
        hd = (fs_ / np.pi) * np.sinc((n - M) * fs_ / np.pi) - \
             (fp / np.pi) * np.sinc((n - M) * fp / np.pi)
        hd = -hd  # Invert BPF
        hd[M] += 1  # Add All-Pass

    return hd


# --- 2. Filter Design Wrapper (Updates to use corrected function) ---
def filter_design(filter_type='LPF', window_type='Hamming', order=100,
                  cutoff_frequency=None, pass_frequency=None, stop_frequency=None,
                  fs=None, label=""):
    # Use the corrected desired_filter
    hd = desired_filter_corrected(filter_type, cutoff_frequency, pass_frequency, stop_frequency, order, fs)

    # Create Window
    # (Assuming filter_window function from your snippet exists)
    # Re-implementing simplified window here for standalone running:
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(order + 1) / order)  # Hamming

    h = hd * w

    # Plotting
    N = len(h)
    freq_axis = np.linspace(0, fs / 2, 1000)  # Frequency in Hz
    H = np.fft.fft(h, 2000)  # High res FFT for smooth plot
    H = H[:1000]  # Keep positive half

    plt.plot(freq_axis, 20 * np.log10(np.abs(H) + 1e-12), label=label)
    return h


# --- 3. Generate Your Filters ---

# Settings (Based on typical EMG/Bio-signal needs)
fs = 836  # Sampling Rate (Hz)
order = 500  # Filter order (Higher = sharper cutoff, more delay)

plt.figure(figsize=(10, 6))

# A. LPF (Low Pass Filter)
# Keeps data below 150Hz (removes high freq noise)
h_lpf = filter_design(filter_type='LPF',
                      cutoff_frequency=10,
                      fs=fs, order=order, label='LPF (10Hz)')

h_hpf = filter_design(filter_type='HPF',
                      cutoff_frequency=10,
                      fs=fs, order=order, label='LPF (10Hz)')

# B. BPF (Band Pass Filter)
# Keeps data between 20Hz and 400Hz (Typical EMG range)
# pass_freq = Lower Cutoff, stop_freq = Upper Cutoff
h_bpf = filter_design(filter_type='BPF',
                      pass_frequency=20, stop_frequency=250,
                      fs=fs, order=order, label='BPF (20-250Hz)')

# C. Notch BSF (Band Stop Filter)
# Removes 50Hz power line hum (Narrow band 48Hz - 52Hz)
h_notch = filter_design(filter_type='BSF',
                        pass_frequency=6, stop_frequency=10,
                        fs=fs, order=500, label='Notch (50Hz)')

plt.title('Filter Frequency Responses')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.ylim(-60, 5)
plt.grid(True)
plt.legend()
plt.show()

for file in files_list:
    data = pd.read_csv(file)
    print(f"Filtering {file.name}...")

    # Create a plot for this file
    plt.figure(figsize=(12, 8))

    # Loop through the specific channels you want
    for i, col in enumerate(['emg']):
        if col in data.columns:
            raw_signal = data[col].values

            # --- APPLYING THE FILTERS ---

            # Step A: Apply Bandpass
            # h_bpf is the 'b' (numerator), 'a' (denominator) is 1 for FIR filters
            # We use filtfilt for zero phase distortion

            filtered_signal = filtfilt(h_hpf, 1.0, raw_signal)

            # Step B: Apply Notch (on top of the bandpassed signal)
            #filtered_signal = filtfilt(h_notch, 1.0, filtered_signal)

            # ----------------------------

            # Plot Raw vs Filtered for comparison
    #         plt.subplot(4, 1, i + 1)
    #         plt.plot(raw_signal, color='lightgray', label='Raw')
    #         plt.plot(filtered_signal, color='blue', linewidth=1, label='Filtered (BPF + Notch)')
    #         plt.title(f'Channel {col} of {file.name}')
    #         plt.legend(loc='upper right')
    #
    # plt.tight_layout()
    # plt.show()



    plt.plot(time_axis, filtered_signal)
    plt.title('EMG signal')
    plt.suptitle(f'Filtered EMG data for {file.name}', fontsize=16)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [μV]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plt.savefig(f"{file.stem}_EMG_Data.png")
    # plt.close()

    plt.plot(time_axis, abs(filtered_signal))
    plt.title('EMG signal')
    plt.suptitle(f'abs of Filtered EMG data for {file.name}', fontsize=16)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [μV]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

################--Function declaration--#######################
def amplitude_normalization(files_list):
    raw_signals_list = []

    # --- Step 1: Read all files and store raw data ---
    print("Loading files...")
    for file in files_list:
        data = pd.read_csv(file)

        # Get the actual signal array (not just the RMS)
        emg = data['emg'].values

        raw_signals_list.append(emg)
        print(f"Loaded {file.name}: {len(emg)} samples")

    # --- Step 2: Find Global Min/Max ---
    # We combine all data temporarily just to find the highest peak across ALL files
    all_data_combined = np.concatenate(raw_signals_list)

    global_min = np.min(all_data_combined)
    global_max = np.max(all_data_combined)

    print(f"\nGlobal Max found: {global_max}")
    print(f"Global Min found: {global_min}")

    # --- Step 3: Normalize each file based on Global limits ---
    normalized_signals = []

    for signal in raw_signals_list:
        # Avoid division by zero if flatline
        if global_max - global_min == 0:
            norm_sig = np.zeros_like(signal)
        else:
            # Apply Min-Max Normalization
            norm_sig = (signal - global_min) / (global_max - global_min)

        normalized_signals.append(norm_sig)

    print("First file data (normalized):", normalized_signals[0][:5])
    return normalized_signals
################--End of Function declaration--################

################--Function declaration--#######################
def zero_crossing(data):

    length = len(data)
    zero_crossing_count = 0

    for i in range(length):
        if i < length - 1:
            if np.sign(data[i]) !=np.sign(data[i+1]):
                zero_crossing_count += 1

    print('number of zero crossings:', zero_crossing_count)

    return zero_crossing_count
################--End of Function declaration--################
zero_crossing(emg)
zero_crossing(filtered_signal)


data = np.sin(time_axis)
zero_crossing(data)


normalized_data = amplitude_normalization(files_list)


noise_detect(emg)
sampling_frequency = 1100
data_length = len(emg)
time_axis = np.linspace(0, data_length/sampling_frequency, data_length)

notch_filter(emg,2, plot = True)
notch_filter(emg,8, plot = True)
notch_filter(emg,8.5, plot = True)
notch_filter(emg,10, plot = True)
notch_filter(emg,1000, plot = True)