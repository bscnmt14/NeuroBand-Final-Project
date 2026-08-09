import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch

def apply_1d_bpf(signal_1d, lowcut, highcut, fs, order):
    """Applies a zero-phase Butterworth Bandpass Filter."""
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    padlen = min(3 * max(len(b), len(a)), len(signal_1d) - 1)
    filtered_signal = filtfilt(b, a, signal_1d, padlen=padlen)
    return filtered_signal

def apply_notch_filter(signal_1d, freq, fs, q=30.0):
    """Applies a zero-phase Notch Filter."""
    b, a = iirnotch(freq, q, fs)
    padlen = min(3 * max(len(b), len(a)), len(signal_1d) - 1)
    filtered_signal = filtfilt(b, a, signal_1d, padlen=padlen)
    return filtered_signal

def apply_margin_removal(signal, segments, margin_samples):
    """
    Sets the boundary samples of each segment to np.nan to remove transition noise.
    Using np.nan prevents these regions from being plotted or included in stats.
    """
    clean_signal = signal.copy()
    if margin_samples <= 0:
        return clean_signal
        
    for _, start_idx, end_idx in segments:
        # Prevent removing more than half the segment if the segment is very short
        actual_margin = min(margin_samples, (end_idx - start_idx) // 2)
        if actual_margin > 0:
            clean_signal[start_idx : start_idx + actual_margin] = np.nan
            clean_signal[end_idx - actual_margin : end_idx] = np.nan
            
    return clean_signal

def generate_stats_text(signal):
    """Generates stats text, ignoring np.nan values created by margin removal."""
    max_peak = np.nanmax(signal)
    min_trough = np.nanmin(signal)
    return (f"Max Peak: {max_peak:.2f} uV\n"
            f"Min Trough: {min_trough:.2f} uV")

def process_and_save_emg(file_path, bpf_params, save_dir, notch_freq=50.0, adc_to_uv_factor=1.0, margin_ms=0):
    """
    Reads a CSV, generates Raw, Processed, and Spectrum plots, and saves them.
    """
    print(f"Loading file: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    sensor_ids = df['unit_id'].dropna().unique()
    all_gestures = df['gesture_label'].dropna().unique()
    active_gestures = [g for g in all_gestures if g not in ['beginning', 'at_rest']]
    active_gesture_name = active_gestures[0] if active_gestures else "Unknown"
    
    emg_cols = [f'emg_{i}' for i in range(8)]
    margin_samples = int((margin_ms / 1000.0) * bpf_params['fs'])
    print(f"Margin set to {margin_ms}ms (Removing {margin_samples} samples from segment boundaries)")
    
    for sensor_id in sensor_ids:
        sensor_df = df[df['unit_id'] == sensor_id].copy()
        
        if sensor_df.empty:
            continue
            
        sensor_df = sensor_df.sort_values(by='timestamp')
        
        raw_continuous_emg = sensor_df[emg_cols].values.flatten().astype(float)
        raw_labels = sensor_df['gesture_label'].values
        continuous_labels = np.repeat(raw_labels, 8)
        
        total_samples = len(raw_continuous_emg)
        time_seconds = np.arange(total_samples) / bpf_params['fs']
        
        segments = []
        current_label = continuous_labels[0]
        start_idx = 0
        for i in range(1, len(continuous_labels)):
            if continuous_labels[i] != current_label:
                segments.append((current_label, start_idx, i))
                current_label = continuous_labels[i]
                start_idx = i
        segments.append((current_label, start_idx, len(continuous_labels)))

        # ==========================================
        # 1. PLOT AND SAVE RAW DATA
        # ==========================================
        raw_emg_uv = raw_continuous_emg * adc_to_uv_factor
        stats_text_raw = generate_stats_text(raw_emg_uv)
        
        fig_raw, ax_raw = plt.subplots(figsize=(14, 6))
        ax_raw.plot(time_seconds, raw_emg_uv, color='#d62728', linewidth=1.0)
        ax_raw.set_title(f"RAW EMG - Sensor {sensor_id}\n", fontsize=14, fontweight='bold')
        ax_raw.set_xlabel("Time (seconds)", fontsize=12)
        ax_raw.set_ylabel(r"Amplitude ($\mu$V)", fontsize=12)
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax_raw.text(0.98, 0.95, stats_text_raw, transform=ax_raw.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='monospace') 
        ax_raw.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        raw_save_path = os.path.join(save_dir, f"{file_name}_Sensor_{sensor_id}_Raw.png")
        plt.savefig(raw_save_path, dpi=150)
        plt.close(fig_raw) 
        
        # ==========================================
        # 2. FILTER DATA
        # ==========================================
        emg_notch = apply_notch_filter(raw_continuous_emg, notch_freq, bpf_params['fs'])
        processed_emg = apply_1d_bpf(emg_notch, bpf_params['lowcut'], bpf_params['highcut'], bpf_params['fs'], bpf_params['order'])
        
        # We save this continuous, gapless version for the frequency spectrum calculation below
        processed_emg_uv_continuous = processed_emg * adc_to_uv_factor
        
        # Apply Margin Removal for the time-domain plot
        processed_emg_uv_gaps = apply_margin_removal(processed_emg_uv_continuous, segments, margin_samples)
        
        # ==========================================
        # 3. PLOT AND SAVE PROCESSED DATA
        # ==========================================
        stats_text_proc = generate_stats_text(processed_emg_uv_gaps)
        
        fig_proc, ax_proc = plt.subplots(figsize=(14, 6))
        ax_proc.plot(time_seconds, processed_emg_uv_gaps, color='#1f77b4', linewidth=1.0)
        ax_proc.set_title(f"FILTERED EMG (BPF + {notch_freq}Hz Notch) - Sensor {sensor_id} - Active Gesture: {active_gesture_name}\n", 
                          fontsize=14, fontweight='bold', pad=45)
        ax_proc.set_xlabel("Time (seconds)", fontsize=12)
        ax_proc.set_ylabel(r"Amplitude ($\mu$V)", fontsize=12) 
        
        ax2 = ax_proc.twiny()
        ax2.set_xlim(ax_proc.get_xlim()) 
        
        midpoints_sec = []
        segment_labels = []
        
        for label, s_idx, e_idx in segments:
            ax_proc.axvline(x=time_seconds[s_idx], color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            mid_idx = (s_idx + e_idx) // 2
            midpoints_sec.append(time_seconds[mid_idx])
            duration_sec = (e_idx - s_idx) / bpf_params['fs']
            segment_labels.append(f"{label.upper()}\n({duration_sec:.2f}s)") 
            
        ax2.set_xticks(midpoints_sec)
        ax2.set_xticklabels(segment_labels, fontsize=9, fontweight='bold', color='#333333', rotation=45, ha='left')
        ax2.tick_params(axis='x', length=0)
        
        ax_proc.text(0.98, 0.95, stats_text_proc, transform=ax_proc.transAxes, fontsize=11,
                     verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='monospace') 
        ax_proc.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        proc_save_path = os.path.join(save_dir, f"{file_name}_Sensor_{sensor_id}_Processed.png")
        plt.savefig(proc_save_path, dpi=150)
        plt.close(fig_proc)

        # ==========================================
        # 4. PLOT AND SAVE FREQUENCY SPECTRUM (Welch's Method)
        # ==========================================
        # Compute PSD using a 1024 sample window (about 1 second of data per window)
        f_raw, Pxx_raw = welch(raw_emg_uv, fs=bpf_params['fs'], nperseg=1024)
        f_proc, Pxx_proc = welch(processed_emg_uv_continuous, fs=bpf_params['fs'], nperseg=1024)
        
        fig_spec, ax_spec = plt.subplots(figsize=(12, 6))
        
        # Plot Raw (Red, partially transparent)
        ax_spec.plot(f_raw, Pxx_raw, color='#d62728', alpha=0.5, label='Raw EMG', linewidth=1.5)
        # Plot Processed (Blue, bold)
        ax_spec.plot(f_proc, Pxx_proc, color='#1f77b4', alpha=0.9, label='Filtered EMG', linewidth=1.5)
        
        ax_spec.set_title(f"Frequency Spectrum - Sensor {sensor_id}", fontsize=14, fontweight='bold')
        ax_spec.set_xlabel("Frequency (Hz)", fontsize=12)
        ax_spec.set_ylabel(r"Power Spectral Density ($\mu V^2$/Hz)", fontsize=12)
        
        # Limit X-axis to the Nyquist frequency (half the sampling rate)
        ax_spec.set_xlim(0, bpf_params['fs'] / 2)
        
        # Log scale on Y is standard for spectral density to see the noise floor properly
        ax_spec.set_yscale('log') 
        
        ax_spec.legend(loc='upper right', fontsize=11)
        ax_spec.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        spec_save_path = os.path.join(save_dir, f"{file_name}_Sensor_{sensor_id}_Spectrum.png")
        plt.savefig(spec_save_path, dpi=150)
        plt.close(fig_spec)

    print(f"Success! Saved raw, processed, and spectrum plots for {file_name} to: {save_dir}\n")


if __name__ == "__main__":
    
    # 1. DIRECTORY CONFIGURATION
    DATA_DIRECTORY = r"B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\Intra-Subject Test\inter_subject_data"
    
    # 2. BPF & NOTCH CONFIGURATION
    bpf_parameters = {
        'lowcut': 35,
        'highcut': 499, 
        'fs': 1200,
        'order': 4
    }
    NOTCH_FREQUENCY = 50.0 
    
    # 3. MARGIN REMOVAL CONFIGURATION (in milliseconds)
    MARGIN_MILLISECONDS = 150 
    
    # 4. MICROVOLT CONVERSION 
    UV_CONVERSION_FACTOR = 1.0 
    
    # 5. BATCH PROCESSING LOOP
    csv_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {DATA_DIRECTORY}")
    else:
        print(f"Found {len(csv_files)} CSV files. Starting batch processing...\n")
        
        for file_path in csv_files:
            base_name = os.path.basename(file_path)
            parts = base_name.split(',')
            
            if len(parts) >= 2:
                gesture_folder = parts[1].strip()
            else:
                gesture_folder = "Unknown_Gesture"
                
            specific_save_dir = os.path.join(DATA_DIRECTORY, "Data plots", gesture_folder)
            
            process_and_save_emg(
                file_path, 
                bpf_parameters,
                save_dir=specific_save_dir,
                notch_freq=NOTCH_FREQUENCY,
                adc_to_uv_factor=UV_CONVERSION_FACTOR,
                margin_ms=MARGIN_MILLISECONDS
            )
        
        print("All files processed successfully!")