import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def apply_1d_bpf(signal_1d, lowcut, highcut, fs, order):
    """
    Applies a zero-phase Butterworth Bandpass Filter to a continuous 1D signal.
    """
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    padlen = min(3 * max(len(b), len(a)), len(signal_1d) - 1)
    filtered_signal = filtfilt(b, a, signal_1d, padlen=padlen)
    return filtered_signal

def plot_emg_from_file(file_path, bpf_params, adc_to_uv_factor=1.0, apply_filter=True):
    """
    Reads a single CSV file and plots the ENTIRE continuous EMG signal.
    Adds a secondary X-axis with gesture labels and vertical lines at transition points.
    """
    print(f"Loading file: {file_path}")
    df = pd.read_csv(file_path)
    
    sensor_ids = df['device_id'].dropna().unique()
    all_gestures = df['gesture_label'].dropna().unique()
    active_gestures = [g for g in all_gestures if g not in ['beginning', 'at_rest']]
    active_gesture_name = active_gestures[0] if active_gestures else "Unknown"
    
    emg_cols = [f'emg_ch_{i}' for i in range(8)]
    
    for sensor_id in sensor_ids:
        sensor_df = df[df['device_id'] == sensor_id].copy()
        
        if sensor_df.empty:
            continue
            
        sensor_df = sensor_df.sort_values(by='timestamp')
        
        # 1. Flatten the entire trial data
        raw_continuous_emg = sensor_df[emg_cols].values.flatten().astype(float)
        
        # ==========================================
        # Expand and Track Gesture Labels
        # ==========================================
        # Because each row has 8 samples, we must repeat the label 8 times 
        # so the label array perfectly matches the length of the flattened EMG array.
        raw_labels = sensor_df['gesture_label'].values
        continuous_labels = np.repeat(raw_labels, 8)
        
        # 2. Route the data based on filter toggle
        if apply_filter:
            processed_emg = apply_1d_bpf(
                raw_continuous_emg, 
                bpf_params['lowcut'], 
                bpf_params['highcut'], 
                bpf_params['fs'], 
                bpf_params['order']
            )
            plot_title = f"FILTERED EMG (Full Trial) - Sensor {sensor_id} - Active Gesture: {active_gesture_name}\n"
            line_color = '#1f77b4' 
        else:
            processed_emg = raw_continuous_emg
            plot_title = f"RAW EMG (Full Trial) - Sensor {sensor_id} - Active Gesture: {active_gesture_name}\n"
            line_color = '#d62728' 
        
        # 3. Convert Amplitude to microVolts
        processed_emg_uv = processed_emg * adc_to_uv_factor
        
        # 4. Generate the main X-axis (Time in Seconds)
        total_samples = len(processed_emg_uv)
        time_seconds = np.arange(total_samples) / bpf_params['fs']
        
        # 5. Calculate stats
        max_peak = np.max(processed_emg_uv)
        min_trough = np.min(processed_emg_uv)
        deviation = np.std(processed_emg_uv) 
        
        stats_text = (f"Max Peak: {max_peak:.2f} uV\n"
                      f"Min Trough: {min_trough:.2f} uV\n"
                      f"Std Dev: {deviation:.2f} uV")
        
        # 6. Plotting
        fig, ax = plt.subplots(figsize=(14, 6)) # Made slightly wider to fit text
        
        ax.plot(time_seconds, processed_emg_uv, color=line_color, linewidth=1.0)
        
        # MODIFIED: pad parameter increased from 20 to 25 to prevent overlapping with multi-line segment labels
        ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=25)
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel(r"Amplitude ($\mu$V)", fontsize=12) 
        
        # ==========================================
        # Find Transitions and Add Dotted Lines
        # ==========================================
        segments = []
        current_label = continuous_labels[0]
        start_idx = 0
        
        # Loop through to find exactly where the label changes
        for i in range(1, len(continuous_labels)):
            if continuous_labels[i] != current_label:
                # Save the segment: (Label, start index, end index)
                segments.append((current_label, start_idx, i))
                
                # Draw the vertical dotted line at the exact second the label changes
                transition_time = time_seconds[i]
                ax.axvline(x=transition_time, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
                
                current_label = continuous_labels[i]
                start_idx = i
                
        # Add the final segment
        segments.append((current_label, start_idx, len(continuous_labels)))

        # ==========================================
        # Add the Secondary X-Axis at the Top
        # ==========================================
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()) # Lock it to the main axis timeframe
        
        midpoints_sec = []
        segment_labels = []
        
        # MODIFIED: Loop now extracts sample sizes, computes duration in seconds, and updates label format
        for label, start_idx, end_idx in segments:
            mid_idx = (start_idx + end_idx) // 2
            midpoints_sec.append(time_seconds[mid_idx])
            
            # Calculate physical duration: (Total Samples inside segment) / (Sampling Frequency)
            duration_sec = (end_idx - start_idx) / bpf_params['fs']
            
            # Use newline character to stack the string labels and timing values cleanly
            segment_labels.append(f"{label.upper()}\n({duration_sec:.2f}s)") 
            
        ax2.set_xticks(midpoints_sec)
        ax2.set_xticklabels(segment_labels, fontsize=10, fontweight='bold', color='#333333')
        
        # Remove the tick marks (the little lines) on the top axis for a cleaner look
        ax2.tick_params(axis='x', length=0)
        
        # Add the Stats Textbox
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props,
                fontfamily='monospace') 
        
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    
    # 1. FILE PATH CONFIGURATION
    SINGLE_FILE_PATH = r"B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\Intra-Subject Test\Files for test\Fixed_Tal, pinch, muscle_mass, 23-21-49, 30-12-25.csv"
    
    # ==========================================
    # 2. FILTER TOGGLE (True = Filtered, False = Raw)
    # ==========================================
    APPLY_FILTER = True  # <--- Change this to True/False to compare!
    
    # 3. BPF CONFIGURATION
    bpf_parameters = {
        'lowcut': 35,
        'highcut': 499, 
        'fs': 1100,
        'order': 4
    }
    
    # 4. MICROVOLT CONVERSION 
    UV_CONVERSION_FACTOR = 1.0 
    
    plot_emg_from_file(
        SINGLE_FILE_PATH, 
        bpf_parameters, 
        adc_to_uv_factor=UV_CONVERSION_FACTOR,
        apply_filter=APPLY_FILTER
    )