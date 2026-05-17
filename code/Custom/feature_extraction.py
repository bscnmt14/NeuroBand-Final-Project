import pandas as pd
import numpy as np

# ========================
# Feature Calculations
# ========================

def calc_rms(emg_signals):
    """
    Root Mean Square (RMS)
    Represents signal power and correlates with muscle force.
    """
    return np.sqrt(np.mean(emg_signals ** 2, axis=0))

def calc_var(emg_signals):
    """
    Variance (VAR)
    Measures signal dispersion around the mean.
    """
    return np.var(emg_signals, axis=0)

def calc_wl(emg_signals):
    """
    Waveform Length (WL)
    Captures amplitude, frequency, and duration information.
    """
    return np.sum(np.abs(np.diff(emg_signals, axis=0)), axis=0)

def calc_emav(emg_signals):
    """
    Enhanced Mean Absolute Value (EMAV)
    Applies weighting to emphasize the middle of the window.
    """
    N = emg_signals.shape[0]
    p = np.full((N, 1), 0.5)

    start_idx = int(0.2 * N)
    end_idx = int(0.8 * N)

    p[start_idx:end_idx + 1] = 0.75

    return np.mean(np.abs(emg_signals) ** p, axis=0)

def calc_zc(emg_signals, threshold):
    """
    Zero Crossings (ZC)
    Counts sign changes above a noise threshold.
    """
    return np.sum(
        (emg_signals[:-1] * emg_signals[1:] < 0) &
        (np.abs(emg_signals[:-1] - emg_signals[1:]) > threshold),
        axis=0
    )

def calc_ssc(emg_signals, threshold):
    """
    Slope Sign Changes (SSC)
    Counts local peaks and valleys with threshold filtering.
    """
    diffs = np.diff(emg_signals, axis=0)
    ssc_products = -diffs[:-1] * diffs[1:]
    return np.sum(ssc_products >= threshold, axis=0)

# ========================
# Feature Extraction Logic
# ========================

def extract_features(sensor_data_dict, margin_ms=100.0, window_ms=300.0, step_ms=None, fs=1100.0, 
                     zc_volt_thresh=1e-6, ssc_volt_thresh=1e-6):
    """
    Applies margins, splits into sliding windows, and extracts features.
    Allows independent voltage thresholds for Zero Crossings (ZC) and Slope Sign Changes (SSC).
    Both inputs are in Volts (e.g., 1e-6 for 1 uV).
    """
    if step_ms is None:
        step_ms = window_ms / 2.0
        
    # Time-to-row math
    margin_rows = int(round((margin_ms / 1000.0) * fs / 8.0))
    window_rows = int(round((window_ms / 1000.0) * fs / 8.0))
    step_rows = int(round((step_ms / 1000.0) * fs / 8.0))
    
    window_rows, step_rows, margin_rows = max(1, window_rows), max(1, step_rows), max(0, margin_rows)

    # =========================================================================
    # SEPARATE THRESHOLD MANAGEMENT:
    # ZC threshold maps 1-to-1 with Volts.
    # SSC threshold is squared internally to match the Volts^2 mathematical dimension.
    # =========================================================================
    zc_threshold_final = zc_volt_thresh
    ssc_threshold_final = ssc_volt_thresh ** 2

    print(f"\nExtracting Features | Window: {window_ms}ms | Step: {step_ms}ms")
    print(f" -> ZC Amplitude Threshold:  {zc_volt_thresh} Volts")
    print(f" -> SSC Amplitude Threshold: {ssc_volt_thresh} Volts (Internal Math: {ssc_threshold_final:.2e} Volts²)")
    
    emg_cols = [f'emg_ch_{i}' for i in range(8)]
    all_features = []

    for sensor_id, gestures in sensor_data_dict.items():
        for gesture_type, df in gestures.items():
            if df is None or df.empty: continue
            
            for trial_idx in df['trial_index'].unique():
                trial_df = df[df['trial_index'] == trial_idx].copy()
                
                # Apply Margins
                if len(trial_df) > (2 * margin_rows):
                    if margin_rows > 0:
                        trial_df = trial_df.iloc[margin_rows : -margin_rows]
                else:
                    continue
                
                # Sliding Windows
                for start_idx in range(0, len(trial_df) - window_rows + 1, step_rows):
                    window_df = trial_df.iloc[start_idx : start_idx + window_rows]
                    
                    window_emg_1d = window_df[emg_cols].values.flatten()
                    emg_matrix = window_emg_1d.reshape(-1, 1)
                    
                    # Calculate features using independent thresholds
                    feat_row = {
                        'sensor_id': sensor_id,
                        'gesture_label': window_df['gesture_label'].iloc[0],
                        'trial_index': trial_idx,
                        'feat_rms': calc_rms(emg_matrix)[0],
                        'feat_var': calc_var(emg_matrix)[0],
                        'feat_wl': calc_wl(emg_matrix)[0],
                        'feat_emav': calc_emav(emg_matrix)[0],
                        'feat_zc': calc_zc(emg_matrix, zc_threshold_final)[0],
                        'feat_ssc': calc_ssc(emg_matrix, ssc_threshold_final)[0],
                        'sp_1': window_df['spectrum_1'].mean(),
                        'sp_2': window_df['spectrum_2'].mean(),
                        'sp_3': window_df['spectrum_3'].mean()
                    }
                    all_features.append(feat_row)

    return pd.DataFrame(all_features)