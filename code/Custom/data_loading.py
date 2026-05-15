"""
===============================================================================
Data Loading and Signal Processing Module
===============================================================================
Handles the extraction of custom EMG CSV files from multi-sensor BLE systems.
Unwraps the 8-sample time buffers per packet into continuous time-series data,
aligns multiple sensors, applies filtering, and trims transients.
"""
import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import hyper_parameters as hp


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut=5.0, highcut=499.0, fs=1000.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


def load_csv_dataset(folder_path, margin_samples=0, apply_filter=True, fs=1000.0):
    all_data = []
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    for file_path in csv_files:
        print(f"Checking file: {os.path.basename(file_path)}")
        df_raw = pd.read_csv(file_path)
        print(f"File: {os.path.basename(file_path)} | Internal Labels: {df_raw['gesture_label'].unique()}")
        
        # Verify multi-sensor structure
        if 'device_id' not in df_raw.columns:
            print(f"Skipping: Missing 'device_id' column")
            continue
            
        # Identify unique sensors and sort them so channel mapping is consistent
        device_ids = sorted(df_raw['device_id'].dropna().unique())
        print(f"Found devices: {device_ids}")
        if not device_ids:
            continue
            
        # Group by contiguous gesture blocks
        df_raw['block_id'] = (df_raw['gesture_label'] != df_raw['gesture_label'].shift()).cumsum()
        
        for _, block_df in df_raw.groupby('block_id'):
            gesture = block_df['gesture_label'].iloc[0]
            
            # Skip unmapped target classes early if defined
            if hasattr(hp, 'TARGET_CLASSES') and gesture not in hp.TARGET_CLASSES:
                continue
                
            trial = block_df['trial_index'].iloc[0] if 'trial_index' in block_df.columns else 0
            
            device_data = {}
            min_len = float('inf')
            
            for i, dev_id in enumerate(device_ids):
                dev_df = block_df[block_df['device_id'] == dev_id]
                
                # Unpack the 8 time-samples per row into a single continuous 1D array
                emg_matrix = dev_df[[f'emg_ch_{c}' for c in range(8)]].values
                emg_flat = emg_matrix.flatten()
                
                # Unpack spectrum data. Since each packet (row) covers 8 EMG samples,
                # we repeat the 4 spectrum bins 8 times to match the temporal resolution.
                sp_matrix = dev_df[[f'spectrum_{c}' for c in range(4)]].values
                sp_repeated = np.repeat(sp_matrix, 8, axis=0)
                
                device_data[i] = {
                    'emg': emg_flat,
                    'sp': sp_repeated
                }
                
                if len(emg_flat) < min_len:
                    min_len = len(emg_flat)
            
            if min_len == 0 or min_len == float('inf'):
                continue
                
            # Build the aligned dataframe for this specific gesture block
            aligned_dict = {
                'gesture_label': [gesture] * min_len,
                'trial_index': [trial] * min_len
            }
            
            emg_block_matrix = np.zeros((min_len, len(device_ids)))
            
            for i in range(len(device_ids)):
                # Store truncated EMG for parallel filtering
                emg_block_matrix[:, i] = device_data[i]['emg'][:min_len]
                
                # Store SP channels (Sensor 0: 0-3, Sensor 1: 4-7, etc.)
                for sp_idx in range(4):
                    col_name = f'sp_ch_{i*4 + sp_idx}'
                    aligned_dict[col_name] = device_data[i]['sp'][:min_len, sp_idx]
            
            # Apply bandpass filter to the continuous EMG block BEFORE trimming margins
            if apply_filter:
                try:
                    emg_block_matrix = apply_bandpass_filter(emg_block_matrix, fs=fs)
                except ValueError:
                    continue # Occurs if block is too short for the filter order
                    
            for i in range(len(device_ids)):
                aligned_dict[f'emg_ch_{i}'] = emg_block_matrix[:, i]
                
            block_aligned_df = pd.DataFrame(aligned_dict)
            
            # Trim the mechanical and filter transients
            if margin_samples > 0:
                if len(block_aligned_df) > 2 * margin_samples:
                    block_aligned_df = block_aligned_df.iloc[margin_samples:-margin_samples]
                else:
                    continue # Block too short after applying margin
                    
            all_data.append(block_aligned_df)

    if not all_data:
        raise ValueError(f"No valid CSV files were found or all data was trimmed in {folder_path}.")

    final_df = pd.concat(all_data, ignore_index=True)
    final_df['Subject'] = 1
    
    return final_df