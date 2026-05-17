# data_loading.py
import os
import glob
import pandas as pd
from scipy.signal import butter, filtfilt

def apply_bpf(df, lowcut, highcut, fs, order, adc_to_volts_factor=1.0):
    """
    Applies a Butterworth Bandpass Filter to the EMG data and converts the output 
    from raw ADC counts directly into Volts.
    """
    if df.empty: return df
    
    emg_cols = [f'emg_ch_{i}' for i in range(8)]
    
    # Create the Butterworth filter using built-in 'fs' normalization
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    
    # Flatten the (N, 8) matrix into a 1D continuous array
    continuous_emg = df[emg_cols].values.flatten()
    
    # Apply zero-phase filter (filtfilt) to avoid phase shifting
    padlen = min(3 * max(len(b), len(a)), len(continuous_emg) - 1)
    filtered_emg = filtfilt(b, a, continuous_emg, padlen=padlen)
    
    # Reshape back to (N, 8)
    reshaped_emg = filtered_emg.reshape(-1, 8)
    
    # =========================================================================
    # CONVERSION TO VOLTS: Multiply the zero-mean filtered data by the scaling factor
    # =========================================================================
    reshaped_emg_volts = reshaped_emg * adc_to_volts_factor
    
    # Assign back to the dataframe
    df_filtered = df.copy()
    df_filtered[emg_cols] = df_filtered[emg_cols].astype(float)
    df_filtered.loc[:, emg_cols] = reshaped_emg_volts
    
    return df_filtered

def load_and_process_folder(folder_path, bpf_params, adc_to_volts_factor=1.0):
    """
    Reads all CSVs in a folder, groups by sensor, applies BPF, converts to Volts, 
    and splits rest/active.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(csv_files)} files in folder: {folder_path}")
    
    master_sensor_dict = {}
    
    for file in csv_files:
        df = pd.read_csv(file)
        sensor_ids = df['device_id'].dropna().unique()
        
        for sensor_id in sensor_ids:
            if sensor_id not in master_sensor_dict:
                master_sensor_dict[sensor_id] = {'rest': [], 'active': []}
                
            sensor_df = df[df['device_id'] == sensor_id].sort_values(by='timestamp')
            cleaned_df = sensor_df[sensor_df['gesture_label'] != 'beginning'].copy()
            
            df_rest = cleaned_df[cleaned_df['gesture_label'] == 'at_rest'].copy()
            df_active = cleaned_df[cleaned_df['gesture_label'] != 'at_rest'].copy()
            
            # Pass the conversion factor to the BPF application step
            df_rest = apply_bpf(df_rest, **bpf_params, adc_to_volts_factor=adc_to_volts_factor)
            df_active = apply_bpf(df_active, **bpf_params, adc_to_volts_factor=adc_to_volts_factor)
            
            master_sensor_dict[sensor_id]['rest'].append(df_rest)
            master_sensor_dict[sensor_id]['active'].append(df_active)

    for sensor_id in master_sensor_dict:
        if master_sensor_dict[sensor_id]['rest']:
            master_sensor_dict[sensor_id]['rest'] = pd.concat(master_sensor_dict[sensor_id]['rest'], ignore_index=True)
        if master_sensor_dict[sensor_id]['active']:
            master_sensor_dict[sensor_id]['active'] = pd.concat(master_sensor_dict[sensor_id]['active'], ignore_index=True)
        
    return master_sensor_dict