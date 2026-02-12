import pandas as pd
import numpy as np
import os
import glob

# --- CONFIGURATION ---
INPUT_FOLDER = r"C:\Users\Nadav\OneDrive - Afeka College Of Engineering\uMyo Python Project - test\Custom_files\wrong data"   # <--- Update this path
OUTPUT_FOLDER = r"C:\Users\Nadav\OneDrive - Afeka College Of Engineering\uMyo Python Project - test\Custom_files"  # <--- Update this path

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def transform_file(file_path):
    try:
        # Load the wrong data
        df = pd.read_csv(file_path)
        
        # Check if it's already in the correct format (has emg_ch_0)
        if 'emg_ch_0' in df.columns:
            print(f"Skipping {os.path.basename(file_path)}: Already in correct format.")
            return

        print(f"Processing: {os.path.basename(file_path)}...")

        transformed_parts = []

        # Process per unit_id to ensure we don't mix devices
        # (If your file has multiple devices, this handles them separately)
        for unit_id, group in df.groupby('unit_id'):
            n_rows = len(group)
            n_packets = n_rows // 8
            
            if n_packets == 0:
                continue
                
            # Truncate to multiple of 8 (drop incomplete end packets)
            group = group.iloc[:n_packets*8]
            
            # --- VECTORIZED TRANSFORMATION ---
            # 1. Reshape EMG: (N, 1) -> (N/8, 8)
            emg_matrix = group['emg'].values.reshape(n_packets, 8)
            
            # 2. Average Timestamp: (N, 1) -> (N/8, 8) -> Mean per row
            # 't_rel_sec' in wrong data maps to 'timestamp' in correct data
            timestamps = group['t_rel_sec'].values.reshape(n_packets, 8).mean(axis=1)
            
            # 3. Copy Metadata (Take 1st value of every 8 rows)
            # We map the old column names to the new column names here
            sp0 = group['sp0'].values[::8]
            sp1 = group['sp1'].values[::8]
            sp2 = group['sp2'].values[::8]
            sp3 = group['sp3'].values[::8]
            ax = group['ax'].values[::8]
            ay = group['ay'].values[::8]
            az = group['az'].values[::8]
            
            # Optional: Preserve labels if they exist (just taking the first one)
            # gesture = group['gesture_label'].values[::8] if 'gesture_label' in group.columns else None
            
            # Create the new DataFrame block
            new_df = pd.DataFrame({
                'timestamp': timestamps,
                'device_id': unit_id,
                'emg_ch_0': emg_matrix[:, 0],
                'emg_ch_1': emg_matrix[:, 1],
                'emg_ch_2': emg_matrix[:, 2],
                'emg_ch_3': emg_matrix[:, 3],
                'emg_ch_4': emg_matrix[:, 4],
                'emg_ch_5': emg_matrix[:, 5],
                'emg_ch_6': emg_matrix[:, 6],
                'emg_ch_7': emg_matrix[:, 7],
                'spectrum_0': sp0,
                'spectrum_1': sp1,
                'spectrum_2': sp2,
                'spectrum_3': sp3,
                'accel_x': ax,
                'accel_y': ay,
                'accel_z': az,
                'emg_sample_count': 8  # Constant
            })
            
            transformed_parts.append(new_df)

        if not transformed_parts:
            print(f"Warning: No valid data blocks found in {file_path}")
            return

        # Concatenate all device parts
        df_transformed = pd.concat(transformed_parts, ignore_index=True)

        # --- FINAL FORMATTING ---
        # Add missing columns found in the "correct" format (filled with NaN)
        target_cols = [
            'timestamp', 'device_id', 'data_id', 'rssi', 'battery_mv', 'emg_sample_count', 
            'emg_ch_0', 'emg_ch_1', 'emg_ch_2', 'emg_ch_3', 'emg_ch_4', 'emg_ch_5', 'emg_ch_6', 'emg_ch_7', 
            'spectrum_0', 'spectrum_1', 'spectrum_2', 'spectrum_3', 
            'quat_w', 'quat_x', 'quat_y', 'quat_z', 
            'accel_x', 'accel_y', 'accel_z', 
            'yaw', 'pitch', 'roll', 'mag_x', 'mag_y', 'mag_z', 'mag_angle'
        ]

        for col in target_cols:
            if col not in df_transformed.columns:
                df_transformed[col] = np.nan

        # Reorder columns to match target exactly
        df_transformed = df_transformed[target_cols]

        # Save to output folder
        out_name = "Fixed_" + os.path.basename(file_path)
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        df_transformed.to_csv(out_path, index=False)
        print(f"Saved: {out_name}")

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")

# --- EXECUTION ---
# Get all csv files
files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
print(f"Found {len(files)} CSV files.")

for f in files:
    transform_file(f)

print("Done!")