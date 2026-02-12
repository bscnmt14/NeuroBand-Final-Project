import numpy as np
import pandas as pd
from pathlib import Path

# 1. Use r"" for paths to avoid errors with \u and \1
folder_path = Path(r"B:\OneDrive - Afeka College Of Engineering\uMyo Python Project - test\Data\מיקום אלקטרודות\1")

# 2. Get the actual list of files (not just the count)
# Check if you need *.csv or *.xlsx
files_list = list(folder_path.glob('*.csv'))

# Create a list to store the results
rms_values = []
spectrum = []
# 3. Iterate through the ACTUAL files


for file in files_list:
    # Read the specific file in the current iteration
    data = pd.read_csv(file)

    # Extract the signal
    sp1 = data['sp1']
    emg = data['emg']

    # 4. Calculate RMS efficiently (Vectorized - much faster)
    # Mean of the squares, then square root
    rms = np.sqrt(np.mean(sp1 ** 2))

    rms_values.append(rms)

# Convert to numpy array if needed
rms_array = np.array(rms_values)

print(f"Processed {len(files_list)} files.")
print("All RMS values:", rms_array)

# Find the value
max_val = np.max(rms_array)

# Find the index (position)
max_index = np.argmax(rms_array)

print(f"File with max RMS: {files_list[max_index].name}")


