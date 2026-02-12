import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# REPLACE THIS with the name of one of your files
FILE_NAME = "session_fist_test.csv"

df = pd.read_csv(FILE_NAME)

# Map labels just like before
LABEL_MAP = {'none': 'at_rest', 'ID_REST_1': 'at_rest'} # Add your IDs here
df['gesture_label'] = df['gesture_label'].replace(LABEL_MAP)

# Create a color map
colors = {'at_rest': 'blue', 'fist': 'red'}
col_list = [colors.get(l, 'gray') for l in df['gesture_label']]

plt.figure(figsize=(15, 6))
plt.title(f"Visual Check: {FILE_NAME}")
plt.plot(df['t_rel_sec'], df['emg'], color='black', alpha=0.3, label='Raw Signal')

# We scatter plot the points to show the labels
# (Plotting every 10th point to keep it fast)
subset = df.iloc[::10]
plt.scatter(subset['t_rel_sec'], subset['emg'], c=[colors.get(l, 'gray') for l in subset['gesture_label']], s=10, alpha=0.8)

# Legend hack
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colors.values()]
plt.legend(markers, colors.keys(), numpoints=1)

plt.xlabel("Time (s)")
plt.ylabel("EMG Amplitude")
plt.show()