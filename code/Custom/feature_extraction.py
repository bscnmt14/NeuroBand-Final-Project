import numpy as np
import pandas as pd


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
# Main Extraction Function
# ========================

def extract_all_features(
    df,
    window_size=200,
    step_size=50,
    zc_thresh=0.01,
    ssc_delta=0.01,
    subject_id=1,
    dataset_type="intra_subject",
    label_map=None
):
    """
    Extracts EMG and SP features using a sliding window approach.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input dataframe containing:
        - EMG channels (emg_ch_*)
        - SP channels (sp_ch_*)
        - gesture_label column

    window_size : int
        Number of samples per window.

    step_size : int
        Step size between consecutive windows.

    zc_thresh : float
        Threshold for Zero Crossing feature.

    ssc_delta : float
        Threshold for Slope Sign Change feature.

    subject_id : int
        Subject identifier.

    dataset_type : str
        Dataset category.

    label_map : dict or None
        Optional label encoding dictionary.

    Returns:
    -------
    features_df : pandas.DataFrame
        Extracted feature matrix.

    label_map : dict
        Mapping from gesture labels to numeric IDs.
    """

    if df.empty:
        return pd.DataFrame(), {}

    # ------------------------
    # Data Cleaning
    # ------------------------

    # Remove non-gesture regions
    df = df[df['gesture_label'] != 'beginning'].copy()

    # Create label encoding automatically if not provided
    if label_map is None:
        unique_labels = sorted(df['gesture_label'].unique())
        label_map = {
            label: idx for idx, label in enumerate(unique_labels)
        }

    # Add numeric labels
    df['label_id'] = df['gesture_label'].map(label_map)

    # Add metadata
    df['Subject'] = subject_id
    df['dataset_type'] = dataset_type

    # ------------------------
    # Channel Detection
    # ------------------------

    emg_cols = [col for col in df.columns if 'emg_ch' in col]
    sp_cols = [col for col in df.columns if 'sp_ch' in col]

    if len(emg_cols) == 0:
        raise ValueError(
            "No EMG channels found "
            "(expected columns containing 'emg_ch')"
        )

    print(f"Found {len(emg_cols)} EMG channels")
    print(f"Found {len(sp_cols)} SP channels")

    features = []

    print("Extracting features...")

    # ------------------------
    # Sliding Window Processing
    # ------------------------

    for start in range(0, len(df) - window_size, step_size):

        end = start + window_size

        window_data = df.iloc[start:end]

        # Skip transition windows
        if window_data['gesture_label'].nunique() > 1:
            continue

        # Extract EMG and SP signals
        emg_signals = window_data[emg_cols].values

        sp_signals = (
            window_data[sp_cols].values
            if len(sp_cols) > 0
            else None
        )

        # ------------------------
        # EMG Feature Extraction
        # ------------------------

        rms = calc_rms(emg_signals)

        var = calc_var(emg_signals)

        wl = calc_wl(emg_signals)

        emav = calc_emav(emg_signals)

        zc = calc_zc(emg_signals, zc_thresh)

        ssc = calc_ssc(emg_signals, ssc_delta)

        # ------------------------
        # SP Temporal Averaging
        # ------------------------

        if sp_signals is not None:
            sp_mean = np.mean(sp_signals, axis=0)

        # ------------------------
        # Create Feature Vector
        # ------------------------

        row_info = {
            'gesture_label': window_data['gesture_label'].iloc[0],
            'label_id': window_data['label_id'].iloc[0],
            'Subject': subject_id,
            'dataset_type': dataset_type
        }

        # Add EMG features
        for i, col in enumerate(emg_cols):

            row_info[f'{col}_RMS'] = rms[i]
            row_info[f'{col}_VAR'] = var[i]
            row_info[f'{col}_WL'] = wl[i]
            row_info[f'{col}_EMAV'] = emav[i]
            row_info[f'{col}_ZC'] = zc[i]
            row_info[f'{col}_SSC'] = ssc[i]

        # Add SP temporal averages
        if sp_signals is not None:

            for i, col in enumerate(sp_cols):

                row_info[f'{col}_MEAN'] = sp_mean[i]

        features.append(row_info)

    features_df = pd.DataFrame(features)

    return features_df, label_map