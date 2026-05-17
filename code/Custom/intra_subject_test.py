# intra_subject_test.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Import functions from our separated modules
from data_loading import load_and_process_folder
from feature_extraction import extract_features


def undersample_rest_features(features_df, enabled=True):
    """
    Balances the dataset by downsampling the 'at_rest' feature windows 
    to match the average window count of the active gesture classes.
    """
    if not enabled or features_df.empty:
        return features_df
        
    # 1. Separate rest windows from active gesture windows
    df_rest = features_df[features_df['gesture_label'] == 'at_rest']
    df_active = features_df[features_df['gesture_label'] != 'at_rest']
    
    # If there's missing data types, return unmodified
    if df_rest.empty or df_active.empty:
        return features_df
        
    # 2. Compute the average window count across all unique active gestures
    # (.value_counts() tracks how many rows/windows exist for each active class)
    active_counts = df_active['gesture_label'].value_counts()
    avg_active_windows = int(round(active_counts.mean()))
    
    print(f"\n--- Class Balance Check ---")
    print(f" Original 'at_rest' windows: {len(df_rest)}")
    print(f" Active classes distribution:\n{active_counts.to_string()}")
    print(f" Target 'at_rest' size (average of active classes): {avg_active_windows}")
    
    # 3. Perform the downsampling if rest is genuinely larger
    if len(df_rest) > avg_active_windows:
        # random_state=42 ensures your runs are perfectly reproducible
        df_rest_downsampled = df_rest.sample(n=avg_active_windows, random_state=42)
        
        # Combine the active data back with our condensed rest data
        balanced_features_df = pd.concat([df_active, df_rest_downsampled], ignore_index=True)
        print(f" -> Successfully downsampled 'at_rest' to {len(df_rest_downsampled)} windows.")
        return balanced_features_df
    else:
        print(" -> 'at_rest' window count is already smaller or equal to target. Skipping.")
        return features_df

def plot_3d_feature_space(features_df):
    """
    Reduces the 9-dimensional feature space into 3 dimensions using PCA 
    and plots an interactive 3D scatter plot to visualize class separation.
    Dynamically scales to support any number of gestures.
    """
    print("\nGenerating 3D Feature Space Map...")
    
    feature_cols = ['feat_rms', 'feat_var', 'feat_wl', 'feat_emav', 'feat_zc', 'feat_ssc', 'sp_1', 'sp_2', 'sp_3']
    
    # 1. Extract the features and labels
    X = features_df[feature_cols]
    y = features_df['gesture_label']
    
    # 2. Scale the data (PCA requires strictly scaled data!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Apply PCA to reduce 9 dimensions down to 3
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate how much of the original data's variance we kept
    total_variance = sum(pca.explained_variance_ratio_) * 100
    
    # 4. Build the 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # =========================================================================
    # FIX 1: Dynamic Categorical Colormap Generation
    # =========================================================================
    gestures = y.unique()
    num_gestures = len(gestures)
    
    # 'tab10' handles up to 10 classes beautifully. If you have more, 'tab20' steps in.
    cmap = plt.get_cmap('tab10' if num_gestures <= 10 else 'tab20')
    colors = [cmap(i) for i in range(num_gestures)]
    
    # Loop using an index tracker instead of zip() to guarantee nothing gets skipped
    for i, gesture in enumerate(gestures):
        # Find the rows that match this specific gesture
        indices = y == gesture
        
        # Plot them in 3D space
        ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2], 
                   color=colors[i], label=gesture, alpha=0.6, edgecolors='w', s=40)
        
    ax.set_title(f'3D Feature Distribution (PCA)\nRetained {total_variance:.1f}% of Original Data Variance')
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_zlabel(f'Principal Component 3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax.legend()
    
    plt.show()

def train_and_evaluate_svm(features_df):
    """
    Splits data based on intra-subject trial indices, trains a Linear SVM,
    and displays a percentage-based Confusion Matrix.
    """
    print("\nTraining SVM Model...")
    
    # Feature columns we extracted
    feature_cols = ['feat_rms', 'feat_var', 'feat_wl', 'feat_emav', 'feat_zc', 'feat_ssc', 'sp_1', 'sp_2', 'sp_3']
    
    # Intra-subject Trial Split
    train_mask = features_df['trial_index'].isin([1, 2, 3])
    val_mask = features_df['trial_index'].isin([4, 5])
    
    X_train = features_df.loc[train_mask, feature_cols]
    y_train = features_df.loc[train_mask, 'gesture_label']
    
    X_val = features_df.loc[val_mask, feature_cols]
    y_val = features_df.loc[val_mask, 'gesture_label']
    
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")
    
    scaler = StandardScaler()

    # Fit the scaler ONLY on the training data (to prevent data leakage), 
    # and then transform the training data.
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform the validation data using the rules learned from the training data
    X_val_scaled = scaler.transform(X_val)
    
    # Train Linear SVM on the SCALED data
    clf = SVC(kernel='rbf', C=1, gamma=1,  random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Predict on the SCALED validation data
    y_pred = clf.predict(X_val_scaled)
    
    # Percentage-based Confusion Matrix (Normalize rows to sum to 100%)
    labels = clf.classes_
    cm_percent = confusion_matrix(y_val, y_pred, labels=labels, normalize='true') * 100
    
    # Print text version to console
    print("\nConfusion Matrix (Percentages %):")
    print(pd.DataFrame(cm_percent, index=labels, columns=labels).round(2))
    
    # Visual Matrix Plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)
    disp.plot(cmap='Blues', values_format='.1f')
    plt.title("Confusion Matrix (%)")
    plt.show()
    
    return clf


if __name__ == "__main__":
    
    # ==========================================
    # PIPELINE CONFIGURATION
    # ==========================================
    
    # 1. Replace with the path to your folder containing the CSVs
    FOLDER_PATH = r"B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\Intra-Subject Test\Files for test" # Currently set to current directory
    
    # 1. BPF & Sensor Hardware Configuration
    bpf_parameters = {
        'lowcut': 25.0,
        'highcut': 499.0, 
        'fs': 1100.0,
        'order': 4
    }
    
    # --- FILL IN YOUR SENSOR FACTOR ---
    # What number converts 1 raw ADC point into 1 literal Volt?
    # (Set to 1.0 if your CSV values are already formatted as true Volts)
    ADC_TO_VOLTS_FACTOR = 1e-6  # Example placeholder scalar
    

    
    # ==========================================
    # RUN PIPELINE
    # ==========================================
    
    # Step 1: Data Loading & Filtering
    processed_data = load_and_process_folder(FOLDER_PATH, bpf_parameters)
    
    # Step 2: Extract features
    features_df = extract_features(
        processed_data, 
        margin_ms=25.0,                   
        window_ms=100.0,                   
        step_ms=None,                     
        fs=bpf_parameters['fs'],           
        zc_volt_thresh=1e-8,   # <--- Threshold for Zero Crossings (e.g., 5 uV)
        ssc_volt_thresh=1e-8   # <--- Independent threshold for Slope Changes (e.g., 2 uV)
    )

    UNDERSAMPLE_REST = True 
    
    if not features_df.empty:
        features_df = undersample_rest_features(features_df, enabled=UNDERSAMPLE_REST) 

    # Step 2.5: Visualize the Feature Space!
    if not features_df.empty:
        plot_3d_feature_space(features_df)
    
    # Step 3: Train and Evaluate SVM
    if not features_df.empty:
        model = train_and_evaluate_svm(features_df)
    else:
        print("Feature DataFrame is empty. Check your data path and margin/window settings.")