# main.py (formerly intra_subject_test.py)
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
    Balances the dataset while strictly preserving time-alignment across all sensors.
    If a rest window is dropped, it drops it for all sensors simultaneously.
    """
    if not enabled or features_df.empty:
        return features_df
        
    # Create a temporary time-index to lock the 3 sensors together
    features_df['temp_time_idx'] = features_df.groupby(['sensor_id', 'trial_index']).cumcount()
        
    df_rest = features_df[features_df['gesture_label'] == 'at_rest']
    df_active = features_df[features_df['gesture_label'] != 'at_rest']
    
    if df_rest.empty or df_active.empty: 
        return features_df.drop(columns=['temp_time_idx'], errors='ignore')
        
    # Base our math on just ONE sensor so we don't triple-count the windows
    first_sensor = df_active['sensor_id'].iloc[0]
    active_counts = df_active[df_active['sensor_id'] == first_sensor]['gesture_label'].value_counts()
    avg_active_windows = int(round(active_counts.mean()))
    
    print(f"\n--- Class Balance Check (Time-Synchronized) ---")
    print(f" Target 'at_rest' size per sensor: {avg_active_windows}")
    
    # Grab the unique rest windows from just the first sensor
    rest_windows = df_rest[df_rest['sensor_id'] == first_sensor][['trial_index', 'temp_time_idx']]
    
    if len(rest_windows) > avg_active_windows:
        # Sample the exact points in time we want to keep
        sampled_time_points = rest_windows.sample(n=avg_active_windows, random_state=42)
        
        # Merge this filter against ALL sensors.
        df_rest_downsampled = df_rest.merge(sampled_time_points, on=['trial_index', 'temp_time_idx'], how='inner')
        
        balanced_features_df = pd.concat([df_active, df_rest_downsampled], ignore_index=True)
        balanced_features_df = balanced_features_df.drop(columns=['temp_time_idx'])
        print(f" -> Successfully synchronized and downsampled 'at_rest'.")
        return balanced_features_df
    else:
        print(" -> 'at_rest' is already balanced. Skipping.")
        return features_df.drop(columns=['temp_time_idx'])
    

def plot_3d_feature_space(features_df):
    """
    Plots a 3D feature space. 
    If SP channels are present, plots a dual view (PCA + Raw SP).
    If pruned, plots only the PCA of Time-Domain features.
    """
    print("\nGenerating 3D Feature Space Map...")
    
    gestures = features_df['gesture_label'].unique()
    
    cmap = plt.get_cmap('tab10' if len(gestures) <= 10 else 'tab20')
    color_map = {gesture: cmap(i) for i, gesture in enumerate(gestures)}

    # DYNAMIC CHECK: Do the SP channels exist in the dataframe?
    has_sp = all(col in features_df.columns for col in ['sp_1', 'sp_2', 'sp_3'])
    
    # Set up the figure layout based on what data is available
    if has_sp:
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
    else:
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111, projection='3d')

    # ==========================================
    # Subplot 1: PCA of Time-Domain Features (Always Runs)
    # ==========================================
    # Dynamically grab whatever time features are available
    all_time_features = ['feat_rms', 'feat_var', 'feat_wl', 'feat_emav', 'feat_zc', 'feat_ssc']
    time_features = [f for f in all_time_features if f in features_df.columns]
    
    X_time = features_df[time_features]
    X_time_scaled = StandardScaler().fit_transform(X_time)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_time_scaled)
    total_variance = sum(pca.explained_variance_ratio_) * 100
    
    for gesture in gestures:
        mask = features_df['gesture_label'] == gesture
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                    c=[color_map[gesture]], label=gesture, alpha=0.6, s=25, edgecolors='w')
        
    ax1.set_title(f'Time-Domain Feature Space (PCA)\nRetained {total_variance:.1f}% Variance')
    ax1.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_zlabel(f'PC 3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    
    # If there's only one plot, put the legend on it directly
    if not has_sp:
        ax1.legend(loc='best')
    
    # ==========================================
    # Subplot 2: Raw Spectral Power Channels (Conditional)
    # ==========================================
    if has_sp:
        for gesture in gestures:
            mask = features_df['gesture_label'] == gesture
            ax2.scatter(features_df.loc[mask, 'sp_1'], 
                        features_df.loc[mask, 'sp_2'], 
                        features_df.loc[mask, 'sp_3'], 
                        c=[color_map[gesture]], label=gesture, alpha=0.6, s=15)
            
        ax2.set_title('Frequency-Domain Feature Space (SP Channels)')
        ax2.set_xlabel('SP_1 (Low Freq Bin)')
        ax2.set_ylabel('SP_2 (Mid Freq Bin)')
        ax2.set_zlabel('SP_3 (High Freq Bin)')
        
        ax2.legend(bbox_to_anchor=(1.15, 1), loc='upper left', title="Gestures")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    # 1. Import the master control board
    import hyper_parameters as hp
    
    # ==========================================
    # DYNAMIC PLUGIN ROUTING
    # ==========================================
    print(f"Loading ML Plugin: {hp.ACTIVE_MODEL.upper()}")
    if hp.ACTIVE_MODEL == 'random_forest':
        from Models.RF_early_fusion import run_model as active_ml_model
    elif hp.ACTIVE_MODEL == 'svm':
        from Models.SVM_grid_search import run_model as active_ml_model
    elif hp.ACTIVE_MODEL == 'knn':
        from Models.KNN_early_fusion import run_model as active_ml_model
    else:
        raise ValueError(f"Unknown ACTIVE_MODEL '{hp.ACTIVE_MODEL}' in hyper_parameters.py")

    # ==========================================
    # RUN ORCHESTRATOR PIPELINE
    # ==========================================
    
    # Step 1: Data Loading
    processed_data = load_and_process_folder(hp.TEST_DATA_PATH)
    
    # Step 2: Extract features
    features_df = extract_features(processed_data)
    
    # Step 2.5: The Centralized SP Pruning Switch
    if not hp.USE_SP_CHANNELS and not features_df.empty:
        print("\n -> Feature Pruning: Dropping SP Channels (Running 6D Time-Domain)")
        features_df = features_df.drop(columns=['sp_1', 'sp_2', 'sp_3'], errors='ignore')
        
    # Step 3: Balance Data
    if not features_df.empty:
        features_df = undersample_rest_features(features_df, enabled=True) 

    # Step 4: Visualize
    if not features_df.empty:
        plot_3d_feature_space(features_df)
    
    # ==========================================
    # Step 5: ROUTE TO THE ACTIVE ML PLUGIN
    # ==========================================
    if not features_df.empty:
        print("\nRouting data to ML Plugin...")
        models_dict, scalers_dict, precision_score = active_ml_model(features_df)
    else:
        print("Feature DataFrame is empty. Check your data path and margin/window settings.")