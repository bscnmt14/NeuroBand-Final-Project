# meta_optimizer.py
import itertools
import pandas as pd
from data_loading import load_and_process_folder
from feature_extraction import extract_features
from Models.SVM_grid_search import run_model as active_ml_model

# We need a stripped-down version of the balancer that doesn't print to the console 50 times
def silent_undersample(features_df):
    """
    Balances the dataset for the meta-optimizer while protecting the main loop
    against empty window extractions or missing indices.
    """
    if features_df.empty:
        return features_df
        
    df_rest = features_df[features_df['gesture_label'] == 'at_rest']
    df_active = features_df[features_df['gesture_label'] != 'at_rest']
    
    # Escape early IF either category is empty BEFORE altering any columns
    if df_rest.empty or df_active.empty: 
        return features_df
    
    # Base our synchronization math on the first physical sensor found
    first_sensor = df_active['sensor_id'].iloc[0]
    active_sensor_df = df_active[df_active['sensor_id'] == first_sensor]
    
    if active_sensor_df.empty:
        return features_df
        
    avg_active = int(round(active_sensor_df['gesture_label'].value_counts().mean()))
    
    # Create a local copy to modify so we don't cause SettingWithCopyWarnings
    working_df = features_df.copy()
    working_df['temp_time_idx'] = working_df.groupby(['sensor_id', 'trial_index']).cumcount()
    
    # Separate the working sets with the time index securely attached
    df_rest_working = working_df[working_df['gesture_label'] == 'at_rest']
    df_active_working = working_df[working_df['gesture_label'] != 'at_rest']
    
    rest_windows = df_rest_working[df_rest_working['sensor_id'] == first_sensor][['trial_index', 'temp_time_idx']]
    
    if len(rest_windows) > avg_active:
        sampled = rest_windows.sample(n=avg_active, random_state=42)
        df_rest_down = df_rest_working.merge(sampled, on=['trial_index', 'temp_time_idx'], how='inner')
        balanced = pd.concat([df_active_working, df_rest_down], ignore_index=True)
        return balanced.drop(columns=['temp_time_idx'], errors='ignore')
        
    return working_df.drop(columns=['temp_time_idx'], errors='ignore')



if __name__ == "__main__":
    FOLDER_PATH = r"B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\Intra-Subject Test\Files for test" 
    bpf_parameters = {'lowcut': 35.0, 'highcut': 499, 'fs': 1100.0, 'order': 4}
    
    print("Loading continuous data (Happens only once)...")
    processed_data = load_and_process_folder(FOLDER_PATH, bpf_parameters)

    # ==========================================
    # DEFINE THE PHYSICAL FEATURE GRID
    # Add or remove values here to expand your search!
    # ==========================================
    physical_grid = {
        'window_ms': [200.0, 300.0, 400.0],
        'margin_ms': [15.0, 25.0, 50.0],
        'zc_thresh': [1e-6, 1e-7, 1e-8],
        'ssc_thresh': [1e-6, 1e-7, 1e-8]
    }

    # Generate all possible combinations
    keys = physical_grid.keys()
    combinations = list(itertools.product(*physical_grid.values()))
    
    print(f"\nInitiating Meta-Search across {len(combinations)} physical configurations...")
    
    results = []

    for idx, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        print(f"\n--- Testing Config {idx + 1}/{len(combinations)} ---")
        print(config)
        
        # 1. Extract features with the current physical config
        features_df = extract_features(
            processed_data, 
            margin_ms=config['margin_ms'],                  
            window_ms=config['window_ms'],        
            step_ms=100, # Keeping step constant to maintain overlap ratios                    
            fs=bpf_parameters['fs'],           
            zc_volt_thresh=config['zc_thresh'],   
            ssc_volt_thresh=config['ssc_thresh']
        )
        
        # 2. Balance
        if not features_df.empty:
            features_df = silent_undersample(features_df)
            
            # 3. Pass to the ML Plugin (which runs its own SVM Grid Search)
            try:
                models, scalers, precision = active_ml_model(features_df)
                
                # Record the result
                results.append({
                    'window_ms': config['window_ms'],
                    'margin_ms': config['margin_ms'],
                    'zc_thresh': config['zc_thresh'],
                    'ssc_thresh': config['ssc_thresh'],
                    'val_precision': precision # <--- Tracking Precision now
                })
                print(f" -> Resulting Macro Precision: {precision * 100:.2f}%")
            except Exception as e:
                print(f" -> Model failed on this config: {e}")

    # ==========================================
    # PRINT THE LEADERBOARD
    # ==========================================
    print("\n==========================================")
    print("        META-OPTIMIZER LEADERBOARD        ")
    print("==========================================")
    # Sort descending by Precision!
    results_df = pd.DataFrame(results).sort_values(by='val_precision', ascending=False)
    print(results_df.head(10).to_string(index=False))