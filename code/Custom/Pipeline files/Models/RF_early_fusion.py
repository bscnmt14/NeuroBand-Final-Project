# Models/RF_early_fusion.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score
import hyper_parameters as hp  # <--- Master Control Board

def run_model(features_df):
    print("\n[PLUGIN] Executing Early Fusion Random Forest...")
    
    # ==========================================
    # 1. Internal Pivot to Whole-Arm format
    # ==========================================
    print(" -> Pivoting data to Early Fusion format...")
    df_working = features_df.copy()
    df_working['window_index'] = df_working.groupby(['sensor_id', 'trial_index']).cumcount()
    
    all_possible_features = ['feat_rms', 'feat_var', 'feat_wl', 'feat_emav', 'feat_zc', 'feat_ssc', 'sp_1', 'sp_2', 'sp_3']
    active_features = [f for f in all_possible_features if f in df_working.columns]
    
    pivot_df = df_working.pivot(
        index=['trial_index', 'window_index', 'gesture_label'],
        columns='sensor_id',
        values=active_features 
    )
    
    pivot_df.columns = [f"sensor_{sensor}_{feat}" for feat, sensor in pivot_df.columns]
    rf_df = pivot_df.reset_index().dropna()
    feature_cols = [col for col in rf_df.columns if col.startswith('sensor_')]
    
    # ==========================================
    # 2. Stratified Dynamic Split
    # ==========================================
    train_trials = []
    val_trials = []
    
    for gesture in rf_df['gesture_label'].unique():
        gesture_trials = sorted(rf_df[rf_df['gesture_label'] == gesture]['trial_index'].unique())
        
        # Reads the split directly from hp
        split_idx = int(len(gesture_trials) * hp.TRAIN_SPLIT_RATIO)
        if split_idx == len(gesture_trials): 
            split_idx = max(1, len(gesture_trials) - 1)
        elif split_idx == 0 and len(gesture_trials) > 0:
            split_idx = 1
            
        train_trials.extend(gesture_trials[:split_idx])
        val_trials.extend(gesture_trials[split_idx:])
        
    train_mask = rf_df['trial_index'].isin(train_trials)
    val_mask = rf_df['trial_index'].isin(val_trials)
    
    X_train = rf_df.loc[train_mask, feature_cols]
    y_train = rf_df.loc[train_mask, 'gesture_label']
    X_val = rf_df.loc[val_mask, feature_cols]
    y_val = rf_df.loc[val_mask, 'gesture_label']
    
    # ==========================================
    # 3. Train the Random Forest (Driven by HP)
    # ==========================================
    print(f" -> Training on {len(X_train)} windows, Validating on {len(X_val)} windows...")
    
    clf = RandomForestClassifier(
        n_estimators=hp.RF_PARAMS['n_estimators'], 
        max_depth=hp.RF_PARAMS['max_depth'], 
        random_state=hp.RF_PARAMS['random_state'], 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    val_probs = clf.predict_proba(X_val)
    classes = clf.classes_
    val_true = y_val.values

    # ==========================================
    # 4. Probabilistic Safety Filter
    # ==========================================
    final_predictions = []
    CONFIDENCE_THRESHOLD = hp.CONFIDENCE_THRESHOLDS['random_forest']
    
    for probs in val_probs:
        max_idx = np.argmax(probs)
        if probs[max_idx] < CONFIDENCE_THRESHOLD:
            final_predictions.append('no_action')
        else:
            final_predictions.append(classes[max_idx])

    # ==========================================
    # 5. Evaluate
    # ==========================================
    labels = list(classes).copy()
    if 'no_action' in final_predictions and 'no_action' not in labels:
        labels.append('no_action')

    val_precision = precision_score(val_true, final_predictions, average='macro', zero_division=0)
    cm_percent = confusion_matrix(val_true, final_predictions, labels=labels, normalize='true') * 100
    
    print("\nEarly Fusion RF Confusion Matrix (Percentages %):")
    print(pd.DataFrame(cm_percent, index=labels, columns=labels).round(2))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)
    disp.plot(cmap='Greens', values_format='.1f') 
    plt.title(f"Early Fusion Random Forest\nConfidence Filter: {CONFIDENCE_THRESHOLD*100}%")
    plt.show()
    
    return {'early_fusion_rf': clf}, {}, val_precision