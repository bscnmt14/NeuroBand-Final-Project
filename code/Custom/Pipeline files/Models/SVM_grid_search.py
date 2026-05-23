# Models/SVM_grid_search.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score
import hyper_parameters as hp

def run_model(features_df):
    print("\n[PLUGIN] Executing Chronological Grid Search SVM...")
    
    all_possible_features = ['feat_rms', 'feat_var', 'feat_wl', 'feat_emav', 'feat_zc', 'feat_ssc', 'sp_1', 'sp_2', 'sp_3']
    feature_cols = [col for col in all_possible_features if col in features_df.columns]
    
    sensors = features_df['sensor_id'].unique()
    models = {}
    scalers = {}
    val_probs_dict = {}
    val_true = None

    # Load grid from HP
    param_grid = hp.SVM_GRID

    for sensor in sensors:
        print(f"\n -> Tuning Sensor: {sensor}...")
        sensor_df = features_df[features_df['sensor_id'] == sensor]
        
        train_trials = []
        val_trials = []
        
        for gesture in sensor_df['gesture_label'].unique():
            gesture_df = sensor_df[sensor_df['gesture_label'] == gesture]
            unique_trials = sorted(gesture_df['trial_index'].unique())
            
            # Use HP split ratio
            split_idx = int(len(unique_trials) * hp.TRAIN_SPLIT_RATIO)
            if split_idx == len(unique_trials): 
                split_idx = max(1, len(unique_trials) - 1)
            elif split_idx == 0 and len(unique_trials) > 0:
                split_idx = 1
                
            train_trials.extend(unique_trials[:split_idx])
            val_trials.extend(unique_trials[split_idx:])
            
        train_mask = sensor_df['trial_index'].isin(train_trials)
        val_mask = sensor_df['trial_index'].isin(val_trials)
        
        X_train = sensor_df.loc[train_mask, feature_cols]
        y_train = sensor_df.loc[train_mask, 'gesture_label']
        groups_train = sensor_df.loc[train_mask, 'trial_index'] 
        X_val = sensor_df.loc[val_mask, feature_cols]
        y_val = sensor_df.loc[val_mask, 'gesture_label']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        base_svm = SVC(kernel='rbf', probability=True, random_state=42)
        logo = LeaveOneGroupOut()
        
        grid_search = GridSearchCV(
            estimator=base_svm, 
            param_grid=param_grid, 
            cv=logo,                   
            scoring='f1_macro', 
            n_jobs=-1                  
        )
        
        grid_search.fit(X_train_scaled, y_train, groups=groups_train)
        
        best_clf = grid_search.best_estimator_
        print(f"    * Best Params: {grid_search.best_params_}")
        print(f"    * Best Internal F1-Score: {grid_search.best_score_:.3f}")
        
        models[sensor] = best_clf
        scalers[sensor] = scaler
        val_probs_dict[sensor] = best_clf.predict_proba(X_val_scaled)
        
        if val_true is None:
            val_true = y_val.values 

    print("\n -> Applying Multi-Sensor Max-Pooling Probability Fusion...")
    final_predictions = []
    num_samples = len(val_true)
    classes = models[sensors[0]].classes_
    
    CONFIDENCE_THRESHOLD = hp.CONFIDENCE_THRESHOLDS['svm']
    
    for i in range(num_samples):
        sample_probs = [val_probs_dict[s][i] for s in sensors]
        max_probs_per_class = np.max(sample_probs, axis=0)
        max_idx = np.argmax(max_probs_per_class)
        max_conf = max_probs_per_class[max_idx]
        
        if max_conf < CONFIDENCE_THRESHOLD:
            final_predictions.append('no_action')
        else:
            final_predictions.append(classes[max_idx])

    unique_true_labels = list(pd.unique(val_true))
    labels = list(unique_true_labels).copy()
    if 'no_action' in final_predictions and 'no_action' not in labels:
        labels.append('no_action')

    val_precision = precision_score(val_true, final_predictions, average='macro', zero_division=0)
    cm_percent = confusion_matrix(val_true, final_predictions, labels=labels, normalize='true') * 100
    
    print("\nOptimized Soft Fusion Confusion Matrix (Percentages %):")
    print(pd.DataFrame(cm_percent, index=labels, columns=labels).round(2))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)
    disp.plot(cmap='Blues', values_format='.1f')
    plt.title(f"Soft Fusion Grid SVM CM (%)\nConfidence Filter Cutoff: {CONFIDENCE_THRESHOLD*100}%")
    plt.show()
    
    return models, scalers, val_precision