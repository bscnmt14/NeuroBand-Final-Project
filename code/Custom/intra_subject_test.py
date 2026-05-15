import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    multilabel_confusion_matrix
)

import data_loading
import feature_extraction
import hyper_parameters as hp

def main():
    # ---------------------------------------------------------
    # 1. Configuration & Hyperparameters
    # ---------------------------------------------------------
    # Path to the folder containing your 8 CSV files
    data_folder = r"B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\Intra-Subject Test\Files for test"  
    
    kernel_type = 'linear'
    svm_params = hp.MODEL_PARAMS[kernel_type]
    data_config = hp.MODEL_DATA_CONFIG[kernel_type]
    
    # --- 240Hz HARDWARE OVERRIDES ---
    # margin=0 ensures short recordings aren't deleted.
    # window=100 is ~0.4s of data at 240Hz.
    margin = 0  
    window = 50 
    step = window//2     
    zc_thresh = data_config['zc']
    ssc_thresh = data_config['ssc']
    
    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Unwrapping multi-sensor packets and applying filters...")
    # Using apply_filter=False temporarily as 1000Hz filters conflict with 240Hz data
    df_raw = data_loading.load_csv_dataset(
        folder_path=data_folder, 
        margin_samples=margin, 
        apply_filter=False, 
        fs=1100.0
    )

    # ---------------------------------------------------------
    # 3. Feature Extraction
    # ---------------------------------------------------------
    print(f"Extracting features (window={window}, step={step})...")
    features_df, label_map = feature_extraction.extract_all_features(
        df=df_raw,
        window_size=window,
        step_size=step,
        zc_thresh=zc_thresh,
        ssc_delta=ssc_thresh,
        subject_id=1,
        dataset_type="intra_subject"
    )
    
    if features_df.empty:
        print("Error: Feature extraction returned an empty DataFrame.")
        print("Suggestion: Try lowering the 'window' size further (e.g., to 50).")
        return

    # ---------------------------------------------------------
    # 4. Data Preparation & Intra-Subject Split (60/40)
    # ---------------------------------------------------------
    X = features_df.drop(columns=['gesture_label', 'label_id', 'Subject', 'dataset_type'])
    y = features_df['label_id']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # 5. Model Training (SVM)
    # ---------------------------------------------------------
    print(f"Training {kernel_type.capitalize()} SVM on 60% of data...")
    clf = SVC(kernel=kernel_type, random_state=42, **svm_params)
    clf.fit(X_train_scaled, y_train)

    # ---------------------------------------------------------
    # 6. Evaluation & Confusion Matrices
    # ---------------------------------------------------------
    print("\nEvaluating Model on remaining 40%...")
    y_pred = clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")
    
    labels = [idx for label, idx in sorted(label_map.items(), key=lambda item: item[1])]
    target_names = [label for label, idx in sorted(label_map.items(), key=lambda item: item[1])]
    
    print("Feature windows extracted per class:")
    print(features_df['gesture_label'].value_counts(), "\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))

    # --- CREATE OUTPUT DIRECTORY ---
    output_dir = "Confusion Matrices"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving visual matrices to '{output_dir}' folder...")

    # --- 1. OVERALL MULTI-CLASS MATRIX ---
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    plt.title(f"Overall Confusion Matrix ({kernel_type.capitalize()}) - 60/40 Split")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_multi_class_matrix.png"), dpi=300)
    plt.close(fig)

    # --- 2. INDIVIDUAL PER-GESTURE MATRICES ---
    mcm = multilabel_confusion_matrix(y_test, y_pred, labels=labels)
    
    for i, label_idx in enumerate(labels):
        gesture_name = target_names[i]
        cm_2x2 = mcm[i]
        
        display_labels = [f"Not {gesture_name}", gesture_name]
        
        fig, ax = plt.subplots(figsize=(6, 5))
        disp_2x2 = ConfusionMatrixDisplay(confusion_matrix=cm_2x2, display_labels=display_labels)
        disp_2x2.plot(cmap=plt.cm.Blues, ax=ax)
        
        plt.title(f"One-vs-Rest: {gesture_name.upper()}")
        plt.tight_layout()
        
        safe_filename = gesture_name.replace(" ", "_").replace("/", "_") + ".png"
        filepath = os.path.join(output_dir, safe_filename)
        
        plt.savefig(filepath, dpi=300)
        plt.close(fig)
        
        tn, fp, fn, tp = cm_2x2.ravel()
        print(f"Saved {safe_filename} | TP:{tp} TN:{tn} FP:{fp} FN:{fn}")

    print("\nAll matrices generated and saved successfully!")

if __name__ == "__main__":
    main()