# hyper_parameters.py
import os

# ==========================================
# 1. PATHS & DIRECTORIES
# ==========================================
BASE_DIR = r"B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\Intra-Subject Test"
RAW_DATA_PATH = os.path.join(BASE_DIR, "Raw")
TEST_DATA_PATH = os.path.join(BASE_DIR, "Files for test_16")

# ==========================================
# 2. HARDWARE & SIGNAL PARAMETERS
# ==========================================
FS = 1100.0
ADC_TO_VOLTS_FACTOR = 1e-6

BPF_PARAMS = {
    'lowcut': 35.0,
    'highcut': 499.0, 
    'fs': FS,
    'order': 4
}

# ==========================================
# 3. FEATURE EXTRACTION & PHYSICAL GRID
# ==========================================
WINDOW_MS = 200.0
STEP_MS = 100.0
MARGIN_MS = 25.0

ZC_VOLT_THRESH = 1e-6
SSC_VOLT_THRESH = 1e-6

# --- THE MASTER FEATURE SWITCH ---
# True  = SVM (Needs frequency data to help separate clusters)
# False = Random Forest / KNN (Drops noisy frequency smear for better micro-movements)
USE_SP_CHANNELS = False  

# ==========================================
# 4. MACHINE LEARNING ARCHITECTURE
# ==========================================
# --- ACTIVE MACHINE LEARNING PLUGIN ---
# Options: 'random_forest' | 'svm' | 'knn'
ACTIVE_MODEL = 'svm'

TRAIN_SPLIT_RATIO = 0.60

# Model-Specific Hardware Safety Filters
CONFIDENCE_THRESHOLDS = {
    'random_forest': 0.35,
    'svm': 0.30,
    'knn': 0.35
}

# Model Hyperparameters
RF_PARAMS = {
    'n_estimators': 200, 
    'max_depth': 5,
    'random_state': 42
}

SVM_GRID = {
    'C': [0.1, 1, 10, 50],             
    'gamma': ['scale', 0.001, 0.01] 
}