import os

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "MICCAI_BraTS2020_TrainingData")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
FINAL_REPORT_PATH = os.path.join(REPORTS_DIR, "final_report.txt")
METRICS_CSV_PATH = os.path.join(REPORTS_DIR, "metrics.csv")
TRAINING_LOG_PATH = os.path.join(LOGS_DIR, "training_log.csv")

# -----------------------------
# DATA SETTINGS
# -----------------------------
IMAGE_SIZE = 128
NUM_CHANNELS = 4
NUM_CLASSES = 4

CLASS_LABELS = {
    "background": 0,
    "necrotic_core": 1,
    "edema": 2,
    "enhancing_tumor": 4
}

# -----------------------------
# TRAINING SETTINGS
# -----------------------------
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_STATE = 42

# -----------------------------
# MEMORY / DATA CONTROL
# -----------------------------
LIMIT_PATIENTS = 30
ARRAY_DTYPE = "float16"

# -----------------------------
# SLICE FILTERING
# -----------------------------
USE_TUMOR_SLICES_ONLY = False
MIN_TUMOR_PIXELS = 20

# -----------------------------
# SYSTEM SETTINGS
# -----------------------------
SEED = 42
VERBOSE = 1