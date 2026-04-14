import os
import random
import numpy as np
import tensorflow as tf

from src.config import (
    OUTPUTS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    PREDICTIONS_DIR,
    PLOTS_DIR,
    SEED
)

def create_directories():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)