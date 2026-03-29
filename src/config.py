import os

# Project root (adjust if needed)
PROJECT_ROOT = "/content/drive/MyDrive/AMP-ML"

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Output paths
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Columns
SEQUENCE_COLUMN = "Sequence"
TARGET_COLUMN = "Activity"

# Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Pathogens
VALID_PATHOGENS = ["klebsiella", "ecoli", "pseudomonas"]

# Models
MODEL_LIST = [
    "random_forest",
    "svm",
    "adaboost",
    "gradient_boosting",
    "xgboost"
]