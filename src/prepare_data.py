import os
import pandas as pd
from config import PROCESSED_DIR, VALID_PATHOGENS

def load_processed_data(pathogen: str):
    if pathogen not in VALID_PATHOGENS:
        raise ValueError(f"Invalid pathogen: {pathogen}. Must be one of {VALID_PATHOGENS}")

    pathogen_dir = os.path.join(PROCESSED_DIR, pathogen)

    X_train_path = os.path.join(pathogen_dir, "X_train.csv")
    X_test_path = os.path.join(pathogen_dir, "X_test.csv")
    y_train_path = os.path.join(pathogen_dir, "y_train.csv")
    y_test_path = os.path.join(pathogen_dir, "y_test.csv")

    required_files = [X_train_path, X_test_path, y_train_path, y_test_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required file: {file_path}")

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    return X_train, X_test, y_train, y_test