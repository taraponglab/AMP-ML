import os
import joblib
import pickle

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import MODEL_DIR

def get_model(name):
    if name == "random_forest":
        return RandomForestClassifier(random_state=42)
    elif name == "svm":
        return SVC(probability=True, random_state=42)
    elif name == "adaboost":
        return AdaBoostClassifier(random_state=42)
    elif name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    elif name == "xgboost":
        return XGBClassifier(random_state=42, eval_metric="logloss")
    else:
        raise ValueError(f"Unknown model name: {name}")

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def save_model(model, pathogen, model_name):
    pathogen_model_dir = os.path.join(MODEL_DIR, pathogen)
    os.makedirs(pathogen_model_dir, exist_ok=True)

    joblib_path = os.path.join(pathogen_model_dir, f"{model_name}.joblib")
    pkl_path = os.path.join(pathogen_model_dir, f"{model_name}.pkl")

    joblib.dump(model, joblib_path)

    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    return joblib_path, pkl_path