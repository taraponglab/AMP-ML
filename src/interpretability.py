import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from config import RESULTS_DIR, FIGURES_DIR, RANDOM_STATE

def run_permutation_importance(model, X_test, y_test, pathogen=None):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="average_precision"
    )

    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    if pathogen is not None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_csv = os.path.join(RESULTS_DIR, f"{pathogen}_rf_permutation_importance.csv")
        perm_df.to_csv(out_csv, index=False)

    return result, perm_df


def run_shap_analysis(model, X_train, X_test, pathogen, max_display=15):
    """
    SHAP analysis for tree-based models.
    Saves:
      - summary bar plot
      - summary beeswarm plot
      - shap values csv (mean absolute shap)
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Sample to keep runtime reasonable
    X_train_sample = X_train.copy()
    X_test_sample = X_test.copy()

    if len(X_train_sample) > 200:
        X_train_sample = X_train_sample.sample(200, random_state=RANDOM_STATE)

    if len(X_test_sample) > 100:
        X_test_sample = X_test_sample.sample(100, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # --- Handle SHAP output shape robustly ---
    if isinstance(shap_values, list):
        # old SHAP style: list of arrays per class
        if len(shap_values) == 2:
            shap_values_to_use = shap_values[1]
        else:
            shap_values_to_use = shap_values[0]

    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            # shape may be (n_samples, n_features, n_classes)
            # for binary classification, take class 1
            shap_values_to_use = shap_values[:, :, 1]
        elif shap_values.ndim == 2:
            shap_values_to_use = shap_values
        else:
            raise ValueError(f"Unexpected SHAP array shape: {shap_values.shape}")
    else:
        raise ValueError(f"Unexpected SHAP output type: {type(shap_values)}")

    # Make sure final array is 2D
    if shap_values_to_use.ndim != 2:
        raise ValueError(f"Processed SHAP values must be 2D, got shape: {shap_values_to_use.shape}")

    # Mean absolute SHAP importance
    mean_abs_shap = np.abs(shap_values_to_use).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": X_test_sample.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_csv = os.path.join(RESULTS_DIR, f"{pathogen}_rf_shap_importance.csv")
    shap_df.to_csv(shap_csv, index=False)

    # Summary bar plot
    plt.figure()
    shap.summary_plot(
        shap_values_to_use,
        X_test_sample,
        plot_type="bar",
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"{pathogen}_rf_shap_bar.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Summary beeswarm plot
    plt.figure()
    shap.summary_plot(
        shap_values_to_use,
        X_test_sample,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"{pathogen}_rf_shap_beeswarm.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    return shap_df