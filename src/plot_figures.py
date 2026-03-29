import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import FIGURES_DIR

def ensure_figure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_roc_curve(y_test, y_prob, pathogen, model_name):
    ensure_figure_dir()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {pathogen} - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"{pathogen}_{model_name}_roc.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_pr_curve(y_test, y_prob, pathogen, model_name):
    ensure_figure_dir()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve: {pathogen} - {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"{pathogen}_{model_name}_pr.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_confusion_matrix_figure(y_test, y_pred, pathogen, model_name):
    ensure_figure_dir()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix: {pathogen} - {model_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"{pathogen}_{model_name}_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_permutation_importance(perm_df, pathogen, top_n=15):
    ensure_figure_dir()

    plot_df = perm_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(6, 5))
    plt.barh(plot_df["feature"], plot_df["importance_mean"], xerr=plot_df["importance_std"])
    plt.xlabel("Permutation Importance (mean AP decrease)")
    plt.ylabel("Feature")
    plt.title(f"Permutation Importance: {pathogen} RF")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"{pathogen}_rf_permutation_importance.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_y_randomization(yrand_df, pathogen, model_name):
    ensure_figure_dir()

    subset = yrand_df[(yrand_df["pathogen"] == pathogen) & (yrand_df["model"] == model_name)].copy()
    if subset.empty:
        return

    plt.figure(figsize=(5, 5))
    plt.plot(subset["run"], subset["pr_auc"], marker="o", label="PR-AUC")
    plt.plot(subset["run"], subset["roc_auc"], marker="s", label="ROC-AUC")
    plt.xlabel("Randomization Run")
    plt.ylabel("Score")
    plt.title(f"Y-randomization: {pathogen} - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"{pathogen}_{model_name}_y_randomization.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()