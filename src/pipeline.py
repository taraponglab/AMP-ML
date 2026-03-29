import os
import pandas as pd

from config import VALID_PATHOGENS, MODEL_LIST, RESULTS_DIR
from prepare_data import load_processed_data
from validators import validate_feature_tables, validate_labels
from train_models import get_model, train_model, save_model
from evaluate_models import evaluate_model
from validation import y_randomization_test
from interpretability import run_permutation_importance, run_shap_analysis
from plot_figures import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix_figure,
    plot_permutation_importance,
    plot_y_randomization
)

def run_pipeline():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    all_yrand = []

    for pathogen in VALID_PATHOGENS:
        print(f"\n{'='*60}")
        print(f"Running pipeline for pathogen: {pathogen}")
        print(f"{'='*60}")

        X_train, X_test, y_train, y_test = load_processed_data(pathogen)

        validate_feature_tables(X_train, X_test)
        validate_labels(y_train, y_test)

        for model_name in MODEL_LIST:
            print(f"\nModel: {model_name}")

            model = get_model(model_name)
            model = train_model(model, X_train, y_train)

            # Save model
            joblib_path, pkl_path = save_model(model, pathogen, model_name)
            print(f"Saved: {joblib_path}")
            print(f"Saved: {pkl_path}")

            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            metrics["pathogen"] = pathogen
            metrics["model"] = model_name
            all_results.append(metrics)
            print(metrics)

            # Predictions for plots
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Curves and confusion matrix
            plot_roc_curve(y_test, y_prob, pathogen, model_name)
            plot_pr_curve(y_test, y_prob, pathogen, model_name)
            plot_confusion_matrix_figure(y_test, y_pred, pathogen, model_name)

            # Y-randomization
            y_rand_results = y_randomization_test(
                model, X_train, y_train, X_test, y_test, n_runs=10
            )
            for row in y_rand_results:
                row["pathogen"] = pathogen
                row["model"] = model_name
                all_yrand.append(row)

        # After all models for this pathogen, save RF interpretation
        rf_model = get_model("random_forest")
        rf_model = train_model(rf_model, X_train, y_train)

        perm_result, perm_df = run_permutation_importance(
            rf_model, X_test, y_test, pathogen=pathogen
        )
        plot_permutation_importance(perm_df, pathogen)

        shap_df = run_shap_analysis(rf_model, X_train, X_test, pathogen)
        print(f"Saved SHAP results for {pathogen}")

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["pathogen", "roc_auc"], ascending=[True, False])
    results_df.to_csv(os.path.join(RESULTS_DIR, "all_model_results.csv"), index=False)

    yrand_df = pd.DataFrame(all_yrand)
    yrand_df.to_csv(os.path.join(RESULTS_DIR, "all_y_randomization_results.csv"), index=False)

    # Plot y-randomization figures
    for pathogen in VALID_PATHOGENS:
        for model_name in MODEL_LIST:
            plot_y_randomization(yrand_df, pathogen, model_name)

    print("\nAll done.")
    print("Saved results, models, SHAP, permutation importance, ROC, PR, confusion matrix, and Y-randomization figures.")

if __name__ == "__main__":
    run_pipeline()