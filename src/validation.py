import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

def y_randomization_test(model, X_train, y_train, X_test, y_test, n_runs=10, random_state=42):
    rng = np.random.default_rng(random_state)
    results = []

    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    for i in range(n_runs):
        y_train_random = rng.permutation(y_train)
        shuffled_model = clone(model)
        shuffled_model.fit(X_train, y_train_random)

        y_prob = shuffled_model.predict_proba(X_test)[:, 1]
        y_pred = shuffled_model.predict(X_test)

        pr_auc = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "run": i + 1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1": f1
        })

    return results