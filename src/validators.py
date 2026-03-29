import pandas as pd

def validate_feature_tables(X_train, X_test):
    assert X_train.shape[1] == X_test.shape[1], "Feature mismatch between train and test."
    assert list(X_train.columns) == list(X_test.columns), "Train/test feature columns are not identical."

def validate_labels(y_train, y_test):
    y_train_values = pd.Series(y_train).squeeze()
    y_test_values = pd.Series(y_test).squeeze()

    assert set(pd.unique(y_train_values)).issubset({0, 1}), "y_train labels must be 0/1"
    assert set(pd.unique(y_test_values)).issubset({0, 1}), "y_test labels must be 0/1"