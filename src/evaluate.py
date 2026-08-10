"""LOSO evaluation utilities — reused across all three stages."""

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


def run_loso(X, y, groups, clf_builder, scale: bool = False):
    """
    clf_builder: zero-arg callable returning a fresh, unfitted classifier
    (fresh instance per fold avoids state leakage across folds).
    """
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(set(y[train_idx])) < 2:
            continue
        X_train, X_test = X[train_idx], X[test_idx]
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        clf = clf_builder()
        clf.fit(X_train, y[train_idx])
        pred = clf.predict(X_test)
        all_true.extend(y[test_idx])
        all_pred.extend(pred)

    labels_sorted = sorted(set(y))
    return {
        'accuracy': accuracy_score(all_true, all_pred),
        'balanced_accuracy': balanced_accuracy_score(all_true, all_pred),
        'confusion_matrix': confusion_matrix(all_true, all_pred, labels=labels_sorted),
        'labels': labels_sorted,
    }
