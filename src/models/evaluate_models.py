"""
Evaluation utilities shared by both Phase 4 notebooks
(03_ml_models.ipynb for validation-set comparison,
04_model_evaluation.ipynb for the one-time final test-set evaluation).

Label convention (same as the rest of the project, see src/preprocessing/load_data.py):
    1 = Fake (the "positive" class for precision/recall/F1/ROC-AUC below)
    0 = Real
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

POSITIVE_LABEL = 1  # Fake - see module docstring


def get_positive_class_scores(model, X):
    """
    Return a 1-D array of "how strongly does the model think this is Fake"
    scores, suitable for roc_auc_score, regardless of whether the model
    exposes predict_proba (Logistic Regression, Random Forest, XGBoost) or
    only decision_function (LinearSVC).

    Returns None if neither is available (so callers can skip ROC-AUC
    cleanly instead of crashing).
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        fake_idx = classes.index(POSITIVE_LABEL)
        return proba[:, fake_idx]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return None


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """
    Compute the standard metric set for this project: accuracy, precision,
    recall, F1 (all with Fake=1 as the positive class), and ROC-AUC if
    y_score is provided.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def get_confusion_matrix(y_true, y_pred):
    """
    Confusion matrix with a FIXED label order [Real(0), Fake(1)], so every
    caller gets the same layout regardless of what labels happen to appear
    in a given batch of predictions:

                    Predicted Real   Predicted Fake
    Actual Real     True Real (TN)   False Fake (FP)
    Actual Fake     False Real (FN)  True Fake (TP)
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def get_classification_report(y_true, y_pred, output_dict: bool = False):
    """Classification report with fixed, human-readable class names."""
    return classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        zero_division=0,
        output_dict=output_dict,
    )
