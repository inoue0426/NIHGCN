from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import pandas as pd
import numpy as np


def print_binary_classification_metrics(targets, predictions):
    """
    A function to print binary classification metrics.
    y_true: true labels
    y_pred: predicted labels
    """
    threshold = 0.5
    predictions_binary = (predictions > threshold).astype(int)

    accuracy = accuracy_score(targets, predictions_binary)
    precision = precision_score(targets, predictions_binary)
    recall = recall_score(targets, predictions_binary)
    f1 = f1_score(targets, predictions_binary)
    f2 = fbeta_score(targets, predictions_binary, beta=2)
    specificity = recall_score(targets, predictions_binary, pos_label=0)
    npv = precision_score(targets, predictions_binary, pos_label=0)

    # Check if there are more than one class in targets to calculate AUC-ROC and AUC-PR
    if len(np.unique(targets)) > 1:
        auc_roc = roc_auc_score(targets, predictions)
        auc_pr = average_precision_score(targets, predictions)
    else:
        print("Only one class in targets. Cannot calculate AUC-ROC and AUC-PR.")
        auc_roc = None
        auc_pr = None

    metrics_data = {
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1": [f1],
        "f2": [f2],
        "AUC-ROC": [auc_roc],
        "AUC-PR": [auc_pr],
        "Specificity": [specificity],
        "NPV": [npv],
    }

    # metrics_df = pd.DataFrame(metrics_data)

    return metrics_data
