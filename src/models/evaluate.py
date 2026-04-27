from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate(y_test, y_pred, y_proba, positive_label=1):
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=positive_label),
        "recall": recall_score(y_test, y_pred, pos_label=positive_label),
        "f1_score": f1_score(y_test, y_pred, pos_label=positive_label),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics
