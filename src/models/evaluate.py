from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate(y_test, y_pred, y_proba, positive_label=1):
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=positive_label),
        "recall": recall_score(y_test, y_pred, pos_label=positive_label),
        "f1_score": f1_score(y_test, y_pred, pos_label=positive_label),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("\n=== Metrics Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics
