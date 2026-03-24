import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(6, 5)):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    if labels is None:
        labels = ["Negative", "Positive"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, cmap="Blues")

    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", fontsize=11)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


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

    print("\n=== Confusion Matrix ===")
    plot_confusion_matrix(y_test, y_pred)

    return metrics
