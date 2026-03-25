from src.models.train import train
from src.models.evaluate import evaluate

if __name__ == "__main__":
    model, X_test, y_test, y_pred, y_proba = train("hr_attrition")
    metrics = evaluate(y_test, y_pred, y_proba)

    print("\n=== Final Metrics ===")
    print(metrics)
