from src.models.train import train
from src.models.evaluate import evaluate
from ml_flow.mlflow import run_experiment


if __name__ == "__main__":
    model, X_test, y_test, y_pred, y_proba = train("hr_attrition")
    metrics = evaluate(y_test, y_pred, y_proba)

    run_experiment()
