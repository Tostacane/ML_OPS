from src.models.train import train
from src.models.evaluate import evaluate
from ml_flow.ml_flow import run_experiment
from models.train import train
from models.evaluate import evaluate

if __name__ == "__main__":
    model, X_test, y_test, y_pred, y_proba = train("hr_attrition")
    metrics = evaluate(y_test, y_pred, y_proba)

    run_experiment()
