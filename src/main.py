from src.models.train import train
from src.models.evaluate import evaluate
from ml_flow.mlflow import run_experiment


if __name__ == "__main__":
    model, X_test, y_test, y_pred, y_proba = train(dataset_name = "hr_attrition", save_file_name = "hr_log_reg")
    # run_experiment()
