from anyio import Path
import mlflow
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.models.train import train
from src.utils.config import FLOW_DIR




experiments = [
    {
        "name": "logistic_regression",
        "logistic_regression": {
            "max_iter": 500,
            "class_weight": "balanced",
            "random_state": 101,
        }
    },
    {
        "name": "random_forest",
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 101,
        }
    }
]

dataset_name = "hr_attrition"

def run_experiment():
    FLOW_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("file:./ml_flow/mlruns")
    mlflow.set_experiment("hr_attrition_experiments")
    for exp in experiments:
        with mlflow.start_run():

            model, X_test, y_test, y_pred, y_proba = train(
                dataset_name=dataset_name,
                model_config=exp
            )

            mlflow.log_param("model_name", exp["name"])
            mlflow.log_params(exp[exp["name"]])
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred)


            plt.savefig(f"./ml_flow/ml_flow_img/{exp['name']}_confusion_matrix.png")
            mlflow.log_artifact(f"./ml_flow/ml_flow_img/{exp['name']}_confusion_matrix.png")

            mlflow.sklearn.log_model(model, exp["name"])
