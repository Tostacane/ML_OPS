import mlflow
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.models.train import train


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
    for exp in experiments:
        with mlflow.start_run():

            model, X_test, y_test, y_pred, y_proba = train(
                dataset_name=dataset_name,
                model_config=exp
            )

            mlflow.log_param("model_name", exp["name"])
            mlflow.log_params(exp[exp["name"]])
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

            # plt.savefig("./ml_flow/ml_flow_img/confusion_matrix.png")
            # mlflow.log_artifact("confusion_matrix.png")

            mlflow.sklearn.log_model(model, exp["name"])
