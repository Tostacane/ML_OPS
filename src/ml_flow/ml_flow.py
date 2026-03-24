import mlflow
import mlflow.sklearn

from models.train import train
from models.evaluate import evaluate


#TODO add parameters to change the model 
def run_experiment():
    mlflow.set_experiment("hr_attrition_experiment")
    with mlflow.start_run():
        print("Running MLflow...")

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 500)
        mlflow.log_param("class_weight", "balanced")

        model, X_test, y_test, y_pred, y_proba = train("hr_attrition")
        
        metrics = evaluate(y_test, y_pred, y_proba)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.sklearn.log_model(model, "model")