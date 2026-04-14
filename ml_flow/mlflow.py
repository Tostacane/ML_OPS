import mlflow
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.models.evaluate import evaluate
from src.models.train import train
from src.utils.config import FLOW_DIR 
from .model_variants import generate_experiment_variants

dataset_name = "hr_attrition"

def run_experiment():

    # Generate variants of experiments 
    experiments = generate_experiment_variants()
    
    MLRUNS_DIR = FLOW_DIR / "mlruns"
    IMG_DIR = FLOW_DIR / "ml_flow_img"
    IMG_DIR.mkdir(exist_ok=True)


    mlflow.set_tracking_uri(f"sqlite:///{FLOW_DIR}/mlflow.db")

    experiment = mlflow.get_experiment_by_name("hr_attrition")

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            "hr_attrition",
            artifact_location=str(MLRUNS_DIR.resolve())
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment("hr_attrition")
    for exp in experiments:
        
        model, X_test, y_test, y_pred, y_proba = train(
            dataset_name=dataset_name,
            model_config=exp
        )
        metrics = evaluate(y_test, y_pred, y_proba)
        with mlflow.start_run(run_name=f"{exp['name']}{metrics["accuracy"]:.4f}"):

            
            
            mlflow.log_param("model_name", exp["name"])
            mlflow.log_params(exp)        
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            disp = ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred,
                cmap="Oranges", 
                ax=ax,
                colorbar=True
                )
            metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            plt.gcf().text(0.75, 0.75, metrics_text, fontsize=10,bbox=dict(facecolor='white', alpha=0.8))

            img_path = IMG_DIR / f"{exp['name']}_confusion_matrix.png"
            plt.savefig(img_path)
            mlflow.log_artifact(str(img_path))

            mlflow.sklearn.log_model(model, exp["name"])
