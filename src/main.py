from src.models.train import train
from ml_flow.mlflow import run_experiment
from logger.logger import configure_logging, get_logger


if __name__ == "__main__":
    # LOGGER SETUP

    configure_logging(
        level = "INFO",
        log_file = "../logs/train.log",
        max_bytes = 2_000_000,
        backup_count = 3,
    )

    log = get_logger(__name__)

    # TRAIN
    log.info("Start Training")
    model, X_test, y_test, y_pred, y_proba = train(dataset_name = "hr_attrition", save_file_name = "hr_log_reg.pkl")
    log.info("Training Complete")

    # MLFLOW
    log.info("Start Evaluating")
    #run_experiment()
    log.info("Evaluation Complete")
