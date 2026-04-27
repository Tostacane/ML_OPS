#!/bin/bash

echo "Running pipeline"
python -W ignore -m src.main


if [ $? -ne 0 ]; then
  echo "Error inside pipeline execution. Please check the logs for more details."
  exit 1
fi

echo "Opening ML_Flow, FastAPI and Streamlit in separate tabs"

konsole -p tabtitle="MLflow" -e bash -c "mlflow ui --backend-store-uri sqlite:///ml_flow/mlflow.db --port 5000; exec bash" &
konsole -p tabtitle="FastAPI" -e bash -c "uvicorn deploy.api:app --host 0.0.0.0 --port 8000; exec bash" &
konsole -p tabtitle="Streamlit" -e bash -c "streamlit run streamlit/app.py; exec bash" &