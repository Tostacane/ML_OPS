#!/bin/bash

echo "Running pipeline"
python -W ignore -m src.main


if [ $? -ne 0 ]; then
  echo "Error inside pipeline execution. Please check the logs for more details."
  exit 1
fi

echo "Opening streamlit and mlflow UI"

konsole -e bash -c "mlflow ui --backend-store-uri sqlite:///ml_flow/mlflow.db --port 5000; exec bash" &
konsole -e bash -c "streamlit run streamlit/app.py --server.port 8501; exec bash" &