import mlflow
import mlflow.pyfunc

# 🔴 CRITICAL: force correct MLflow backend
mlflow.set_tracking_uri("sqlite:///C:/kafka_project/training/mlflow.db")

print("Tracking URI:", mlflow.get_tracking_uri())

model = mlflow.pyfunc.load_model(
    "models:/SalesPredictionModel/Production"
)

print("✅ Model loaded successfully:", model)
