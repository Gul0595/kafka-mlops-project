import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import mlflow
import mlflow.sklearn

# =========================
# 1️⃣ MLFLOW SETUP
# =========================
mlflow.set_tracking_uri("sqlite:///C:/kafka_project/training/mlflow.db")
mlflow.set_experiment("15_min_parallel_model_training")

# =========================
# 2️⃣ MYSQL CONNECTION
# =========================
db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="kafka_db",
    port=3307
)

# =========================
# 3️⃣ LOAD LAST 15 MIN DATA
# =========================
end_time = datetime.now()
start_time = end_time - timedelta(minutes=15)

query = f"""
SELECT price, quantity, sales
FROM sales
WHERE event_time BETWEEN '{start_time}' AND '{end_time}'
"""

df = pd.read_sql(query, db)

print(f"Rows fetched: {len(df)}")

if len(df) < 30:
    print("❌ Not enough data to train models")
    exit()

# =========================
# 4️⃣ FEATURES & TARGET
# =========================
X = df[["price", "quantity"]]
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5️⃣ DEFINE MODELS (PARALLEL)
# =========================
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, random_state=42
    )
}

results = []

# =========================
# 6️⃣ TRAIN & LOG EACH MODEL
# =========================
for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        mape = mean_absolute_percentage_error(y_test, preds) * 100

        # Log params & metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mape", mape)
        mlflow.log_param("rows_used", len(df))

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="SalesPredictionModel"
        )

        results.append((model_name, rmse, mape))

        print(f"✅ {model_name} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

# =========================
# 7️⃣ SHOW COMPARISON
# =========================
results_df = pd.DataFrame(
    results, columns=["Model", "RMSE", "MAPE"]
).sort_values("MAPE")

print("\n🏆 Model Comparison (Best → Worst)")
print(results_df)

