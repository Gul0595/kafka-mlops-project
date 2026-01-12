import time
from datetime import datetime

import pandas as pd
import numpy as np
import mysql.connector
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==================================================
# CONFIGURATION
# ==================================================
WINDOW_SECONDS = 900          # 15 minutes
MIN_ROWS = 10                # minimum rows to train
EXPERIMENT_NAME = "15_min_sales_training"

# ==================================================
# MYSQL CONNECTION (EMPTY PASSWORD)
# ==================================================
def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",          # ✅ EMPTY password
        database="kafka_db",
        port=3307             # change to 3306 if needed
    )

# ==================================================
# MLFLOW SETUP
# ==================================================
mlflow.set_experiment(EXPERIMENT_NAME)

print("🚀 15-minute tumbling window training started")

# ==================================================
# MAIN TRAINING LOOP
# ==================================================
while True:
    now = int(time.time())

    # Calculate FIXED 15-minute window
    window_start = now - (now % WINDOW_SECONDS)
    window_end = window_start + WINDOW_SECONDS

    print(
        f"\n🕒 Training window: "
        f"{datetime.fromtimestamp(window_start)} "
        f"to {datetime.fromtimestamp(window_end)}"
    )

    # ==================================================
    # FETCH DATA FOR THIS WINDOW
    # ==================================================
    conn = get_mysql_connection()

    query = f"""
    SELECT price, quantity
    FROM sales
    WHERE timestamp >= FROM_UNIXTIME({window_start})
      AND timestamp <  FROM_UNIXTIME({window_end})
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print("📊 Rows fetched:", len(df))

    if len(df) < MIN_ROWS:
        print("⚠️ Not enough data. Skipping this window.")
    else:
        # ==================================================
        # FEATURE ENGINEERING
        # ==================================================
        df["target"] = df["price"] * df["quantity"]

        X = df[["price", "quantity"]]
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ==================================================
        # MODEL TRAINING
        # ==================================================
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # RMSE (manual – compatible with all sklearn versions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"✅ Model trained | RMSE = {rmse:.2f}")

        # ==================================================
        # MLFLOW LOGGING
        # ==================================================
        with mlflow.start_run():
            mlflow.log_param("window_start", window_start)
            mlflow.log_param("window_end", window_end)
            mlflow.log_param("rows_used", len(df))

            mlflow.log_metric("rmse", rmse)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="SalesPredictionModel"
            )

            print("📦 Model logged to MLflow")

    # ==================================================
    # WAIT FOR NEXT WINDOW
    # ==================================================
    sleep_time = window_end - int(time.time())
    if sleep_time > 0:
        print(f"😴 Sleeping {sleep_time} seconds till next window...")
        time.sleep(sleep_time)
