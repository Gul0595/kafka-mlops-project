# ============================================
# Kafka MLOps – Parallel Model Training Script
# CI Safe | Beginner Friendly
# ============================================

import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import mysql.connector


# ============================================
# 1️⃣ MLflow Setup
# ============================================
mlflow.set_tracking_uri("sqlite:///training/mlflow.db")
mlflow.set_experiment("15_min_parallel_model_training")


# ============================================
# 2️⃣ Detect CI Environment
# ============================================
IS_CI = os.getenv("CI", "false").lower() == "true"


# ============================================
# 3️⃣ MySQL Connection (CI-safe)
# ============================================
def get_mysql_connection():
    if IS_CI:
        print("CI environment detected → skipping MySQL connection")
        return None

    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="kafka_db",
            port=3307
        )
    except Exception as e:
        print("MySQL connection failed:", e)
        return None


# ============================================
# 4️⃣ Load Last 15 Minutes Data
# ============================================
end_time = datetime.now()
start_time = end_time - timedelta(minutes=15)

query = f"""
SELECT price, quantity, sales
FROM sales
WHERE event_time BETWEEN '{start_time}' AND '{end_time}'
"""

conn = get_mysql_connection()

if conn is None:
    print("Using dummy data for CI run")
    df = pd.DataFrame({
        "price": [1000, 2000, 3000, 4000, 5000, 6000],
        "quantity": [1, 2, 1, 3, 2, 4],
        "sales": [1000, 4000, 3000, 12000, 10000, 24000]
    })
else:
    df = pd.read_sql(query, conn)

print(f"Rows fetched: {len(df)}")


# ====================
