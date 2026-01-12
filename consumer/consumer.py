from kafka import KafkaConsumer
import json
import mysql.connector

# =========================
# 1️⃣ CONNECT TO MySQL
# =========================
db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="kafka_db",
    port=3307
)


cursor = db.cursor()
print("Connected to MySQL successfully")

# =========================
# 2️⃣ KAFKA CONSUMER
# =========================
consumer = KafkaConsumer(
    "sales_events",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="sales-mysql-group",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("Kafka Consumer started... Storing data into MySQL")

# =========================
# 3️⃣ INSERT INTO MySQL
# =========================
from datetime import datetime

for message in consumer:
    data = message.value

    # ✅ derive sales
    sales = data["price"] * data["quantity"]

    # ✅ handle schema evolution safely
    event_time = (
        data.get("event_time") or
        data.get("timestamp") or
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    sql = """
    INSERT INTO sales 
    (order_id, product, city, price, quantity, sales, event_time)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        data["order_id"],
        data["product"],
        data["city"],
        data["price"],
        data["quantity"],
        sales,
        event_time
    )

    cursor.execute(sql, values)
    db.commit()

    print("Inserted into MySQL:", data)


