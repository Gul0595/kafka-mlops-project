from kafka import KafkaProducer
import json
import random
import time
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("Kafka Producer started...")

products = ["Shoes", "T-shirt", "Laptop", "Mobile", "Watch"]
cities = ["Delhi", "Mumbai", "Bangalore", "Pune", "Chennai"]

base_prices = {
    "Shoes": 2000,
    "T-shirt": 800,
    "Laptop": 50000,
    "Mobile": 20000,
    "Watch": 5000
}

city_multiplier = {
    "Delhi": 1.1,
    "Mumbai": 1.2,
    "Bangalore": 1.15,
    "Pune": 1.0,
    "Chennai": 0.95
}

order_id = 1

while True:
    product = random.choice(products)
    city = random.choice(cities)

    price = int(
        base_prices[product]
        * city_multiplier[city]
        * random.uniform(0.9, 1.1)
    )

    quantity = random.randint(1, 4)

    data = {
        "order_id": order_id,
        "product": product,
        "city": city,
        "price": price,
        "quantity": quantity,
        "sales": price * quantity,
        "event_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    producer.send("sales_events", value=data)
    producer.flush()

    print("Sent:", data)

    order_id += 1
    time.sleep(3)   # every 3 seconds
