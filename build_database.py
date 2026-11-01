
import sqlite3
from pathlib import Path
import pandas as pd

# Project paths
DATA_DIR = Path("data")
CLEAN_DIR = DATA_DIR / "clean"
DB_PATH = DATA_DIR / "instamart.db"

# Ensure the data folder exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Read cleaned CSV files
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Create SQLite database and tables
with sqlite3.connect(DB_PATH) as connection:
    customers.to_sql(
        "customers", connection, if_exists="replace", index=False
    )
    products.to_sql(
        "products", connection, if_exists="replace", index=False
    )
    orders.to_sql(
        "orders", connection, if_exists="replace", index=False
    )

    # Verify table row counts
    cursor = connection.cursor()

    for table in ["customers", "products", "orders"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count} rows")

print(f"\nDatabase created: {DB_PATH}")
