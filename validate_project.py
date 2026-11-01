import sqlite3
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
customers = pd.read_csv(root / "data/clean/customers.csv")
orders = pd.read_csv(root / "data/clean/orders.csv")
products = pd.read_csv(root / "data/clean/products.csv")

checks = []

def check(name, condition):
    assert condition, f"FAILED: {name}"
    checks.append(name)

check("Customers row count", len(customers) == 1200)
check("Customer ages have no missing values", customers["age"].isna().sum() == 0)
check("Customer age range", customers["age"].min() == 18 and customers["age"].max() == 60)
check("Customer gender counts", customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30})
check("Orders row count", len(orders) == 12309)
check("Orders contain no pending status", not orders["status"].eq("pending").any())
check("Order amounts have no missing values", orders["order_amount"].isna().sum() == 0)
check("Completed and cancelled counts", orders["status"].value_counts().to_dict() == {"completed": 11110, "cancelled": 1199})
check("Completed delivery times have no missing values", orders.loc[orders["status"] == "completed", "delivery_minutes"].isna().sum() == 0)
check("Cancelled delivery times are missing", orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().sum() == 1199)
check("Completed revenue", round(orders.loc[orders["status"] == "completed", "order_amount"].sum(), 2) == 2998668.80)
check("Products row count", len(products) == 40)

db_path = root / "data/instamart.db"
with sqlite3.connect(db_path) as conn:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check("SQLite required tables", {"customers", "orders", "products"}.issubset(tables))
    check("SQLite customer count", conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 1200)
    check("SQLite order count", conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309)
    check("SQLite product count", conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 40)

print(f"CHECKPOINTS PASSED: {len(checks)}")
print("ALL CHECKPOINTS PASSED")
