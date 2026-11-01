import pandas as pd

# Load raw data
orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")

# Convert order timestamp
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# Remove exact duplicate rows
orders = orders.drop_duplicates()

# Keep the latest record for each order_id
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(
    subset="order_id", keep="last"
).copy()

# Remove pending orders
orders = orders[orders["status"] != "pending"].copy()

# Convert order amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Match products to orders
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Fill missing amounts using price, quantity and discount
missing_amount = orders["order_amount"].isna()

orders.loc[missing_amount, "order_amount"] = (
    orders.loc[missing_amount, "price_inr"]
    * orders.loc[missing_amount, "quantity"]
    * (
        1
        - orders.loc[missing_amount, "discount_pct"] / 100
    )
)

# Convert delivery time to numeric
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Negative delivery times are invalid
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

# Calculate category medians using completed orders
completed = orders["status"] == "completed"

category_medians = orders.loc[completed].groupby(
    "category"
)["delivery_minutes"].median()

# Impute missing delivery times for completed orders
missing_completed = (
    completed & orders["delivery_minutes"].isna()
)

orders.loc[
    missing_completed, "delivery_minutes"
] = orders.loc[
    missing_completed, "category"
].map(category_medians)

# Cancelled orders should have missing delivery times
orders.loc[
    orders["status"] == "cancelled",
    "delivery_minutes"
] = float("nan")

# Remove helper columns
orders = orders.drop(
    columns=["price_inr", "category"]
)

# Save cleaned data
orders.to_csv("data/clean/orders.csv", index=False)

# Validation
print("Cleaned order rows:", len(orders))
print("\nStatus counts:")
print(orders["status"].value_counts().sort_index())

print(
    "\nMissing order amounts:",
    orders["order_amount"].isna().sum()
)

print(
    "Missing completed delivery times:",
    orders.loc[
        orders["status"] == "completed",
        "delivery_minutes"
    ].isna().sum()
)

print(
    "Cancelled rows with missing delivery times:",
    orders.loc[
        orders["status"] == "cancelled",
        "delivery_minutes"
    ].isna().sum()
)

print(
    "Completed order amount total:",
    round(
        orders.loc[
            orders["status"] == "completed",
            "order_amount"
        ].sum(),
        2
    )
)