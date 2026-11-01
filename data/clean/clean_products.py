
import pandas as pd

# Load raw product data
products = pd.read_csv("data/raw/products.csv")

print("Raw products shape:", products.shape)
print("Missing values before cleaning:")
print(products.isna().sum())

# Remove exact duplicate rows
products = products.drop_duplicates().copy()

# Clean text columns
for column in ["product_id", "product_name", "category"]:
    products[column] = (
        products[column]
        .astype("string")
        .str.strip()
    )

# Convert price to numeric
products["price_inr"] = pd.to_numeric(
    products["price_inr"],
    errors="coerce"
)

# Remove rows with missing essential fields
products = products.dropna(
    subset=[
        "product_id",
        "product_name",
        "category",
        "price_inr"
    ]
).copy()

# Remove invalid non-positive prices
products = products[products["price_inr"] > 0].copy()

# Save cleaned products
products.to_csv(
    "data/clean/products.csv",
    index=False
)

# Validation
print("\nCleaned products shape:", products.shape)
print("\nMissing values after cleaning:")
print(products.isna().sum())
print("\nDuplicate product IDs:", products["product_id"].duplicated().sum())
print("\nProducts saved to data/clean/products.csv")
