import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

FILES = {
    "customers": RAW_DIR / "customers.csv",
    "orders": RAW_DIR / "orders.csv",
    "products": RAW_DIR / "products.csv",
    "tickets": RAW_DIR / "tickets.csv",
}

for name, path in FILES.items():
    if name == "tickets":
        df = pd.read_csv(path, keep_default_na=False)
    else:
        df = pd.read_csv(path)

    print(f"\n{'=' * 60}")
    print(f"TABLE: {name}")
    print(f"Shape: {df.shape}")

    print("\nColumn audit:")
    audit = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_percentage": (
            df.isna().sum() / len(df) * 100
        ).round(2),
    })

    print(audit.to_string())

# Specific raw-data reference checks
customers = pd.read_csv(FILES["customers"])
orders = pd.read_csv(FILES["orders"])

print(f"\n{'=' * 60}")
print("RAW DATA CHECKPOINTS")

print("Raw order rows:", len(orders))
print("Exact duplicate order rows:", orders.duplicated().sum())
print(
    "Pending order rows:",
    orders["status"].astype(str).str.strip().str.lower()
    .eq("pending").sum()
)
print("Distinct order IDs:", orders["order_id"].nunique())

amount_numeric = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
print("Unparseable order amounts:", amount_numeric.isna().sum())

delivery_numeric = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)
print("Negative delivery times:", (delivery_numeric < 0).sum())
print(
    "Blank delivery times:",
    orders["delivery_minutes"].isna().sum()
)

age_numeric = pd.to_numeric(customers["age"], errors="coerce")
print(
    "Blank ages:",
customers["age"].isna().sum()
)
print(
    "Ages outside 10-100:",
    ((age_numeric < 10) | (age_numeric > 100)).sum()
)
print(
    "Distinct raw city spellings:",
    customers["city"].dropna().astype(str).nunique()
)
print(
    "Distinct nonblank raw gender spellings:",
    customers.loc[
        customers["gender"].astype(str).str.strip().ne(""),
        "gender"
    ].astype(str).nunique()
)
print(
    "Blank genders:",
    customers["gender"].isna().sum()
)

