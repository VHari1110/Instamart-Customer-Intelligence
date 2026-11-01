
import pandas as pd

# Load the original customer data
customers = pd.read_csv("data/raw/customers.csv")

# 1. Standardise city names
customers["city"] = (
    customers["city"]
    .astype("string")
    .str.strip()
    .str.title()
)

# 2. Standardise gender values
gender_map = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female",
}

customers["gender"] = (
    customers["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna("unknown")
)

# 3. Convert age to numeric
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

# Ages outside 10–100 are invalid
customers.loc[
    ~customers["age"].between(10, 100), "age"
] = pd.NA

# Fill missing ages using each city's median
city_median_age = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median_age)

# 4. Convert signup_date to datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# Save cleaned data; keep the raw file unchanged
customers.to_csv("data/clean/customers.csv", index=False)

# Basic validation
print("Cleaned customer rows:", len(customers))
print("\nCity counts:")
print(customers["city"].value_counts().sort_index())

print("\nGender counts:")
print(customers["gender"].value_counts().sort_index())

print("\nMissing ages:", customers["age"].isna().sum())
print("Age range:", customers["age"].min(), "to", customers["age"].max())
print("Signup date dtype:", customers["signup_date"].dtype)
