
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
DATA_DIR = Path("data/clean")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load cleaned data
# --------------------------------------------------
customers = pd.read_csv(DATA_DIR / "customers.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

completed_orders = orders[
    (orders["status"] == "completed")
    & orders["order_ts"].notna()
    & orders["order_amount"].notna()
].copy()

print("Customers loaded:", len(customers))
print("Orders loaded:", len(orders))
print("Completed orders used:", len(completed_orders))

# --------------------------------------------------
# 3. Calculate RFM
# --------------------------------------------------
if completed_orders.empty:
    raise ValueError("No valid completed orders found.")

analysis_date = pd.Timestamp("2025-10-01")
rfm = (
    completed_orders.groupby("customer_id")
    .agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (
    analysis_date - rfm["last_order"].dt.normalize()
).dt.days

rfm["monetary"] = rfm["monetary"].round(2)

# --------------------------------------------------
# 4. Add customers who have no completed orders
# --------------------------------------------------
rfm = customers[["customer_id"]].merge(
    rfm,
    on="customer_id",
    how="left",
    validate="one_to_one",
)

rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
rfm["monetary"] = rfm["monetary"].fillna(0).round(2)

# Recency is undefined for customers without completed orders.
# Keep it missing instead of inventing a last-order date.
rfm["recency"] = rfm["recency"].astype("Int64")

# --------------------------------------------------
# 5. Create RFM scores
# --------------------------------------------------
# Higher recency score means a more recent purchase.
# If all values are identical, assign the middle score.
def make_score(series, higher_is_better=True):
    valid = series.dropna()

    if valid.nunique() < 2:
        result = pd.Series(3, index=series.index, dtype="Int64")
        result.loc[series.isna()] = pd.NA
        return result

    ranks = valid.rank(method="average", pct=True)

    if not higher_is_better:
        ranks = 1 - ranks + (1 / len(valid))

    scores = np.ceil(ranks * 5).clip(1, 5).astype(int)

    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    result.loc[valid.index] = scores.astype("Int64")
    return result

rfm["R_score"] = make_score(rfm["recency"], higher_is_better=False)
rfm["F_score"] = make_score(rfm["frequency"], higher_is_better=True)
rfm["M_score"] = make_score(rfm["monetary"], higher_is_better=True)

rfm["RFM_score"] = (
    rfm["R_score"].astype("string")
    + rfm["F_score"].astype("string")
    + rfm["M_score"].astype("string")
)

# --------------------------------------------------
# 6. Prepare data for K-Means clustering
# --------------------------------------------------

# Keep only customers with valid RFM recency.
rfm = rfm.dropna(subset=["recency"]).copy()

cluster_features = rfm[
    ["recency", "frequency", "monetary"]
].copy()

# Scale the original RFM features without log transformation.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)


# Compare K=2 to K=8.
from sklearn.metrics import silhouette_score

cluster_results = []

for k in range(2, 9):
    if np.unique(scaled_features, axis=0).shape[0] < k:
        continue

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(scaled_features)

    score = silhouette_score(scaled_features, labels)

    cluster_results.append({
        "K": k,
        "Inertia": model.inertia_,
        "Silhouette": score,
    })

cluster_comparison = pd.DataFrame(cluster_results)

print("\nK-Means comparison (K=2 to K=8):")
print(cluster_comparison.round(4).to_string(index=False))

# Select the best silhouette score among K=3, 4, and 5.
eligible = cluster_comparison[
    cluster_comparison["K"].isin([3, 4, 5])
]

if eligible.empty:
    raise ValueError("No valid K available among 3, 4, and 5.")

best_k = int(
    eligible.loc[eligible["Silhouette"].idxmax(), "K"]
)

print("\nSelected K:", best_k)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10,
)

rfm["segment"] = kmeans.fit_predict(scaled_features)

# --------------------------------------------------
# --------------------------------------------------
# 7. PCA: reduce the scaled features to two dimensions
# --------------------------------------------------
if scaled_features.shape[0] >= 2 and scaled_features.shape[1] >= 2:
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled_features)
    rfm["PCA1"] = coordinates[:, 0]
    rfm["PCA2"] = coordinates[:, 1]
else:
    rfm["PCA1"] = 0.0
    rfm["PCA2"] = 0.0

# --------------------------------------------------
# 8. Add customer details
# --------------------------------------------------
rfm = rfm.merge(
    customers[["customer_id", "city", "age", "gender"]],
    on="customer_id",
    how="left",
    validate="one_to_one",
)
# --------------------------------------------------
# 8A. Identify Premium customers
# --------------------------------------------------
segment_monetary = rfm.groupby("segment")["monetary"].mean()
premium_segment = segment_monetary.idxmax()

rfm["premium_flag"] = (
    rfm["segment"] == premium_segment
).astype(int)

print("\nAverage monetary value by segment:")
print(segment_monetary.round(2))

print("\nPremium segment:", premium_segment)
print("Premium customers:", rfm["premium_flag"].sum())
# --------------------------------------------------
# --------------------------------------------------
# 9. Prepare assignment-required output columns
# --------------------------------------------------

rfm["segment_id"] = rfm["segment"]

# Assign segment names based on average monetary value.
segment_order = (
    rfm.groupby("segment")["monetary"]
    .mean()
    .sort_values()
    .index.tolist()
)

if len(segment_order) == 3:
    names = [
        "Low Activity Customers",
        "Regular Customers",
        "Premium Customers",
    ]
elif len(segment_order) == 4:
    names = [
        "Low Activity Customers",
        "Occasional Customers",
        "Regular Customers",
        "Premium Customers",
    ]
elif len(segment_order) == 5:
    names = [
        "Low Activity Customers",
        "Occasional Customers",
        "Regular Customers",
        "High Value Customers",
        "Premium Customers",
    ]
else:
    raise ValueError("Unexpected number of customer segments.")

segment_name_map = dict(zip(segment_order, names))
rfm["segment_name"] = rfm["segment"].map(segment_name_map)

rfm["is_premium"] = rfm["premium_flag"].astype(int)

output_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium",
]

output_path = OUTPUT_DIR / "customer_segments.csv"
rfm[output_columns].to_csv(output_path, index=False)

# --------------------------------------------------
# 10. Display validation and summary
# --------------------------------------------------
print("\nRFM preview:")
print(rfm[output_columns].head(10).to_string(index=False))

print("\nCustomers in output:", len(rfm))
print("Missing recency values:", rfm["recency"].isna().sum())
print("Missing frequency values:", rfm["frequency"].isna().sum())
print("Missing monetary values:", rfm["monetary"].isna().sum())

print("\nSegment counts:")
print(rfm["segment_id"].value_counts().sort_index())

print("\nPremium flag counts:")
print(rfm["is_premium"].value_counts().sort_index())

print("\nPCA explained variance ratio:")
if "pca" in locals():
    print(pca.explained_variance_ratio_)
else:
    print("PCA not calculated: insufficient rows or features.")

print("\nSaved:", output_path)