
from pathlib import Path
import sqlite3
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
OUTPUT_DIR = ROOT / "outputs"
DB_PATH = DATA_DIR / "instamart.db"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")
RANDOM_STATE = 42


def load_data():
    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    orders["status"] = (
        orders["status"].astype(str).str.strip().str.lower()
    )

    products["product_id"] = products["product_id"].astype(str)
    orders["product_id"] = orders["product_id"].astype(str)

    return customers, orders, products


def build_lapse_dataset(customers, orders, products):
    # Features use only orders strictly before the cutoff.
    historical = orders[
        (orders["order_ts"] < CUTOFF)
        & orders["order_ts"].notna()
    ].copy()

    completed_history = historical[
        historical["status"] == "completed"
    ].copy()

    # Cohort: customers with at least one completed order before cutoff.
    cohort_ids = completed_history["customer_id"].unique()
    cohort = customers[
        customers["customer_id"].isin(cohort_ids)
    ].copy()

    # Label: no completed order from September 1 through September 30.
    september_completed = orders[
        (orders["order_ts"] >= CUTOFF)
        & (orders["order_ts"] < LABEL_END)
        & (orders["status"] == "completed")
    ]

    active_in_september = set(
        september_completed["customer_id"].unique()
    )

    cohort["lapsed"] = (
        ~cohort["customer_id"].isin(active_in_september)
    ).astype(int)

    # Completed-order RFM and behavior features, all before cutoff.
    completed_history["order_amount"] = pd.to_numeric(
        completed_history["order_amount"], errors="coerce"
    )
    completed_history["delivery_minutes"] = pd.to_numeric(
        completed_history["delivery_minutes"], errors="coerce"
    )
    completed_history["discount_pct"] = pd.to_numeric(
        completed_history["discount_pct"], errors="coerce"
    )

    last_order = completed_history.groupby("customer_id")[
        "order_ts"
    ].max()

    frequency = completed_history.groupby("customer_id")[
        "order_id"
    ].nunique()

    monetary = completed_history.groupby("customer_id")[
        "order_amount"
    ].sum(min_count=1)

    avg_discount = completed_history.groupby("customer_id")[
        "discount_pct"
    ].mean()

    avg_delivery = completed_history.groupby("customer_id")[
        "delivery_minutes"
    ].mean()

    # Cancellation rate among all finalized historical orders.
    finalized = historical[
        historical["status"].isin(["completed", "cancelled"])
    ]

    status_counts = finalized.pivot_table(
        index="customer_id",
        columns="status",
        values="order_id",
        aggfunc="nunique",
        fill_value=0,
    )

    for status in ["completed", "cancelled"]:
        if status not in status_counts.columns:
            status_counts[status] = 0

    status_counts["cancellation_rate"] = (
        status_counts["cancelled"]
        / (
            status_counts["cancelled"]
            + status_counts["completed"]
        ).replace(0, np.nan)
    )

    # Most frequently ordered product category before cutoff.
    category_orders = completed_history.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
    )

    favorite_category = (
        category_orders.groupby(["customer_id", "category"])
        .size()
        .reset_index(name="category_order_count")
        .sort_values(
            ["customer_id", "category_order_count", "category"],
            ascending=[True, False, True],
        )
        .drop_duplicates("customer_id")
        .set_index("customer_id")["category"]
    )

    features = cohort.set_index("customer_id").copy()

    features["recency"] = (
        CUTOFF - last_order.reindex(features.index)
    ).dt.days

    features["frequency"] = frequency.reindex(features.index)
    features["monetary"] = monetary.reindex(features.index)
    features["avg_discount_pct"] = avg_discount.reindex(features.index)
    features["avg_delivery_minutes"] = avg_delivery.reindex(
        features.index
    )
    features["cancellation_rate"] = status_counts[
        "cancellation_rate"
    ].reindex(features.index)
    features["favorite_category"] = favorite_category.reindex(
        features.index
    )

    features["tenure_days"] = (
        CUTOFF - features["signup_date"]
    ).dt.days

    features["age"] = pd.to_numeric(
        features["age"], errors="coerce"
    )

    # Keep customer_id and label separately from model features.
    labels = features["lapsed"].astype(int)
    customer_ids = features.index.to_series().rename("customer_id")

    feature_columns = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "favorite_category",
        "city",
        "age",
        "gender",
        "tenure_days",
    ]

    X = features[feature_columns].copy()

    # Ensure numeric columns are numeric.
    numeric_columns = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "age",
        "tenure_days",
    ]

    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    # Remove invalid ages/tenure values rather than leaking labels.
    X["age"] = X["age"].where(X["age"].between(0, 120))
    X["tenure_days"] = X["tenure_days"].where(
        X["tenure_days"] >= 0
    )

    return X, labels, customer_ids, features


def make_preprocessor():
    numeric_features = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "age",
        "tenure_days",
    ]

    categorical_features = [
        "city",
        "gender",
        "favorite_category",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Not lapsed", "Lapsed"],
            zero_division=0,
        )
    )

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }


def main():
    print("Loading cleaned Instamart data...")
    customers, orders, products = load_data()

    X, y, customer_ids, full_features = build_lapse_dataset(
        customers, orders, products
    )

    print("\nLapse dataset")
    print("-" * 40)
    print("Cohort size:", len(X))
    print("Lapsed customers:", int(y.sum()))
    print("Lapsed percentage:", round(y.mean() * 100, 2), "%")
    print("Class counts:")
    print(y.value_counts().sort_index())

    if y.nunique() != 2:
        raise ValueError(
            "Lapse dataset must contain both classes (0 and 1)."
        )

    # Baseline: always predict not lapsed.
    baseline_predictions = np.zeros(len(y), dtype=int)
    print("\nBaseline (always predict not lapsed)")
    print("Accuracy:", round(accuracy_score(y, baseline_predictions), 4))
    print(
        "Lapsed-class F1:",
        round(
            f1_score(y, baseline_predictions, zero_division=0),
            4,
        ),
    )

    X_train, X_test, y_train, y_test, ids_train, ids_test = (
        train_test_split(
            X,
            y,
            customer_ids,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    print("\nTrain rows:", len(X_train))
    print("Test rows:", len(X_test))

    # The preprocessing pipeline is fitted only inside CV/training folds.
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    lr_grid = GridSearchCV(
        estimator=lr_pipeline,
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )

    print("\nTuning Logistic Regression...")
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    print("Best LR parameters:", lr_grid.best_params_)
    print("Best LR CV F1:", round(lr_grid.best_score_, 4))

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    rf_grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid={
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20],
        },
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )

    print("\nTuning Random Forest...")
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print("Best RF parameters:", rf_grid.best_params_)
    print("Best RF CV F1:", round(rf_grid.best_score_, 4))

    lr_metrics = evaluate_model(
        "Tuned Logistic Regression", best_lr, X_test, y_test
    )
    rf_metrics = evaluate_model(
        "Tuned Random Forest", best_rf, X_test, y_test
    )

    # Select the model using training-only cross-validation F1.
    if rf_grid.best_score_ >= lr_grid.best_score_:
        selected_model = best_rf
        selected_name = "Random Forest"
    else:
        selected_model = best_lr
        selected_name = "Logistic Regression"

    print("\nSelected model:", selected_name)
    print(
        "Business reason: F1 balances precision and recall. "
        "For retention, missing a genuinely lapsed customer can mean "
        "lost future revenue, while false positives can waste outreach "
        "discounts and campaign effort."
    )

    # Oversample only the training split for a separate logistic-regression
    # experiment. Test rows remain unchanged.
    print("\nTraining-only oversampling experiment...")
    train_frame = X_train.copy()
    train_frame["_label"] = y_train.to_numpy()

    class_counts = train_frame["_label"].value_counts()
    minority_class = class_counts.idxmin()
    majority_count = int(class_counts.max())
    minority_count = int(class_counts.min())

    minority_rows = train_frame[
        train_frame["_label"] == minority_class
    ]

    extra_rows = minority_rows.sample(
        n=majority_count - minority_count,
        replace=True,
        random_state=RANDOM_STATE,
    )

    balanced_train = pd.concat(
        [train_frame, extra_rows],
        ignore_index=True,
    )

    X_balanced = balanced_train.drop(columns="_label")
    y_balanced = balanced_train["_label"].astype(int)

    balanced_lr = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    # Keep the best C found on the original training data.
    best_c = best_lr.named_steps["model"].C
    balanced_lr.set_params(model__C=best_c)
    balanced_lr.fit(X_balanced, y_balanced)

    print("Original training class counts:")
    print(y_train.value_counts().sort_index())
    print("Balanced training class counts:")
    print(y_balanced.value_counts().sort_index())

    unbalanced_lr_metrics = evaluate_model(
        "Unbalanced Logistic Regression",
        best_lr,
        X_test,
        y_test,
    )
    balanced_lr_metrics = evaluate_model(
        "Oversampled Logistic Regression",
        balanced_lr,
        X_test,
        y_test,
    )

    print("\nOversampling comparison (lapsed class)")
    print(
        pd.DataFrame(
            [
                {
                    "model": "Unbalanced LR",
                    "precision": unbalanced_lr_metrics["precision"],
                    "recall": unbalanced_lr_metrics["recall"],
                    "f1": unbalanced_lr_metrics["f1"],
                },
                {
                    "model": "Oversampled LR",
                    "precision": balanced_lr_metrics["precision"],
                    "recall": balanced_lr_metrics["recall"],
                    "f1": balanced_lr_metrics["f1"],
                },
            ]
        ).round(4).to_string(index=False)
    )

    # Explain the scaler's training-only frequency mean.
    fitted_scaler = (
        best_lr.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )
    numeric_features = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "age",
        "tenure_days",
    ]
    frequency_position = numeric_features.index("frequency")
    scaler_frequency_mean = fitted_scaler.mean_[frequency_position]

    print("\nFrequency mean comparison")
    print(
        "Full cohort frequency mean:",
        round(X["frequency"].mean(), 4),
    )
    print(
        "Training-only fitted scaler frequency mean:",
        round(float(scaler_frequency_mean), 4),
    )
    print(
        "These may differ because the scaler learns only from training "
        "rows (and CV fits only on each training fold), not the full cohort."
    )

    # Logistic regression coefficient interpretation.
    lr_preprocessor = best_lr.named_steps["preprocessor"]
    lr_classifier = best_lr.named_steps["model"]
    try:
        transformed_names = lr_preprocessor.get_feature_names_out()
        coefficients = pd.Series(
            lr_classifier.coef_[0],
            index=transformed_names,
        ).sort_values()

        print("\nLogistic Regression coefficient interpretation")
        for feature, coefficient in pd.concat(
            [coefficients.head(2), coefficients.tail(1)]
        ).items():
            direction = "increases" if coefficient > 0 else "decreases"
            print(
                f"{feature}: coefficient={coefficient:.4f}; "
                f"associated with {direction} odds of lapse, "
                "holding other model features constant."
            )
    except Exception as exc:
        print("Could not extract LR coefficients:", exc)

    # Random Forest feature importance.
    rf_preprocessor = best_rf.named_steps["preprocessor"]
    rf_classifier = best_rf.named_steps["model"]
    rf_feature_names = rf_preprocessor.get_feature_names_out()
    importances = pd.Series(
        rf_classifier.feature_importances_,
        index=rf_feature_names,
    ).sort_values(ascending=False)

    print("\nTop 5 Random Forest features")
    for feature, importance in importances.head(5).items():
        print(
            f"{feature}: importance={importance:.4f}. "
            "This feature contributes to the model's splits; "
            "importance alone does not establish causation."
        )

    # Save the selected tuned model and verify reload predictions.
    model_path = OUTPUT_DIR / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)
    reloaded_model = joblib.load(model_path)

    original_predictions = selected_model.predict(X_test)
    reloaded_predictions = reloaded_model.predict(X_test)

    print("\nModel saved:", model_path)
    print(
        "Reloaded predictions identical:",
        np.array_equal(original_predictions, reloaded_predictions),
    )

    # Retention list: test-set customers in the whole-window Premium segment.
    # This Premium flag is post-hoc and uses whole-window RFM, including
    # September. It is only for an illustrative list, not model features.
    segments_path = OUTPUT_DIR / "customer_segments.csv"
    if not segments_path.exists():
        raise FileNotFoundError(
            "outputs/customer_segments.csv not found. "
            "Run customer_segments.py first."
        )

    segments = pd.read_csv(segments_path)

    if "customer_id" not in segments.columns:
        raise ValueError(
            "customer_segments.csv must contain customer_id."
        )

    if "premium_flag" in segments.columns:
        premium_ids = set(
            segments.loc[
                pd.to_numeric(
                    segments["premium_flag"], errors="coerce"
                ).fillna(0).astype(int) == 1,
                "customer_id",
            ]
        )
    elif "is_premium" in segments.columns:
        premium_ids = set(
            segments.loc[
                pd.to_numeric(
                    segments["is_premium"], errors="coerce"
                ).fillna(0).astype(int) == 1,
                "customer_id",
            ]
        )
    else:
        raise ValueError(
            "customer_segments.csv needs premium_flag or is_premium."
        )

    test_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

    retention = pd.DataFrame(
        {
            "customer_id": ids_test.to_numpy(),
            "lapse_probability": test_probabilities,
            "actually_lapsed": y_test.to_numpy(),
        }
    )

    retention = retention[
        retention["customer_id"].isin(premium_ids)
    ].sort_values("lapse_probability", ascending=False).head(10)

    retention_path = OUTPUT_DIR / "retention_list.csv"
    retention.to_csv(retention_path, index=False)

    print("\nRetention list saved:", retention_path)
    print(retention.to_string(index=False))
    print(
        "\nLIMITATION: Premium membership is based on whole-window RFM "
        "and can include September data. The retention list is illustrative "
        "and is not a leakage-free evaluation. Model features and labels "
        "are kept separate; features use only pre-cutoff orders."
    )


if __name__ == "__main__":
    main()
