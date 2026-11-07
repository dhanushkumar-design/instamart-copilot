
"""Part B: Customer segmentation and lapse prediction."""

import json
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from .foundation import BASE, CLEAN


AS_OF = pd.Timestamp("2025-10-01")
SEPTEMBER_START = pd.Timestamp("2025-09-01")

COLS = ["recency_days", "frequency", "monetary"]


def features(orders, customers, cutoff):
    good = orders[
        (orders["status"] == "completed")
        & (orders["order_ts"] < cutoff)
    ].copy()

    good["order_amount"] = pd.to_numeric(
        good["order_amount"], errors="coerce"
    )

    grouped = good.groupby("customer_id")

    result = customers[["customer_id"]].copy()
    result = result.set_index("customer_id")

    result["frequency"] = grouped.size().reindex(
        result.index, fill_value=0
    )

    result["monetary"] = (
        grouped["order_amount"]
        .sum(min_count=1)
        .reindex(result.index)
        .fillna(0)
        .clip(lower=0)
    )

    last_order = grouped["order_ts"].max().reindex(
        result.index
    )

    result["recency_days"] = (
        (cutoff - last_order)
        .dt.days
        .fillna(999)
        .clip(lower=0)
    )

    return result.reset_index()


def customer_segmentation(orders, customers):
    rfm = features(orders, customers, AS_OF)

    X = np.log1p(rfm[COLS])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_stats = (
        rfm.groupby("cluster")[COLS]
        .mean()
        .round(2)
    )

    order = (
        cluster_stats["frequency"]
        .sort_values()
        .index.tolist()
    )

    labels = {
        order[0]: "low_activity",
        order[1]: "occasional",
        order[2]: "regular",
        order[3]: "high_frequency",
    }

    rfm["segment"] = rfm["cluster"].map(labels)

    # PCA for two-dimensional visualization
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled)

    rfm["pca_1"] = coordinates[:, 0]
    rfm["pca_2"] = coordinates[:, 1]

    rfm.to_csv(
        BASE / "reports" / "customer_segments.csv",
        index=False
    )

    return {
        "segment_counts": (
            rfm["segment"].value_counts().to_dict()
        ),
        "pca_explained_variance": [
            round(float(value), 4)
            for value in pca.explained_variance_ratio_
        ],
    }


def lapse_prediction(orders, customers):
    # Features use orders strictly before September.
    customer_features = features(
        orders, customers, SEPTEMBER_START
    )

    # September completed orders are used only for labels.
    september_orders = orders[
        (orders["status"] == "completed")
        & (orders["order_ts"] >= SEPTEMBER_START)
        & (orders["order_ts"] < AS_OF)
    ]

    active_customers = set(
        september_orders["customer_id"]
    )

    # 1 = no completed order in September
    y = (
        ~customer_features["customer_id"].isin(
            active_customers
        )
    ).astype(int)

    X = np.log1p(customer_features[COLS])

    # Hold out test data before model selection.
    X_trainval, X_test, y_trainval, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    # Separate validation data for threshold selection.
    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_trainval,
            y_trainval,
            test_size=0.25,
            random_state=42,
            stratify=y_trainval,
        )
    )

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=42
            ),
            {
                "C": [0.01, 0.1, 1, 10],
                "class_weight": [
                    None, "balanced"
                ],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [4, 8, None],
                "min_samples_leaf": [1, 5, 10],
                "class_weight": [
                    None, "balanced"
                ],
            },
        ),
    }

    best_model = None
    best_name = None
    best_threshold = 0.5
    best_validation_f1 = -1
    best_parameters = {}

    comparison = {}

    for name, (estimator, parameters) in models.items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            scoring="f1",
            cv=cv,
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        model = search.best_estimator_
        val_probabilities = model.predict_proba(
            X_val
        )[:, 1]

        model_best_f1 = -1
        model_best_threshold = 0.5

        for threshold in np.arange(
            0.10, 0.91, 0.05
        ):
            predictions = (
                val_probabilities >= threshold
            ).astype(int)

            score = f1_score(
                y_val,
                predictions,
                zero_division=0
            )

            if score > model_best_f1:
                model_best_f1 = score
                model_best_threshold = float(
                    threshold
                )

        comparison[name] = {
            "cv_f1": round(
                float(search.best_score_), 4
            ),
            "validation_f1": round(
                float(model_best_f1), 4
            ),
        }

        if model_best_f1 > best_validation_f1:
            best_validation_f1 = model_best_f1
            best_model = model
            best_name = name
            best_threshold = model_best_threshold
            best_parameters = search.best_params_

    # Evaluate the selected model once on held-out test data.
    probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    metrics = {
        "selected_model": best_name,
        "best_threshold": round(
            float(best_threshold), 2
        ),
        "f1": round(
            float(f1_score(
                y_test,
                predictions,
                zero_division=0
            )), 4
        ),
        "accuracy": round(
            float(accuracy_score(
                y_test, predictions
            )), 4
        ),
        "precision": round(
            float(precision_score(
                y_test,
                predictions,
                zero_division=0
            )), 4
        ),
        "recall": round(
            float(recall_score(
                y_test,
                predictions,
                zero_division=0
            )), 4
        ),
        "roc_auc": round(
            float(roc_auc_score(
                y_test, probabilities
            )), 4
        ),
        "lapse_rate": round(
            float(y.mean()), 4
        ),
        "validation_f1": round(
            float(best_validation_f1), 4
        ),
        "best_parameters": best_parameters,
        "model_comparison": comparison,
    }

    return metrics


def run():
    orders = pd.read_csv(
        CLEAN / "orders.csv",
        parse_dates=["order_ts"]
    )

    customers = pd.read_csv(
        CLEAN / "customers.csv"
    )

    segmentation = customer_segmentation(
        orders, customers
    )

    prediction = lapse_prediction(
        orders, customers
    )

    metrics = {
        **prediction,
        **segmentation,
    }

    report_path = (
        BASE / "reports" / "model_metrics.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path.write_text(
        json.dumps(metrics, indent=2)
    )

    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
