"""
src/feature_engineering.py
─────────────────────────────────────────────────────────────────────────────
End-to-end feature engineering pipeline built on scikit-learn.

Transformations applied
────────────────────────
1. Impute missing numeric values (median strategy).
2. Scale numeric features with RobustScaler (outlier-resilient).
3. One-hot-encode categoricals (drop first to avoid dummy trap).
4. Engineer domain-specific interaction features:
     • tenure × monthly_charges  → lifetime_value_proxy
     • num_products               → already computed in data gen; kept as-is
     • charges_per_product        → monthly_charges / (num_products + 1)
     • is_fiber_no_security       → fibre user without online_security
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature lists (mirrors config.yaml for easy validation)
# ─────────────────────────────────────────────────────────────────────────────

CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

NUMERICAL_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_products",
    "support_calls",
    "avg_monthly_gb_download",
    "senior_citizen",
]

TARGET = "churn"
DROP_COLUMNS = ["customer_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Domain feature engineering (pre-pipeline step)
# ─────────────────────────────────────────────────────────────────────────────

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hand-crafted interaction and ratio features that carry signal
    beyond what the raw columns provide.

    Returns a *copy* with new columns appended.
    """
    df = df.copy()

    # Lifetime value proxy
    df["lifetime_value_proxy"] = df["tenure"] * df["monthly_charges"]

    # Average charges per subscribed product
    df["charges_per_product"] = df["monthly_charges"] / (df["num_products"] + 1)

    # High-risk segment: fibre internet user with no security add-on
    df["is_fiber_no_security"] = (
        (df["internet_service"] == "Fiber optic") &
        (df["online_security"] == "No")
    ).astype(int)

    # Contract risk score (ordinal encode)
    contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk"] = df["contract"].map(contract_map)

    # Support call intensity (calls per tenure month)
    df["support_call_rate"] = df["support_calls"] / (df["tenure"] + 1)

    logger.debug("Domain features added: %s",
                 ["lifetime_value_proxy", "charges_per_product",
                  "is_fiber_no_security", "contract_risk", "support_call_rate"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# scikit-learn preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────

ENGINEERED_NUMERICAL = [
    "lifetime_value_proxy",
    "charges_per_product",
    "is_fiber_no_security",
    "contract_risk",
    "support_call_rate",
]


def build_feature_pipeline() -> Pipeline:
    """
    Return an unfitted sklearn Pipeline that:
      1. Applies domain feature engineering (FunctionTransformer).
      2. Imputes + scales numerical columns.
      3. One-hot-encodes categorical columns.
    """
    all_numerical = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", drop="first",
                                  sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,  all_numerical),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])

    logger.info("Feature pipeline built (numerical=%d, categorical=%d)",
                len(all_numerical), len(CATEGORICAL_FEATURES))
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def split_X_y(
    df: pd.DataFrame,
    target: str = TARGET,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features X and target y."""
    drop_cols = drop_cols or DROP_COLUMNS
    cols_to_drop = [c for c in drop_cols + [target] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target].astype(int)
    return X, y


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract output feature names after fitting the preprocessor."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())
