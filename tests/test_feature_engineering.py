"""
tests/test_feature_engineering.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the feature engineering pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_generation import generate_dataset
from src.feature_engineering import (
    add_domain_features,
    build_feature_pipeline,
    get_feature_names,
    split_X_y,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ENGINEERED_NUMERICAL,
)


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return generate_dataset(n_samples=300, random_state=7)


@pytest.fixture(scope="module")
def X_y(raw_df):
    return split_X_y(raw_df)


class TestAddDomainFeatures:
    def test_new_columns_added(self, raw_df):
        X, _ = split_X_y(raw_df)
        X_eng = add_domain_features(X)
        for col in ["lifetime_value_proxy", "charges_per_product",
                    "is_fiber_no_security", "contract_risk", "support_call_rate"]:
            assert col in X_eng.columns, f"Missing engineered column: {col}"

    def test_no_mutation(self, raw_df):
        X, _ = split_X_y(raw_df)
        X_copy = X.copy()
        _ = add_domain_features(X)
        pd.testing.assert_frame_equal(X, X_copy)

    def test_lifetime_value_proxy(self, raw_df):
        X, _ = split_X_y(raw_df)
        X_eng = add_domain_features(X)
        expected = X["tenure"] * X["monthly_charges"]
        pd.testing.assert_series_equal(
            X_eng["lifetime_value_proxy"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_contract_risk_values(self, raw_df):
        X, _ = split_X_y(raw_df)
        X_eng = add_domain_features(X)
        assert set(X_eng["contract_risk"].unique()).issubset({0, 1, 2})


class TestBuildFeaturePipeline:
    def test_returns_pipeline(self):
        pipe = build_feature_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_fit_transform_shape(self, raw_df):
        X, y = split_X_y(raw_df)
        X_eng = add_domain_features(X)
        pipe = build_feature_pipeline()
        X_tr = pipe.fit_transform(X_eng)
        # Expect columns > raw feature count (OHE expands categoricals)
        n_raw = len(NUMERICAL_FEATURES) + len(ENGINEERED_NUMERICAL)
        assert X_tr.shape[0] == len(X_eng)
        assert X_tr.shape[1] > n_raw

    def test_no_nans_after_transform(self, raw_df):
        X, _ = split_X_y(raw_df)
        X_eng = add_domain_features(X)
        pipe = build_feature_pipeline()
        X_tr = pipe.fit_transform(X_eng)
        assert not np.isnan(X_tr).any(), "Transformed features contain NaN."


class TestSplitXY:
    def test_target_is_series(self, raw_df):
        _, y = split_X_y(raw_df)
        assert isinstance(y, pd.Series)

    def test_customer_id_dropped(self, raw_df):
        X, _ = split_X_y(raw_df)
        assert "customer_id" not in X.columns

    def test_churn_dropped_from_X(self, raw_df):
        X, _ = split_X_y(raw_df)
        assert "churn" not in X.columns

    def test_y_binary(self, raw_df):
        _, y = split_X_y(raw_df)
        assert set(y.unique()).issubset({0, 1})
