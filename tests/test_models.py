"""
tests/test_models.py
─────────────────────────────────────────────────────────────────────────────
Integration tests for the training and evaluation modules.
Uses a small 600-sample dataset so tests finish quickly in CI.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_generation import generate_dataset
from src.evaluate import compare_models, evaluate_model, threshold_search
from src.feature_engineering import split_X_y
from src.train import train_random_forest, train_xgboost


@pytest.fixture(scope="module")
def splits():
    df = generate_dataset(n_samples=600, random_state=42)
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


@pytest.fixture(scope="module")
def rf_pipe(splits):
    X_train, _, _, y_train, _, _ = splits
    return train_random_forest(
        X_train, y_train,
        params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
        run_cv=False,
    )


@pytest.fixture(scope="module")
def xgb_pipe(splits):
    X_train, _, _, y_train, _, _ = splits
    return train_xgboost(
        X_train, y_train,
        params={"n_estimators": 50, "max_depth": 4, "random_state": 42},
        run_cv=False,
    )


class TestTrainRandomForest:
    def test_returns_pipeline(self, rf_pipe):
        assert isinstance(rf_pipe, Pipeline)

    def test_has_predict_proba(self, rf_pipe, splits):
        X_train, _, _, _, _, _ = splits
        from src.feature_engineering import add_domain_features
        X_eng = add_domain_features(X_train)
        proba = rf_pipe.predict_proba(X_eng)
        assert proba.shape[1] == 2

    def test_proba_sums_to_one(self, rf_pipe, splits):
        X_train, _, _, _, _, _ = splits
        from src.feature_engineering import add_domain_features
        X_eng = add_domain_features(X_train)
        proba = rf_pipe.predict_proba(X_eng)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestTrainXGBoost:
    def test_returns_pipeline(self, xgb_pipe):
        assert isinstance(xgb_pipe, Pipeline)

    def test_proba_in_zero_one(self, xgb_pipe, splits):
        X_train, _, _, _, _, _ = splits
        from src.feature_engineering import add_domain_features
        X_eng = add_domain_features(X_train)
        proba = xgb_pipe.predict_proba(X_eng)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()


class TestEvaluateModel:
    def test_metrics_keys(self, rf_pipe, splits):
        _, _, X_test, _, _, y_test = splits
        m = evaluate_model(rf_pipe, X_test, y_test, verbose=False)
        for key in ["accuracy", "roc_auc", "f1", "recall", "precision"]:
            assert key in m, f"Missing metric: {key}"

    def test_accuracy_reasonable(self, rf_pipe, splits):
        _, _, X_test, _, _, y_test = splits
        m = evaluate_model(rf_pipe, X_test, y_test, verbose=False)
        assert m["accuracy"] >= 0.60, "Accuracy below 60% – something is wrong."

    def test_roc_auc_above_random(self, xgb_pipe, splits):
        _, _, X_test, _, _, y_test = splits
        m = evaluate_model(xgb_pipe, X_test, y_test, verbose=False)
        assert m["roc_auc"] >= 0.70, "AUC below 0.70 – model performs worse than expected."


class TestCompareModels:
    def test_returns_dataframe(self, rf_pipe, xgb_pipe, splits):
        _, _, X_test, _, _, y_test = splits
        df = compare_models(
            {"RF": rf_pipe, "XGB": xgb_pipe}, X_test, y_test
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_sorted_by_auc(self, rf_pipe, xgb_pipe, splits):
        _, _, X_test, _, _, y_test = splits
        df = compare_models(
            {"RF": rf_pipe, "XGB": xgb_pipe}, X_test, y_test
        )
        aucs = df["roc_auc"].tolist()
        assert aucs == sorted(aucs, reverse=True)


class TestThresholdSearch:
    def test_threshold_in_range(self, xgb_pipe, splits):
        _, X_val, _, _, y_val, _ = splits
        t = threshold_search(xgb_pipe, X_val, y_val, metric="f1")
        assert 0.10 <= t <= 0.90
