"""
tests/test_data_generation.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the data generation module.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generation import generate_dataset


class TestGenerateDataset:
    """Tests for generate_dataset()."""

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return generate_dataset(n_samples=500, random_state=0)

    def test_row_count(self, df):
        assert len(df) == 500, "Expected exactly 500 rows."

    def test_required_columns(self, df):
        required = [
            "customer_id", "gender", "tenure", "monthly_charges",
            "total_charges", "churn", "contract", "internet_service",
        ]
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_churn_is_binary(self, df):
        assert set(df["churn"].unique()).issubset({0, 1}), \
            "Churn column must contain only 0 and 1."

    def test_churn_rate_in_range(self, df):
        rate = df["churn"].mean()
        assert 0.15 <= rate <= 0.40, \
            f"Churn rate {rate:.2%} outside expected range [15%, 40%]."

    def test_tenure_positive(self, df):
        assert (df["tenure"] >= 1).all(), "Tenure must be >= 1."

    def test_monthly_charges_positive(self, df):
        assert (df["monthly_charges"] > 0).all(), "Monthly charges must be positive."

    def test_total_charges_has_some_nulls(self, df):
        """We intentionally inject ~1.5% missingness into total_charges."""
        null_pct = df["total_charges"].isna().mean()
        assert 0.001 <= null_pct <= 0.05, \
            f"Expected small % of NaN in total_charges, got {null_pct:.2%}."

    def test_customer_ids_unique(self, df):
        assert df["customer_id"].nunique() == len(df), \
            "customer_id must be unique for every row."

    def test_reproducibility(self):
        df1 = generate_dataset(n_samples=200, random_state=99)
        df2 = generate_dataset(n_samples=200, random_state=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_dataset(n_samples=200, random_state=1)
        df2 = generate_dataset(n_samples=200, random_state=2)
        assert not df1["churn"].equals(df2["churn"]), \
            "Different seeds should produce different churn labels."
