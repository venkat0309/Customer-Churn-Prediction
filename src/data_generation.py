"""
src/data_generation.py
─────────────────────────────────────────────────────────────────────────────
Generates a realistic synthetic telecom churn dataset with 10 000+ records.

Key design decisions
────────────────────
• Relationships mirror real Telco patterns (e.g. month-to-month contract →
  higher churn probability, longer tenure → lower churn probability).
• Continuous features are generated from mixture distributions so the data
  isn't trivially separable.
• A small amount of realistic missingness is injected and then documented.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import get_logger, ensure_dirs

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    n_samples: int = 10_000,
    random_state: int = 42,
    churn_rate: float = 0.265,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic telecom churn dataset.

    Parameters
    ----------
    n_samples:     Number of customer records to generate.
    random_state:  NumPy random seed for reproducibility.
    churn_rate:    Target proportion of churned customers (~26.5 %).
    output_path:   If provided, save the CSV here.

    Returns
    -------
    pd.DataFrame with 22 columns including the binary 'churn' target.
    """
    rng = np.random.default_rng(random_state)
    logger.info("Generating %d synthetic customer records …", n_samples)

    # ── Demographic & account features ────────────────────────────────────
    gender          = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen  = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner         = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents      = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])

    # ── Service subscription features ─────────────────────────────────────
    phone_service   = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines  = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58]),
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_samples,
        p=[0.34, 0.44, 0.22],
    )

    def internet_addon(name: str, yes_p: float = 0.50) -> np.ndarray:
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n_samples, p=[yes_p, 1 - yes_p]),
        )

    online_security   = internet_addon("online_security",  yes_p=0.29)
    online_backup     = internet_addon("online_backup",     yes_p=0.34)
    device_protection = internet_addon("device_protection", yes_p=0.34)
    tech_support      = internet_addon("tech_support",      yes_p=0.29)
    streaming_tv      = internet_addon("streaming_tv",      yes_p=0.38)
    streaming_movies  = internet_addon("streaming_movies",  yes_p=0.39)

    contract        = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.21, 0.24],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method  = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # ── Numerical features ─────────────────────────────────────────────────
    # Tenure: mix of short-term (high churn risk) and long-term customers
    tenure = np.clip(
        np.where(
            rng.random(n_samples) < 0.40,
            rng.integers(1, 12, size=n_samples),            # short-tenured
            rng.integers(12, 72, size=n_samples),           # established
        ),
        1, 72,
    ).astype(int)

    # Monthly charges: fibre customers pay more
    base_charge = np.where(
        internet_service == "Fiber optic",
        rng.normal(85, 15, n_samples),
        np.where(internet_service == "DSL", rng.normal(55, 12, n_samples),
                 rng.normal(25, 8, n_samples)),
    )
    # Add-ons inflate charge
    addon_surcharge = (
        (multiple_lines  == "Yes").astype(float) * rng.uniform(5, 15, n_samples) +
        (online_security == "Yes").astype(float) * rng.uniform(3, 8,  n_samples) +
        (online_backup   == "Yes").astype(float) * rng.uniform(3, 8,  n_samples) +
        (tech_support    == "Yes").astype(float) * rng.uniform(3, 8,  n_samples) +
        (streaming_tv    == "Yes").astype(float) * rng.uniform(4, 10, n_samples) +
        (streaming_movies== "Yes").astype(float) * rng.uniform(4, 10, n_samples)
    )
    monthly_charges = np.clip(base_charge + addon_surcharge, 18.0, 120.0).round(2)
    total_charges   = (monthly_charges * tenure * rng.uniform(0.92, 1.02, n_samples)).round(2)

    # Engineered numerical features
    num_products   = (
        (phone_service   == "Yes").astype(int) +
        (multiple_lines  == "Yes").astype(int) +
        (internet_service != "No").astype(int) +
        (online_security == "Yes").astype(int) +
        (online_backup   == "Yes").astype(int) +
        (device_protection == "Yes").astype(int) +
        (tech_support    == "Yes").astype(int) +
        (streaming_tv    == "Yes").astype(int) +
        (streaming_movies== "Yes").astype(int)
    )
    support_calls = np.clip(
        rng.poisson(lam=1.8, size=n_samples), 0, 10
    ).astype(int)
    avg_monthly_gb_download = np.clip(
        np.where(
            internet_service == "Fiber optic",
            rng.normal(120, 40, n_samples),
            np.where(internet_service == "DSL",
                     rng.normal(50, 20, n_samples),
                     rng.normal(5, 3, n_samples)),
        ),
        0, 300,
    ).round(1)

    # ── Churn label (logistic probability model) ───────────────────────────
    log_odds = (
        -2.50
        + 0.03  * np.clip(12 - tenure, 0, 12)          # short tenure → churn
        + 0.015 * (monthly_charges - 50)                # high bill → churn
        + 0.80  * (contract == "Month-to-month").astype(float)
        - 0.60  * (contract == "Two year").astype(float)
        + 0.40  * (internet_service == "Fiber optic").astype(float)
        + 0.35  * (payment_method == "Electronic check").astype(float)
        + 0.25  * (paperless_billing == "Yes").astype(float)
        - 0.25  * (tech_support == "Yes").astype(float)
        - 0.20  * (online_security == "Yes").astype(float)
        + 0.15  * (senior_citizen == 1).astype(float)
        - 0.10  * (partner == "Yes").astype(float)
        + 0.08  * support_calls
        + rng.normal(0, 0.4, n_samples)                 # noise
    )
    churn_prob = 1.0 / (1.0 + np.exp(-log_odds))

    # Calibrate to target churn rate
    threshold = np.percentile(churn_prob, 100 * (1 - churn_rate))
    churn     = (churn_prob >= threshold).astype(int)

    # ── Customer IDs ───────────────────────────────────────────────────────
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_samples + 1)]

    # ── Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        "customer_id":            customer_ids,
        "gender":                 gender,
        "senior_citizen":         senior_citizen,
        "partner":                partner,
        "dependents":             dependents,
        "tenure":                 tenure,
        "phone_service":          phone_service,
        "multiple_lines":         multiple_lines,
        "internet_service":       internet_service,
        "online_security":        online_security,
        "online_backup":          online_backup,
        "device_protection":      device_protection,
        "tech_support":           tech_support,
        "streaming_tv":           streaming_tv,
        "streaming_movies":       streaming_movies,
        "contract":               contract,
        "paperless_billing":      paperless_billing,
        "payment_method":         payment_method,
        "monthly_charges":        monthly_charges,
        "total_charges":          total_charges,
        "num_products":           num_products,
        "support_calls":          support_calls,
        "avg_monthly_gb_download": avg_monthly_gb_download,
        "churn":                  churn,
    })

    # ── Inject realistic missingness (<2 %) ───────────────────────────────
    miss_idx = rng.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    df.loc[miss_idx, "total_charges"] = np.nan

    actual_rate = churn.mean()
    logger.info(
        "Dataset ready: %d rows × %d cols | churn rate = %.1f%%",
        len(df), len(df.columns), actual_rate * 100,
    )

    if output_path is not None:
        output_path = Path(output_path)
        ensure_dirs(output_path.parent)
        df.to_csv(output_path, index=False)
        logger.info("Saved raw data -> %s", output_path)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/telecom_churn.csv"
    generate_dataset(output_path=out)
