"""
src/predict.py
─────────────────────────────────────────────────────────────────────────────
Inference module – load a saved pipeline and score new customers.

Usage (CLI)
-----------
python -m src.predict \\
    --model  models/xgboost_pipeline.pkl \\
    --input  data/raw/new_customers.csv  \\
    --output data/predictions/scored.csv \\
    --threshold 0.40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .feature_engineering import add_domain_features
from .utils import ensure_dirs, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(model_path: str | Path):
    """Deserialise and return a joblib-saved sklearn Pipeline."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info("Loading pipeline from %s …", model_path)
    return joblib.load(model_path)


def score_customers(
    pipeline,
    df: pd.DataFrame,
    threshold: float = 0.40,
    id_col: str = "customer_id",
) -> pd.DataFrame:
    """
    Score a DataFrame of raw customer features.

    Returns
    -------
    DataFrame with columns:
        customer_id, churn_probability, churn_prediction, risk_segment
    """
    X = df.copy()
    X_eng = add_domain_features(X)

    proba  = pipeline.predict_proba(X_eng)[:, 1]
    pred   = (proba >= threshold).astype(int)

    def _risk(p: float) -> str:
        if p >= 0.70:
            return "High Risk"
        elif p >= 0.40:
            return "Medium Risk"
        else:
            return "Low Risk"

    result = pd.DataFrame({
        id_col:               df[id_col] if id_col in df.columns else range(len(df)),
        "churn_probability":  proba.round(4),
        "churn_prediction":   pred,
        "risk_segment":       [_risk(p) for p in proba],
    })

    logger.info(
        "Scored %d customers | High=%.1f%% | Medium=%.1f%% | Low=%.1f%%",
        len(result),
        (result["risk_segment"] == "High Risk").mean()   * 100,
        (result["risk_segment"] == "Medium Risk").mean() * 100,
        (result["risk_segment"] == "Low Risk").mean()    * 100,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score customers with a saved churn model.")
    p.add_argument("--model",     required=True, help="Path to joblib pipeline file.")
    p.add_argument("--input",     required=True, help="Path to raw customer CSV.")
    p.add_argument("--output",    default="data/predictions/scored.csv",
                   help="Where to save scored CSV (default: data/predictions/scored.csv).")
    p.add_argument("--threshold", type=float, default=0.40,
                   help="Decision threshold (default: 0.40).")
    p.add_argument("--id-col",    default="customer_id",
                   help="Name of the customer-ID column.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    pipe   = load_pipeline(args.model)
    df_raw = pd.read_csv(args.input)

    results = score_customers(pipe, df_raw,
                              threshold=args.threshold,
                              id_col=args.id_col)

    out = Path(args.output)
    ensure_dirs(out.parent)
    results.to_csv(out, index=False)
    logger.info("Predictions saved -> %s", out)
    print(results.head(10).to_string(index=False))
