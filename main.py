"""
main.py
─────────────────────────────────────────────────────────────────────────────
End-to-end churn prediction pipeline.

Steps
─────
1.  Generate / load synthetic dataset  (10 000 customers)
2.  Exploratory Data Analysis          (save figures)
3.  Feature engineering
4.  Train / test split (80 / 20, stratified)
5.  Train Random Forest
6.  Train XGBoost
7.  Evaluate both models
8.  Compare models, pick winner
9.  SHAP feature importance
10. Save artefacts

Run
───
    python main.py                    # full pipeline with defaults
    python main.py --skip-eda         # skip EDA plots (faster)
    python main.py --tune             # run hyperparameter search (slow)
    python main.py --data path/to/csv # use existing CSV
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Project imports ───────────────────────────────────────────────────────────
from src.data_generation import generate_dataset
from src.evaluate import compare_models, evaluate_model, shap_summary, threshold_search
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    add_domain_features,
    get_feature_names,
    split_X_y,
)
from src.train import train_random_forest, train_xgboost, tune_random_forest, tune_xgboost
from src.utils import ensure_dirs, get_logger, load_config
from src.visualize import (
    plot_categorical_churn_rates,
    plot_churn_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_comparison,
    plot_numerical_distributions,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_threshold_analysis,
)

logger = get_logger("churn_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Directory layout
# ─────────────────────────────────────────────────────────────────────────────
DIRS = [
    "data/raw", "data/processed", "data/predictions",
    "models",
    "reports/figures/eda",
    "reports/figures/models",
    "reports/metrics",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    p.add_argument("--data",      default=None,
                   help="Path to an existing CSV (skips data generation).")
    p.add_argument("--skip-eda",  action="store_true",
                   help="Skip EDA visualisation step.")
    p.add_argument("--tune",      action="store_true",
                   help="Run RandomizedSearchCV hyperparameter tuning.")
    p.add_argument("--threshold", type=float, default=0.40,
                   help="Decision threshold (default 0.40).")
    p.add_argument("--config",    default="config.yaml",
                   help="Path to config YAML.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def _save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved -> %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def stage_data(args, cfg) -> pd.DataFrame:
    _banner("STAGE 1 – Data Acquisition")
    if args.data:
        logger.info("Loading existing CSV: %s", args.data)
        df = pd.read_csv(args.data)
    else:
        df = generate_dataset(
            n_samples    = cfg.get("N_SAMPLES", 10_000),
            random_state = cfg.get("RANDOM_STATE", 42),
            churn_rate   = cfg.get("CHURN_RATE", 0.265),
            output_path  = cfg.get("RAW_DATA_PATH", "data/raw/telecom_churn.csv"),
        )
    logger.info("Dataset shape: %s | churn rate: %.2f%%",
                df.shape, df["churn"].mean() * 100)
    return df


def stage_eda(df: pd.DataFrame) -> None:
    _banner("STAGE 2 – Exploratory Data Analysis")
    fig_dir = Path("reports/figures/eda")

    plot_churn_distribution(
        df, save_path=fig_dir / "01_churn_distribution.png")

    num_feats = ["tenure", "monthly_charges", "total_charges",
                 "num_products", "support_calls", "avg_monthly_gb_download"]
    plot_numerical_distributions(
        df, features=num_feats,
        save_path=fig_dir / "02_numerical_distributions.png")

    cat_feats_subset = [
        "contract", "internet_service", "payment_method",
        "paperless_billing", "tech_support", "online_security",
    ]
    plot_categorical_churn_rates(
        df, features=cat_feats_subset,
        save_path=fig_dir / "03_categorical_churn_rates.png")

    plot_correlation_heatmap(
        df, save_path=fig_dir / "04_correlation_heatmap.png")

    logger.info("EDA figures saved to reports/figures/eda/")


def stage_split(df: pd.DataFrame, cfg: dict, threshold: float):
    _banner("STAGE 3 – Feature Engineering & Train/Test Split")

    X, y = split_X_y(df)
    logger.info("Features: %d | Target: churn | Positives: %d (%.1f%%)",
                X.shape[1], y.sum(), y.mean() * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = cfg.get("TEST_SIZE", 0.20),
        random_state = cfg.get("RANDOM_STATE", 42),
        stratify     = y,
    )

    # Validation split from training set (for threshold tuning)
    val_size = cfg.get("VALIDATION_SIZE", 0.10)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size    = val_size / (1 - cfg.get("TEST_SIZE", 0.20)),
        random_state = cfg.get("RANDOM_STATE", 42),
        stratify     = y_train,
    )

    logger.info("Train=%d | Val=%d | Test=%d",
                len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def stage_train(X_train, y_train, X_val, y_val, cfg, args):
    _banner("STAGE 4 – Model Training")

    rf_params  = cfg.get("RANDOM_FOREST", {})
    xgb_params = cfg.get("XGBOOST", {})

    if args.tune:
        logger.info("Running hyperparameter search (this may take several minutes) …")
        rf_params  = tune_random_forest(X_train, y_train)
        xgb_params = tune_xgboost(X_train, y_train)

    t0 = time.time()
    rf_pipeline = train_random_forest(
        X_train, y_train,
        params    = rf_params,
        run_cv    = True,
        save_path = "models/random_forest_pipeline.pkl",
    )
    logger.info("Random Forest trained in %.1f s", time.time() - t0)

    t0 = time.time()
    xgb_pipeline = train_xgboost(
        X_train, y_train,
        params    = xgb_params,
        run_cv    = True,
        save_path = "models/xgboost_pipeline.pkl",
    )
    logger.info("XGBoost trained in %.1f s", time.time() - t0)

    return rf_pipeline, xgb_pipeline


def stage_evaluate(rf_pipeline, xgb_pipeline, X_test, y_test,
                   X_val, y_val, threshold, cfg):
    _banner("STAGE 5 – Model Evaluation")
    fig_dir = Path("reports/figures/models")

    # --- Threshold optimisation on validation set
    opt_threshold = threshold_search(xgb_pipeline, X_val, y_val, metric="f1")
    logger.info("Using optimised threshold: %.3f (cmd arg: %.3f)",
                opt_threshold, threshold)
    final_threshold = opt_threshold  # use optimised value

    # --- Full evaluation
    pipelines = {"Random Forest": rf_pipeline, "XGBoost": xgb_pipeline}

    rf_metrics  = evaluate_model(rf_pipeline,  X_test, y_test,
                                 threshold=final_threshold, model_name="Random Forest")
    xgb_metrics = evaluate_model(xgb_pipeline, X_test, y_test,
                                 threshold=final_threshold, model_name="XGBoost")

    _save_metrics(rf_metrics,  "reports/metrics/random_forest_metrics.json")
    _save_metrics(xgb_metrics, "reports/metrics/xgboost_metrics.json")

    # --- Comparison
    cmp_df = compare_models(pipelines, X_test, y_test, threshold=final_threshold)
    cmp_df.to_csv("reports/metrics/model_comparison.csv", index=False)

    # --- Plots
    plot_roc_curves(pipelines, X_test, y_test,
                    save_path=fig_dir / "roc_curves.png")
    plot_precision_recall_curves(pipelines, X_test, y_test,
                                 save_path=fig_dir / "pr_curves.png")
    plot_confusion_matrix(rf_pipeline,  X_test, y_test, final_threshold,
                          "Random Forest",
                          save_path=fig_dir / "rf_confusion_matrix.png")
    plot_confusion_matrix(xgb_pipeline, X_test, y_test, final_threshold,
                          "XGBoost",
                          save_path=fig_dir / "xgb_confusion_matrix.png")
    plot_model_comparison(cmp_df,
                          save_path=fig_dir / "model_comparison.png")
    plot_threshold_analysis(xgb_pipeline, X_val, y_val,
                            model_name="XGBoost",
                            save_path=fig_dir / "threshold_analysis.png")

    return rf_metrics, xgb_metrics, cmp_df, final_threshold


def stage_feature_importance(rf_pipeline, xgb_pipeline, X_test, y_test):
    _banner("STAGE 6 – Feature Importance & SHAP")
    fig_dir = Path("reports/figures/models")

    # Get feature names from fitted preprocessor
    X_eng = add_domain_features(X_test)
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())

    plot_feature_importance(rf_pipeline,  feature_names, top_n=20,
                            model_name="Random Forest",
                            save_path=fig_dir / "rf_feature_importance.png")
    plot_feature_importance(xgb_pipeline, feature_names, top_n=20,
                            model_name="XGBoost",
                            save_path=fig_dir / "xgb_feature_importance.png")

    # SHAP (on a 500-sample subset for speed)
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_df = shap_summary(xgb_pipeline, sample, feature_names=feature_names,
                           max_display=20,
                           save_path="reports/metrics/shap_importance.csv")
    if not shap_df.empty:
        logger.info("Top 5 SHAP features:\n%s", shap_df.head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    args    = _parse_args()
    cfg     = load_config(args.config)

    _banner("Customer Churn Prediction Pipeline  v1.0")
    logger.info("Config: %s | Tune: %s | Skip-EDA: %s",
                args.config, args.tune, args.skip_eda)

    # Create all output directories
    ensure_dirs(*DIRS)

    # ── Stages ──────────────────────────────────────────────────────────────
    df = stage_data(args, cfg)
    df.to_csv(cfg.get("PROCESSED_DATA_PATH", "data/processed/churn_processed.csv"),
              index=False)

    if not args.skip_eda:
        stage_eda(df)

    X_train, X_val, X_test, y_train, y_val, y_test = stage_split(
        df, cfg, args.threshold
    )

    rf_pipeline, xgb_pipeline = stage_train(
        X_train, y_train, X_val, y_val, cfg, args
    )

    rf_metrics, xgb_metrics, cmp_df, final_threshold = stage_evaluate(
        rf_pipeline, xgb_pipeline,
        X_test, y_test, X_val, y_val,
        args.threshold, cfg,
    )

    stage_feature_importance(rf_pipeline, xgb_pipeline, X_test, y_test)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    winner  = cmp_df.iloc[0]["model"]

    _banner("PIPELINE COMPLETE")
    print(f"  Best model   : {winner}")
    print(f"  Accuracy     : {cmp_df.iloc[0]['accuracy']:.4f}")
    print(f"  ROC-AUC      : {cmp_df.iloc[0]['roc_auc']:.4f}")
    print(f"  F1-Score     : {cmp_df.iloc[0]['f1']:.4f}")
    print(f"  Threshold    : {final_threshold:.3f}")
    print(f"  Total time   : {elapsed:.1f} s")
    print(f"\n  Saved artefacts:")
    print(f"    models/random_forest_pipeline.pkl")
    print(f"    models/xgboost_pipeline.pkl")
    print(f"    reports/figures/  (all plots)")
    print(f"    reports/metrics/  (JSON + CSV)")
    print()


if __name__ == "__main__":
    main()
