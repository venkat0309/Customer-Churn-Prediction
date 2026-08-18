"""
src/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Model evaluation utilities.

Provides
────────
• evaluate_model()   – full metric suite + classification report
• compare_models()   – side-by-side comparison of multiple fitted pipelines
• threshold_search() – find the decision threshold that maximises F1
• shap_summary()     – SHAP-based global feature importance
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .feature_engineering import add_domain_features
from .utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.40,
    model_name: str = "Model",
    verbose: bool = True,
) -> dict[str, float]:
    """
    Compute a comprehensive metric suite for a fitted pipeline.

    Parameters
    ----------
    pipeline:   Fitted sklearn Pipeline (preprocessor + classifier).
    X_test:     Raw (untransformed) test features.
    y_test:     True binary labels.
    threshold:  Decision threshold for classification (default 0.40, tuned
                to boost recall on the minority churn class).
    model_name: Label used in log output.
    verbose:    Print classification report to stdout.

    Returns
    -------
    dict with keys: accuracy, roc_auc, pr_auc, f1, precision, recall,
                    specificity, tp, fp, tn, fn.
    """
    X_eng = add_domain_features(X_test)

    y_prob = pipeline.predict_proba(X_eng)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy":    accuracy_score(y_test, y_pred),
        "roc_auc":     roc_auc_score(y_test, y_prob),
        "pr_auc":      average_precision_score(y_test, y_prob),
        "f1":          f1_score(y_test, y_pred),
        "precision":   precision_score(y_test, y_pred),
        "recall":      recall_score(y_test, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }

    if verbose:
        sep = "-" * 60
        print(f"\n{sep}")
        print(f"  {model_name} -- Evaluation Results (threshold={threshold})")
        print(sep)
        print(f"  Accuracy   : {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC     : {metrics['pr_auc']:.4f}")
        print(f"  F1-Score   : {metrics['f1']:.4f}")
        print(f"  Precision  : {metrics['precision']:.4f}")
        print(f"  Recall     : {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN={tn:>5}  FP={fp:>5}")
        print(f"    FN={fn:>5}  TP={tp:>5}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=["No Churn", "Churn"]))
        print(sep)

    logger.info(
        "%s | ACC=%.4f | AUC=%.4f | F1=%.4f | REC=%.4f",
        model_name,
        metrics["accuracy"], metrics["roc_auc"],
        metrics["f1"], metrics["recall"],
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(
    pipelines: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.40,
) -> pd.DataFrame:
    """
    Evaluate multiple pipelines and return a sorted comparison DataFrame.

    Parameters
    ----------
    pipelines:  {'model_name': fitted_pipeline, ...}
    X_test, y_test: Test split.
    threshold:  Shared decision threshold.

    Returns
    -------
    pd.DataFrame sorted by ROC-AUC descending.
    """
    rows = []
    for name, pipe in pipelines.items():
        m = evaluate_model(pipe, X_test, y_test, threshold=threshold,
                           model_name=name, verbose=False)
        rows.append({"model": name, **m})

    cmp = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    logger.info("\nModel comparison (sorted by ROC-AUC):\n%s", cmp.to_string(index=False))
    return cmp


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimisation
# ─────────────────────────────────────────────────────────────────────────────

def threshold_search(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "f1",
    n_thresholds: int = 100,
) -> float:
    """
    Sweep decision thresholds and return the one that maximises *metric*.

    Parameters
    ----------
    metric: one of 'f1', 'recall', 'precision', 'accuracy'.

    Returns
    -------
    Optimal threshold float.
    """
    X_eng  = add_domain_features(X_val)
    y_prob = pipeline.predict_proba(X_eng)[:, 1]
    thresholds = np.linspace(0.10, 0.90, n_thresholds)

    best_t, best_score = 0.5, -np.inf
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_val, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_val, y_pred, zero_division=0)
        else:
            score = accuracy_score(y_val, y_pred)
        if score > best_score:
            best_score, best_t = score, t

    logger.info("Optimal threshold (max %s=%.4f): %.3f", metric, best_score, best_t)
    return float(best_t)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP importance (optional – graceful fallback if shap not installed)
# ─────────────────────────────────────────────────────────────────────────────

def shap_summary(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    feature_names: list[str] | None = None,
    max_display: int = 20,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Compute SHAP values for a tree-based model and return a feature
    importance DataFrame sorted by mean |SHAP|.

    Returns an empty DataFrame if SHAP is unavailable.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed – skipping SHAP analysis.")
        return pd.DataFrame()

    X_eng = add_domain_features(X_sample)
    # Transform through preprocessor only
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_eng)

    classifier = pipeline.named_steps["classifier"]
    explainer  = shap.TreeExplainer(classifier)
    shap_vals  = explainer.shap_values(X_transformed)

    # For binary classifiers shap_values returns list[array] or single array
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # positive class

    mean_abs = np.abs(shap_vals).mean(axis=0)

    if feature_names is None:
        feature_names = (
            list(preprocessor.get_feature_names_out())
            if hasattr(preprocessor, "get_feature_names_out")
            else [f"f{i}" for i in range(len(mean_abs))]
        )

    importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_shap": mean_abs})
        .sort_values("mean_shap", ascending=False)
        .head(max_display)
        .reset_index(drop=True)
    )

    if save_path is not None:
        importance_df.to_csv(save_path, index=False)
        logger.info("SHAP importance saved -> %s", save_path)

    return importance_df
