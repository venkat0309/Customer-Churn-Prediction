"""
src/train.py
─────────────────────────────────────────────────────────────────────────────
Model training routines for Random Forest and XGBoost classifiers.

Each trainer returns a fitted sklearn Pipeline (preprocessor + model) so
the same object can be used for inference without re-applying transforms.

GridSearchCV / RandomizedSearchCV hyperparameter tuning is also provided
as optional fast-path helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ENGINEERED_NUMERICAL,
    add_domain_features,
    build_feature_pipeline,
    split_X_y,
)
from .utils import ensure_dirs, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Default hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

RF_DEFAULTS: dict[str, Any] = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

XGB_DEFAULTS: dict[str, Any] = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=2.8,
    eval_metric="auc",
    use_label_encoder=False,
    random_state=42,
    verbosity=0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_pipeline(model: Any) -> Pipeline:
    """Wrap preprocessing + model into a single Pipeline."""
    preprocessor = build_feature_pipeline()
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])


def _cv_report(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
               cv: int = 5) -> dict[str, float]:
    """Run stratified k-fold CV and return a metrics dict."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    metrics: dict[str, float] = {}
    for metric in ("accuracy", "roc_auc", "f1", "precision", "recall"):
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring=metric, n_jobs=-1)
        metrics[metric] = float(scores.mean())
        logger.info("  CV %-12s = %.4f ± %.4f", metric, scores.mean(), scores.std())
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Public trainers
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
    run_cv: bool = True,
    save_path: str | Path | None = None,
) -> Pipeline:
    """
    Train a Random Forest classifier inside a full preprocessing pipeline.

    Parameters
    ----------
    X_train:    Raw feature DataFrame (before any preprocessing).
    y_train:    Binary target Series.
    params:     Override default hyperparameters.
    run_cv:     If True, run 5-fold stratified CV and log results.
    save_path:  If provided, serialise the fitted pipeline with joblib.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    hp = {**RF_DEFAULTS, **(params or {})}
    logger.info("Training Random Forest | params: %s", hp)

    X_eng = add_domain_features(X_train)
    rf    = RandomForestClassifier(**hp)
    pipe  = _build_full_pipeline(rf)
    pipe.fit(X_eng, y_train)

    if run_cv:
        logger.info("5-fold cross-validation (Random Forest):")
        _cv_report(pipe, X_eng, y_train)

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dirs(save_path.parent)
        joblib.dump(pipe, save_path)
        logger.info("Random Forest pipeline saved -> %s", save_path)

    return pipe


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict[str, Any] | None = None,
    run_cv: bool = True,
    save_path: str | Path | None = None,
) -> Pipeline:
    """
    Train an XGBoost classifier inside a full preprocessing pipeline.

    Parameters
    ----------
    X_train / y_train:  Training data.
    X_val   / y_val:    Optional held-out validation set for early stopping.
    params:             Override default XGBoost hyperparameters.
    run_cv:             If True, run 5-fold stratified CV and log results.
    save_path:          If provided, serialise the fitted pipeline.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    hp = {**XGB_DEFAULTS, **(params or {})}
    logger.info("Training XGBoost | params: %s", hp)

    X_eng = add_domain_features(X_train)
    xgb   = XGBClassifier(**hp)
    pipe  = _build_full_pipeline(xgb)
    pipe.fit(X_eng, y_train)

    if run_cv:
        logger.info("5-fold cross-validation (XGBoost):")
        _cv_report(pipe, X_eng, y_train)

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dirs(save_path.parent)
        joblib.dump(pipe, save_path)
        logger.info("XGBoost pipeline saved -> %s", save_path)

    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Optional: Hyperparameter search
# ─────────────────────────────────────────────────────────────────────────────

def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
) -> dict[str, Any]:
    """
    RandomizedSearchCV over the most impactful RF hyperparameters.
    Returns the best parameter dict (ready to pass to train_random_forest).
    """
    param_dist = {
        "classifier__n_estimators":    [100, 200, 300, 500],
        "classifier__max_depth":       [8, 10, 12, 15, None],
        "classifier__min_samples_split":[2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features":    ["sqrt", "log2"],
    }
    X_eng = add_domain_features(X_train)
    pipe  = _build_full_pipeline(RandomForestClassifier(
        class_weight="balanced", n_jobs=-1, random_state=42
    ))
    search = RandomizedSearchCV(
        pipe, param_dist,
        n_iter=n_iter, cv=cv,
        scoring="roc_auc",
        n_jobs=-1, random_state=42,
        verbose=1,
    )
    search.fit(X_eng, y_train)
    logger.info("Best RF AUC: %.4f", search.best_score_)
    logger.info("Best RF params: %s", search.best_params_)
    # Strip "classifier__" prefix so dict can be used with train_random_forest
    best = {k.replace("classifier__", ""): v
            for k, v in search.best_params_.items()}
    return best


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
) -> dict[str, Any]:
    """RandomizedSearchCV over XGBoost hyperparameters."""
    param_dist = {
        "classifier__n_estimators":  [200, 300, 400, 500],
        "classifier__max_depth":     [4, 5, 6, 7, 8],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.15],
        "classifier__subsample":     [0.6, 0.7, 0.8, 0.9],
        "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    }
    X_eng = add_domain_features(X_train)
    pipe  = _build_full_pipeline(XGBClassifier(
        scale_pos_weight=2.8, eval_metric="auc",
        use_label_encoder=False, verbosity=0, random_state=42
    ))
    search = RandomizedSearchCV(
        pipe, param_dist,
        n_iter=n_iter, cv=cv,
        scoring="roc_auc",
        n_jobs=-1, random_state=42,
        verbose=1,
    )
    search.fit(X_eng, y_train)
    logger.info("Best XGB AUC: %.4f", search.best_score_)
    logger.info("Best XGB params: %s", search.best_params_)
    best = {k.replace("classifier__", ""): v
            for k, v in search.best_params_.items()}
    return best
