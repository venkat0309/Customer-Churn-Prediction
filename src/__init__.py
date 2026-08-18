"""
src/__init__.py
Package initialiser – exposes high-level pipeline helpers.
"""

from .data_generation import generate_dataset
from .feature_engineering import build_feature_pipeline
from .train import train_random_forest, train_xgboost
from .evaluate import evaluate_model, compare_models
from .utils import load_config, get_logger, ensure_dirs

__all__ = [
    "generate_dataset",
    "build_feature_pipeline",
    "train_random_forest",
    "train_xgboost",
    "evaluate_model",
    "compare_models",
    "load_config",
    "get_logger",
    "ensure_dirs",
]
