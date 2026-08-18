"""
src/visualize.py
─────────────────────────────────────────────────────────────────────────────
All Matplotlib / Seaborn visualisation functions used by the EDA notebook
and the main pipeline.  Every function saves its figure to disk AND returns
the Figure object so it can be embedded in notebooks.

Colour palette: dark background ("seaborn-v0_8-darkgrid") with a
custom accent palette that looks great on GitHub README previews.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .feature_engineering import add_domain_features
from .utils import ensure_dirs, get_logger

logger = get_logger(__name__)

# ── Global style ──────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.dpi":       120,
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "axes.titlecolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "legend.facecolor": "#21262d",
    "legend.edgecolor": "#30363d",
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})

PALETTE       = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff"]
CHURN_COLORS  = {"No Churn": "#3fb950", "Churn": "#f78166"}
ACCENT        = "#58a6ff"


def _save(fig: plt.Figure, path: str | Path | None) -> None:
    if path is not None:
        path = Path(path)
        ensure_dirs(path.parent)
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Figure saved -> %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 1. EDA plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_churn_distribution(
    df: pd.DataFrame,
    target: str = "churn",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Donut chart of churn vs. no-churn."""
    counts = df[target].value_counts().sort_index()
    labels = ["No Churn", "Churn"]
    colors = [CHURN_COLORS["No Churn"], CHURN_COLORS["Churn"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # -- Donut
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="#0d1117", linewidth=2),
        textprops=dict(color="#e6edf3", fontsize=11),
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
    axes[0].set_title("Churn Distribution", fontsize=14, pad=14)

    # -- Count bar
    bar_colors = [CHURN_COLORS["No Churn"], CHURN_COLORS["Churn"]]
    bars = axes[1].bar(labels, counts.values, color=bar_colors,
                       edgecolor="#0d1117", linewidth=1.5, width=0.5)
    axes[1].set_title("Customer Counts", fontsize=14, pad=14)
    axes[1].set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{val:,}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold", color="#e6edf3")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))

    fig.suptitle("Customer Churn Overview", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_numerical_distributions(
    df: pd.DataFrame,
    features: list[str],
    target: str = "churn",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """KDE plots split by churn label for each numerical feature."""
    n     = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    label_map = {0: "No Churn", 1: "Churn"}

    for i, feat in enumerate(features):
        ax = axes[i]
        for label, color in CHURN_COLORS.items():
            val = 0 if label == "No Churn" else 1
            subset = df[df[target] == val][feat].dropna()
            ax.hist(subset, bins=25, density=True, alpha=0.45,
                    color=color, label=label, edgecolor="#0d1117")
        ax.set_title(feat.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("")
        ax.legend(fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions by Churn", fontsize=15,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_categorical_churn_rates(
    df: pd.DataFrame,
    features: list[str],
    target: str = "churn",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of churn rate per category value."""
    n     = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        rates = (
            df.groupby(feat)[target]
              .mean()
              .sort_values(ascending=True)
              .reset_index()
        )
        rates.columns = [feat, "churn_rate"]
        colors = [
            CHURN_COLORS["Churn"] if r > df[target].mean() else CHURN_COLORS["No Churn"]
            for r in rates["churn_rate"]
        ]
        bars = ax.barh(rates[feat], rates["churn_rate"] * 100,
                       color=colors, edgecolor="#0d1117", height=0.6)
        ax.axvline(df[target].mean() * 100, color=ACCENT,
                   linestyle="--", linewidth=1.4, label="Overall avg")
        for bar, rate in zip(bars, rates["churn_rate"]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{rate * 100:.1f}%", va="center", fontsize=8.5)
        ax.set_title(feat.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Churn Rate (%)")
        ax.legend(fontsize=8, loc="lower right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Churn Rate by Categorical Feature", fontsize=15,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Correlation heatmap of all numerical columns."""
    num_df = df.select_dtypes(include=np.number).drop(columns=["senior_citizen"],
                                                        errors="ignore")
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=~mask,          # show lower triangle
        annot=True, fmt=".2f",
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0, square=True,
        linewidths=0.5, linecolor="#21262d",
        ax=ax, cbar_kws={"shrink": 0.75},
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=14)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model evaluation plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    pipelines: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay ROC curves for multiple fitted pipelines."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, pipe), color in zip(pipelines.items(), PALETTE):
        X_eng  = add_domain_features(X_test)
        y_prob = pipe.predict_proba(X_eng)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5, label="Random classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="white")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – Churn Prediction Models", fontsize=14,
                 fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_precision_recall_curves(
    pipelines: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay Precision-Recall curves for multiple fitted pipelines."""
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.mean()

    for (name, pipe), color in zip(pipelines.items(), PALETTE):
        X_eng  = add_domain_features(X_test)
        y_prob = pipe.predict_proba(X_eng)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{name}  (PR-AUC = {pr_auc:.4f})")

    ax.axhline(baseline, color="white", linestyle="--", lw=1, alpha=0.5,
               label=f"Baseline (churn rate = {baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves – Churn Prediction Models",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_confusion_matrix(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.40,
    model_name: str = "Model",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Annotated confusion matrix with percentage labels."""
    X_eng  = add_domain_features(X_test)
    y_prob = pipeline.predict_proba(X_eng)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=False, fmt="d",
        cmap=sns.light_palette(ACCENT, as_cmap=True),
        linewidths=2, linecolor="#0d1117",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax, cbar=False,
    )
    labels = ["TN", "FP", "FN", "TP"]
    for (r, c), val, pct, lbl in zip(
        np.ndindex(cm.shape), cm.flatten(), cm_pct.flatten(), labels
    ):
        ax.text(c + 0.5, r + 0.35, f"{val:,}", ha="center", va="center",
                fontsize=16, fontweight="bold", color="#0d1117")
        ax.text(c + 0.5, r + 0.62, f"({pct:.1f}%)", ha="center", va="center",
                fontsize=10, color="#0d1117", alpha=0.85)
        ax.text(c + 0.5, r + 0.15, lbl, ha="center", va="center",
                fontsize=9, color="#0d1117", alpha=0.65)

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix – {model_name}\n(threshold={threshold})",
                 fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
    top_n: int = 20,
    model_name: str = "Model",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances."""
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "feature_importances_"):
        logger.warning("%s has no feature_importances_ attribute.", type(clf).__name__)
        return plt.figure()

    importances = clf.feature_importances_
    n_feat = min(top_n, len(importances), len(feature_names))

    idx   = np.argsort(importances)[-n_feat:]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    colors = [
        PALETTE[int(v * (len(PALETTE) - 1) / vals.max())] if vals.max() > 0
        else ACCENT
        for v in vals
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, n_feat * 0.4)))
    bars = ax.barh(names, vals, color=colors, edgecolor="#0d1117",
                   height=0.75)
    ax.set_xlabel("Feature Importance (Gini / Gain)", fontsize=11)
    ax.set_title(f"Top {n_feat} Feature Importances – {model_name}",
                 fontsize=14, fontweight="bold", pad=12)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing key metrics across models."""
    metrics = ["accuracy", "roc_auc", "pr_auc", "f1", "precision", "recall"]
    available = [m for m in metrics if m in comparison_df.columns]

    x   = np.arange(len(available))
    n   = len(comparison_df)
    w   = 0.8 / n

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        offset = (i - n / 2 + 0.5) * w
        vals   = [row[m] for m in available]
        bars   = ax.bar(x + offset, vals, width=w, label=row["model"],
                        color=PALETTE[i % len(PALETTE)],
                        edgecolor="#0d1117", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in available], fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Performance Comparison", fontsize=14,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_threshold_analysis(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "Model",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot F1, Precision, and Recall vs. decision threshold."""
    X_eng    = add_domain_features(X_val)
    y_prob   = pipeline.predict_proba(X_eng)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 200)

    f1s, precs, recs = [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_val, y_pred, zero_division=0))
        precs.append(precision_score(y_val, y_pred, zero_division=0))
        recs.append(recall_score(y_val, y_pred, zero_division=0))

    best_idx = int(np.argmax(f1s))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1s,   color=PALETTE[0], lw=2, label="F1")
    ax.plot(thresholds, precs, color=PALETTE[1], lw=2, label="Precision")
    ax.plot(thresholds, recs,  color=PALETTE[2], lw=2, label="Recall")
    ax.axvline(thresholds[best_idx], color="white", linestyle="--", lw=1.5,
               label=f"Best F1 threshold = {thresholds[best_idx]:.2f}")
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Threshold Analysis – {model_name}", fontsize=14,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    _save(fig, save_path)
    return fig
