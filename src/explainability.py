"""
SHAP Explainability Module for Financial Risk Assessment.

Provides functions to explain individual predictions, generate SHAP
summary / waterfall / dependence plots, and extract raw SHAP values.
"""

import os
import logging
import pickle

import numpy as np
import pandas as pd
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_shap_values(model, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values for a dataset using TreeExplainer.

    Falls back to KernelExplainer when tree-based explanation is not available.

    Args:
        model: Fitted model (tree-based preferred).
        X: Feature matrix.

    Returns:
        SHAP values array of shape (n_samples, n_features).
    """
    logger.info("Computing SHAP values for %d samples", X.shape[0])
    target_model = model
    if hasattr(model, "estimators_") and len(model.estimators_) > 0:
        target_model = model.estimators_[0]
    elif hasattr(model, "named_estimators_") and "xgboost" in model.named_estimators_:
        target_model = model.named_estimators_["xgboost"]

    if not HAS_SHAP:
        logger.info("Computing surrogate feature contributions (SHAP library not loaded)")
        if hasattr(target_model, "feature_importances_"):
            importances = target_model.feature_importances_
        else:
            importances = np.ones(X.shape[1]) / X.shape[1]
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0) + 1e-6
        norm_diff = (X - mean_X) / std_X
        return norm_diff * importances

    try:
        explainer = shap.TreeExplainer(target_model)
        sv = explainer.shap_values(X)
        # For binary classifiers TreeExplainer may return a list; take class-1 values
        if isinstance(sv, list):
            sv = sv[1]
    except Exception:
        logger.warning("TreeExplainer failed; falling back to KernelExplainer (slower)")
        background = shap.sample(X, min(100, X.shape[0]))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
    return sv


def explain_prediction(
    model,
    X_single: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Explain a single prediction by returning per-feature SHAP contributions.

    Args:
        model: Fitted model.
        X_single: Feature vector for one sample (1-D or shape (1, n)).
        feature_names: List of feature names.

    Returns:
        Dictionary of {feature_name: shap_value} sorted by absolute contribution.
    """
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)

    sv = get_shap_values(model, X_single)[0]
    contributions = dict(zip(feature_names, sv))
    sorted_contributions = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    logger.info("Top contributing feature: %s", next(iter(sorted_contributions)))
    return sorted_contributions


def generate_shap_plots(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    output_dir: str = "static/shap",
    max_display: int = 15,
) -> None:
    """Generate and save SHAP summary, waterfall, and dependence plots.

    Args:
        model: Fitted model.
        X_test: Test feature matrix.
        feature_names: List of feature names.
        output_dir: Directory to write PNG files.
        max_display: Max features shown per plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Generating SHAP plots → %s", output_dir)

    # Compute SHAP values
    sv = get_shap_values(model, X_test)

    if not HAS_SHAP:
        logger.info("Generating explainability plots using matplotlib")
        mean_abs = np.abs(sv).mean(axis=0)
        indices = np.argsort(mean_abs)[::-1][:max_display]

        # Summary Plot
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in reversed(indices)], [mean_abs[i] for i in reversed(indices)], color='#e94560')
        plt.xlabel("Mean |Feature Impact| (Risk Contribution)")
        plt.title("Feature Impact Summary (Explainability)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150)
        plt.close()

        # Bar plot
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in reversed(indices)], [mean_abs[i] for i in reversed(indices)], color='#3b82f6')
        plt.xlabel("Importance")
        plt.title("Model Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=150)
        plt.close()

        # Waterfall Plot
        sample_vals = sv[0][indices]
        sample_names = [feature_names[i] for i in indices]
        plt.figure(figsize=(10, 6))
        colors = ['#e94560' if v >= 0 else '#10b981' for v in sample_vals]
        plt.barh(sample_names[::-1], sample_vals[::-1], color=colors[::-1])
        plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
        plt.xlabel("Impact on Default Risk")
        plt.title("Sample Prediction Feature Attribution")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_waterfall.png"), dpi=150)
        plt.close()

        # Dependence Plot
        top_idx = indices[0]
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test[:, top_idx], sv[:, top_idx], alpha=0.6, color='#0f3460')
        plt.xlabel(feature_names[top_idx])
        plt.ylabel("Risk Contribution Value")
        plt.title(f"Dependence: {feature_names[top_idx]}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_dependence_top.png"), dpi=150)
        plt.close()

        logger.info("All explainability plots generated successfully")
        return

    # --- Summary (bee-swarm) plot ---
    plt.figure(figsize=(12, 8))
    shap.summary_plot(sv, X_test, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150)
    plt.close()
    logger.info("Saved shap_summary.png")

    # --- Bar summary plot ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test, feature_names=feature_names,
                      plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=150)
    plt.close()
    logger.info("Saved shap_bar.png")

    # --- Waterfall plot for the first test sample ---
    if not HAS_SHAP:
        logger.info("Computing surrogate feature contributions (SHAP library not loaded)")
        if hasattr(target_model, "feature_importances_"):
            importances = target_model.feature_importances_
        else:
            importances = np.ones(X.shape[1]) / X.shape[1]
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0) + 1e-6
        norm_diff = (X - mean_X) / std_X
        return norm_diff * importances

    try:
        target_model = model
        if hasattr(model, "estimators_") and len(model.estimators_) > 0:
            target_model = model.estimators_[0]
        elif hasattr(model, "named_estimators_") and "xgboost" in model.named_estimators_:
            target_model = model.named_estimators_["xgboost"]

        explainer = shap.TreeExplainer(target_model)
        explanation = explainer(X_test[:1])
        if hasattr(explanation, "values") and explanation.values.ndim == 3:
            explanation = explanation[:, :, 1]
        explanation.feature_names = feature_names
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_waterfall.png"), dpi=150)
        plt.close()
        logger.info("Saved shap_waterfall.png")
    except Exception as exc:
        logger.warning("Waterfall plot skipped: %s", exc)

    # --- Dependence plot for the top feature ---
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_idx = int(np.argmax(mean_abs_shap))
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(top_idx, sv, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_dependence_top.png"), dpi=150)
    plt.close()
    logger.info("Saved shap_dependence_top.png")

    logger.info("All SHAP plots generated successfully")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sample_credit_data.csv")

    # Load model (prefer XGBoost for SHAP compatibility)
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    if not os.path.exists(model_path):
        logger.error("Model not found at %s — run model_training.py first", model_path)
        raise SystemExit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded model from %s", model_path)

    # Preprocess test data
    from data_preprocessing import load_data, preprocess_data, get_feature_names

    df = load_data(DATA_PATH)
    _, X_test, _, y_test, preprocessor = preprocess_data(df, save_preprocessor=False)
    feature_names = get_feature_names(preprocessor)

    # Generate all plots
    output_dir = os.path.join(PROJECT_ROOT, "static", "shap")
    generate_shap_plots(model, X_test, feature_names, output_dir=output_dir)

    # Explain a single prediction
    sample = X_test[0]
    contributions = explain_prediction(model, sample, feature_names)
    print("\nSingle-sample explanation (top 10 features):")
    for feat, val in list(contributions.items())[:10]:
        print(f"  {feat:>30s}: {val:+.4f}")
