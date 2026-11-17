"""
Training script entry point for the RiskModel ensemble model.

Orchestrates the full training pipeline:
1. Load and preprocess data
2. Train XGBoost, LightGBM, and ensemble models
3. Evaluate all models
4. Generate SHAP explainability plots
5. Save models, preprocessor, and reports
"""
import os
import sys
import logging
import time
import json
from datetime import datetime, timezone

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the full model training pipeline."""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("RiskModel — Training Pipeline")
    logger.info("=" * 60)

    # Ensure output directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('static/shap', exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load and preprocess data
    # ------------------------------------------------------------------
    logger.info("Step 1/5: Loading and preprocessing data...")
    from src.data_preprocessing import load_data, preprocess_data

    data_path = os.path.join('data', 'sample_credit_data.csv')
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}. "
                      "Please ensure the data file exists.")
        sys.exit(1)

    df = load_data(data_path)
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    logger.info(f"  Training set: {X_train.shape}")
    logger.info(f"  Test set:     {X_test.shape}")
    logger.info(f"  Class distribution (train): "
                f"{dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ------------------------------------------------------------------
    # Step 2: Feature names
    # ------------------------------------------------------------------
    logger.info("Step 2/5: Extracting feature names...")
    from src.data_preprocessing import get_feature_names

    feature_names = get_feature_names(preprocessor)
    logger.info(f"  {len(feature_names)} features after preprocessing")

    # ------------------------------------------------------------------
    # Step 3: Train models
    # ------------------------------------------------------------------
    logger.info("Step 3/5: Training models...")
    from src.model_training import (
        train_xgboost,
        train_lightgbm,
        train_ensemble,
        evaluate_model,
        save_model,
    )

    logger.info("  Training XGBoost...")
    xgb_model, xgb_cv_auc, xgb_params = train_xgboost(X_train, y_train)

    logger.info("  Training LightGBM...")
    lgbm_model, lgbm_cv_auc, lgbm_params = train_lightgbm(X_train, y_train)

    logger.info("  Training Ensemble (soft voting)...")
    ensemble_model = train_ensemble(X_train, y_train, xgb_model=xgb_model, lgbm_model=lgbm_model)

    from sklearn.base import clone
    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import cross_val_score
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        ens_cv_pipe = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', VotingClassifier(
                [('xgboost', clone(xgb_model)), ('lightgbm', clone(lgbm_model))],
                voting='soft'
            ))
        ])
        ens_cv_scores = cross_val_score(ens_cv_pipe, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
        ens_cv_auc = round(float(np.mean(ens_cv_scores)), 4)
    except Exception as e:
        logger.warning("Could not compute ensemble CV score: %s", e)
        ens_cv_auc = round(float((xgb_cv_auc + lgbm_cv_auc) / 2.0), 4)

    # ------------------------------------------------------------------
    # Step 4: Evaluate models
    # ------------------------------------------------------------------
    logger.info("Step 4/5: Evaluating models...")
    cv_auc_map = {
        'XGBoost': xgb_cv_auc,
        'LightGBM': lgbm_cv_auc,
        'Ensemble': ens_cv_auc,
    }
    results = {}
    for name, model in [('XGBoost', xgb_model),
                         ('LightGBM', lgbm_model),
                         ('Ensemble', ensemble_model)]:
        metrics = evaluate_model(model, X_test, y_test, name)
        cv_auc = cv_auc_map[name]
        metrics['cv_roc_auc'] = cv_auc
        results[name] = metrics
        logger.info(f"  {name}: Accuracy={metrics.get('accuracy', 0):.4f}  "
                     f"ROC-AUC={metrics.get('roc_auc', 0):.4f}  "
                     f"CV ROC-AUC={cv_auc:.4f}  "
                     f"F1={metrics.get('f1_score', 0):.4f}")

    # Save models
    save_model(xgb_model, os.path.join('models', 'xgboost_model.pkl'))
    save_model(lgbm_model, os.path.join('models', 'lightgbm_model.pkl'))
    save_model(ensemble_model, os.path.join('models', 'ensemble_model.pkl'))
    logger.info("  Models saved to models/")

    # Save evaluation report
    report_path = os.path.join('reports', 'model_evaluation.txt')
    with open(report_path, 'w') as f:
        f.write("RiskModel — Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        for name, metrics in results.items():
            f.write(f"Model: {name}\n")
            f.write("-" * 30 + "\n")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    f.write(f"  {metric}: {value:.4f}\n")
                elif metric == "classification_report":
                    f.write(f"\n{value}\n")
                else:
                    f.write(f"  {metric}: {value}\n")
            f.write("\n")
    logger.info(f"  Evaluation report saved to {report_path}")

    # Save metrics JSON
    metrics_json_path = os.path.join('reports', 'metrics.json')
    metrics_data = {
        "xgboost": {
            "accuracy": results["XGBoost"]["accuracy"],
            "roc_auc": results["XGBoost"]["roc_auc"],
            "cv_roc_auc": results["XGBoost"]["cv_roc_auc"],
            "f1": results["XGBoost"]["f1_score"],
            "f1_score": results["XGBoost"]["f1_score"],
            "precision": results["XGBoost"]["precision"],
            "recall": results["XGBoost"]["recall"],
        },
        "lightgbm": {
            "accuracy": results["LightGBM"]["accuracy"],
            "roc_auc": results["LightGBM"]["roc_auc"],
            "cv_roc_auc": results["LightGBM"]["cv_roc_auc"],
            "f1": results["LightGBM"]["f1_score"],
            "f1_score": results["LightGBM"]["f1_score"],
            "precision": results["LightGBM"]["precision"],
            "recall": results["LightGBM"]["recall"],
        },
        "ensemble": {
            "accuracy": results["Ensemble"]["accuracy"],
            "roc_auc": results["Ensemble"]["roc_auc"],
            "cv_roc_auc": results["Ensemble"]["cv_roc_auc"],
            "f1": results["Ensemble"]["f1_score"],
            "f1_score": results["Ensemble"]["f1_score"],
            "precision": results["Ensemble"]["precision"],
            "recall": results["Ensemble"]["recall"],
        },
        "accuracy": results["Ensemble"]["accuracy"],
        "roc_auc": results["Ensemble"]["roc_auc"],
        "cv_roc_auc": results["Ensemble"]["cv_roc_auc"],
        "f1": results["Ensemble"]["f1_score"],
        "f1_score": results["Ensemble"]["f1_score"],
        "precision": results["Ensemble"]["precision"],
        "recall": results["Ensemble"]["recall"],
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"  Metrics JSON saved to {metrics_json_path}")

    # ------------------------------------------------------------------
    # Step 5: SHAP explainability
    # ------------------------------------------------------------------
    logger.info("Step 5/5: Generating SHAP explainability plots...")
    from src.explainability import generate_shap_plots

    generate_shap_plots(
        ensemble_model,
        X_test,
        feature_names,
        output_dir='static/shap'
    )
    logger.info("  SHAP plots saved to static/shap/")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Best model: {max(results, key=lambda k: results[k].get('roc_auc', 0))}")
    logger.info(f"  Artifacts saved:")
    logger.info(f"    - models/ensemble_model.pkl")
    logger.info(f"    - models/xgboost_model.pkl")
    logger.info(f"    - models/lightgbm_model.pkl")
    logger.info(f"    - models/preprocessor.pkl")
    logger.info(f"    - reports/model_evaluation.txt")
    logger.info(f"    - reports/metrics.json")
    logger.info(f"    - static/shap/ (plots)")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
