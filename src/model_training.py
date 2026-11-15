"""
Model Training Module for Financial Risk Assessment.

Trains XGBoost, LightGBM, and an ensemble VotingClassifier with
hyperparameter tuning via GridSearchCV.  Evaluates models, saves
artifacts, and logs experiments to MLflow.
"""

import os
import logging
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as LGBMClassifier
    HAS_LGBM = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training results and functions
# ---------------------------------------------------------------------------

class TrainingResult(tuple):
    """Tuple containing (clf, cv_roc_auc, best_params) that also delegates attribute access to clf."""
    def __new__(cls, clf, cv_roc_auc, best_params):
        return super().__new__(cls, (clf, cv_roc_auc, best_params))

    def __getattr__(self, name):
        return getattr(self[0], name)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
):
    """Train an XGBoost classifier with GridSearchCV hyperparameter tuning and SMOTE in-fold.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        random_state: Seed for reproducibility.

    Returns:
        TrainingResult with (best_estimator_.named_steps['clf'], cv_roc_auc, best_params).
    """
    logger.info("Training XGBoost with GridSearchCV (native=%s)", HAS_XGB)
    if HAS_XGB:
        base = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )
        param_grid = {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [4, 6],
            "clf__learning_rate": [0.05, 0.1],
        }
    else:
        base = XGBClassifier(random_state=random_state)
        param_grid = {
            "clf__max_iter": [50, 100],
            "clf__max_depth": [4, 6],
            "clf__learning_rate": [0.05, 0.1],
        }

    if HAS_SMOTE:
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("clf", base),
        ])
    else:
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("clf", base)])

    grid = GridSearchCV(
        pipe, param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    logger.info("Best XGBoost params: %s (AUC=%.4f)", grid.best_params_, grid.best_score_)
    best_clf = grid.best_estimator_.named_steps["clf"]
    best_cv_auc = round(float(grid.best_score_), 4)
    return TrainingResult(best_clf, best_cv_auc, grid.best_params_)


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
):
    """Train a LightGBM classifier with GridSearchCV hyperparameter tuning and SMOTE in-fold.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        random_state: Seed for reproducibility.

    Returns:
        TrainingResult with (best_estimator_.named_steps['clf'], cv_roc_auc, best_params).
    """
    logger.info("Training LightGBM with GridSearchCV (native=%s)", HAS_LGBM)
    if HAS_LGBM:
        base = LGBMClassifier(
            random_state=random_state,
            verbose=-1,
        )
        param_grid = {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [4, 6],
            "clf__learning_rate": [0.05, 0.1],
        }
    else:
        base = LGBMClassifier(random_state=random_state)
        param_grid = {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
        }

    if HAS_SMOTE:
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("clf", base),
        ])
    else:
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("clf", base)])

    grid = GridSearchCV(
        pipe, param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    logger.info("Best LightGBM params: %s (AUC=%.4f)", grid.best_params_, grid.best_score_)
    best_clf = grid.best_estimator_.named_steps["clf"]
    best_cv_auc = round(float(grid.best_score_), 4)
    return TrainingResult(best_clf, best_cv_auc, grid.best_params_)


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    xgb_model=None,
    lgbm_model=None,
    random_state: int = 42,
) -> VotingClassifier:
    """Train a soft-voting ensemble of XGBoost and LightGBM without GridSearchCV.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        xgb_model: Optional tuned XGBoost classifier to clone.
        lgbm_model: Optional tuned LightGBM classifier to clone.
        random_state: Seed for reproducibility.

    Returns:
        Fitted VotingClassifier with estimators_ [XGBClassifier, LGBMClassifier].
    """
    logger.info("Training Ensemble (VotingClassifier, soft voting)")
    if xgb_model is None:
        if HAS_XGB:
            xgb_est = XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state,
                verbosity=0,
            )
        else:
            xgb_est = XGBClassifier(random_state=random_state)
    else:
        if isinstance(xgb_model, tuple) and hasattr(xgb_model, "__getattr__"):
            xgb_model = xgb_model[0]
        xgb_est = clone(xgb_model)

    if lgbm_model is None:
        if HAS_LGBM:
            lgbm_est = LGBMClassifier(
                random_state=random_state,
                verbose=-1,
            )
        else:
            lgbm_est = LGBMClassifier(random_state=random_state)
    else:
        if isinstance(lgbm_model, tuple) and hasattr(lgbm_model, "__getattr__"):
            lgbm_model = lgbm_model[0]
        lgbm_est = clone(lgbm_model)

    ensemble = VotingClassifier(
        estimators=[("xgboost", xgb_est), ("lightgbm", lgbm_est)],
        voting="soft",
    )
    if HAS_SMOTE:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
    ensemble.fit(X_train_res, y_train_res)
    logger.info("Ensemble training complete")
    return ensemble


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
) -> dict:
    """Evaluate a trained classifier and return a metrics dictionary.

    Args:
        model: Fitted sklearn-compatible classifier.
        X_test: Test feature matrix.
        y_test: Test target vector.
        model_name: Human-readable model name for logging.

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc, and
        the full classification_report string.
    """
    logger.info("Evaluating %s", model_name)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
        "classification_report": classification_report(y_test, y_pred),
    }

    logger.info(
        "%s — Accuracy: %.4f | F1: %.4f | AUC: %s",
        model_name,
        metrics["accuracy"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, filepath: str) -> None:
    """Save a model to the given filepath.

    Args:
        model: Fitted model to save.
        filepath: Destination path (e.g. ``models/xgboost.pkl``).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    import joblib
    if isinstance(model, tuple) and hasattr(model, "__getattr__"):
        model = model[0]
    joblib.dump(model, filepath)
    logger.info("Model saved to %s", filepath)


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data, get_feature_names

    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sample_credit_data.csv")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load and preprocess
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    feature_names = get_feature_names(preprocessor)

    # MLflow experiment
    if HAS_MLFLOW:
        try:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", os.path.join(PROJECT_ROOT, "mlruns"))
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("RiskModel_Training")
        except Exception as e:
            logger.warning("MLflow init warning: %s", e)

    all_metrics: list[dict] = []

    # --- XGBoost ---
    if HAS_MLFLOW:
        with mlflow.start_run(run_name="XGBoost"):
            xgb_model = train_xgboost(X_train, y_train)
            xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
            all_metrics.append(xgb_metrics)
            save_model(xgb_model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
            try:
                mlflow.log_params(xgb_model.get_params())
                mlflow.log_metrics({k: v for k, v in xgb_metrics.items() if isinstance(v, (int, float)) and v is not None})
                mlflow.sklearn.log_model(xgb_model, "xgboost_model")
            except Exception as e:
                logger.warning("MLflow logging error: %s", e)
    else:
        xgb_model = train_xgboost(X_train, y_train)
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        all_metrics.append(xgb_metrics)
        save_model(xgb_model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))

    # --- LightGBM ---
    if HAS_MLFLOW:
        with mlflow.start_run(run_name="LightGBM"):
            lgbm_model = train_lightgbm(X_train, y_train)
            lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test, "LightGBM")
            all_metrics.append(lgbm_metrics)
            save_model(lgbm_model, os.path.join(MODELS_DIR, "lightgbm_model.pkl"))
            try:
                mlflow.log_params(lgbm_model.get_params())
                mlflow.log_metrics({k: v for k, v in lgbm_metrics.items() if isinstance(v, (int, float)) and v is not None})
                mlflow.sklearn.log_model(lgbm_model, "lightgbm_model")
            except Exception as e:
                logger.warning("MLflow logging error: %s", e)
    else:
        lgbm_model = train_lightgbm(X_train, y_train)
        lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test, "LightGBM")
        all_metrics.append(lgbm_metrics)
        save_model(lgbm_model, os.path.join(MODELS_DIR, "lightgbm_model.pkl"))

    # --- Ensemble ---
    if HAS_MLFLOW:
        with mlflow.start_run(run_name="Ensemble"):
            ensemble_model = train_ensemble(X_train, y_train)
            ens_metrics = evaluate_model(ensemble_model, X_test, y_test, "Ensemble")
            all_metrics.append(ens_metrics)
            save_model(ensemble_model, os.path.join(MODELS_DIR, "ensemble_model.pkl"))
            try:
                mlflow.log_metrics({k: v for k, v in ens_metrics.items() if isinstance(v, (int, float)) and v is not None})
                mlflow.sklearn.log_model(ensemble_model, "ensemble_model")
            except Exception as e:
                logger.warning("MLflow logging error: %s", e)
    else:
        ensemble_model = train_ensemble(X_train, y_train)
        ens_metrics = evaluate_model(ensemble_model, X_test, y_test, "Ensemble")
        all_metrics.append(ens_metrics)
        save_model(ensemble_model, os.path.join(MODELS_DIR, "ensemble_model.pkl"))

    # --- Write evaluation report ---
    report_path = os.path.join(REPORTS_DIR, "model_evaluation.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  FINANCIAL RISK MODEL — EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        for m in all_metrics:
            f.write(f"--- {m['model_name']} ---\n")
            f.write(f"Accuracy : {m['accuracy']}\n")
            f.write(f"Precision: {m['precision']}\n")
            f.write(f"Recall   : {m['recall']}\n")
            f.write(f"F1 Score : {m['f1_score']}\n")
            f.write(f"ROC AUC  : {m['roc_auc']}\n\n")
            f.write(m["classification_report"])
            f.write("\n\n")
    logger.info("Evaluation report saved to %s", report_path)

    # Print summary
    print("\n" + "=" * 60)
    for m in all_metrics:
        print(f"{m['model_name']:>12} | Acc={m['accuracy']:.4f} F1={m['f1_score']:.4f} AUC={m['roc_auc']}")
    print("=" * 60)
