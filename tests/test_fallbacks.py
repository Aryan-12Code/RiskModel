"""
Tests verifying removal of silent fallbacks and fallback branches.
"""
import os
import sys
import importlib
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'sample_credit_data.csv'
)


def test_xgboost_unavailable_raises_import_error(monkeypatch):
    """With xgboost unavailable, importing src.model_training raises ImportError."""
    import src.model_training
    monkeypatch.setitem(sys.modules, "xgboost", None)
    try:
        with pytest.raises(ImportError):
            importlib.reload(src.model_training)
    finally:
        monkeypatch.undo()
        importlib.reload(src.model_training)


def test_lightgbm_unavailable_raises_import_error(monkeypatch):
    """With lightgbm unavailable, importing src.model_training raises ImportError."""
    import src.model_training
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    try:
        with pytest.raises(ImportError):
            importlib.reload(src.model_training)
    finally:
        monkeypatch.undo()
        importlib.reload(src.model_training)


def test_shap_unavailable_raises_import_error(monkeypatch):
    """With shap unavailable, importing src.explainability raises ImportError."""
    import src.explainability
    monkeypatch.setitem(sys.modules, "shap", None)
    try:
        with pytest.raises(ImportError):
            importlib.reload(src.explainability)
    finally:
        monkeypatch.undo()
        importlib.reload(src.explainability)


def test_imblearn_unavailable_raises_import_error(monkeypatch):
    """With imblearn unavailable, importing src.model_training raises ImportError."""
    import src.model_training
    monkeypatch.setitem(sys.modules, "imblearn", None)
    monkeypatch.setitem(sys.modules, "imblearn.pipeline", None)
    monkeypatch.setitem(sys.modules, "imblearn.over_sampling", None)
    try:
        with pytest.raises(ImportError):
            importlib.reload(src.model_training)
    finally:
        monkeypatch.undo()
        importlib.reload(src.model_training)


def test_preprocess_data_no_apply_smote_parameter():
    """preprocess_data no longer accepts the apply_smote parameter."""
    from src.data_preprocessing import load_data, preprocess_data
    df = load_data(DATA_PATH)
    with pytest.raises(TypeError):
        preprocess_data(df, apply_smote=True)


def test_preprocess_data_returns_original_class_balance():
    """preprocess_data returns a training set whose minority-class share matches original data within 0.02."""
    from src.data_preprocessing import load_data, preprocess_data
    df = load_data(DATA_PATH)
    original_minority_share = float((df["default"] == 1).mean())
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, save_preprocessor=False)
    train_minority_share = float((pd.Series(y_train) == 1).mean())
    assert abs(train_minority_share - original_minority_share) <= 0.02


def test_train_model_fails_if_shap_fails(monkeypatch, tmp_path):
    """train_model's main run raises if generate_shap_plots raises."""
    import train_model
    import src.explainability
    from sklearn.dummy import DummyClassifier

    # Fast-track model training to quickly test SHAP step in main()
    dummy = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    dummy.cv_roc_auc = 0.8
    monkeypatch.setattr("src.model_training.train_xgboost", lambda *args, **kwargs: (dummy, 0.8, {}))
    monkeypatch.setattr("src.model_training.train_lightgbm", lambda *args, **kwargs: (dummy, 0.8, {}))
    monkeypatch.setattr("src.model_training.train_ensemble", lambda *args, **kwargs: dummy)
    monkeypatch.setattr("sklearn.model_selection.cross_val_score", lambda *args, **kwargs: np.array([0.8]))
    monkeypatch.setattr("src.model_training.save_model", lambda *args, **kwargs: None)
    monkeypatch.setattr("joblib.dump", lambda *args, **kwargs: None)
    from src.data_preprocessing import preprocess_data as real_preprocess
    monkeypatch.setattr("src.data_preprocessing.preprocess_data", lambda df, *args, **kw: real_preprocess(df, *args, **{**kw, "save_preprocessor": False}))

    real_open = open
    def safe_open(file, *args, **kwargs):
        if 'reports' in str(file):
            return real_open(os.path.join(str(tmp_path), os.path.basename(file)), *args, **kwargs)
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr("builtins.open", safe_open)

    def failing_shap(*args, **kwargs):
        raise RuntimeError("SHAP generation failed")

    monkeypatch.setattr(src.explainability, "generate_shap_plots", failing_shap)

    with pytest.raises(RuntimeError, match="SHAP generation failed"):
        train_model.main()
