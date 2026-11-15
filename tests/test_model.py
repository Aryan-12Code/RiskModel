"""
Tests for model training, evaluation, and persistence.
"""
import os
import sys
import tempfile
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'sample_credit_data.csv'
)


def _get_train_data():
    """Helper: load and preprocess data for model tests."""
    from src.data_preprocessing import load_data, preprocess_data

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    return X_train, X_test, y_train, y_test


class TestXGBoostTraining:
    """Tests for XGBoost model training."""

    def test_xgboost_training(self):
        """Verify XGBoost model trains and has a predict method."""
        from src.model_training import train_xgboost

        X_train, _, y_train, _ = _get_train_data()
        model = train_xgboost(X_train, y_train)

        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_xgboost_predictions_shape(self):
        """Verify XGBoost predictions have correct shape."""
        from src.model_training import train_xgboost

        X_train, X_test, y_train, _ = _get_train_data()
        model = train_xgboost(X_train, y_train)
        preds = model.predict(X_test)

        assert preds.shape[0] == X_test.shape[0]


class TestLightGBMTraining:
    """Tests for LightGBM model training."""

    def test_lightgbm_training(self):
        """Verify LightGBM model trains and has a predict method."""
        from src.model_training import train_lightgbm

        X_train, _, y_train, _ = _get_train_data()
        model = train_lightgbm(X_train, y_train)

        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_lightgbm_predictions_shape(self):
        """Verify LightGBM predictions have correct shape."""
        from src.model_training import train_lightgbm

        X_train, X_test, y_train, _ = _get_train_data()
        model = train_lightgbm(X_train, y_train)
        preds = model.predict(X_test)

        assert preds.shape[0] == X_test.shape[0]


class TestEnsembleTraining:
    """Tests for the ensemble (VotingClassifier) model."""

    def test_ensemble_training(self):
        """Verify ensemble model trains and has a predict method."""
        from src.model_training import train_ensemble

        X_train, _, y_train, _ = _get_train_data()
        model = train_ensemble(X_train, y_train)

        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_ensemble_predictions_binary(self):
        """Verify ensemble predictions are binary (0 or 1)."""
        from src.model_training import train_ensemble

        X_train, X_test, y_train, _ = _get_train_data()
        model = train_ensemble(X_train, y_train)
        preds = model.predict(X_test)

        assert set(np.unique(preds)).issubset({0, 1})


class TestModelEvaluation:
    """Tests for model evaluation metrics."""

    def test_model_evaluation_has_required_keys(self):
        """Verify metrics dict has required keys."""
        from src.model_training import train_xgboost, evaluate_model

        X_train, X_test, y_train, y_test = _get_train_data()
        model = train_xgboost(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, 'XGBoost')

        required_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_model_evaluation_values_in_range(self):
        """Verify metric values are between 0 and 1."""
        from src.model_training import train_xgboost, evaluate_model

        X_train, X_test, y_train, y_test = _get_train_data()
        model = train_xgboost(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, 'XGBoost')

        numeric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for key in numeric_keys:
            value = metrics.get(key)
            if value is not None:
                assert 0.0 <= value <= 1.0, (
                    f"Metric {key}={value} out of [0, 1] range"
                )


class TestModelSaveLoad:
    """Tests for model save/load roundtrip."""

    def test_model_save_load(self):
        """Verify model can be saved and loaded correctly."""
        from src.model_training import train_xgboost, save_model

        X_train, X_test, y_train, _ = _get_train_data()
        model = train_xgboost(X_train, y_train)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            save_model(model, temp_path)
            assert os.path.exists(temp_path)

            # Load model
            import joblib
            loaded_model = joblib.load(temp_path)

            # Predictions should match
            original_preds = model.predict(X_test)
            loaded_preds = loaded_model.predict(X_test)
            np.testing.assert_array_equal(original_preds, loaded_preds)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestCVLeakageAndEnsembleFix:
    """Tests for SMOTE in CV pipeline and ensemble fitting without GridSearchCV."""

    def test_train_ensemble_does_not_call_gridsearch(self, monkeypatch):
        """Verify train_ensemble does not invoke GridSearchCV."""
        from src import model_training

        def fail_gridsearch(*args, **kwargs):
            raise AssertionError("GridSearchCV should not be called by train_ensemble")

        monkeypatch.setattr(model_training, "GridSearchCV", fail_gridsearch)

        X_train, _, y_train, _ = _get_train_data()
        ensemble = model_training.train_ensemble(X_train, y_train)
        assert ensemble is not None

    def test_saved_ensemble_estimators_types(self):
        """Verify saved ensemble has estimators_ of types XGBClassifier and LGBMClassifier."""
        from src.model_training import train_ensemble
        from sklearn.pipeline import Pipeline
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        X_train, _, y_train, _ = _get_train_data()
        ensemble = train_ensemble(X_train, y_train)

        assert hasattr(ensemble, "estimators_")
        assert len(ensemble.estimators_) == 2
        for est in ensemble.estimators_:
            assert not isinstance(est, Pipeline)
        assert isinstance(ensemble.estimators_[0], XGBClassifier)
        assert isinstance(ensemble.estimators_[1], LGBMClassifier)

    def test_xgboost_lightgbm_return_non_pipeline_classifier_and_cv_auc(self):
        """Verify xgboost and lightgbm training functions return a non-Pipeline classifier, CV AUC, and best params."""
        from src.model_training import train_xgboost, train_lightgbm
        from sklearn.pipeline import Pipeline
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        X_train, _, y_train, _ = _get_train_data()

        xgb_res = train_xgboost(X_train, y_train)
        xgb_clf, xgb_cv_auc, xgb_params = xgb_res
        assert not isinstance(xgb_clf, Pipeline)
        assert isinstance(xgb_clf, XGBClassifier)
        assert isinstance(xgb_cv_auc, (float, int))
        assert 0.0 <= xgb_cv_auc <= 1.0
        assert isinstance(xgb_params, dict)
        assert any(k.startswith("clf__") for k in xgb_params)

        lgb_res = train_lightgbm(X_train, y_train)
        lgb_clf, lgb_cv_auc, lgb_params = lgb_res
        assert not isinstance(lgb_clf, Pipeline)
        assert isinstance(lgb_clf, LGBMClassifier)
        assert isinstance(lgb_cv_auc, (float, int))
        assert 0.0 <= lgb_cv_auc <= 1.0
        assert isinstance(lgb_params, dict)
        assert any(k.startswith("clf__") for k in lgb_params)

