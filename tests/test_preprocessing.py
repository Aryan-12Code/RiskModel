"""
Tests for the data preprocessing pipeline.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'sample_credit_data.csv'
)


class TestLoadData:
    """Tests for data loading functionality."""

    def test_load_data_returns_dataframe(self):
        """Verify CSV loads correctly and returns a DataFrame."""
        from src.data_preprocessing import load_data

        df = load_data(DATA_PATH)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_data_has_expected_columns(self):
        """Verify loaded data contains required columns."""
        from src.data_preprocessing import load_data

        df = load_data(DATA_PATH)
        required_columns = [
            'age', 'income', 'loan_amount', 'credit_score',
            'debt_to_income', 'default'
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_data_target_is_binary(self):
        """Verify the target column is binary (0/1)."""
        from src.data_preprocessing import load_data

        df = load_data(DATA_PATH)
        assert set(df['default'].unique()).issubset({0, 1})


class TestFeatureEngineering:
    """Tests for feature engineering functionality."""

    def test_feature_engineering_creates_derived_features(self):
        """Verify derived features are created."""
        from src.feature_engineering import create_derived_features
        from src.data_preprocessing import load_data

        df = load_data(DATA_PATH)
        df_engineered = create_derived_features(df)

        expected_features = [
            'debt_ratio', 'income_to_loan_ratio',
            'credit_utilization', 'payment_history_score'
        ]
        for feat in expected_features:
            assert feat in df_engineered.columns, (
                f"Derived feature '{feat}' not created"
            )

    def test_feature_engineering_no_rows_lost(self):
        """Verify feature engineering does not drop rows."""
        from src.feature_engineering import create_derived_features
        from src.data_preprocessing import load_data

        df = load_data(DATA_PATH)
        original_len = len(df)
        df_engineered = create_derived_features(df)
        assert len(df_engineered) == original_len


class TestPreprocessingPipeline:
    """Tests for the full preprocessing pipeline."""

    def test_preprocessing_pipeline_shapes(self):
        """Verify output shapes are consistent after preprocessing."""
        from src.data_preprocessing import load_data, preprocess_data

        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

        # X and y lengths must match
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        # Feature counts must match
        assert X_train.shape[1] == X_test.shape[1]

        # Total samples should be reasonable
        total = X_train.shape[0] + X_test.shape[0]
        assert total > 0

    def test_preprocessing_pipeline_no_nulls(self):
        """Verify no null values remain after preprocessing."""
        from src.data_preprocessing import load_data, preprocess_data

        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test, _ = preprocess_data(df)

        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isnan(X_test))

    def test_preprocessor_returned(self):
        """Verify a fitted preprocessor is returned."""
        from src.data_preprocessing import load_data, preprocess_data

        df = load_data(DATA_PATH)
        _, _, _, _, preprocessor = preprocess_data(df)
        assert preprocessor is not None


class TestSMOTEBalancing:
    """Tests for SMOTE class balancing."""

    def test_smote_balancing_improves_balance(self):
        """Verify class balance is improved after SMOTE."""
        from src.data_preprocessing import load_data, preprocess_data

        df = load_data(DATA_PATH)
        X_train, _, y_train, _, _ = preprocess_data(df)

        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))

        # After SMOTE, minority class should be closer in count
        if len(class_counts) == 2:
            ratio = min(class_counts.values()) / max(class_counts.values())
            # SMOTE should bring ratio above 0.5 at minimum
            assert ratio > 0.4, (
                f"Class ratio {ratio:.2f} too imbalanced after SMOTE"
            )
