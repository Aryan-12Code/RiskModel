"""
Data Preprocessing Pipeline for Financial Risk Assessment.

Provides functions to load, clean, engineer features, encode categoricals,
scale numerics, handle class imbalance with SMOTE, and split data for training.
"""

import os
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COL = "default"
CATEGORICAL_COLS = ["loan_purpose", "home_ownership"]
NUMERIC_COLS = [
    "age", "income", "employment_length", "loan_amount", "interest_rate",
    "credit_score", "debt_to_income", "num_credit_lines", "num_late_payments",
    "annual_revenue", "monthly_expenses", "savings_balance",
]
ENGINEERED_NUMERIC_COLS = [
    "debt_ratio", "income_to_loan_ratio", "credit_utilization",
    "risk_score_bin", "payment_history_score",
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        pd.DataFrame with raw data.
    """
    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features used in preprocessing.

    Args:
        df: DataFrame with at least the original numeric columns.

    Returns:
        DataFrame with additional engineered columns.
    """
    df = df.copy()

    # debt_ratio: loan_amount / income (guarded against zero/nan)
    df["debt_ratio"] = np.where(
        df["income"].gt(0), df["loan_amount"] / df["income"], 0.0
    )

    # income_to_loan_ratio: inverse of debt_ratio
    df["income_to_loan_ratio"] = np.where(
        df["loan_amount"].gt(0), df["income"] / df["loan_amount"], 0.0
    )

    # credit_utilization: proxy using debt_to_income * num_credit_lines
    df["credit_utilization"] = df["debt_to_income"] * df["num_credit_lines"] / 100.0

    # risk_score_bin: binned credit_score (1–5)
    df["risk_score_bin"] = pd.cut(
        df["credit_score"],
        bins=[0, 500, 600, 700, 750, 900],
        labels=[1, 2, 3, 4, 5],
    ).astype(float)

    # payment_history_score: higher is worse — late payments relative to credit lines
    df["payment_history_score"] = np.where(
        df["num_credit_lines"].gt(0),
        df["num_late_payments"] / df["num_credit_lines"],
        df["num_late_payments"].astype(float),
    )

    logger.info("Engineered %d derived features", len(ENGINEERED_NUMERIC_COLS))
    return df


def _build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build a ColumnTransformer with imputation, scaling, and encoding.

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: A fitted ColumnTransformer.

    Returns:
        List of output feature names.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn
        num_features = preprocessor.transformers_[0][2]
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_features = list(cat_encoder.get_feature_names_out())
        return list(num_features) + cat_features


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_preprocessor: bool = True,
) -> tuple:
    """Run the full preprocessing pipeline.

    Steps:
        1. Engineer derived features.
        2. Separate features / target.
        3. Build and fit ColumnTransformer (impute, scale, encode).
        4. Train/test split (stratified).
        5. Apply SMOTE on training set.
        6. Optionally save the preprocessor to disk.

    Args:
        df: Raw DataFrame (must contain TARGET_COL).
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.
        save_preprocessor: Whether to persist the preprocessor as a pickle.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor).
    """
    logger.info("Starting preprocessing pipeline")

    # 1. Feature engineering
    df = _engineer_features(df)

    # 2. Separate X / y
    all_numeric = NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
    X = df[all_numeric + CATEGORICAL_COLS]
    y = df[TARGET_COL]
    logger.info("Features: %d numeric + %d categorical", len(all_numeric), len(CATEGORICAL_COLS))

    # 3. Build preprocessor
    preprocessor = _build_preprocessor(all_numeric, CATEGORICAL_COLS)

    # 4. Train / test split (stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Train/test split: %d train, %d test", len(X_train_raw), len(X_test_raw)
    )

    # Fit on training data, transform both
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # 5. SMOTE on training set (with fallback balancing)
    if HAS_SMOTE:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    else:
        y_train_arr = np.array(y_train)
        idx_0 = np.where(y_train_arr == 0)[0]
        idx_1 = np.where(y_train_arr == 1)[0]
        if len(idx_1) > 0 and len(idx_0) > len(idx_1):
            extra_idx = np.random.choice(idx_1, size=(len(idx_0) - len(idx_1)), replace=True)
            all_idx = np.concatenate([np.arange(len(y_train_arr)), extra_idx])
            X_train = X_train[all_idx]
            y_train = y_train_arr[all_idx]
    logger.info(
        "After class balancing: %d training samples (class balance: %s)",
        len(X_train),
        dict(pd.Series(y_train).value_counts()),
    )

    # 6. Save preprocessor
    if save_preprocessor:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(models_dir, exist_ok=True)
        pkl_path = os.path.join(models_dir, "preprocessor.pkl")
        import joblib
        joblib.dump(preprocessor, pkl_path)
        logger.info("Preprocessor saved to %s", pkl_path)

    return X_train, X_test, y_train, y_test, preprocessor


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_credit_data.csv"
    )
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    feature_names = get_feature_names(preprocessor)
    print(f"Feature count: {len(feature_names)}")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Feature names: {feature_names[:10]} ...")
