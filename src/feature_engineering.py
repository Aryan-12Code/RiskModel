"""
Feature Engineering Module for Financial Risk Assessment.

Provides functions to create derived features, perform feature selection
using mutual information, extract feature importance from trained models,
and run correlation analysis.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional derived features from the raw dataset.

    Engineered features:
        - debt_ratio: loan_amount / income
        - income_to_loan_ratio: income / loan_amount
        - credit_utilization: (debt_to_income * num_credit_lines) / 100
        - risk_score_bin: credit_score bucketed into 5 risk tiers
        - payment_history_score: num_late_payments / num_credit_lines
        - expense_ratio: monthly_expenses * 12 / income
        - savings_to_loan: savings_balance / loan_amount
        - net_monthly_income: (income / 12) - monthly_expenses
        - loan_to_annual_revenue: loan_amount / annual_revenue
        - credit_lines_per_year: num_credit_lines / (employment_length + 1)

    Args:
        df: Raw DataFrame with original columns.

    Returns:
        DataFrame with all original columns plus derived features.
    """
    df = df.copy()
    logger.info("Creating derived features")

    # Core ratios
    df["debt_ratio"] = np.where(
        df["income"].gt(0), df["loan_amount"] / df["income"], 0.0
    )
    df["income_to_loan_ratio"] = np.where(
        df["loan_amount"].gt(0), df["income"] / df["loan_amount"], 0.0
    )
    df["credit_utilization"] = (
        df["debt_to_income"] * df["num_credit_lines"] / 100.0
    )
    df["risk_score_bin"] = pd.cut(
        df["credit_score"].fillna(df["credit_score"].median()),
        bins=[0, 500, 600, 700, 750, 900],
        labels=[1, 2, 3, 4, 5],
    ).astype(float)
    df["payment_history_score"] = np.where(
        df["num_credit_lines"].gt(0),
        df["num_late_payments"] / df["num_credit_lines"],
        df["num_late_payments"].astype(float),
    )

    # Extra derived features
    df["expense_ratio"] = np.where(
        df["income"].gt(0), (df["monthly_expenses"] * 12) / df["income"], 0.0
    )
    df["savings_to_loan"] = np.where(
        df["loan_amount"].gt(0), df["savings_balance"] / df["loan_amount"], 0.0
    )
    df["net_monthly_income"] = (df["income"] / 12) - df["monthly_expenses"]
    df["loan_to_annual_revenue"] = np.where(
        df["annual_revenue"].gt(0),
        df["loan_amount"] / df["annual_revenue"],
        0.0,
    )
    df["credit_lines_per_year"] = df["num_credit_lines"] / (
        df["employment_length"].fillna(0) + 1
    )

    logger.info("Created %d derived features", 10)
    return df


def select_features_mutual_info(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    k: int = 15,
    feature_names: list[str] | None = None,
) -> list[str]:
    """Select the top-k features by mutual information with the target.

    Args:
        X: Feature matrix (array or DataFrame).
        y: Target vector.
        k: Number of features to select.
        feature_names: Optional list of names matching columns of X.

    Returns:
        List of selected feature names sorted by descending MI score.
    """
    logger.info("Selecting top %d features by mutual information", k)

    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or list(X.columns)
        X_arr = X.values
    else:
        X_arr = X
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    mi_scores = mutual_info_classif(X_arr, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

    selected = list(mi_series.head(k).index)
    logger.info("Selected features: %s", selected)
    return selected


def get_feature_importance(
    model, feature_names: list[str]
) -> dict[str, float]:
    """Extract feature importances from a fitted model.

    Supports tree-based models with ``feature_importances_`` and linear models
    with ``coef_``.  For ensemble/voting models, attempts to average importances
    from sub-estimators.

    Args:
        model: Fitted sklearn-compatible model.
        feature_names: List of feature names matching the model input.

    Returns:
        Dictionary of {feature_name: importance} sorted descending by value.
    """
    logger.info("Extracting feature importances")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    elif hasattr(model, "estimators_"):
        # VotingClassifier / ensemble with sub-estimators
        all_imp = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                all_imp.append(est.feature_importances_)
        if all_imp:
            importances = np.mean(all_imp, axis=0)
        else:
            logger.warning("No importances found in sub-estimators")
            return {}
    else:
        logger.warning("Model type %s has no importances attribute", type(model))
        return {}

    imp_dict = dict(zip(feature_names, importances))
    sorted_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
    return sorted_dict


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix for numeric columns.

    Args:
        df: DataFrame (only numeric columns are used).

    Returns:
        Correlation matrix as a DataFrame.
    """
    logger.info("Computing correlation matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    logger.info("Correlation matrix shape: %s", corr_matrix.shape)
    return corr_matrix


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_credit_data.csv"
    )
    df = pd.read_csv(data_path)
    df = create_derived_features(df)
    print("Derived features added. New columns:")
    print([c for c in df.columns if c not in pd.read_csv(data_path).columns])

    # Correlation analysis
    corr = correlation_analysis(df)
    print("\nTop correlations with 'default':")
    if "default" in corr.columns:
        print(corr["default"].sort_values(ascending=False).head(10))

    # Mutual information (numeric only, no NaN)
    target = "default"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors="ignore")
    X_num = df[numeric_cols].fillna(0)
    y = df[target]
    selected = select_features_mutual_info(X_num, y, k=10, feature_names=list(numeric_cols))
    print(f"\nTop 10 features by MI: {selected}")
