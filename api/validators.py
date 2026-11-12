"""
Input validation module for the RiskModel API.

Provides functions to validate and sanitize applicant data before
it is passed to the prediction pipeline.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Required fields with their expected types and (min, max) ranges.
# None means no range constraint for that bound.
REQUIRED_FIELDS: Dict[str, Dict[str, Any]] = {
    "age": {"type": (int, float), "min": 18, "max": 120},
    "income": {"type": (int, float), "min": 0, "max": None},
    "employment_length": {"type": (int, float), "min": 0, "max": 60},
    "loan_amount": {"type": (int, float), "min": 0, "max": None},
    "interest_rate": {"type": (int, float), "min": 0, "max": 100},
    "credit_score": {"type": (int, float), "min": 300, "max": 850},
    "debt_to_income": {"type": (int, float), "min": 0, "max": None},
    "num_credit_lines": {"type": (int, float), "min": 0, "max": None},
    "num_late_payments": {"type": (int, float), "min": 0, "max": None},
    "loan_purpose": {
        "type": (str,),
        "allowed": [
            "personal", "education", "medical",
            "business", "home_improvement", "debt_consolidation",
        ],
    },
    "home_ownership": {
        "type": (str,),
        "allowed": ["own", "rent", "mortgage"],
    },
    "annual_revenue": {"type": (int, float), "min": 0, "max": None},
    "monthly_expenses": {"type": (int, float), "min": 0, "max": None},
    "savings_balance": {"type": (int, float), "min": 0, "max": None},
}


def validate_prediction_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize a single prediction input dictionary.

    Checks that all required fields are present with correct types
    and that numeric values fall within acceptable ranges. String
    fields are stripped and lower-cased.

    Args:
        data: Raw input dictionary from the API request.

    Returns:
        A cleaned copy of the input dictionary with sanitized values.

    Raises:
        ValueError: If any required field is missing, has the wrong
            type, or has a value outside its permitted range.
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object")

    validated: Dict[str, Any] = {}
    errors: list = []

    for field, rules in REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
            continue

        value = data[field]

        # Type check
        if not isinstance(value, rules["type"]):
            expected = " or ".join(t.__name__ for t in rules["type"])
            errors.append(
                f"Field '{field}' must be of type {expected}, "
                f"got {type(value).__name__}"
            )
            continue

        # Sanitize strings
        if isinstance(value, str):
            value = value.strip().lower()
            allowed = rules.get("allowed")
            if allowed and value not in allowed:
                errors.append(
                    f"Field '{field}' must be one of {allowed}, got '{value}'"
                )
                continue

        # Range checks for numerics
        if isinstance(value, (int, float)):
            min_val = rules.get("min")
            max_val = rules.get("max")
            if min_val is not None and value < min_val:
                errors.append(
                    f"Field '{field}' must be >= {min_val}, got {value}"
                )
                continue
            if max_val is not None and value > max_val:
                errors.append(
                    f"Field '{field}' must be <= {max_val}, got {value}"
                )
                continue

        validated[field] = value

    if errors:
        raise ValueError("; ".join(errors))

    logger.debug("Input validated successfully")
    return validated
