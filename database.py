"""
Database module for the RiskModel application.

Provides SQLite-based storage for prediction logging and retrieval
using the built-in sqlite3 module for simplicity.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.

    Args:
        db_path: Path to the SQLite database file or sqlite:/// URI.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    import os
    if db_path.startswith("sqlite:///"):
        db_path = db_path[10:]
    elif db_path.startswith("sqlite://"):
        db_path = db_path[9:]

    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_tables(db_path: str) -> None:
    """
    Create the required database tables if they do not already exist.

    Creates:
        predictions_log — stores every prediction request and result.

    Args:
        db_path: Absolute path to the SQLite database file.
    """
    conn = get_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_data TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                shap_values TEXT
            )
        """)
        conn.commit()
        logger.info("Database tables created / verified at %s", db_path)
    except sqlite3.Error as exc:
        logger.error("Failed to create tables: %s", exc)
        raise
    finally:
        conn.close()


def log_prediction(
    db_path: str,
    input_data: Dict[str, Any],
    risk_score: float,
    risk_level: str,
    shap_values: Optional[Dict[str, float]] = None,
) -> int:
    """
    Insert a prediction record into the predictions_log table.

    Args:
        db_path: Absolute path to the SQLite database file.
        input_data: Dictionary of applicant input features.
        risk_score: Predicted probability of default (0-1).
        risk_level: Categorical risk level ('low', 'medium', or 'high').
        shap_values: Optional dictionary mapping feature names to SHAP values.

    Returns:
        The row id of the inserted record.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO predictions_log
                (timestamp, input_data, risk_score, risk_level, shap_values)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                json.dumps(input_data),
                risk_score,
                risk_level,
                json.dumps(shap_values) if shap_values else None,
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        logger.debug("Logged prediction id=%d risk_level=%s", row_id, risk_level)
        return row_id
    except sqlite3.Error as exc:
        logger.error("Failed to log prediction: %s", exc)
        raise
    finally:
        conn.close()


def get_prediction_history(db_path: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent prediction records.

    Args:
        db_path: Absolute path to the SQLite database file.
        limit: Maximum number of records to return (default 100).

    Returns:
        List of dictionaries, each representing a prediction record,
        ordered by most recent first.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, timestamp, input_data, risk_score, risk_level, shap_values
            FROM predictions_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        results = []
        for row in rows:
            record = {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "input_data": json.loads(row["input_data"]),
                "risk_score": row["risk_score"],
                "risk_level": row["risk_level"],
                "shap_values": json.loads(row["shap_values"]) if row["shap_values"] else None,
            }
            results.append(record)

        return results
    except sqlite3.Error as exc:
        logger.error("Failed to retrieve prediction history: %s", exc)
        raise
    finally:
        conn.close()


def get_prediction_stats(db_path: str) -> Dict[str, Any]:
    """
    Compute aggregate statistics over all logged predictions.

    Args:
        db_path: Absolute path to the SQLite database file.

    Returns:
        Dictionary with keys:
            total_predictions, avg_risk_score, min_risk_score, max_risk_score,
            risk_level_counts (dict mapping level → count).
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*)      AS total,
                AVG(risk_score) AS avg_score,
                MIN(risk_score) AS min_score,
                MAX(risk_score) AS max_score
            FROM predictions_log
            """
        ).fetchone()

        level_rows = conn.execute(
            """
            SELECT risk_level, COUNT(*) AS cnt
            FROM predictions_log
            GROUP BY risk_level
            """
        ).fetchall()

        level_counts = {r["risk_level"]: r["cnt"] for r in level_rows}

        stats = {
            "total_predictions": row["total"],
            "avg_risk_score": round(row["avg_score"], 4) if row["avg_score"] is not None else None,
            "min_risk_score": round(row["min_score"], 4) if row["min_score"] is not None else None,
            "max_risk_score": round(row["max_score"], 4) if row["max_score"] is not None else None,
            "risk_level_counts": level_counts,
        }
        return stats
    except sqlite3.Error as exc:
        logger.error("Failed to compute prediction stats: %s", exc)
        raise
    finally:
        conn.close()
