"""
Authentication module for the RiskModel API.

Provides JWT-based authentication with role-based access control.
Users are stored in the SQLite database with passwords hashed via werkzeug.
"""

import datetime
import functools
import logging
from typing import Any, Callable, Dict, Optional

import jwt
from flask import Blueprint, current_app, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

from database import get_connection

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')


# ---------------------------------------------------------------------------
# User management helpers
# ---------------------------------------------------------------------------

def create_user(db_path: str, username: str, password: str, role: str = 'analyst') -> None:
    """
    Create a new user in the database.

    Args:
        db_path: Path to the SQLite database.
        username: Unique username.
        password: Plaintext password (will be hashed before storage).
        role: User role ('admin' or 'analyst').

    Raises:
        ValueError: If the username already exists or role is invalid.
    """
    if role not in ('admin', 'analyst'):
        raise ValueError(f"Role must be 'admin' or 'analyst', got '{role}'")

    password_hash = generate_password_hash(password)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role),
        )
        conn.commit()
    except Exception as exc:
        if 'UNIQUE constraint' in str(exc):
            raise ValueError(f"User '{username}' already exists") from exc
        raise
    finally:
        conn.close()


def _get_user(db_path: str, username: str) -> Optional[Dict[str, str]]:
    """Look up a user by username. Returns dict with password_hash and role, or None."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT password_hash, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return None
        return {"password_hash": row["password_hash"], "role": row["role"]}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _generate_token(username: str, role: str) -> str:
    """
    Generate a JWT token for the given user.

    Args:
        username: The authenticated username.
        role: The user's role ('admin' or 'analyst').

    Returns:
        Encoded JWT string.
    """
    expiration_hours = current_app.config.get('JWT_EXPIRATION_HOURS', 24)
    payload = {
        "sub": username,
        "role": role,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=expiration_hours),
    }
    token = jwt.encode(payload, current_app.config['SECRET_KEY'], algorithm='HS256')
    return token


def _decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify a JWT token.

    Args:
        token: The raw JWT string.

    Returns:
        Decoded payload dictionary.

    Raises:
        jwt.ExpiredSignatureError: If the token has expired.
        jwt.InvalidTokenError: If the token is malformed.
    """
    return jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])


def login(username: str, password: str) -> Optional[str]:
    """
    Authenticate a user and return a JWT token.

    Args:
        username: The username to authenticate.
        password: The plaintext password.

    Returns:
        JWT token string on success, or None on failure.
    """
    db_path = current_app.config['DATABASE_URI']
    user = _get_user(db_path, username)
    if user is None:
        return None
    if not check_password_hash(user["password_hash"], password):
        return None
    return _generate_token(username, user["role"])


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def token_required(f: Callable) -> Callable:
    """
    Decorator that requires a valid JWT in the Authorization header.

    Adds ``current_user`` (dict with 'username' and 'role') as the
    first positional argument to the wrapped function.
    """
    @functools.wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({"status": "error", "message": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(' ', 1)[1]
        try:
            payload = _decode_token(token)
        except jwt.ExpiredSignatureError:
            return jsonify({"status": "error", "message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        current_user = {"username": payload["sub"], "role": payload["role"]}
        return f(current_user, *args, **kwargs)

    return decorated


def role_required(*allowed_roles: str) -> Callable:
    """
    Decorator factory that restricts access to specific roles.

    Must be applied **after** ``@token_required`` so that
    ``current_user`` is available.

    Args:
        *allowed_roles: One or more role strings (e.g. 'admin', 'analyst').
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated(current_user: Dict[str, str], *args: Any, **kwargs: Any) -> Any:
            if current_user["role"] not in allowed_roles:
                return jsonify({
                    "status": "error",
                    "message": "Insufficient permissions",
                }), 403
            return f(current_user, *args, **kwargs)
        return decorated
    return decorator


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@auth_bp.route('/login', methods=['POST'])
def login_route():
    """
    Authenticate a user and return a JWT token.

    Request JSON:
        {"username": "...", "password": "..."}

    Response JSON:
        {"status": "success", "data": {"token": "..."}, "message": "Login successful"}
    """
    body = request.get_json(silent=True)
    if not body or 'username' not in body or 'password' not in body:
        return jsonify({
            "status": "error",
            "data": None,
            "message": "Username and password are required",
        }), 400

    token = login(body['username'], body['password'])
    if token is None:
        return jsonify({
            "status": "error",
            "data": None,
            "message": "Invalid username or password",
        }), 401

    db_path = current_app.config['DATABASE_URI']
    user = _get_user(db_path, body['username'])
    logger.info("User '%s' logged in", body['username'])
    return jsonify({
        "status": "success",
        "token": token,
        "data": {
            "token": token,
            "user": {
                "username": body['username'],
                "role": user['role']
            }
        },
        "message": "Login successful",
    }), 200


@auth_bp.route('/register', methods=['POST'])
@token_required
@role_required('admin')
def register_route(current_user: Dict[str, str]):
    """
    Register a new user (admin only).

    Request JSON:
        {"username": "...", "password": "...", "role": "analyst"}

    Response JSON:
        {"status": "success", "data": null, "message": "User registered successfully"}
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({
            "status": "error",
            "data": None,
            "message": "Request body is required",
        }), 400

    username = body.get('username', '').strip()
    password = body.get('password', '')
    role = body.get('role', 'analyst').strip().lower()

    if not username or not password:
        return jsonify({
            "status": "error",
            "data": None,
            "message": "Username and password are required",
        }), 400

    if role not in ('admin', 'analyst'):
        return jsonify({
            "status": "error",
            "data": None,
            "message": "Role must be 'admin' or 'analyst'",
        }), 400

    db_path = current_app.config['DATABASE_URI']
    if _get_user(db_path, username) is not None:
        return jsonify({
            "status": "error",
            "data": None,
            "message": f"User '{username}' already exists",
        }), 409

    create_user(db_path, username, password, role)
    logger.info("Admin '%s' registered new user '%s' with role '%s'",
                current_user['username'], username, role)

    return jsonify({
        "status": "success",
        "data": None,
        "message": "User registered successfully",
    }), 201
