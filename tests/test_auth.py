"""
Tests for the auth configuration changes.

Covers:
- App startup without SECRET_KEY raises an error (outside testing config)
- Token signed with wrong key is rejected 401
- Login with wrong password returns 401
- Login works for a user created via create_user logic
"""
import json
import os
import sys
import tempfile

import jwt
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestSecretKeyRequired:
    """SECRET_KEY must be set outside of TestingConfig."""

    def test_missing_secret_key_raises(self, monkeypatch):
        """Creating the app without SECRET_KEY (non-testing) must raise."""
        monkeypatch.delenv('SECRET_KEY', raising=False)
        # Config.SECRET_KEY is evaluated at import time from os.environ,
        # so we must also patch the class attribute for the current process.
        from config import Config, DevelopmentConfig
        monkeypatch.setattr(Config, 'SECRET_KEY', None)
        monkeypatch.setattr(DevelopmentConfig, 'SECRET_KEY', None)
        from app import create_app

        with pytest.raises(RuntimeError):
            create_app('development')

    def test_testing_config_does_not_require_env_secret(self, monkeypatch):
        """TestingConfig supplies its own key — no error."""
        monkeypatch.delenv('SECRET_KEY', raising=False)
        from app import create_app

        app = create_app('testing')
        assert app.config['SECRET_KEY']


class TestTokenValidation:
    """A token signed with a different key must be rejected."""

    def test_wrong_key_token_rejected(self):
        """Token signed with a wrong secret returns 401."""
        from app import create_app

        app = create_app('testing')
        with app.test_client() as client:
            # Forge a token with a different key
            import datetime
            payload = {
                "sub": "attacker",
                "role": "admin",
                "iat": datetime.datetime.utcnow(),
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
            }
            bad_token = jwt.encode(payload, "wrong-secret-key", algorithm="HS256")
            resp = client.get(
                '/api/prediction-history',
                headers={'Authorization': f'Bearer {bad_token}'},
            )
            assert resp.status_code == 401


class TestDatabaseAuth:
    """Auth using database-backed users with werkzeug password hashing."""

    @pytest.fixture(autouse=True)
    def _app(self, tmp_path):
        """Create a testing app with a temp database."""
        from app import create_app

        db_path = str(tmp_path / 'test_auth.db')
        app = create_app('testing')
        app.config['DATABASE_URI'] = db_path

        # Create tables (including users table)
        from database import create_tables
        create_tables(db_path)

        self.app = app
        self.db_path = db_path

    def _create_user(self, username, password, role='analyst'):
        """Use the same function that create_user.py exposes."""
        from auth import create_user
        create_user(self.db_path, username, password, role)

    def test_login_wrong_password_returns_401(self):
        self._create_user('alice', 'correct-password', 'analyst')
        with self.app.test_client() as client:
            resp = client.post(
                '/api/auth/login',
                data=json.dumps({'username': 'alice', 'password': 'wrong-password'}),
                content_type='application/json',
            )
            assert resp.status_code == 401

    def test_login_works_after_create_user(self):
        self._create_user('bob', 'good-pass-123', 'admin')
        with self.app.test_client() as client:
            resp = client.post(
                '/api/auth/login',
                data=json.dumps({'username': 'bob', 'password': 'good-pass-123'}),
                content_type='application/json',
            )
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert 'token' in data

    def test_created_user_token_is_valid(self):
        self._create_user('carol', 's3cret!', 'analyst')
        with self.app.test_client() as client:
            resp = client.post(
                '/api/auth/login',
                data=json.dumps({'username': 'carol', 'password': 's3cret!'}),
                content_type='application/json',
            )
            token = json.loads(resp.data)['token']
            # Use the token on a protected endpoint
            resp2 = client.get(
                '/api/prediction-history',
                headers={'Authorization': f'Bearer {token}'},
            )
            assert resp2.status_code == 200
