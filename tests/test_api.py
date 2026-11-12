"""
Tests for the Flask REST API endpoints.
"""
import os
import sys
import json
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def client():
    """Create a Flask test client."""
    from app import create_app

    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def auth_token(client):
    """Obtain a JWT token for authenticated requests."""
    response = client.post(
        '/api/auth/login',
        data=json.dumps({
            'username': 'admin',
            'password': 'admin123'
        }),
        content_type='application/json'
    )
    if response.status_code == 200:
        data = json.loads(response.data)
        return data.get('token', '')
    return ''


VALID_APPLICANT = {
    "age": 35,
    "income": 65000,
    "employment_length": 8,
    "loan_amount": 15000,
    "interest_rate": 7.5,
    "credit_score": 720,
    "debt_to_income": 0.28,
    "num_credit_lines": 5,
    "num_late_payments": 0,
    "loan_purpose": "home_improvement",
    "home_ownership": "mortgage",
    "annual_revenue": 65000,
    "monthly_expenses": 3200,
    "savings_balance": 12000
}


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_endpoint_returns_200(self, client):
        """GET /api/health returns 200."""
        response = client.get('/api/health')
        assert response.status_code == 200

    def test_health_endpoint_returns_json(self, client):
        """GET /api/health returns valid JSON with status field."""
        response = client.get('/api/health')
        data = json.loads(response.data)
        assert 'status' in data


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""

    def test_predict_valid_input(self, client, auth_token):
        """POST /api/predict with valid data returns prediction."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {auth_token}'
        }
        response = client.post(
            '/api/predict',
            data=json.dumps(VALID_APPLICANT),
            headers=headers
        )
        # Accept 200 (success) or 503 (model not loaded in test env)
        assert response.status_code in (200, 503)

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'risk_score' in data or 'risk_level' in data

    def test_predict_invalid_input_returns_400(self, client, auth_token):
        """POST /api/predict with bad data returns 400."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {auth_token}'
        }
        response = client.post(
            '/api/predict',
            data=json.dumps({}),
            headers=headers
        )
        assert response.status_code in (400, 422)

    def test_predict_no_body_returns_error(self, client, auth_token):
        """POST /api/predict with no body returns an error."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {auth_token}'
        }
        response = client.post('/api/predict', headers=headers)
        assert response.status_code in (400, 415, 422)


class TestModelInfoEndpoint:
    """Tests for the model info endpoint."""

    def test_model_info(self, client):
        """GET /api/model-info returns model metadata."""
        response = client.get('/api/model-info')
        # Accept 200 or 503 if model not loaded
        assert response.status_code in (200, 503)

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'model_type' in data or 'type' in data


class TestAuthEndpoint:
    """Tests for the authentication endpoint."""

    def test_auth_login_valid(self, client):
        """POST /api/auth/login with valid credentials returns token."""
        response = client.post(
            '/api/auth/login',
            data=json.dumps({
                'username': 'admin',
                'password': 'admin123'
            }),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'token' in data

    def test_auth_login_invalid(self, client):
        """POST /api/auth/login with invalid credentials returns 401."""
        response = client.post(
            '/api/auth/login',
            data=json.dumps({
                'username': 'wrong',
                'password': 'wrong'
            }),
            content_type='application/json'
        )
        assert response.status_code == 401
