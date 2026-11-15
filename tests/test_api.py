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

    app = create_app('testing')
    app.config['TESTING'] = True

    # Ensure users table exists and seed a test user
    from database import create_tables
    create_tables(app.config['DATABASE_URI'])
    from auth import create_user
    try:
        create_user(app.config['DATABASE_URI'], 'admin', 'admin123', 'admin')
    except ValueError:
        pass  # user already exists from a prior fixture invocation

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

    def test_predict_no_token_returns_401(self, client):
        """POST /api/predict without token returns 401."""
        response = client.post(
            '/api/predict',
            data=json.dumps(VALID_APPLICANT),
            content_type='application/json'
        )
        assert response.status_code == 401

    def test_predict_valid_token_returns_200(self, client, auth_token):
        """POST /api/predict with valid token returns 200."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {auth_token}'
        }
        response = client.post(
            '/api/predict',
            data=json.dumps(VALID_APPLICANT),
            headers=headers
        )
        assert response.status_code == 200


class TestPredictionHistoryEndpoint:
    """Tests for the prediction history endpoint."""

    def test_prediction_history_no_token_returns_401(self, client):
        """GET /api/prediction-history without token returns 401."""
        response = client.get('/api/prediction-history')
        assert response.status_code == 401

    def test_prediction_history_valid_token_returns_200(self, client, auth_token):
        """GET /api/prediction-history with valid token returns 200."""
        headers = {'Authorization': f'Bearer {auth_token}'}
        response = client.get('/api/prediction-history', headers=headers)
        assert response.status_code == 200


class TestPredictionStatsEndpoint:
    """Tests for the prediction stats endpoint."""

    def test_prediction_stats_no_token_returns_401(self, client):
        """GET /api/prediction-stats without token returns 401."""
        response = client.get('/api/prediction-stats')
        assert response.status_code == 401

    def test_prediction_stats_valid_token_returns_200(self, client, auth_token):
        """GET /api/prediction-stats with valid token returns 200."""
        headers = {'Authorization': f'Bearer {auth_token}'}
        response = client.get('/api/prediction-stats', headers=headers)
        assert response.status_code == 200

    def test_prediction_stats_empty_db_returns_zero_and_no_fabricated_numbers(self, client, auth_token, tmp_path):
        """Empty database returns total_predictions 0 and null averages, no fabricated numbers."""
        empty_db = str(tmp_path / "empty.db")
        from database import create_tables
        create_tables(empty_db)
        client.application.config['DATABASE_URI'] = empty_db
        headers = {'Authorization': f'Bearer {auth_token}'}

        response = client.get('/api/prediction-stats', headers=headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data.get('total_predictions') == 0
        assert data.get('average_risk_score') is None
        assert data.get('high_risk_percentage') is None
        fabricated = [1247, 0.34, 0.28, 18.3, 94.7]
        for val in fabricated:
            assert data.get('total_predictions') != val
            assert data.get('average_risk_score') != val
            assert data.get('high_risk_percentage') != val
            assert data.get('model_accuracy') != val

    def test_prediction_stats_model_accuracy_matches_file_or_null_if_missing(self, client, auth_token, tmp_path):
        """model_accuracy equals the file's ensemble accuracy, or null if missing."""
        headers = {'Authorization': f'Bearer {auth_token}'}

        # Case 1: metrics file exists
        metrics_file = tmp_path / "metrics.json"
        metrics_content = {
            "ensemble": {"accuracy": 0.912},
            "accuracy": 0.912
        }
        metrics_file.write_text(json.dumps(metrics_content))
        client.application.config['METRICS_PATH'] = str(metrics_file)

        response = client.get('/api/prediction-stats', headers=headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data.get('model_accuracy') == 0.912

        # Case 2: metrics file is missing
        client.application.config['METRICS_PATH'] = str(tmp_path / "nonexistent.json")
        response = client.get('/api/prediction-stats', headers=headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data.get('model_accuracy') is None

    def test_prediction_stats_returns_500_on_db_query_failure(self, client, auth_token, monkeypatch):
        """When database query fails, return 500 and no fake data."""
        headers = {'Authorization': f'Bearer {auth_token}'}
        import database
        def fail_get_stats(db_path):
            raise Exception("Database query failed")
        monkeypatch.setattr(database, 'get_prediction_stats', fail_get_stats)

        response = client.get('/api/prediction-stats', headers=headers)
        assert response.status_code == 500
        data = json.loads(response.data)
        for val in [1247, 0.34, 18.3, 94.7]:
            assert data.get('total_predictions') != val
            assert data.get('average_risk_score') != val
            assert data.get('high_risk_percentage') != val
            assert data.get('model_accuracy') != val


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

    def test_model_info_returns_accuracy_from_metrics_file(self, client, tmp_path):
        """/api/model-info returns the same accuracy as the metrics file."""
        metrics_file = tmp_path / "metrics.json"
        metrics_content = {
            "ensemble": {"accuracy": 0.895, "roc_auc": 0.94, "f1": 0.88, "precision": 0.87, "recall": 0.89},
            "xgboost": {"accuracy": 0.88, "roc_auc": 0.93, "f1": 0.86, "precision": 0.85, "recall": 0.87},
            "lightgbm": {"accuracy": 0.87, "roc_auc": 0.92, "f1": 0.85, "precision": 0.84, "recall": 0.86},
            "accuracy": 0.895,
            "n_train": 800,
            "n_test": 200,
            "trained_at": "2026-11-15T00:00:00Z"
        }
        metrics_file.write_text(json.dumps(metrics_content))
        client.application.config['METRICS_PATH'] = str(metrics_file)

        response = client.get('/api/model-info')
        assert response.status_code == 200
        data = json.loads(response.data)
        metrics = data.get('metrics') or (data.get('data') and data['data'].get('metrics'))
        assert metrics is not None
        assert isinstance(metrics, dict)
        acc = metrics.get('ensemble', {}).get('accuracy') or metrics.get('accuracy')
        assert acc == 0.895

    def test_model_info_returns_null_metrics_when_file_missing(self, client, tmp_path):
        """/api/model-info returns metrics null (not invented numbers) when the file is missing."""
        missing_file = tmp_path / "nonexistent_metrics.json"
        client.application.config['METRICS_PATH'] = str(missing_file)

        response = client.get('/api/model-info')
        assert response.status_code == 200
        data = json.loads(response.data)
        metrics = data.get('metrics')
        data_metrics = data.get('data', {}).get('metrics') if data.get('data') else None
        assert metrics is None
        assert data_metrics is None


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


class TestLoginRoute:
    """Tests for the login route and form."""

    def test_get_login_returns_200_and_contains_username_and_password_fields(self, client):
        """GET /login returns 200 and contains a username and a password field."""
        response = client.get('/login')
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert 'name="username"' in html
        assert 'name="password"' in html

    def test_get_login_does_not_contain_admin123(self, client):
        """GET /login does not contain 'admin123'."""
        response = client.get('/login')
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert 'admin123' not in html


class TestPagePlaceholderData:
    """Tests ensuring placeholder dates are not rendered in HTML templates."""

    def test_history_page_does_not_contain_placeholder_dates(self, client):
        """GET /history does not contain placeholder dates 2026-11-08 to 2026-11-12."""
        response = client.get('/history')
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        placeholder_dates = [
            '2026-11-08', '2026-11-09', '2026-11-10', '2026-11-11', '2026-11-12'
        ]
        for dt in placeholder_dates:
            assert dt not in html

    def test_index_page_does_not_contain_placeholder_dates(self, client):
        """GET / does not contain placeholder dates 2026-11-08 to 2026-11-12."""
        response = client.get('/')
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        placeholder_dates = [
            '2026-11-08', '2026-11-09', '2026-11-10', '2026-11-11', '2026-11-12'
        ]
        for dt in placeholder_dates:
            assert dt not in html

