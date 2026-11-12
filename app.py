"""
Main Flask application for the RiskModel Financial Risk Assessment System.

Uses the application factory pattern. Run directly or import create_app().
"""

import datetime
import logging
import os
import sys
from typing import Optional

from flask import Flask, jsonify, render_template
from flask_cors import CORS

from api.routes import api_bp
from auth import auth_bp
from config import config_by_name
from database import create_tables


def _setup_logging(app: Flask) -> None:
    """
    Configure application-wide logging.

    Writes to both stdout and the configured log file.

    Args:
        app: The Flask application instance.
    """
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
    log_file = app.config.get('LOG_FILE', 'riskmodel.log')

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Stream handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    # Apply to the root logger so all modules pick it up
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def create_app(config_name: Optional[str] = None) -> Flask:
    """
    Application factory for the RiskModel Flask app.

    Args:
        config_name: One of 'development', 'production', 'testing'.
            Defaults to the FLASK_ENV environment variable or
            'development'.

    Returns:
        Configured Flask application instance.
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    app = Flask(__name__, static_folder='static')

    # Load configuration
    config_class = config_by_name.get(config_name, config_by_name['default'])
    app.config.from_object(config_class)

    # Logging
    _setup_logging(app)
    app.logger.info("Starting RiskModel app with '%s' configuration", config_name)

    # CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Database
    create_tables(app.config['DATABASE_URI'])

    # Register blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(auth_bp)

    # Context processor for Jinja templates
    @app.context_processor
    def inject_globals():
        return {'now': datetime.datetime.now}

    # ------------------------------------------------------------------
    # Frontend Web Routes
    # ------------------------------------------------------------------
    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['GET'])
    def predict():
        return render_template('predict.html')

    @app.route('/history', methods=['GET'])
    def history():
        return render_template('history.html')

    @app.route('/about', methods=['GET'])
    def about():
        return render_template('about.html')

    @app.route('/login', methods=['GET'])
    def login():
        return render_template('login.html')

    # ------------------------------------------------------------------
    # Health-check endpoint
    # ------------------------------------------------------------------
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Return a simple health-check response."""
        return jsonify({
            "status": "success",
            "data": {"service": "RiskModel API", "state": "healthy"},
            "message": "Service is running",
        }), 200

    # ------------------------------------------------------------------
    # Error handlers (JSON responses)
    # ------------------------------------------------------------------
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "status": "error",
            "data": None,
            "message": "Bad request",
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "status": "error",
            "data": None,
            "message": "Resource not found",
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error("Internal server error: %s", error)
        return jsonify({
            "status": "error",
            "data": None,
            "message": "Internal server error",
        }), 500

    return app


# ------------------------------------------------------------------
# Direct execution
# ------------------------------------------------------------------
if __name__ == '__main__':
    application = create_app()
    application.run(host='0.0.0.0', port=5000, debug=True)
