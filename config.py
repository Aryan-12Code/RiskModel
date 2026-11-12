"""
Configuration module for the RiskModel Flask application.

Provides base and environment-specific configuration classes for
development, testing, and production environments.
"""

import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration class with default settings."""

    SECRET_KEY = os.environ.get('SECRET_KEY', 'riskmodel-dev-secret-key-change-in-prod')
    DEBUG = False
    TESTING = False

    # Database
    DATABASE_URI = os.environ.get(
        'DATABASE_URI',
        os.path.join(BASE_DIR, 'riskmodel.db')
    )

    # Model paths
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ensemble_model.pkl')
    PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor.pkl')
    SHAP_DIR = os.path.join(BASE_DIR, 'static', 'shap')

    # MLflow
    MLFLOW_TRACKING_URI = os.environ.get(
        'MLFLOW_TRACKING_URI',
        os.path.join(BASE_DIR, 'mlruns')
    )

    # JWT settings
    JWT_EXPIRATION_HOURS = 24

    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = os.path.join(BASE_DIR, 'riskmodel.log')


class DevelopmentConfig(Config):
    """Development environment configuration."""

    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production environment configuration."""

    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY', None)
    LOG_LEVEL = 'WARNING'

    def __init__(self):
        if self.SECRET_KEY is None:
            raise ValueError("SECRET_KEY environment variable must be set in production")


class TestingConfig(Config):
    """Testing environment configuration."""

    TESTING = True
    DEBUG = True
    DATABASE_URI = os.path.join(BASE_DIR, 'test_riskmodel.db')
    LOG_LEVEL = 'DEBUG'


# Map of environment names to config classes
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig,
}
