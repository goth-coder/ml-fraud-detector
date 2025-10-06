"""
Configurações da aplicação Flask.
Define configurações para diferentes ambientes (dev/prod/test).
"""
import os
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent.parent


class Config:
    """Configurações base (comuns a todos os ambientes)."""
    
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'postgresql://fraud_user:fraud_pass_dev@localhost:5432/fraud_detection'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    
    # ML Model Paths
    MODEL_PATH = BASE_DIR / 'models' / 'xgboost_v2.1.0.pkl'
    SCALER_PATH = BASE_DIR / 'models' / 'scalers.pkl'
    
    # API Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    JSON_SORT_KEYS = False
    
    # Pagination
    DEFAULT_PAGE_SIZE = 50
    MAX_PAGE_SIZE = 1000


class DevelopmentConfig(Config):
    """Configurações para ambiente de desenvolvimento."""
    
    DEBUG = True
    TESTING = False
    SQLALCHEMY_ECHO = True


class ProductionConfig(Config):
    """Configurações para ambiente de produção."""
    
    DEBUG = False
    TESTING = False
    
    @property
    def SECRET_KEY(self):
        key = os.environ.get('SECRET_KEY')
        if not key:
            raise ValueError("SECRET_KEY environment variable must be set in production")
        return key


class TestingConfig(Config):
    """Configurações para testes."""
    
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
