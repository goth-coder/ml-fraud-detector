"""
Flask Application Factory
Cria e configura a aplicação Flask para o dashboard de detecção de fraudes.
"""
import os
from flask import Flask, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


def create_app(config_name='development'):
    """
    Factory pattern para criar aplicação Flask.
    
    Args:
        config_name: Nome da configuração ('development', 'production', 'testing')
        
    Returns:
        Flask app configurada
    """
    # Define paths para templates e static
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    services_dir = os.path.dirname(backend_dir)
    frontend_dir = os.path.join(services_dir, 'frontend')
    
    app = Flask(
        __name__,
        template_folder=os.path.join(frontend_dir, 'templates'),
        static_folder=os.path.join(frontend_dir, 'static')
    )
    
    from src.services.backend.config import config
    app.config.from_object(config[config_name])
    
    # CORS: Permite apenas frontend local acessar a API
    CORS(app, origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ])
    
    db.init_app(app)
    
    from src.services.backend.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Rota principal: Serve o dashboard HTML
    @app.route('/')
    def index():
        """Página principal do dashboard."""
        return render_template('index.html')
    
    @app.route('/health')
    def health_check():
        """Health check endpoint para monitoramento."""
        return {'status': 'healthy', 'model_version': 'v2.1.0'}, 200
    
    return app
