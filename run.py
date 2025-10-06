"""
Entry point para executar a aplicação Flask do Dashboard de Detecção de Fraudes.

Usage:
    python run.py
    
Environment Variables:
    FLASK_ENV: development|production|testing (default: development)
    FLASK_PORT: Porta do servidor (default: 5000)
    FLASK_HOST: Host do servidor (default: 127.0.0.1)
"""
import os
from src.api import create_app


config_name = os.environ.get('FLASK_ENV', 'development')
app = create_app(config_name)


if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    print(f"🚀 Starting Flask app in {config_name} mode")
    print(f"📍 Server running at http://{host}:{port}")
    print(f"🤖 Model: XGBoost v2.1.0")
    print(f"📊 Dashboard: http://{host}:{port}/")
    print(f"🔧 Health check: http://{host}:{port}/health")
    
    app.run(host=host, port=port, debug=(config_name == 'development'))
