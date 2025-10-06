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
import subprocess
import signal
import sys
from src.services.backend import create_app


def kill_process_on_port(port):
    """
    Mata qualquer processo rodando na porta especificada (exceto este processo).
    
    Args:
        port: Número da porta
    """
    try:
        current_pid = os.getpid()
        
        # Busca PID do processo na porta
        result = subprocess.run(
            f"lsof -ti:{port}",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid_str in pids:
                try:
                    pid = int(pid_str)
                    # Não mata o próprio processo (evita suicídio no reload do Flask)
                    if pid != current_pid:
                        os.kill(pid, signal.SIGKILL)
                        print(f"🔪 Processo {pid} na porta {port} foi encerrado")
                except (ProcessLookupError, ValueError):
                    pass
    except Exception as e:
        print(f"⚠️  Erro ao tentar liberar porta {port}: {e}")


# Só executa o kill se for o processo principal (não no reload do Flask)
if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
    port = int(os.environ.get('FLASK_PORT', 5000))
    kill_process_on_port(port)

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
