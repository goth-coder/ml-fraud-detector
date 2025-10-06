"""
API Routes - Endpoints REST para o dashboard de detecção de fraudes.
"""
from flask import Blueprint, jsonify, request
import logging

from src.services.ml.transaction_generator import transaction_generator
from src.services.ml.model_service import model_service
from src.services.database.database_service import database_service


logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)


@api_bp.route('/simulate', methods=['POST'])
def simulate_transaction():
    """
    POST /api/simulate - Simula transação (legítima ou fraudulenta).
    
    Body:
        {
            "transaction_type": "legitimate" | "fraud"
        }
        
    Response:
        {
            "success": true,
            "transaction_id": 123,
            "classification_id": 456,
            "is_fraud": false,
            "fraud_probability": 0.0023,
            "confidence": "Baixa",
            "predicted_at": "2025-10-06T10:30:00",
            "transaction_type": "legitimate"
        }
    """
    try:
        data = request.get_json(silent=True)
        
        if not data or 'transaction_type' not in data:
            return jsonify({
                'success': False,
                'error': 'Campo "transaction_type" é obrigatório (body JSON: {"transaction_type": "legitimate" | "fraud"})'
            }), 400
        
        transaction_type = data['transaction_type']
        
        if transaction_type not in ['legitimate', 'fraud']:
            return jsonify({
                'success': False,
                'error': 'transaction_type deve ser "legitimate" ou "fraud"'
            }), 400
        
        # 1. Gerar transação real do PostgreSQL
        transaction_features = transaction_generator.generate(transaction_type)
        
        # 2. Classificar com modelo
        prediction = model_service.predict(transaction_features)
        
        # 3. Salvar classificação no banco
        classification_id = database_service.save_classification(
            is_fraud=prediction['is_fraud'],
            fraud_probability=prediction['probability'],
            transaction_features=transaction_features,
            model_version=prediction['model_version'],
            source='webapp'
        )
        
        # 4. Salvar transação
        transaction_id = database_service.save_transaction(
            transaction_type=transaction_type,
            features=transaction_features,
            classification_id=classification_id
        )
        
        # 5. Buscar timestamp da classificação
        history = database_service.get_history(limit=1)
        predicted_at = history[0]['predicted_at'] if history else None
        
        return jsonify({
            'success': True,
            'transaction_id': transaction_id,
            'classification_id': classification_id,
            'is_fraud': prediction['is_fraud'],
            'fraud_probability': round(prediction['probability'], 4),
            'fraud_probability_percent': prediction['probability_percent'],
            'confidence': prediction['confidence'],
            'model_version': prediction['model_version'],
            'predicted_at': predicted_at,
            'transaction_type': transaction_type
        }), 200
        
    except Exception as e:
        logger.error(f"Erro em /api/simulate: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    GET /api/stats - Retorna estatísticas agregadas.
    
    Query Params:
        ?hours=24 (opcional, padrão: 24)
        
    Response:
        {
            "success": true,
            "stats": {
                "total": 150,
                "fraud_count": 30,
                "fraud_percentage": 20.0,
                "avg_probability": 0.4523,
                "max_probability": 0.9987,
                "min_probability": 0.0001,
                "by_hour": [...],
                "period_hours": 24
            }
        }
    """
    try:
        hours = request.args.get('hours', default=24, type=int)
        
        if hours < 1 or hours > 168:  # máximo 7 dias
            return jsonify({
                'success': False,
                'error': 'hours deve estar entre 1 e 168'
            }), 400
        
        stats = database_service.get_stats(hours=hours)
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Erro em /api/stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/history', methods=['GET'])
def get_history():
    """
    GET /api/history - Retorna histórico de classificações.
    
    Query Params:
        ?limit=50 (opcional, padrão: 50)
        
    Response:
        {
            "success": true,
            "count": 50,
            "history": [
                {
                    "id": 456,
                    "predicted_at": "2025-10-06T10:30:00",
                    "is_fraud": true,
                    "fraud_probability": 0.9876,
                    "model_version": "v2.1.0",
                    "source": "webapp"
                },
                ...
            ]
        }
    """
    try:
        limit = request.args.get('limit', default=50, type=int)
        
        if limit < 1 or limit > 1000:
            return jsonify({
                'success': False,
                'error': 'limit deve estar entre 1 e 1000'
            }), 400
        
        history = database_service.get_history(limit=limit)
        
        return jsonify({
            'success': True,
            'count': len(history),
            'history': history
        }), 200
        
    except Exception as e:
        logger.error(f"Erro em /api/history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
