"""
Database Service - Camada de acesso aos dados do webapp.

Gerencia persistência de predições e transações simuladas.
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from src.models.database_models import ClassificationResult, Transaction
from src.services.database.connection import get_engine


class DatabaseService:
    """
    Serviço de persistência para o webapp.
    
    Gerencia operações CRUD em classification_results e simulated_transactions.
    """
    
    def __init__(self):
        """Inicializa serviço com engine do PostgreSQL."""
        self._engine = get_engine()
    
    def save_classification(
        self,
        is_fraud: bool,
        fraud_probability: float,
        transaction_features: Dict[str, float],
        model_version: str = "v2.1.0",
        source: str = "webapp"
    ) -> int:
        """
        Salva resultado de classificação no banco.
        
        Args:
            is_fraud: Se foi classificado como fraude
            fraud_probability: Probabilidade de fraude (0.0 a 1.0)
            transaction_features: Dicionário com 33 features
            model_version: Versão do modelo usado
            source: Origem da predição (webapp, api, batch)
            
        Returns:
            ID da classificação salva
            
        Example:
            >>> service = DatabaseService()
            >>> features = {'V1': 0.5, 'V2': -1.2, ...}
            >>> id = service.save_classification(
            ...     is_fraud=True,
            ...     fraud_probability=0.9876,
            ...     transaction_features=features
            ... )
        """
        from sqlalchemy.orm import sessionmaker
        
        Session = sessionmaker(bind=self._engine)
        session = Session()
        
        try:
            result = ClassificationResult(
                model_version=model_version,
                is_fraud=is_fraud,
                fraud_probability=fraud_probability,
                transaction_features=transaction_features,
                source=source
            )
            
            session.add(result)
            session.commit()
            session.refresh(result)
            
            return result.id
            
        except Exception as e:
            session.rollback()
            raise Exception(f"Erro ao salvar classificação: {e}")
        finally:
            session.close()
    
    def save_transaction(
        self,
        transaction_type: str,
        features: Dict[str, float],
        classification_id: Optional[int] = None
    ) -> int:
        """
        Salva transação simulada no banco.
        
        Args:
            transaction_type: 'legitimate' ou 'fraud'
            features: Dicionário com 33 features
            classification_id: ID da classificação associada (opcional)
            
        Returns:
            ID da transação salva
        """
        from sqlalchemy.orm import sessionmaker
        
        Session = sessionmaker(bind=self._engine)
        session = Session()
        
        try:
            transaction = Transaction(
                transaction_type=transaction_type,
                features=features,
                classification_id=classification_id
            )
            
            session.add(transaction)
            session.commit()
            session.refresh(transaction)
            
            return transaction.id
            
        except Exception as e:
            session.rollback()
            raise Exception(f"Erro ao salvar transação: {e}")
        finally:
            session.close()
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """
        Retorna histórico de classificações recentes.
        
        Args:
            limit: Número máximo de registros (padrão: 50)
            
        Returns:
            Lista de dicionários com classificações
            
        Example:
            >>> service = DatabaseService()
            >>> history = service.get_history(limit=10)
            >>> print(history[0])
            {
                'id': 1,
                'predicted_at': '2025-10-06 10:30:00',
                'is_fraud': True,
                'fraud_probability': 0.9876,
                'model_version': 'v2.1.0'
            }
        """
        from sqlalchemy.orm import sessionmaker
        
        Session = sessionmaker(bind=self._engine)
        session = Session()
        
        try:
            results = session.query(ClassificationResult)\
                .order_by(desc(ClassificationResult.predicted_at))\
                .limit(limit)\
                .all()
            
            return [
                {
                    'id': r.id,
                    'predicted_at': r.predicted_at.isoformat(),
                    'is_fraud': r.is_fraud,
                    'fraud_probability': round(r.fraud_probability, 4),
                    'model_version': r.model_version,
                    'source': r.source
                }
                for r in results
            ]
            
        finally:
            session.close()
    
    def get_stats(self, hours: int = 24) -> Dict:
        """
        Retorna estatísticas de classificações.
        
        Args:
            hours: Período em horas (padrão: últimas 24h)
            
        Returns:
            Dicionário com estatísticas:
            - total: Total de classificações
            - fraud_count: Quantidade de fraudes detectadas
            - fraud_percentage: Percentual de fraudes
            - avg_probability: Probabilidade média de fraude
            - by_hour: Contagem por hora
            
        Example:
            >>> service = DatabaseService()
            >>> stats = service.get_stats(hours=24)
            >>> print(f"Fraudes detectadas: {stats['fraud_count']}")
        """
        from sqlalchemy.orm import sessionmaker
        
        Session = sessionmaker(bind=self._engine)
        session = Session()
        
        try:
            # Período de análise
            since = datetime.now() - timedelta(hours=hours)
            
            # Query base
            query = session.query(ClassificationResult)\
                .filter(ClassificationResult.predicted_at >= since)
            
            # Contagens
            total = query.count()
            fraud_count = query.filter(ClassificationResult.is_fraud == True).count()
            
            # Estatísticas agregadas
            stats_query = session.query(
                func.avg(ClassificationResult.fraud_probability).label('avg_prob'),
                func.max(ClassificationResult.fraud_probability).label('max_prob'),
                func.min(ClassificationResult.fraud_probability).label('min_prob')
            ).filter(ClassificationResult.predicted_at >= since).first()
            
            # Contagem por hora
            hourly = session.query(
                func.date_trunc('hour', ClassificationResult.predicted_at).label('hour'),
                func.count().label('count')
            ).filter(
                ClassificationResult.predicted_at >= since
            ).group_by('hour').order_by('hour').all()
            
            return {
                'total': total,
                'fraud_count': fraud_count,
                'fraud_percentage': round((fraud_count / total * 100) if total > 0 else 0, 2),
                'avg_probability': round(stats_query.avg_prob or 0, 4),
                'max_probability': round(stats_query.max_prob or 0, 4),
                'min_probability': round(stats_query.min_prob or 0, 4),
                'by_hour': [
                    {
                        'hour': h.hour.isoformat(),
                        'count': h.count
                    }
                    for h in hourly
                ],
                'period_hours': hours
            }
            
        finally:
            session.close()
    
    def clear_old_data(self, days: int = 30) -> int:
        """
        Remove dados antigos do banco.
        
        Args:
            days: Manter apenas dados dos últimos N dias
            
        Returns:
            Número de registros deletados
        """
        from sqlalchemy.orm import sessionmaker
        
        Session = sessionmaker(bind=self._engine)
        session = Session()
        
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            deleted = session.query(ClassificationResult)\
                .filter(ClassificationResult.predicted_at < cutoff)\
                .delete()
            
            session.commit()
            
            return deleted
            
        except Exception as e:
            session.rollback()
            raise Exception(f"Erro ao limpar dados: {e}")
        finally:
            session.close()


# Instância global
database_service = DatabaseService()
