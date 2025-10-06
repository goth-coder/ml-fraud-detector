"""
Database Models - SQLAlchemy models para PostgreSQL
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class RawTransaction(Base):
    """Tabela de transações raw (do CSV)"""
    __tablename__ = 'raw_transactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    Time = Column(Float, nullable=False)
    Amount = Column(Float, nullable=False)
    Class = Column(Integer, nullable=False)
    
    # Features V1-V28 (PCA)
    V1 = Column(Float)
    V2 = Column(Float)
    V3 = Column(Float)
    V4 = Column(Float)
    V5 = Column(Float)
    V6 = Column(Float)
    V7 = Column(Float)
    V8 = Column(Float)
    V9 = Column(Float)
    V10 = Column(Float)
    V11 = Column(Float)
    V12 = Column(Float)
    V13 = Column(Float)
    V14 = Column(Float)
    V15 = Column(Float)
    V16 = Column(Float)
    V17 = Column(Float)
    V18 = Column(Float)
    V19 = Column(Float)
    V20 = Column(Float)
    V21 = Column(Float)
    V22 = Column(Float)
    V23 = Column(Float)
    V24 = Column(Float)
    V25 = Column(Float)
    V26 = Column(Float)
    V27 = Column(Float)
    V28 = Column(Float)


class ModelMetadata(Base):
    """Tabela de metadados de modelos treinados"""
    __tablename__ = 'trained_models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # decision_tree, svm_rbf, xgboost
    version = Column(String(20))
    file_path = Column(String(200))
    trained_at = Column(DateTime, server_default=func.now())
    
    # Hiperparâmetros (JSON)
    hyperparameters = Column(JSON)
    
    # Métricas
    pr_auc = Column(Float)
    roc_auc = Column(Float)
    best_threshold = Column(Float)
    recall = Column(Float)
    precision = Column(Float)
    f1_score = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=False)


class PredictionResult(Base):
    """Tabela de resultados de predições"""
    __tablename__ = 'prediction_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(Integer)
    model_name = Column(String(50))
    predicted_class = Column(Integer)  # 0 ou 1
    predicted_probability = Column(Float)
    actual_class = Column(Integer, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    predicted_at = Column(DateTime, server_default=func.now())


class MetricsHistory(Base):
    """Tabela de histórico de métricas"""
    __tablename__ = 'metrics_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50))
    metric_name = Column(String(50))  # precision, recall, f1, pr_auc, etc
    metric_value = Column(Float)
    threshold = Column(Float, nullable=True)
    recorded_at = Column(DateTime, server_default=func.now())


class ClassificationResult(Base):
    """
    Tabela de resultados de classificação do webapp.
    
    Armazena cada predição feita via dashboard para histórico e estatísticas.
    
    IMPORTANTE: 
    - is_fraud = GROUND TRUTH (label real da transação, não predição)
    - fraud_probability = predição do modelo (0.0 a 1.0)
    """
    __tablename__ = 'classification_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Metadados da predição
    model_version = Column(String(20), nullable=False)  # v2.1.0
    predicted_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Ground Truth e Predição
    is_fraud = Column(Boolean, nullable=False)  # ⚠️ GROUND TRUTH (label real)
    fraud_probability = Column(Float, nullable=False)  # Predição do modelo (0-1)
    
    # Performance
    latency_ms = Column(Float, nullable=True)  # Tempo de inferência em ms
    
    # Features da transação (JSON para flexibilidade)
    transaction_features = Column(JSON, nullable=False)
    
    # Metadata
    source = Column(String(20), default='webapp')  # webapp, api, batch
    
    def __repr__(self):
        return f"<ClassificationResult(id={self.id}, is_fraud={self.is_fraud}, probability={self.fraud_probability:.4f})>"


class Transaction(Base):
    """
    Tabela de transações simuladas pelo webapp.
    
    Armazena transações geradas pelo simulador para análise posterior.
    """
    __tablename__ = 'simulated_transactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Metadados
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    transaction_type = Column(String(20), nullable=False)  # legitimate, fraud
    
    # Features completas (33 features)
    features = Column(JSON, nullable=False)
    
    # Link com resultado de classificação (opcional)
    classification_id = Column(Integer, nullable=True)
    
    def __repr__(self):
        return f"<Transaction(id={self.id}, type={self.transaction_type})>"
