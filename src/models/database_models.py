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
