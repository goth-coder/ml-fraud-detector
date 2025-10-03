"""
Splitters - Funções para dividir dados em train/test
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def stratified_train_test_split(
    df: pd.DataFrame,
    target_column: str = 'Class',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide dados em train/test mantendo proporção de classes
    
    Args:
        df: DataFrame completo
        target_column: Nome da coluna target
        test_size: Proporção do test set (0.2 = 20%)
        random_state: Seed para reprodutibilidade
        
    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    print(f"\n✂️  Dividindo dados em train/test (estratificado)...")
    
    # Separar features e target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Manter proporção de classes
    )
    
    # Validar proporções
    train_fraud_pct = (y_train == 1).sum() / len(y_train) * 100
    test_fraud_pct = (y_test == 1).sum() / len(y_test) * 100
    
    print(f"\n📊 Split Statistics:")
    print(f"   Train Set:")
    print(f"      - Total: {len(X_train):,} ({(1-test_size)*100:.0f}%)")
    print(f"      - Legítimas: {(y_train == 0).sum():,}")
    print(f"      - Fraudes: {(y_train == 1).sum():,} ({train_fraud_pct:.3f}%)")
    print(f"   Test Set:")
    print(f"      - Total: {len(X_test):,} ({test_size*100:.0f}%)")
    print(f"      - Legítimas: {(y_test == 0).sum():,}")
    print(f"      - Fraudes: {(y_test == 1).sum():,} ({test_fraud_pct:.3f}%)")
    
    return X_train, X_test, y_train, y_test


def validate_split_integrity(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> None:
    """
    Valida integridade do split (sem data leakage)
    
    Args:
        X_train: Features de treino
        X_test: Features de teste
        y_train: Target de treino
        y_test: Target de teste
    """
    print(f"\n🔍 Validando integridade do split...")
    
    # Verificar se não há overlap de índices
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    overlap = train_indices.intersection(test_indices)
    
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} índices duplicados entre train e test"
    print(f"   ✅ Sem data leakage (0 índices duplicados)")
    
    # Verificar proporções de classes
    train_fraud_pct = (y_train == 1).sum() / len(y_train) * 100
    test_fraud_pct = (y_test == 1).sum() / len(y_test) * 100
    
    # Diferença deve ser < 0.01%
    diff = abs(train_fraud_pct - test_fraud_pct)
    assert diff < 0.01, f"Proporções de classes muito diferentes: train={train_fraud_pct:.3f}%, test={test_fraud_pct:.3f}%"
    print(f"   ✅ Proporções balanceadas (diff={diff:.4f}%)")
    
    # Verificar tamanhos
    total = len(X_train) + len(X_test)
    print(f"   ✅ Total de amostras: {total:,}")


def save_split_to_postgresql(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    engine
) -> None:
    """
    Salva split no PostgreSQL usando COPY otimizado
    
    Args:
        X_train: Features de treino
        X_test: Features de teste
        y_train: Target de treino
        y_test: Target de teste
        engine: SQLAlchemy engine
    """
    from src.ml.processing.loader import save_to_postgresql
    
    print(f"\n💾 Salvando split no PostgreSQL (COPY otimizado)...")
    
    # Combinar features e target
    train_df = X_train.copy()
    train_df['Class'] = y_train.values
    
    test_df = X_test.copy()
    test_df['Class'] = y_test.values
    
    # Salvar usando COPY otimizado (padrão)
    save_to_postgresql(train_df, engine, 'train_data', if_exists='replace')
    save_to_postgresql(test_df, engine, 'test_data', if_exists='replace')
    
    print(f"   ✅ Train e Test salvos no PostgreSQL")
