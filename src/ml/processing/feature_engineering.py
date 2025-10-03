"""
Feature Engineering - Criação de novas features
"""

import pandas as pd
import numpy as np
from typing import List


def create_time_period_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features baseadas em períodos de tempo
    
    Args:
        df: DataFrame com coluna 'Time'
        
    Returns:
        DataFrame com novas features temporais
        
    Features criadas:
        - Time_Hours: Hora do dia (0-47h no dataset de 2 dias)
        - Time_Period_of_Day: Período do dia (0=Madrugada, 1=Manhã, 2=Tarde, 3=Noite) - NUMÉRICO
    """
    df_eng = df.copy()
    
    # Converter Time (segundos) para horas
    df_eng['Time_Hours'] = df_eng['Time'] / 3600
    
    # Criar binning temporal NUMÉRICO (0, 1, 2, 3)
    # 0-6h: 0 (Madrugada), 6-12h: 1 (Manhã), 12-18h: 2 (Tarde), 18-24h: 3 (Noite)
    df_eng['Time_Period_of_Day'] = pd.cut(
        df_eng['Time_Hours'] % 24,
        bins=[0, 6, 12, 18, 24],
        labels=[0, 1, 2, 3],  # NUMÉRICO para ML
        include_lowest=True
    ).astype(int)  # Converter para int
    
    print(f"   ✅ Time_Period_of_Day criado (0=Madrugada, 1=Manhã, 2=Tarde, 3=Noite)")
    
    return df_eng


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features baseadas em Amount
    
    Args:
        df: DataFrame com coluna 'Amount'
        
    Returns:
        DataFrame com novas features de Amount
        
    Features criadas:
        - Amount_Log: Log(Amount + 1) para normalizar distribuição
        - Amount_Bin: Categorização (0=Very Low, 1=Low, 2=Medium, 3=High) - NUMÉRICO
    """
    df_eng = df.copy()
    
    # Log transformation (para normalizar distribuição)
    df_eng['Amount_Log'] = np.log1p(df_eng['Amount'])  # log1p evita log(0)
    
    # Categorização de Amount NUMÉRICA (0, 1, 2, 3)
    # <$10: 0 (Very Low), $10-100: 1 (Low), $100-500: 2 (Medium), >$500: 3 (High)
    df_eng['Amount_Bin'] = pd.cut(
        df_eng['Amount'],
        bins=[-np.inf, 10, 100, 500, np.inf],
        labels=[0, 1, 2, 3],  # NUMÉRICO para ML
        include_lowest=True
    ).astype(int)  # Converter para int
    
    print(f"   ✅ Amount_Log e Amount_Bin criados (numéricos)")
    
    return df_eng


def create_v_feature_interactions(df: pd.DataFrame, top_features: List[str] = None) -> pd.DataFrame:
    """
    Cria interações entre features V mais importantes
    
    Args:
        df: DataFrame com features V1-V28
        top_features: Lista de features para criar interações
                     Se None, usa ['V17', 'V14', 'V12', 'V10']
        
    Returns:
        DataFrame com features de interação
    """
    df_eng = df.copy()
    
    if top_features is None:
        top_features = ['V17', 'V14', 'V12', 'V10']
    
    # Criar interações multiplicativas
    interactions = []
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            interaction_name = f"{feat1}_{feat2}_Interaction"
            df_eng[interaction_name] = df_eng[feat1] * df_eng[feat2]
            interactions.append(interaction_name)
    
    print(f"   ✅ {len(interactions)} interações criadas: {', '.join(interactions[:3])}...")
    
    return df_eng


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features estatísticas das features V
    
    Args:
        df: DataFrame com features V1-V28
        
    Returns:
        DataFrame com features estatísticas
    """
    df_eng = df.copy()
    
    v_cols = [col for col in df_eng.columns if col.startswith('V')]
    
    # Estatísticas descritivas
    df_eng['V_Mean'] = df_eng[v_cols].mean(axis=1)
    df_eng['V_Std'] = df_eng[v_cols].std(axis=1)
    df_eng['V_Min'] = df_eng[v_cols].min(axis=1)
    df_eng['V_Max'] = df_eng[v_cols].max(axis=1)
    df_eng['V_Range'] = df_eng['V_Max'] - df_eng['V_Min']
    
    print(f"   ✅ Features estatísticas criadas (Mean, Std, Min, Max, Range)")
    
    return df_eng


def engineer_all_features(df: pd.DataFrame, include_interactions: bool = True) -> pd.DataFrame:
    """
    Aplica todas as transformações de feature engineering
    
    Args:
        df: DataFrame original
        include_interactions: Se True, inclui interações entre features
        
    Returns:
        DataFrame com todas as novas features
    """
    print(f"\n🔨 Feature Engineering...")
    
    df_eng = df.copy()
    
    # Features temporais
    df_eng = create_time_period_features(df_eng)
    
    # Features de Amount
    df_eng = create_amount_features(df_eng)
    
    # Features estatísticas
    df_eng = create_statistical_features(df_eng)
    
    # Interações (opcional, pode aumentar muito dimensionalidade)
    if include_interactions:
        df_eng = create_v_feature_interactions(df_eng)
    
    print(f"\n   📊 Features criadas:")
    print(f"      - Original: {len(df.columns)} colunas")
    print(f"      - Após engineering: {len(df_eng.columns)} colunas")
    print(f"      - Novas features: {len(df_eng.columns) - len(df.columns)}")
    
    return df_eng
