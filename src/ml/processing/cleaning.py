"""
Cleaning - Funções de limpeza e análise de outliers
"""

import pandas as pd
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_iqr_bounds(series: pd.Series) -> Tuple[float, float, float, float, float]:
    """
    Calcula bounds usando IQR (Tukey's Method)
    
    Args:
        series: Série pandas com valores numéricos
        
    Returns:
        Tupla (Q1, Q3, IQR, lower_bound, upper_bound)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return Q1, Q3, IQR, lower_bound, upper_bound


def identify_outliers(df: pd.DataFrame, column: str = 'Amount') -> pd.DataFrame:
    """
    Identifica outliers usando IQR (apenas análise, não remove)
    
    Args:
        df: DataFrame com os dados
        column: Nome da coluna para análise de outliers
        
    Returns:
        DataFrame com análise de outliers
    """
    print(f"\n🔍 Identificando outliers de {column} usando IQR (Tukey's Method)...")
    
    # Calcular IQR bounds
    Q1, Q3, IQR, lower_bound, upper_bound = calculate_iqr_bounds(df[column])
    
    print(f"\n📊 Estatísticas IQR:")
    print(f"   - Q1 (25%): ${Q1:.2f}")
    print(f"   - Q3 (75%): ${Q3:.2f}")
    print(f"   - IQR: ${IQR:.2f}")
    print(f"   - Lower Bound (Q1 - 1.5*IQR): ${lower_bound:.2f}")
    print(f"   - Upper Bound (Q3 + 1.5*IQR): ${upper_bound:.2f}")
    
    # Identificar outliers
    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outliers = df[outliers_mask]
    
    # Separar outliers baixos e altos
    outliers_low = df[df[column] < lower_bound]
    outliers_high = df[df[column] > upper_bound]
    
    print(f"\n📊 Análise de Outliers:")
    print(f"   - Total outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.3f}%)")
    print(f"   - Outliers baixos (< ${lower_bound:.2f}): {len(outliers_low):,}")
    print(f"   - Outliers altos (> ${upper_bound:.2f}): {len(outliers_high):,}")
    
    return outliers


def analyze_fraud_distribution_in_outliers(
    df: pd.DataFrame,
    outliers: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analisa distribuição de fraudes em outliers
    
    Args:
        df: DataFrame completo
        outliers: DataFrame apenas com outliers
        
    Returns:
        Dictionário com estatísticas de fraudes em outliers
    """
    total_frauds = (df['Class'] == 1).sum()
    frauds_in_outliers = (outliers['Class'] == 1).sum()
    fraud_pct_in_outliers = frauds_in_outliers / total_frauds * 100 if total_frauds > 0 else 0
    
    print(f"\n⚠️  Impacto em Fraudes:")
    print(f"   - Total de fraudes no dataset: {total_frauds:,}")
    print(f"   - Fraudes em outliers: {frauds_in_outliers:,}")
    print(f"   - Percentual de fraudes em outliers: {fraud_pct_in_outliers:.2f}%")
    
    if fraud_pct_in_outliers > 15:
        print(f"\n❌ DECISÃO: NÃO REMOVER OUTLIERS")
        print(f"   - Perderíamos {fraud_pct_in_outliers:.1f}% das fraudes ({frauds_in_outliers} de {total_frauds})")
        print(f"   - Solução: RobustScaler no Step 04")
    
    return {
        'total_frauds': int(total_frauds),
        'frauds_in_outliers': int(frauds_in_outliers),
        'fraud_pct_in_outliers': round(float(fraud_pct_in_outliers), 2)
    }


def handle_missing_values(df: pd.DataFrame, strategy: dict = None) -> pd.DataFrame:
    """
    Trata missing values de acordo com estratégia definida
    
    Args:
        df: DataFrame com os dados
        strategy: Dicionário com estratégia por coluna
                 {'column': 'median', 'column2': 'drop', etc}
                 
    Returns:
        DataFrame com missing values tratados
    """
    df_clean = df.copy()
    
    if strategy is None:
        # Estratégia padrão
        strategy = {
            'Class': 'drop',  # Não pode imputar target
            'Time': 'median',
            'Amount': 'median_by_class'
        }
    
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if col in strategy:
                strat = strategy[col]
                
                if strat == 'drop':
                    df_clean = df_clean[df_clean[col].notnull()]
                    print(f"   - {col}: Linhas com missing removidas")
                    
                elif strat == 'median':
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    print(f"   - {col}: Imputado com mediana ({median_val:.2f})")
                    
                elif strat == 'median_by_class':
                    for class_val in df_clean['Class'].unique():
                        mask = df_clean['Class'] == class_val
                        median_val = df_clean.loc[mask, col].median()
                        df_clean.loc[mask & df_clean[col].isnull(), col] = median_val
                    print(f"   - {col}: Imputado com mediana por classe")
    
    return df_clean
