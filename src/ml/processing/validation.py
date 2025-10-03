"""
Validation - Funções para validar schema e integridade dos dados
"""

import pandas as pd


def validate_csv_schema(df: pd.DataFrame, expected_columns: int = 31) -> None:
    """
    Valida schema do DataFrame de fraudes
    
    Args:
        df: DataFrame a ser validado
        expected_columns: Número esperado de colunas
        
    Raises:
        AssertionError: Se o schema não for válido
    """
    expected_cols = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
    
    assert len(df.columns) == expected_columns, (
        f"Esperado {expected_columns} colunas, encontrado {len(df.columns)}"
    )
    assert set(df.columns) == set(expected_cols), "Colunas não correspondem ao esperado"
    assert df['Class'].isin([0, 1]).all(), "Coluna Class deve conter apenas 0 ou 1"
    
    print(f"✅ Schema validado: {len(df.columns)} colunas, {len(df)} linhas")


def validate_class_distribution(df: pd.DataFrame, expected_fraud_pct: float = 0.172) -> None:
    """
    Valida distribuição de classes (fraudes vs legítimas)
    
    Args:
        df: DataFrame a ser validado
        expected_fraud_pct: Percentual esperado de fraudes
    """
    fraud_pct = (df['Class'] == 1).sum() / len(df) * 100
    print(f"📊 Distribuição de classes:")
    print(f"   - Legítimas: {(df['Class'] == 0).sum():,} ({100-fraud_pct:.3f}%)")
    print(f"   - Fraudes: {(df['Class'] == 1).sum():,} ({fraud_pct:.3f}%)")
    
    # Validação flexível (±0.01%)
    assert abs(fraud_pct - expected_fraud_pct) < 0.01, (
        f"Distribuição anormal: esperado ~{expected_fraud_pct}%, encontrado {fraud_pct:.3f}%"
    )


def validate_no_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida ausência de valores faltantes e retorna relatório
    
    Args:
        df: DataFrame a ser validado
        
    Returns:
        DataFrame com relatório de missing values
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Percentage': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    
    if len(missing_df) == 0:
        print("✅ Nenhum missing value encontrado!")
    else:
        print(f"⚠️  Missing values encontrados:")
        print(missing_df.to_string(index=False))
    
    return missing_df
