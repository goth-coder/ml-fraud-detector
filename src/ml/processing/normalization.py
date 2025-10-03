"""
Normalization - Funções para normalização de features
"""

import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Tuple, Dict


def create_scalers() -> Dict[str, object]:
    """
    Cria scalers para diferentes features
    
    Returns:
        Dicionário com scalers por feature
    """
    scalers = {
        'amount': RobustScaler(),  # Resistente a outliers (usa mediana/IQR)
        'time': StandardScaler()    # Para distribuição aproximadamente normal
    }
    return scalers


def fit_and_transform_features(
    df: pd.DataFrame,
    scalers: Dict[str, object] = None
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Ajusta scalers e transforma features
    
    Args:
        df: DataFrame com os dados
        scalers: Dicionário com scalers (se None, cria novos)
        
    Returns:
        Tupla (DataFrame normalizado, dicionário de scalers)
    """
    df_scaled = df.copy()
    
    if scalers is None:
        scalers = create_scalers()
    
    print(f"\n🔧 Normalizando features...")
    
    # Normalizar Amount com RobustScaler (resistente a outliers)
    if 'Amount' in df_scaled.columns:
        df_scaled['Amount'] = scalers['amount'].fit_transform(
            df_scaled[['Amount']]
        )
        print(f"   ✅ Amount normalizado (RobustScaler - usa mediana/IQR)")
    
    # Normalizar Time com StandardScaler
    if 'Time' in df_scaled.columns:
        df_scaled['Time'] = scalers['time'].fit_transform(
            df_scaled[['Time']]
        )
        print(f"   ✅ Time normalizado (StandardScaler)")
    
    # V1-V28 já são PCA, não normalizar
    print(f"   ℹ️  V1-V28 mantidos (já são PCA normalizados)")
    
    return df_scaled, scalers


def transform_features(df: pd.DataFrame, scalers: Dict[str, object]) -> pd.DataFrame:
    """
    Transforma features usando scalers já ajustados (para test set)
    
    Args:
        df: DataFrame com os dados
        scalers: Dicionário com scalers já ajustados
        
    Returns:
        DataFrame normalizado
    """
    df_scaled = df.copy()
    
    print(f"\n🔧 Aplicando normalização...")
    
    if 'Amount' in df_scaled.columns:
        df_scaled['Amount'] = scalers['amount'].transform(df_scaled[['Amount']])
        print(f"   ✅ Amount transformado")
    
    if 'Time' in df_scaled.columns:
        df_scaled['Time'] = scalers['time'].transform(df_scaled[['Time']])
        print(f"   ✅ Time transformado")
    
    return df_scaled


def save_scalers(scalers: Dict[str, object], output_path: Path) -> None:
    """
    Salva scalers em arquivo pickle
    
    Args:
        scalers: Dicionário com scalers
        output_path: Caminho para salvar o arquivo
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(scalers, f)
    
    print(f"💾 Scalers salvos em: {output_path}")


def load_scalers(input_path: Path) -> Dict[str, object]:
    """
    Carrega scalers de arquivo pickle
    
    Args:
        input_path: Caminho do arquivo pickle
        
    Returns:
        Dicionário com scalers
    """
    with open(input_path, 'rb') as f:
        scalers = pickle.load(f)
    
    print(f"📥 Scalers carregados de: {input_path}")
    return scalers


def get_scaler_statistics(scalers: Dict[str, object]) -> None:
    """
    Exibe estatísticas dos scalers
    
    Args:
        scalers: Dicionário com scalers
    """
    print(f"\n📊 Estatísticas dos Scalers:")
    
    # RobustScaler (Amount)
    if 'amount' in scalers:
        scaler = scalers['amount']
        print(f"\n   Amount (RobustScaler):")
        print(f"      - Mediana: {scaler.center_[0]:.2f}")
        print(f"      - IQR: {scaler.scale_[0]:.2f}")
    
    # StandardScaler (Time)
    if 'time' in scalers:
        scaler = scalers['time']
        print(f"\n   Time (StandardScaler):")
        print(f"      - Média: {scaler.mean_[0]:.2f}")
        print(f"      - Desvio padrão: {scaler.scale_[0]:.2f}")
