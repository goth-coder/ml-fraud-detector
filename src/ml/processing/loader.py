"""
Loader - Funções para carregar dados de diferentes fontes
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import text
from typing import Optional
import tempfile
import os
from io import StringIO


def load_csv_to_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    Carrega CSV em DataFrame pandas
    
    Args:
        csv_path: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com os dados carregados
    """
    print(f"📁 Carregando CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"📊 Dataset carregado:")
    print(f"   - Linhas: {len(df):,}")
    print(f"   - Colunas: {len(df.columns)}")
    print(f"   - Memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def load_from_postgresql(engine, table_name: str, query: Optional[str] = None) -> pd.DataFrame:
    """
    Carrega dados do PostgreSQL
    
    Args:
        engine: SQLAlchemy engine
        table_name: Nome da tabela
        query: Query SQL customizada (opcional)
        
    Returns:
        DataFrame com os dados
    """
    print(f"📥 Carregando {table_name} do PostgreSQL...")
    
    if query is None:
        query = f"SELECT * FROM {table_name}"
    
    df = pd.read_sql(query, engine)
    print(f"✅ Dados carregados: {len(df):,} linhas")
    
    return df
    
    return df


def save_to_postgresql(
    df: pd.DataFrame, 
    engine, 
    table_name: str, 
    if_exists: str = 'replace',
    use_copy: bool = True
) -> int:
    """
    Salva DataFrame no PostgreSQL usando COPY (70-80% mais rápido)
    
    Performance:
        - COPY: ~15-20s para 284k linhas × 31 colunas
        - to_sql fallback: ~90s para 284k linhas × 31 colunas
    
    Args:
        df: DataFrame a ser salvo
        engine: SQLAlchemy engine
        table_name: Nome da tabela de destino
        if_exists: Estratégia ('replace', 'append', 'fail')
        use_copy: Se True, usa COPY otimizado. Se False, usa to_sql tradicional
        
    Returns:
        Número de linhas inseridas
    """
    # Se use_copy=False, usar método tradicional
    if not use_copy:
        print(f"💾 Salvando dados em PostgreSQL (to_sql): {table_name}")
        
        if if_exists == 'replace':
            with engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                conn.commit()
        
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            method='multi',
            chunksize=10000
        )
        
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
        
        print(f"✅ Dados salvos: {count:,} linhas")
        return count
    
    # Método otimizado com COPY
    print(f"🚀 Salvando dados (COPY otimizado): {table_name}")
    
    try:
        # Drop table se existir e estratégia for replace
        if if_exists == 'replace':
            with engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                conn.commit()
        
        # Obter raw connection do SQLAlchemy
        raw_conn = engine.raw_connection()
        cursor = raw_conn.cursor()
        
        try:
            # Criar tabela primeiro usando pandas schema inference
            df.head(0).to_sql(
                name=table_name,
                con=engine,
                if_exists=if_exists,
                index=False
            )
            
            # Preparar CSV em memória
            output = StringIO()
            df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
            output.seek(0)
            
            # Usar PostgreSQL COPY FROM
            cursor.copy_from(
                file=output,
                table=table_name,
                sep='\t',
                null='\\N',
                columns=list(df.columns)
            )
            
            raw_conn.commit()
            
            # Validar inserção
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar()
            
            print(f"✅ Dados salvos (COPY): {count:,} linhas")
            return count
            
        finally:
            cursor.close()
            raw_conn.close()
    
    except Exception as e:
        print(f"⚠️  COPY falhou: {e}")
        print(f"🔄 Usando fallback to_sql()...")
        
        # Fallback recursivo com use_copy=False
        return save_to_postgresql(df, engine, table_name, if_exists, use_copy=False)


def validate_row_count(engine, table_name: str, expected_count: int) -> None:
    """
    Valida contagem de linhas em uma tabela
    
    Args:
        engine: SQLAlchemy engine
        table_name: Nome da tabela
        expected_count: Contagem esperada
        
    Raises:
        AssertionError: Se a contagem não corresponder
    """
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        actual_count = result.scalar()
    
    assert actual_count == expected_count, f"Erro: esperado {expected_count}, encontrado {actual_count}"
    print(f"✅ Validação: {actual_count:,} linhas em {table_name}")
