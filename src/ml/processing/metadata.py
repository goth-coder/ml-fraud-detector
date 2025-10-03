"""
Metadata - Funções para registrar metadata de processamento
Salva estatísticas de cada step no PostgreSQL sem duplicar dados
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Engine, text


def save_pipeline_metadata(
    engine: Engine,
    step_number: int,
    step_name: str,
    rows_processed: int,
    rows_output: int,
    data_modified: bool,
    metadata: Dict[str, Any],
    duration_seconds: float = None,
    status: str = 'success'
) -> None:
    """
    Salva metadata de um step do pipeline
    
    Args:
        engine: SQLAlchemy engine
        step_number: Número do step (1-6)
        step_name: Nome do step (ex: 'outlier_analysis')
        rows_processed: Número de linhas processadas
        rows_output: Número de linhas na saída
        data_modified: Se dados foram alterados (True) ou apenas analisados (False)
        metadata: Dicionário com estatísticas do step
        duration_seconds: Duração da execução em segundos
        status: Status do step ('success', 'failed', 'skipped')
    """
    metadata_json = json.dumps(metadata)
    
    query = text("""
        INSERT INTO pipeline_metadata (
            step_number, 
            step_name, 
            rows_processed, 
            rows_output, 
            data_modified,
            metadata,
            duration_seconds,
            status
        ) VALUES (
            :step_number,
            :step_name,
            :rows_processed,
            :rows_output,
            :data_modified,
            CAST(:metadata AS jsonb),
            :duration_seconds,
            :status
        )
    """)
    
    with engine.begin() as conn:
        conn.execute(query, {
            'step_number': step_number,
            'step_name': step_name,
            'rows_processed': rows_processed,
            'rows_output': rows_output,
            'data_modified': data_modified,
            'metadata': metadata_json,
            'duration_seconds': duration_seconds,
            'status': status
        })
    
    print(f"📊 Metadata salvo: Step {step_number} ({step_name})")


def get_last_pipeline_execution(engine: Engine) -> Dict[str, Any]:
    """
    Retorna informações da última execução do pipeline
    
    Returns:
        Dicionário com estatísticas da última execução
    """
    query = text("""
        SELECT 
            step_number,
            step_name,
            timestamp,
            rows_processed,
            rows_output,
            data_modified,
            metadata,
            duration_seconds,
            status
        FROM pipeline_metadata
        ORDER BY timestamp DESC
        LIMIT 6
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
        
        if not rows:
            return {}
        
        return {
            'steps': [
                {
                    'step_number': row[0],
                    'step_name': row[1],
                    'timestamp': row[2],
                    'rows_processed': row[3],
                    'rows_output': row[4],
                    'data_modified': row[5],
                    'metadata': row[6],
                    'duration_seconds': row[7],
                    'status': row[8]
                }
                for row in rows
            ]
        }


def get_latest_data_table(engine: Engine, target_step: int) -> str:
    """
    Identifica qual tabela contém os dados mais recentes para um step
    
    Lógica:
    - Busca no pipeline_metadata qual foi o último step que MODIFICOU dados
    - Se step anterior não modificou dados, busca o anterior recursivamente
    - Retorna nome da tabela com dados válidos
    
    Args:
        engine: SQLAlchemy engine
        target_step: Step que vai processar (ex: 4 = normalize)
        
    Returns:
        Nome da tabela com dados mais recentes (ex: 'raw_transactions', 'normalized_transactions')
    """
    # Mapeamento step → tabela de output
    step_to_table = {
        1: 'raw_transactions',
        2: 'cleaned_transactions',  # Não usada (step 02 não modifica)
        3: 'imputed_transactions',  # Não usada (step 03 não modifica)
        4: 'normalized_transactions',
        5: 'engineered_transactions',
        6: 'train_test_split'  # Caso especial
    }
    
    # Buscar último step que MODIFICOU dados antes do target_step
    query = text("""
        SELECT step_number, step_name, data_modified
        FROM pipeline_metadata
        WHERE step_number < :target_step 
          AND status = 'success'
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {'target_step': target_step})
        rows = result.fetchall()
        
        if not rows:
            # Nenhum step executado antes, usar raw
            return 'raw_transactions'
        
        # Procurar último step que MODIFICOU dados
        for row in rows:
            step_num = row[0]
            data_modified = row[2]
            
            if data_modified:
                # Este step modificou dados, usar sua tabela de output
                return step_to_table.get(step_num, 'raw_transactions')
        
        # Nenhum step modificou dados ainda, usar raw
        return 'raw_transactions'


def check_table_exists(engine: Engine, table_name: str) -> bool:
    """
    Verifica se tabela existe no PostgreSQL
    
    Args:
        engine: SQLAlchemy engine
        table_name: Nome da tabela
        
    Returns:
        True se existe, False caso contrário
    """
    query = text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = :table_name
        )
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {'table_name': table_name})
        return result.scalar()
