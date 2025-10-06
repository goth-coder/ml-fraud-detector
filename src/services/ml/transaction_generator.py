"""
Transaction Generator - Retorna transações reais do dataset de teste.

Busca transações aleatórias (legítimas ou fraudulentas) do dataset de teste
armazenado no PostgreSQL para uso no simulador do dashboard.
"""
import numpy as np
import pandas as pd
from typing import Dict, Literal
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.ml.models.configs import config
from src.services.database.connection import get_engine


class TransactionGenerator:
    """
    Gerador de transações baseado em dados reais do dataset de teste.
    
    Busca transações aleatórias do PostgreSQL (tabela test_data) para
    garantir que as transações simuladas sejam realistas e já validadas.
    """
    
    def __init__(self):
        """Inicializa gerador e carrega pool de transações."""
        self._engine = None
        self._fraud_pool = None
        self._legit_pool = None
        self._load_pools()
    
    def _load_pools(self):
        """
        Carrega pools de transações fraudulentas e legítimas do PostgreSQL.
        
        Mantém em memória para performance (evita queries repetidas).
        """
        try:
            self._engine = get_engine()
            
            # Carregar todas as transações de teste
            query = """
                SELECT * FROM test_data
                ORDER BY RANDOM()
            """
            df = pd.read_sql(query, self._engine)
            
            # Separar por tipo
            self._fraud_pool = df[df['Class'] == 1].drop('Class', axis=1).to_dict('records')
            self._legit_pool = df[df['Class'] == 0].drop('Class', axis=1).to_dict('records')
            
            print(f"✅ Transaction pools carregados:")
            print(f"   - Legítimas: {len(self._legit_pool)} transações")
            print(f"   - Fraudulentas: {len(self._fraud_pool)} transações")
            
        except Exception as e:
            print(f"❌ Erro ao carregar pools: {e}")
            print("⚠️  Usando fallback: pools vazios")
            self._fraud_pool = []
            self._legit_pool = []
    
    def generate(
        self, 
        transaction_type: Literal['legitimate', 'fraud'] = 'legitimate'
    ) -> Dict[str, float]:
        """
        Retorna uma transação aleatória real do dataset de teste.
        
        Args:
            transaction_type: 'legitimate' ou 'fraud'
            
        Returns:
            Dicionário com 33 features da transação
            
        Raises:
            ValueError: Se pool estiver vazio
            
        Example:
            >>> generator = TransactionGenerator()
            >>> legit = generator.generate('legitimate')
            >>> fraud = generator.generate('fraud')
        """
        pool = self._legit_pool if transaction_type == 'legitimate' else self._fraud_pool
        
        if not pool:
            raise ValueError(f"Pool de transações {transaction_type} está vazio. Verifique se test_data existe no PostgreSQL.")
        
        # Selecionar transação aleatória
        transaction = pool[np.random.randint(0, len(pool))].copy()
        
        return transaction
    
    def generate_batch(
        self, 
        count: int,
        fraud_ratio: float = 0.5
    ) -> list[Dict[str, float]]:
        """
        Retorna múltiplas transações aleatórias.
        
        Args:
            count: Número total de transações
            fraud_ratio: Proporção de fraudes (0.0 a 1.0)
            
        Returns:
            Lista de transações
            
        Example:
            >>> generator = TransactionGenerator()
            >>> batch = generator.generate_batch(100, fraud_ratio=0.1)  # 10% fraudes
        """
        fraud_count = int(count * fraud_ratio)
        legit_count = count - fraud_count
        
        transactions = []
        
        # Gerar legítimas
        for _ in range(legit_count):
            transactions.append(self.generate('legitimate'))
        
        # Gerar fraudes
        for _ in range(fraud_count):
            transactions.append(self.generate('fraud'))
        
        # Embaralhar
        np.random.shuffle(transactions)
        
        return transactions
    
    def reload_pools(self):
        """
        Recarrega pools do PostgreSQL.
        
        Útil se o dataset for atualizado durante execução.
        """
        self._load_pools()


# Instância global
transaction_generator = TransactionGenerator()
