"""
Model Service - Singleton para carregar e usar modelo XGBoost v2.1.0.

Carrega o modelo uma única vez na primeira chamada e mantém em memória.
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import sys

# Adicionar path do projeto para imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.ml.models.configs import config

logger = logging.getLogger(__name__)


class ModelService:
    """
    Singleton para gerenciar modelo XGBoost e scalers.
    
    Carrega automaticamente na primeira chamada de predict().
    """
    
    _instance = None
    _model = None
    _scalers = None
    _model_version = "v2.1.0"
    
    # Features esperadas (importadas de configs.py)
    EXPECTED_FEATURES = config.feature_selection.included_features
    
    def __new__(cls):
        """Singleton pattern: sempre retorna a mesma instância."""
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance
    
    def _load_model(self, model_path: Path, scaler_path: Path):
        """
        Carrega modelo e scalers do disco.
        
        Args:
            model_path: Caminho para xgboost_v2.1.0.pkl
            scaler_path: Caminho para scalers.pkl
        """
        if self._model is not None:
            logger.info("Modelo já carregado, reutilizando instância")
            return
        
        logger.info(f"Carregando modelo: {model_path}")
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)
        
        logger.info(f"Carregando scalers: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self._scalers = pickle.load(f)
        
        logger.info(f"✅ Modelo {self._model_version} carregado com sucesso!")
    
    def _validate_features(self, features: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """
        Valida se features estão completas e corretas.
        
        Args:
            features: Dicionário com features da transação
            
        Returns:
            (is_valid, error_message)
        """
        provided_features = set(features.keys())
        expected_features = set(self.EXPECTED_FEATURES)
        
        missing = expected_features - provided_features
        if missing:
            return False, f"Features faltando: {sorted(missing)}"
        
        extra = provided_features - expected_features
        if extra:
            logger.warning(f"Features extras ignoradas: {sorted(extra)}")
        
        for feature in self.EXPECTED_FEATURES:
            if not isinstance(features[feature], (int, float)):
                return False, f"Feature '{feature}' deve ser numérica"
        
        return True, None
    
    def predict(
        self, 
        features: Dict[str, float],
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        Executa predição em uma transação.
        
        Args:
            features: Dicionário com 33 features da transação
            model_path: Caminho do modelo (usa padrão se None)
            scaler_path: Caminho dos scalers (usa padrão se None)
            
        Returns:
            {
                'prediction': 0 ou 1,
                'probability': float (0-1),
                'confidence': 'Alta' | 'Média' | 'Baixa',
                'is_fraud': bool
            }
            
        Raises:
            ValueError: Se features inválidas
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent.parent / 'models' / 'xgboost_v2.1.0.pkl'
        if scaler_path is None:
            scaler_path = Path(__file__).parent.parent.parent.parent / 'models' / 'scalers.pkl'
        
        if self._model is None:
            self._load_model(model_path, scaler_path)
        
        is_valid, error = self._validate_features(features)
        if not is_valid:
            raise ValueError(error)
        
        feature_array = np.array([[features[f] for f in self.EXPECTED_FEATURES]])
        
        prediction = int(self._model.predict(feature_array)[0])
        probability = float(self._model.predict_proba(feature_array)[0][1])
        
        if probability >= 0.7:
            confidence = "Alta"
        elif probability >= 0.3:
            confidence = "Média"
        else:
            confidence = "Baixa"
        
        return {
            'prediction': prediction,
            'probability': probability,
            'probability_percent': f"{probability * 100:.2f}%",
            'confidence': confidence,
            'is_fraud': prediction == 1,
            'model_version': self._model_version
        }
    
    def predict_batch(
        self, 
        features_list: List[Dict[str, float]],
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None
    ) -> List[Dict[str, any]]:
        """
        Executa predição em múltiplas transações (batch).
        
        Args:
            features_list: Lista de dicionários com features
            model_path: Caminho do modelo (usa padrão se None)
            scaler_path: Caminho dos scalers (usa padrão se None)
            
        Returns:
            Lista de resultados de predição
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent.parent / 'models' / 'xgboost_v2.1.0.pkl'
        if scaler_path is None:
            scaler_path = Path(__file__).parent.parent.parent.parent / 'models' / 'scalers.pkl'
        
        if self._model is None:
            self._load_model(model_path, scaler_path)
        
        results = []
        for features in features_list:
            try:
                result = self.predict(features, model_path, scaler_path)
                results.append(result)
            except ValueError as e:
                results.append({'error': str(e)})
        
        return results
    
    @property
    def is_loaded(self) -> bool:
        """Verifica se modelo está carregado."""
        return self._model is not None
    
    @property
    def model_version(self) -> str:
        """Retorna versão do modelo."""
        return self._model_version


model_service = ModelService()
