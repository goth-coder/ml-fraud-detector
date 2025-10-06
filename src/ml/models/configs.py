"""
Configs - Configurações centralizadas do projeto
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
import json


@dataclass
class DatabaseConfig:
    """Configurações do banco de dados"""
    host: str = "localhost"
    port: int = 5432
    database: str = "fraud_detection"
    user: str = "fraud_user"
    password: str = "fraud_pass_dev"
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class PathConfig:
    """Configurações de caminhos"""
    project_root: Path = Path(__file__).parent.parent.parent.parent
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    reports_dir: Path = project_root / "reports"
    docs_dir: Path = project_root / "docs"
    images_dir: Path = docs_dir / "images"
    
    # Arquivos de dados
    raw_csv: Path = data_dir / "creditcard.csv"
    
    # Arquivos de modelos (com versionamento)
    scalers_pkl: Path = models_dir / "scalers.pkl"
    best_model_pkl: Path = models_dir / "best_model.pkl"
    
    # Reports
    feature_selection_report: Path = reports_dir / "feature_selection_analysis.json"
    
    def __post_init__(self):
        """Cria diretórios se não existirem"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def get_versioned_scaler_path(self, version: str) -> Path:
        """
        Retorna path versionado para scaler
        
        Example:
            >>> paths.get_versioned_scaler_path('1.0.0')
            Path('models/scalers_v1.0.0.pkl')
        """
        return self.models_dir / f"scalers_v{version}.pkl"


@dataclass
class DataConfig:
    """Configurações de dados"""
    # Schema esperado
    expected_columns: int = 31
    feature_columns: List[str] = None
    target_column: str = "Class"
    
    # Proporções
    expected_fraud_percentage: float = 0.172
    train_test_split_ratio: float = 0.8
    
    # Validação
    random_state: int = 42
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]


@dataclass
class ModelConfig:
    """Configurações de modelos"""
    
    # Validação cruzada
    cv_folds: int = 5
    cv_scoring: str = "average_precision"  # PR-AUC
    
    # Grid Search
    grid_search_cv: int = 3
    grid_search_n_jobs: int = -1
    
    # Class weights
    class_weight: str = "balanced"
    
    # Random state
    random_state: int = 42
    
    # XGBoost Hyperparameters (loaded from JSON)
    xgboost_params: Dict = None
    hyperparams_json_path: Path = None
    
    def __post_init__(self):
        # Load hyperparameters from JSON
        if self.hyperparams_json_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            self.hyperparams_json_path = project_root / "data" / "xgboost_hyperparameters.json"
        
        if self.xgboost_params is None:
            self.xgboost_params = self._load_xgboost_params()
    
    def _load_xgboost_params(self) -> Dict:
        """
        Carrega hiperparâmetros do arquivo JSON
        
        Fallback para valores padrão se arquivo não existir
        """
        if self.hyperparams_json_path.exists():
            try:
                with open(self.hyperparams_json_path, 'r') as f:
                    config_data = json.load(f)
                
                params = config_data.get('hyperparameters', {})
                
                # Garantir random_state e eval_metric
                if 'random_state' not in params:
                    params['random_state'] = self.random_state
                if 'eval_metric' not in params:
                    params['eval_metric'] = 'aucpr'
                
                return params
            except Exception as e:
                print(f"⚠️  Erro ao carregar {self.hyperparams_json_path}: {e}")
                return self._get_default_params()
        else:
            print(f"⚠️  {self.hyperparams_json_path} não encontrado. Usando defaults.")
            return self._get_default_params()
    
    def _get_default_params(self) -> Dict:
        """Parâmetros padrão (fallback)"""
        return {
            'colsample_bytree': 0.7,
            'learning_rate': 0.3,
            'max_depth': 6,
            'min_child_weight': 1,
            'n_estimators': 100,
            'scale_pos_weight': 577,
            'subsample': 0.7,
            'random_state': self.random_state,
            'eval_metric': 'aucpr'
        }


@dataclass
class PipelineConfig:
    """Configurações de pipeline"""
    # Tabelas PostgreSQL
    table_raw: str = "raw_transactions"
    table_cleaned: str = "cleaned_transactions"
    table_imputed: str = "imputed_transactions"
    table_normalized: str = "normalized_transactions"
    table_engineered: str = "engineered_transactions"
    table_selected: str = "selected_features"  # Após feature selection
    table_train: str = "train_data"
    table_test: str = "test_data"
    
    # Feature engineering
    include_time_features: bool = True
    include_amount_features: bool = True
    include_statistical_features: bool = True
    include_interactions: bool = False  # Pode aumentar muito dimensionalidade
    
    # Normalização
    amount_scaler: str = "robust"  # robust ou standard
    time_scaler: str = "standard"
    normalize_v_features: bool = False  # V1-V28 já são PCA


@dataclass
class FeatureSelectionConfig:
    """Configuração de seleção de features (Config-Driven)"""
    
    # Features a serem REMOVIDAS (manualmente configurado após análise)
    # Edite esta lista após revisar reports/feature_selection_analysis.json
    excluded_features: List[str] = field(default_factory=lambda: [
        'Time',  # Feature crua sem sentido real (apenas offset no dataset de 2 dias)
        'Time_Period_of_Day',        
        'V8',
        'V23',
        'Amount',
        'V13' 
    ])
    
    # Features INCLUÍDAS após feature engineering e seleção (33 features)
    # Usado pelo Model Service para validação
    included_features: List[str] = field(default_factory=lambda: [
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11',
        'V12', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
        'V22', 'V24', 'V25', 'V26', 'V27', 'V28',
        'Time_Hours', 'Amount_Log', 'Amount_Bin',
        'V_Mean', 'V_Std', 'V_Min', 'V_Max', 'V_Range'
    ])
    
    # Thresholds para análise automática (ajuda na decisão)
    min_pearson_correlation: float = 0.01
    min_spearman_correlation: float = 0.01
    min_mutual_info: float = 0.001
    
    # Análise de multicolinearidade
    max_vif_threshold: float = 10.0  # Variance Inflation Factor
    max_correlation_threshold: float = 0.95  # Features redundantes entre si
    
    # Versionamento
    model_version: str = "2.1.0"  # Current production model


@dataclass
class ProjectConfig:
    """Configuração global do projeto"""
    database: DatabaseConfig = None
    paths: PathConfig = None
    data: DataConfig = None
    models: ModelConfig = None
    pipeline: PipelineConfig = None
    feature_selection: FeatureSelectionConfig = None
    
    def __post_init__(self):
        """Inicializa subconfigs após criação"""
        if self.database is None:
            self.database = DatabaseConfig()
        if self.paths is None:
            self.paths = PathConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.models is None:
            self.models = ModelConfig()
        if self.pipeline is None:
            self.pipeline = PipelineConfig()
        if self.feature_selection is None:
            self.feature_selection = FeatureSelectionConfig()
    
    @classmethod
    def load(cls) -> 'ProjectConfig':
        """Carrega configuração (pode ser extendido para ler de arquivo)"""
        return cls()


# Instância global
config = ProjectConfig.load()


# Grid Search Parameters por modelo
GRID_SEARCH_PARAMS = {
    'decision_tree': {
        'max_depth': [3, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None]
    },
    'svm_rbf': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf'],
        'class_weight': ['balanced', None]
    },
    'xgboost': {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'scale_pos_weight': [1, 5, 10, 20, 50],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
}
