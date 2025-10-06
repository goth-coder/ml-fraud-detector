# 🏗️ Arquitetura MVC-ML - Fraud Detection System (OTIMIZADO)

## 📁 Estrutura do Projeto

```
src/ml/
├── __init__.py
├── processing/           # Funções reutilizáveis (OTIMIZADAS!)
│   ├── __init__.py
│   ├── loader.py                  # OTIMIZADO: PostgreSQL COPY (81.6% mais rápido)
│   ├── validation.py              # Validação de schema e integridade
│   ├── cleaning.py                # Análise de outliers (preserva todos os dados)
│   ├── normalization.py           # RobustScaler + StandardScaler ✅
│   ├── feature_engineering.py     # 9 novas features (Time_Period, Amount_Bin, etc) ✅
│   ├── feature_selection.py       # Análise automatizada (Pearson, Spearman, MI, VIF) ✅
│   ├── splitters.py               # Stratified train/test split ✅
│   └── metadata.py                # Pipeline metadata tracking
│
├── pipelines/            # Scripts que encadeiam funções (OTIMIZADOS!)
│   ├── __init__.py
│   └── data_pipeline.py           # Pipeline completo (Steps 01-07) ✅
│                                   # - PostgreSQL COPY para bulk inserts
│                                   # - Paralelização Steps 02-03 (ThreadPoolExecutor)
│                                   # - Metadata-only para steps sem alteração
│                                   # - Feature Selection Analysis (Step 05.5)
│                                   # - Config-driven feature removal (Step 06)
│
├── models/               # Configurações, schemas
│   ├── __init__.py
│   └── configs.py                 # Configurações centralizadas (dataclasses)
│                                   # - Carrega hiperparâmetros de data/xgboost_hyperparameters.json
│                                   # - Define feature selection (excluded_features)
│                                   # - Config-driven approach (sem hardcoded values)
│
├── training/             # Scripts de treinamento
│   ├── __init__.py
│   ├── train.py                   # Treinamento com hiperparâmetros do JSON
│   ├── tune.py                    # Grid Search + atualização automática de JSON
│   └── archive/                   # Scripts obsoletos (histórico)
```

---

## 🎯 Princípios da Arquitetura

### 1. **Separação de Responsabilidades**
- **Processing**: Funções puras, reutilizáveis, testáveis
- **Pipelines**: Encadeiam funções de processing
- **Models**: Configurações e schemas (carrega de JSON)
- **Training**: Treino e otimização de modelos
- **Validators**: [FUTURO] Métricas ML

### 2. **Modularidade**
- Cada função em `processing/` faz UMA coisa
- Fácil de testar isoladamente
- Fácil de compor em pipelines diferentes

### 3. **Configuração Centralizada + JSON Versionado**
- Hiperparâmetros em `data/xgboost_hyperparameters.json` (versionado)
- Feature selection em `models/configs.py` (hardcoded mas documentado)
- Dataclasses tipadas para type safety
- Fácil de modificar sem alterar código
- Archive automático de versões antigas (`data/archive/`)

### 4. **Idempotência**
- Cada step do pipeline pode ser re-executado
- Sem efeitos colaterais
- Tabelas PostgreSQL sempre recriadas

### 5. **Production-Ready**
- Nenhum valor hardcoded em comparações
- Baselines dinâmicas carregadas de `data/archive/`
- Versionamento automático de modelos e hiperparâmetros
- Sistema escalável para qualquer número de versões

### 5. **Performance Otimizada** ⚡
- PostgreSQL COPY para bulk inserts (70-80% mais rápido)
- Paralelização de steps independentes (ThreadPoolExecutor)
- Metadata tracking evita duplicação de dados
- Pipeline completo: **~62s** (antes: ~130s) - **52% mais rápido!**

---

## 🚀 Como Usar

### Executar Pipeline Completo
```bash  
# Via pipeline direto
python -m src.ml.pipelines.data_pipeline
```

### Importar Funções em Scripts
```python
# Carregar dados (OTIMIZADO com COPY!)
from src.ml.processing.loader import load_from_postgresql, save_to_postgresql

# Normalizar (RobustScaler para outliers)
from src.ml.processing.normalization import fit_and_transform_features

# Feature engineering (9 novas features)
from src.ml.processing.feature_engineering import engineer_all_features

# Train/test split estratificado
from src.ml.processing.splitters import stratified_train_test_split

# Usar configurações
from src.ml.models.configs import config
print(config.database.connection_string)
```

---

## 📊 Pipeline de Dados (7 Steps) - COMPLETO ✅

### Step 01: Load Raw Data ✅
- **Entrada**: `data/creditcard.csv`
- **Saída**: PostgreSQL `raw_transactions` (284,807 linhas)
- **Função**: `processing/loader.py::load_csv_to_dataframe()` + `save_to_postgresql()`
- **Performance**: ~20s (com PostgreSQL COPY otimizado)

### Step 02: Outlier Analysis ✅
- **Entrada**: `raw_transactions`
- **Saída**: **Metadata apenas** (dados NÃO duplicados!)
- **Função**: `processing/cleaning.py::identify_outliers()`
- **Decisão**: Preservar 100% dos dados (18.5% fraudes em outliers)
- **Performance**: ~2.4s
- **Paralelização**: Roda em paralelo com Step 03 ⚡

### Step 03: Handle Missing ✅
- **Entrada**: `raw_transactions`
- **Saída**: **Metadata apenas** (dados NÃO duplicados!)
- **Função**: `processing/validation.py::validate_no_missing_values()`
- **Resultado**: Nenhum missing value encontrado (dataset 100% completo)
- **Performance**: ~2.4s
- **Paralelização**: Roda em paralelo com Step 02 ⚡

### Step 04: Normalize ✅  
- **Entrada**: `raw_transactions`
- **Saída**: `normalized_transactions` + `models/scalers.pkl`
- **Função**: `processing/normalization.py::fit_and_transform_features()`
- **Scalers**: RobustScaler (Amount), StandardScaler (Time)
- **Performance**: ~16.5s (ANTES: ~90s) - **81.6% mais rápido!** 🔥
- **Otimização**: PostgreSQL COPY em vez de to_sql()
- **Versionamento**: `models/scalers_v1.0.0.pkl` (opcional)

### Step 05: Feature Engineering ✅
- **Entrada**: `normalized_transactions`
- **Saída**: `engineered_transactions` (284,807 × 40 colunas)
- **Função**: `processing/feature_engineering.py::engineer_all_features()`
- **Features**: Time_Period, Amount_Log, Amount_Bin, V-statistics (Mean/Std/Min/Max/Range)
- **Performance**: ~20.6s (com PostgreSQL COPY otimizado)

### Step 05.5: Feature Selection Analysis ✅
- **Entrada**: `engineered_transactions`
- **Saída**: `reports/feature_selection_analysis.json` (relatório JSON)
- **Função**: `processing/feature_selection.py::analyze_feature_importance()`
- **Análises**:
  - Pearson Correlation (linear)
  - Spearman Correlation (monotônica)
  - Mutual Information (não-linear)
  - VIF - Variance Inflation Factor (multicolinearidade)
  - Correlation Matrix (redundância entre features)
- **Output**: Relatório JSON com candidatos a remoção
- **Performance**: ~15s
- **Ação Requerida**: Revisar relatório e atualizar `configs.py`

### Step 06: Apply Feature Selection ✅
- **Entrada**: `engineered_transactions`
- **Saída**: `selected_features` (SE houver exclusões configuradas)
- **Função**: `processing/feature_selection.py::apply_feature_selection()`
- **Config-Driven**: Remove features listadas em `configs.py::FeatureSelectionConfig.excluded_features`
- **Performance**: ~5s (ou metadata-only se lista vazia)
- **Versionamento**: Incrementar `model_version` quando alterar exclusões

### Step 07: Train/Test Split
- **Entrada**: `selected_features` (se Step 06 aplicou) OU `engineered_transactions`
- **Saída**: `train_data` (227,845) + `test_data` (56,962)
- **Função**: `processing/splitters.py::stratified_train_test_split()`
- **Split**: 80/20 estratificado (preserva proporção 0.172%)
- **Performance**: ~22.2s (com PostgreSQL COPY para ambas tabelas)

---

## 🎯 Feature Selection Strategy (Config-Driven)

### Filosofia: Análise Automatizada + Decisão Manual

**Por que NÃO automatizar completamente?**
- ❌ Remoção automática pode descartar features úteis não-lineares
- ❌ Contexto de negócio importa (fraudes de alto valor, padrões temporais)
- ❌ A/B testing manual permite comparar performance

**Solução: Config-Driven Approach**
1. ✅ **Step 05.5**: Analisa automaticamente (Pearson, Spearman, MI, VIF)
2. ✅ **Relatório JSON**: Salva candidatos a remoção em `reports/feature_selection_analysis.json`
3. ✅ **Desenvolvedor revisa**: Entende por que features têm baixa correlação
4. ✅ **Atualiza configs.py**: Adiciona features à lista `excluded_features`
5. ✅ **Step 06**: Aplica remoção automaticamente no próximo run

### Workflow de Desenvolvimento

```bash
# 1. Primeira execução: gera relatório
python test_pipeline.py  # Executa Steps 01-07 (incluindo 05.5)

# 2. Desenvolvedor revisa relatório
cat reports/feature_selection_analysis.json

# Output exemplo:
{
  "candidates_for_removal": {
    "low_correlation_all_methods": ["V13", "V15", "V22"],
    "count": 3,
    "features_detail": {
      "V13": {
        "pearson": 0.0046,
        "spearman": 0.0037,
        "mutual_info": 0.0004
      },
      ...
    }
  },
  "multicollinearity": {
    "high_vif_features": [],
    "redundant_pairs": []
  }
}

# 3. Atualiza configs.py manualmente
vim src/ml/models/configs.py

# Descomentar features para remover:
excluded_features: List[str] = field(default_factory=lambda: [
    'V13',  # Baixa correlação (decisão: 2025-10-02)
    'V15',  # Baixa correlação (decisão: 2025-10-02)
    'V22',  # Baixa correlação (decisão: 2025-10-02)
])

# Incrementar versão:
model_version: str = "1.1.0"

# 4. Re-executa pipeline
python test_pipeline.py  # Step 06 aplica remoção automaticamente

# 5. Treina modelos com features otimizadas
python -m src.ml.pipelines.training_pipeline  # [FUTURO]

# 6. Compara performance (com vs sem feature selection)
# Se melhor: commit configs.py
# Se pior: reverte changes
```

### Vantagens da Abordagem

| Aspecto | Vantagem |
|---------|----------|
| **Rastreabilidade** | Git versiona decisões (`configs.py`) |
| **Reprodutibilidade** | Mesmo código + mesma config = mesmo resultado |
| **A/B Testing** | Fácil comentar/descomentar features |
| **Context-Aware** | Desenvolvedor considera contexto de negócio |
| **Produção-Ready** | Config management usado em empresas reais |
| **Educativo** | Relatório JSON ensina sobre feature importance |

---

## 🔖 Versionamento de Modelos

### Scalers (Pickle)

**MVP/POC (atual)**:
```python
# Simples e eficaz
models/scalers.pkl  # Versão ativa
```

**Produção (recomendado)**:
```python
# Versionamento explícito
models/
  scalers_v1.0.0.pkl  # Sem feature selection
  scalers_v1.1.0.pkl  # Com feature selection (V13, V15, V22 removidos)
  scalers_v1.2.0.pkl  # Novas features adicionadas

# Carregar versão específica
scaler_path = config.paths.get_versioned_scaler_path('1.1.0')
scalers = joblib.load(scaler_path)
```

**Enterprise (produção real)**:
```python
# MLflow (tracking completo)
import mlflow

with mlflow.start_run():
    mlflow.sklearn.log_model(scaler, "scaler")
    mlflow.log_param("scaler_type", "RobustScaler")
    mlflow.log_param("model_version", "1.1.0")
    mlflow.log_param("excluded_features", ["V13", "V15", "V22"])

# Carregar versão específica
scaler = mlflow.sklearn.load_model("runs:/<run_id>/scaler")
```

### Por que Pickle para Scalers?

| Método | Prós | Contras | Uso |
|--------|------|---------|-----|
| **Pickle** | ✅ Preserva objeto completo<br>✅ Rápido (nativo)<br>✅ Integração scikit-learn | ❌ Python-only<br>❌ Sem segurança | ✅ **MVP/POC** |
| **ONNX** | ✅ Cross-platform<br>✅ Otimizado para inferência | ❌ Conversão complexa<br>❌ Nem todos modelos suportados | Produção multi-linguagem |
| **MLflow** | ✅ Versionamento automático<br>✅ Tracking completo<br>✅ UI visual | ❌ Infra adicional<br>❌ Overhead | ✅ **Enterprise** |
| **SQL** | ✅ Centralizado<br>✅ Backup automático | ❌ Perder métodos do objeto<br>❌ Precisa recriar | ❌ Não recomendado |

**Decisão para este projeto**: Pickle + versionamento manual (MVP), com path para MLflow documentado para evolução futura.

---

## ⚙️ Configurações

Todas as configurações estão em `src/ml/models/configs.py`:

```python
from src.ml.models.configs import config

# Database
config.database.connection_string
config.database.pool_size

# Paths
config.paths.raw_csv
config.paths.scalers_pkl
config.paths.feature_selection_report
config.paths.get_versioned_scaler_path('1.0.0')  # models/scalers_v1.0.0.pkl

# Data
config.data.expected_columns
config.data.target_column
config.data.train_test_split_ratio  # 0.8

# Pipeline
config.pipeline.table_raw
config.pipeline.table_normalized
config.pipeline.table_engineered
config.pipeline.table_selected  # Após feature selection
config.pipeline.include_interactions  # False (default)

# Feature Selection (Config-Driven)
config.feature_selection.excluded_features  # Lista para editar manualmente
config.feature_selection.min_pearson_correlation  # 0.01
config.feature_selection.max_vif_threshold  # 10.0
config.feature_selection.model_version  # "1.0.0" (incrementar ao alterar)
```

---

## ⚡ Otimizações Implementadas

### 1. PostgreSQL COPY (81.6% mais rápido) 🔥
```python
# src/ml/processing/loader.py
def save_to_postgresql(df, engine, table_name, use_copy=True):
    """
    Usa PostgreSQL COPY nativo para bulk inserts
    Fallback automático para to_sql() se falhar
    """
    # COPY: ~16s para 284k × 31 colunas
    # to_sql: ~90s para 284k × 31 colunas
```

**Impacto**:
- Step 04: 90s → 16.5s
- Step 05: Beneficiado (40 colunas)
- Step 06: Beneficiado (2 tabelas grandes)

### 2. Paralelização Steps 02-03 (18.6% mais rápido) ⚡
```python
# src/ml/pipelines/data_pipeline.py
with ThreadPoolExecutor(max_workers=2) as executor:
    future_02 = executor.submit(run_step_02_outlier_analysis)
    future_03 = executor.submit(run_step_03_handle_missing)
    # Ambos rodam em paralelo!
```

**Impacto**:
- Antes: 2.43s + 2.36s = 4.79s (sequencial)
- Depois: max(2.43s, 2.36s) = 3.90s (paralelo)

### 3. Metadata-Only (Zero I/O desnecessário)
- Steps 02-03 salvam apenas metadata, NÃO duplicam dados
- Rastreabilidade mantida em `pipeline_metadata` table
- Dados permanecem em `raw_transactions`

### 4. Performance Total
| Métrica | Antes | Depois | Ganho |
|---------|-------|--------|-------|
| Step 04 | 90s | 16.5s | **81.6%** 🔥 |
| Steps 02-03 | 4.8s | 3.9s | **18.6%** ⚡ |
| **Pipeline Completo** | **~130s** | **~62s** | **52%** 🚀 |

---

## 🧪 Testes

```bash
# Testar pipeline completo (OTIMIZADO)
python test_pipeline.py

# Testar imports
python -c "from src.ml.pipelines.data_pipeline import DataPipeline; print('OK')"

# Testar funções individuais
python -c "from src.ml.processing.loader import save_to_postgresql; print('OK')"
python -c "from src.ml.processing.feature_engineering import engineer_all_features; print('OK')"
```

---

## 📈 Próximos Passos

1. ✅ **Pipeline de Dados Completo** (Steps 01-07) - CONCLUÍDO ✅
2. ✅ **Otimizações de Performance** - CONCLUÍDO ✅
3. ✅ **Feature Selection Analysis** - CONCLUÍDO ✅
4. ⏳ **Training Pipeline** (training/01_train_models.py)
   - Treinar 3 modelos robustos (Decision Tree, SVM, XGBoost)
   - StratifiedKFold validation
   - Grid Search para hiperparâmetros
   - Threshold tuning
4. ⏳ **Validators** (src/ml/validators/)
   - metrics.py (Precision, Recall, F1, PR-AUC)
   - kfold_validator.py (StratifiedKFold)
   - model_comparator.py (Comparação estatística)
5. ⏳ **Dashboard Flask** (src/views/)
   - Interface interativa
   - Simulação em tempo real
   - WebSocket streaming
6. ⏳ **[OPCIONAL] Kafka Enhancement** (plan_kafka.md)

---

## 📚 Referências

- **MVC Pattern**: Model-View-Controller
- **ML Pipeline**: Modular, testável, reprodutível
- **SOLID Principles**: Single Responsibility, Open/Closed
- **PostgreSQL COPY**: Bulk insert nativo (70-80% mais rápido)
- **ThreadPoolExecutor**: Paralelização de tarefas I/O-bound
- **RobustScaler**: Normalização resistente a outliers (mediana + IQR)

---

**Status Atual**: ✅ **Pipeline de Dados COMPLETO E OTIMIZADO (Steps 01-07)**  
**Performance**: 52% mais rápido (~62s vs ~130s)  
**Feature Selection**: Config-driven, Time removido, **38 features** prontas  
**Versionamento**: Scalers com suporte a versionamento (get_versioned_scaler_path)  
**Próximo**: ⏳ **Treinamento de modelos ML em andamento** (Decision Tree, SVM RBF, XGBoost)  

### 🎯 Estratégia de Modelagem

#### Fase 1: Baseline Training (38 Features) - EM ANDAMENTO ⏳
- **Features**: V1-V28 (28), Amount/Amount_Log/Amount_Bin (3), Time_Hour/Time_Period_of_Day (2), V-Statistics (5)
- **Time removido**: Apenas offset temporal, sem significado em produção
- **3 Modelos Robustos**: Decision Tree, SVM RBF, XGBoost (Regressão Logística REMOVIDA - sensível a outliers)
- **Validação**: StratifiedKFold (k=5)
- **Métricas**: PR-AUC (principal), Recall, F1, Precision

#### Fase 2: Feature Importance Analysis (Pós-Treino)
- XGBoost fornece `feature_importances_` nativo
- Identificar features com importance ≈ 0
- **Decisão data-driven**: Remover features inúteis baseado em evidência real
- A/B testing: Comparar performance com/sem features

#### Fase 3: Refinamento (Se Necessário)
- Grid Search para hiperparâmetros (scoring='average_precision')
- Threshold tuning (0.2-0.3 esperado, não 0.5)
- Re-treinamento com features otimizadas
- Comparação estatística entre modelos

#### Fase 4: Dashboard Flask (MVP)
- Interface interativa
- Simulação em tempo real
- WebSocket streaming
