# 🚨 Detecção de Fraude em Cartão de Crédito
## Tech Challenge Fase 3 - FIAP

Sistema completo de **Machine Learning para detecção de fraude** com pipeline modular, PostgreSQL para persistência e CLI interativo.

---

## 🏆 Modelo Final

**Quando usar**: Primeira vez ou após mudanças no dataset original.

---

### Modo 1: Train (Treinamento Rápido)t v2.1.0** - Modelo otimizado em produção:

| Métrica | Valor | Benefício |
|---------|-------|-----------|
| **PR-AUC** | 0.8772 | Métrica robusta para desbalanceamento |
| **Precision** | 86.60% | 86% das predições de fraude estão corretas |
| **Recall** | 81.63% | Detecta 82% das fraudes reais |
| **F1-Score** | 84.04% | Equilíbrio entre Precision e Recall |
| **Falsos Positivos** | 13 | Apenas R$ 260/dia de custo operacional |
| **Features** | 33 | Modelo enxuto e rápido (11% redução) |

**Dataset**: creditcard.csv (Kaggle) - 284.807 transações, 492 fraudes (0.172%)

---

## 🎯 Objetivo

Desenvolver sistema de detecção de fraude usando **XGBoost otimizado** com:
- ✅ Tratamento de dados **altamente desbalanceados** (ratio 1:578)
- ✅ Pipeline modular MVC-ML (processamento → treinamento → inferência)
- ✅ Validação robusta com **StratifiedKFold**
- ✅ **Grid Search automatizado** com versionamento de hiperparâmetros
- ✅ **PostgreSQL** (dados) + **Pickle** (modelos) para rastreabilidade
- ✅ **CLI completo** (pipeline/train/tune/predict) com 4 modos de operação
---

## 🏗️ Arquitetura do Projeto

### Estrutura MVC-ML Modular
```
src/ml/
├── processing/           # 🔧 Funções reutilizáveis (OTIMIZADAS!)
│   ├── loader.py         # PostgreSQL COPY (81.6% mais rápido)
│   ├── validation.py     # Schema e integridade
│   ├── cleaning.py       # Análise de outliers (preserva 100%)
│   ├── normalization.py  # RobustScaler + StandardScaler
│   ├── feature_engineering.py  # 9 novas features
│   ├── feature_selection.py    # Análise automatizada
│   ├── splitters.py      # Stratified train/test split
│   └── metadata.py       # Pipeline metadata tracking
│
├── pipelines/            # 🚀 Encadeamento de processos
│   └── data_pipeline.py  # Steps 01-07 (completo, 52% mais rápido)
│
├── training/             # 🎓 Scripts de treinamento ML
│   ├── train.py          # ⭐ Treino com hiperparâmetros do JSON
│   └── tune.py           # ⭐ Grid Search + atualização automática de JSON
│
└── models/               # ⚙️ Configurações centralizadas
    └── configs.py        # Dataclasses tipadas + carregamento de JSON
```

### Pipeline de Dados (7 Steps - 52% mais rápido ⚡)
```
┌──────────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING PIPELINE (~62s)                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CSV Raw (284,807) → [01] Load Raw → PostgreSQL (raw_transactions)  │
│                         ↓ (~20s)                                     │
│                    [02] Outlier Analysis → Metadata only             │
│                         ↓ (paralelo ~2.4s)                          │
│                    [03] Missing Values → Metadata only               │
│                         ↓ (paralelo ~2.4s)                          │
│                    [04] Normalize → PostgreSQL + scalers.pkl         │
│                         ↓ (~16.5s - OTIMIZADO 81.6% 🔥)             │
│                    [05] Feature Engineering → PostgreSQL (40 cols)   │
│                         ↓ (~20.6s - Time_Period, Amount_Log, stats) │
│                    [05.5] Feature Selection Analysis → JSON Report   │
│                         ↓ (~15s - Pearson, Spearman, MI, VIF)       │
│                    [06] Apply Feature Selection → PostgreSQL (33)    │
│                         ↓ (~5s - Config-driven removal)             │
│                    [07] Train/Test Split → train_data + test_data    │
│                         ↓ (~22.2s - StratifiedKFold 80/20)          │
│                                                                      │
│  TOTAL: ~62s (antes: ~130s) - GANHO: 52% ⚡                         │
└──────────────────────────────────────────────────────────────────────┘
```

### Arquitetura de Persistência

**PostgreSQL - Tabelas do Pipeline**:
- `raw_transactions` (284,807 linhas): CSV original
- `cleaned_transactions` (284,804 linhas): Sem outliers
- `imputed_transactions` (284,804 linhas): Sem missing values
- `normalized_transactions` (284,804 linhas): Features normalizadas
- `engineered_transactions` (284,804 linhas): Features novas (40 colunas)
- `train_features` (~227,843 linhas): 80% treino
- `test_features` (~56,961 linhas): 20% teste

**PostgreSQL - Tabelas de Metadados** (🔮 FUTURO - não implementadas):
- `metrics_history`: Métricas de modelos ao longo do tempo
- `trained_models`: Registro de modelos treinados com timestamps
- `data_splits`: Histórico de splits train/test

**Pickle (modelos ML)**:
- `models/scalers.pkl`: RobustScaler + StandardScaler (fit em train)
- `models/xgboost_v2.1.0.pkl`: Modelo final em produção ⭐

**JSON (hiperparâmetros versionados)**:
- `data/xgboost_hyperparameters.json`: Hiperparâmetros ativos
- `data/archive/`: Versões anteriores com timestamp automático

**Por que híbrido?**
- **PostgreSQL**: Rastreabilidade, SQL analytics, backups automáticos
- **Pickle**: Fast loading (~0.1s), métodos do objeto preservados, scikit-learn nativo
- **JSON**: Versionamento de hiperparâmetros, comparações dinâmicas, production-ready

### Stack Tecnológico
- **Data Pipeline**: Python 3.13 + Pandas + PostgreSQL 15
- **ML Model**: XGBoost 3.0.5 (scale_pos_weight=577, max_depth=6)
- **Persistência**: PostgreSQL (dados) + Pickle (modelos)
- **Otimização**: PostgreSQL COPY, ThreadPoolExecutor, metadata-only steps
- **Infraestrutura**: Docker Compose (PostgreSQL)

---

## 📊 Dataset

- **Fonte**: Kaggle - Credit Card Fraud Detection
- **Transações**: 284.807 (2 dias, setembro 2013)
- **Features**: 30 (Time, Amount, V1-V28 via PCA)
- **Target**: Class (0=legítima, 1=fraude)
- **Desbalanceamento**: 492 fraudes (0.172%) vs 284.315 legítimas

---

## 🚀 Quick Start

### 1. Pré-requisitos
- Python 3.13+
- PostgreSQL 15+ (ou Docker)
- 2GB+ RAM disponível

### 2. Instalação

```bash
# Clone do repositório
git clone <repo-url>
cd ml-fraud-detector

# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt

# OU usando uv (mais rápido)
uv pip install -e .
```

### 3. Configurar PostgreSQL

**Opção A: Docker (Recomendado)**
```bash
cd database
docker-compose up -d

# Validar conexão
docker exec -it fraud_detection_postgres psql -U fraud_user -d fraud_detection -c "\dt"
```

**Opção B: PostgreSQL Local**
```bash
# Criar banco e usuário
createdb fraud_detection
createuser fraud_user --password fraud_pass_dev

# Aplicar schema
psql -U fraud_user -d fraud_detection -f database/schema.sql
```

### 4. Executar Pipeline de Dados

```bash
# Opção 1: Via CLI (RECOMENDADO)
python main.py pipeline

# Opção 2: Direto (legacy)
python -m src.ml.pipelines.data_pipeline

# Resultado esperado (~62s):
# ✅ raw_transactions: 284,807 linhas
# ✅ cleaned_transactions: 284,804 linhas (outliers removidos)
# ✅ imputed_transactions: 284,804 linhas (missing tratados)
# ✅ normalized_transactions: 284,804 linhas (RobustScaler aplicado)
# ✅ engineered_transactions: 284,804 linhas (40 features)
# ✅ train_features: ~227,843 linhas (394 fraudes)
# ✅ test_features: ~56,961 linhas (98 fraudes)
# ✅ scalers.pkl salvo em models/
```

---

## 🖥️ CLI Principal (`main.py`)

O projeto inclui um **CLI completo** com 4 modos de operação:

### Modo 1: Pipeline (Processamento de Dados)

Executa pipeline completo de dados (Steps 01-07, ~62s):

```bash
# Pipeline completo
python main.py pipeline

# Saída esperada:
# 🚀 MODO: PIPELINE DE DADOS
# Data: 2025-10-03 14:35:22
# ⏱️  Tempo estimado: ~62 segundos
# 
# 🚀 Executando pipeline completo...
#    Script: data_pipeline.py
#    Steps: 01-07
# 
# ✅ Pipeline completo!
# 📊 Resultado esperado:
#    ✅ raw_transactions: 284,807 linhas
#    ✅ cleaned_transactions: 284,804 linhas
#    ✅ imputed_transactions: 284,804 linhas
#    ✅ normalized_transactions: 284,804 linhas
#    ✅ engineered_transactions: 284,804 linhas (40 features)
#    ✅ train_features: ~227,843 linhas (394 fraudes)
#    ✅ test_features: ~56,961 linhas (98 fraudes)
#    ✅ scalers.pkl salvo em models/
# 
# 💡 Próximo passo: python main.py train
```

**Quando usar**: Primeira vez ou após mudanças no dataset original.
 


### Modo 2: Train (Treinamento Rápido)

Treina modelo usando hiperparâmetros de `data/xgboost_hyperparameters.json` (~1 min):

```bash
# Treinar XGBoost com JSON ativo (assume dados já no PostgreSQL)
python main.py train

# Especificar versão personalizada
python main.py train --model-version 2.1.1

# Overwrite modelo existente (move antigo para models/archive/)
python main.py train --model-version 2.1.0 -ow

# Saída esperada:
# 🚀 MODO: TREINAMENTO
# ✅ Train: 227,845 linhas (394 fraudes)
# ✅ Test: 56,962 linhas (98 fraudes)
# 📊 Cross-Validation (StratifiedKFold k=5)...
#    Precision (CV): 0.8123 ± 0.0234
#    Recall (CV): 0.8456 ± 0.0189
# ✅ Modelo salvo: models/xgboost_v2.1.0.pkl
#    PR-AUC: 0.8772
#    Precision: 86.60%
#    False Positives: 13
```

**Versionamento Automático:**
- **Sem `-ow`**: Salva em `models/archive/xgboost_v{version}_{timestamp}.pkl`
- **Com `-ow`**: Move modelo antigo para archive e cria novo em `models/`

**Quando usar**: Treinar rapidamente com parâmetros conhecidos bons (JSON ativo).

---

### Modo 3: Tune (Grid Search Automático)

Executa **Grid Search** (~20 min) e **atualiza `data/xgboost_hyperparameters.json`** automaticamente:

```bash
# Grid Search (modo archive - padrão)
python main.py tune

# Grid Search + overwrite do JSON ativo
python main.py tune -ow

# Especificar versão otimizada
python main.py tune --model-version 3.0.0 -ow

# Saída esperada:
# 🔧 MODO: TUNING (Grid Search)
# ⚠️  ATENÇÃO: Grid Search pode levar ~20 minutos
# ✅ Train: 227,845 linhas
# 
# 📌 Versões anteriores disponíveis:
#    ✅ v2.0.0: PR-AUC 0.8847, Precision 85.42%, FP 14, Features 37
#       v1.1.0: PR-AUC 0.8719, Precision 72.27%, FP 33, Features 37
# 
# 🔧 Executando Grid Search...
#    Combinações: 729 (3×3×3×3×3×3×1)
#    CV: StratifiedKFold k=3
#    Total fits: 2,187
# ✅ Grid Search completo!
# 🏆 Melhores Parâmetros:
#    colsample_bytree: 0.7
#    learning_rate: 0.3
#    max_depth: 6
#    ...
#    PR-AUC: 0.8772
# 
# � Atualizando configs.py com melhores hiperparâmetros...
#    Backup criado: configs_backup_20251003_143522.py
#    ✅ configs.py atualizado!
```

**Quando usar**: 
- Otimizar hiperparâmetros após mudanças no dataset (novas features, etc.)
- Experimentar novos grids de busca

**Segurança**:
- Cria **backup automático** de `configs.py` antes de atualizar
- Modo `--auto-update` para pipelines CI/CD
- Modo interativo pede confirmação antes de atualizar

---

### Modo 4: Predict (Inferência)

Executa predições em **CSV** ou **JSON**:

**Opção A: CSV**
```bash
python main.py predict --file data/new_transactions.csv

# Saída esperada:
# 🔮 MODO: PREDIÇÃO (CSV)
# 📥 Carregando modelo: xgboost_v2.1.0.pkl
# ✅ 1,000 transações carregadas
# ✅ Predições completas!
# 📊 Resumo:
#    Total transações: 1,000
#    Fraudes detectadas: 3 (0.30%)
#    Legítimas: 997
# 💾 Resultados salvos: data/new_transactions_predictions.csv
#
# ⚠️  Top 5 Fraudes com Maior Probabilidade:
#    Linha 42: 95.23% probabilidade
#    Linha 157: 89.45% probabilidade
#    Linha 891: 78.12% probabilidade
```

**Opção B: JSON (arquivo)**
```bash
python main.py predict --json-file input.json

# input.json exemplo:
# [
#   {"V1": -0.5, "V2": 1.2, "V3": 0.8, ..., "Amount_Log": 4.5},
#   {"V1": 0.3, "V2": -0.9, "V3": 1.1, ..., "Amount_Log": 3.2}
# ]

# Saída:
# 🔮 MODO: PREDIÇÃO (JSON)
# ✅ 2 transações carregadas
# 📋 Resultados:
#    🚨 Transação 0: FRAUDE (95.23%)
#    ✅ Transação 1: LEGÍTIMA (2.45%)
# 💾 Resultados salvos: prediction_results.json
```

**Opção C: JSON (inline)**
```bash
python main.py predict --json '{"V1": -0.5, "V2": 1.2, "V3": 0.8, "Amount_Log": 4.5}'

# Saída:
# 🔮 MODO: PREDIÇÃO (JSON)
# ✅ 1 transação carregada
# 📋 Resultados:
#    🚨 Transação 0: FRAUDE (89.34%)
```

**Quando usar**:
- **CSV**: Batch processing de muitas transações
- **JSON file**: Integração com APIs/sistemas externos
- **JSON inline**: Testes rápidos, CI/CD pipelines

---

## 📚 Documentação Completa

### Planos Estratégicos
- **`docs/plan_main.md`**: Plano MVP completo com XGBoost v1.1.0, v2.0.0, v2.1.0
- **`docs/plan_kafka.md`**: Enhancement Kafka (opcional, não implementado)
- **`docs/changelog.md`**: Registro de todas as mudanças

### Relatórios Técnicos
- **`docs/EDA_REPORT.md`**: Análise exploratória de dados completa
- **`docs/MODEL_SELECTION.md`**: Comparação de 4 modelos + 3 versões XGBoost
- **`docs/DATA_ARCHITECTURE.md`**: Arquitetura PostgreSQL + Pickle
 
### Arquitetura MVC-ML
- **`src/ml/README.md`**: Documentação completa da arquitetura modular

---
    
---

## 🔧 Configurações Importantes

### `src/ml/models/configs.py`

```python
# Hiperparâmetros XGBoost v2.1.0 (Production)
xgboost_params = {
    'colsample_bytree': 0.7,      # 70% features por árvore
    'learning_rate': 0.3,          # Taxa de aprendizado alta
    'max_depth': 6,                # Árvores profundas (não-linear)
    'min_child_weight': 1,         # Mínimo peso folha
    'n_estimators': 100,           # 100 árvores
    'scale_pos_weight': 577,       # Compensa desbalanceamento
    'subsample': 0.7,              # 70% amostras por árvore
    'eval_metric': 'aucpr'         # PR-AUC como métrica
}

# Features Removidas (Config-Driven)
excluded_features = [
    'Time',              # Offset temporal sem sentido real
    'Time_Period_of_Day', # Derivada de Time
    'V8', 'V23',         # Baixa correlação (<0.01)
    'Amount',            # Amount_Log é melhor
    'V13'                # Redundante (alta correlação com V17)
]
```

---

## 📊 Métricas de Performance

### Pipeline de Tratamento de Dados (52% mais rápido ⚡)

| Step | Antes | Depois | Ganho | Otimização |
|------|-------|--------|-------|------------|
| **04 Normalize** | 90s | 16.5s | **81.6%** 🔥 | PostgreSQL COPY |
| **02+03 Parallel** | 4.8s | 3.9s | **18.6%** ⚡ | ThreadPoolExecutor |
| **Pipeline Total** | ~130s | **~62s** | **52%** 🚀 | Multi-otimização |

### Modelos XGBoost (Test Set)

| Métrica | v1.1.0 | v2.0.0 | v2.1.0 ⭐ |
|---------|--------|--------|----------|
| **PR-AUC** | 0.8719 | 0.8847 | 0.8772 |
| **ROC-AUC** | 0.9738 | 0.9801 | 0.9765 |
| **Precision** | 72.27% | 85.42% | **86.60%** |
| **Recall** | 87.76% | 83.67% | 81.63% |
| **F1-Score** | 79.26% | 84.54% | 84.04% |
| **True Positives** | 86 | 82 | 80 |
| **False Positives** | 33 | 14 | **13** |
| **False Negatives** | 12 | 16 | 18 |
| **True Negatives** | 56,831 | 56,850 | 56,851 |

---

## 🚧 Próximos Passos (Opcional)
- [ ] **SHAP Analysis**: Explicabilidade de predições individuais
- [ ] **Kafka Streaming**: Real-time fraud detection (ver `docs/plan_kafka.md`)
- [ ] **Dashboard Flask**: Interface web para inferência interativa
- [ ] **Docker Full Stack**: Container com PostgreSQL + Flask + ML
- [ ] **CI/CD Pipeline**: GitHub Actions para testes + deploy automático

---

## 📖 Referências

- **Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **XGBoost**: Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"
- **PR-AUC**: Saito & Rehmsmeier (2015) - "The Precision-Recall Plot Is More Informative"
- **RobustScaler**: Scikit-learn - Normalização resistente a outliers (mediana + IQR)
- **PostgreSQL COPY**: PostgreSQL Documentation - Bulk loading optimization

---

## 👥 Autores
- Victor Lucas Santos de Oliveira
- Adrianny Lelis da Silva

--- 


## 📊 Treinamento de Modelos
  
## 4. EDA (Análise Exploratória)
```bash
# Gráficos gerados em docs/images/
ls docs/images/*.png
```

---

## 📁 Estrutura do Projeto

```
ml-fraud-detector/
├── main.py                      # 🎯 CLI principal (pipeline/train/tune/predict)
├── pyproject.toml               # Configuração do projeto (uv/pip)
├── requirements.txt             # Dependências Python
├── README.md                    # Este arquivo
│
├── data/                        # 📊 Datasets e configurações
│   ├── creditcard.csv           # Dataset original (284,807 transações)
│   ├── xgboost_hyperparameters.json  # Hiperparâmetros ativos
│   ├── archive/                 # Versões antigas de hiperparâmetros
│   └── examples/                # Exemplos de transações para teste
│       ├── fraud_transaction.json
│       └── legitimate_transaction.json
│
├── models/                      # 🤖 Modelos treinados
│   ├── scalers.pkl              # RobustScaler + StandardScaler
│   ├── xgboost_v2.1.0.pkl       # ⭐ Modelo em produção
│   └── archive/                 # Versões antigas de modelos
│
├── reports/                     # 📈 Relatórios gerados (JSON)
│   └── feature_selection_report.json
│
├── database/                    # 🗄️ Configuração PostgreSQL
│   ├── docker-compose.yml       # Docker setup (PostgreSQL 15)
│   └── schema.sql               # Schema completo (7 tabelas pipeline + 3 metadata)
│
├── docs/                        # 📚 Documentação técnica
│   ├── EDA_REPORT.md            # Análise exploratória
│   ├── MODEL_SELECTION.md       # Comparação de 4 modelos
│   ├── DATA_ARCHITECTURE.md     # Arquitetura PostgreSQL + Pickle
│   ├── DECISOES_TECNICAS.md     # Decisões ML + otimizações
│   ├── TRANSACTION_EXAMPLES.md  # Exemplos práticos de transações (fraude vs legítima)
│   └── images/                  # Gráficos EDA
│       ├── 01_class_distribution.png
│       ├── 02_amount_analysis.png
│       ├── 03_correlation_heatmap.png
│       └── ...
│
└── src/                         # 💻 Código fonte
    ├── __init__.py
    │
    ├── ml/                      # 🧠 Machine Learning
    │   ├── __init__.py
    │   ├── README.md            # Documentação da arquitetura MVC-ML
    │   │
    │   ├── processing/          # 🔧 Funções de processamento
    │   │   ├── __init__.py
    │   │   ├── loader.py        # PostgreSQL COPY (otimizado 81.6%)
    │   │   ├── validation.py    # Schema e integridade
    │   │   ├── cleaning.py      # Análise de outliers
    │   │   ├── normalization.py # RobustScaler + StandardScaler
    │   │   ├── feature_engineering.py  # 9 novas features
    │   │   ├── feature_selection.py    # Análise automatizada
    │   │   ├── splitters.py     # Stratified train/test split
    │   │   └── metadata.py      # Pipeline metadata tracking
    │   │
    │   ├── pipelines/           # 🚀 Orquestração
    │   │   ├── __init__.py
    │   │   └── data_pipeline.py # Steps 01-07 (completo)
    │   │
    │   ├── training/            # 🎓 Treinamento ML
    │   │   ├── __init__.py
    │   │   ├── train.py         # Treino com JSON configs
    │   │   └── tune.py          # Grid Search + auto-update JSON
    │   │
    │   └── models/              # ⚙️ Configurações
    │       ├── __init__.py
    │       └── configs.py       # Dataclasses + carregamento JSON
    │
    ├── models/                  # 📦 Data models
    │   ├── __init__.py
    │   └── database_models.py   # SQLAlchemy models
    │
    └── services/                # 🔌 Serviços
        └── database/            # 🗄️ Conexão PostgreSQL
            ├── __init__.py
            └── connection.py    # Engine + connection pooling
```

---

## 🔄 Pipeline de Tratamento de Dados (Modular)

Cada step é **isolado e reutilizável**, podendo ser orquestrado como **DAG no Airflow**:

### ✅ Step 01: Load Raw Data
- **Input**: `data/creditcard.csv`
- **Output**: PostgreSQL `raw_transactions` (284,807 linhas)
- **Função**: Carregar e validar schema

### ✅ Step 02: Remove Outliers
- **Input**: `raw_transactions`
- **Output**: `cleaned_transactions` (284,804 linhas)
- **Função**: Remover Amount > $15k (3 transações, todas legítimas)
- **Justificativa**: 73.6% das fraudes têm Amount < $100 (mediana: $9.25)

### ✅ Step 03: Handle Missing Values
- **Input**: `cleaned_transactions`
- **Output**: `imputed_transactions`
- **Função**: Imputação (mediana) ou remoção (Class)

### ✅ Step 04: Normalize Features 
- **Input**: `imputed_transactions`
- **Output**: `normalized_transactions` + `models/scalers.pkl`
- **Função**: RobustScaler (Amount), StandardScaler (Time)

### ✅ Step 05: Feature Engineering 
- **Input**: `normalized_transactions`
- **Output**: `engineered_transactions`
- **Função**: Time_Period, Amount_Bin, Amount_Log, V_Interactions

### ✅ Step 06: Train/Test Split
- **Input**: `engineered_transactions`
- **Output**: `train_features`, `test_features`, `train_target`, `test_target`
- **Função**: StratifiedShuffleSplit (80/20)

📚 **Documentação completa**: [`src/ml/README.md`](src/ml/README.md)

---
## 🤖 Modelos de Machine Learning

**Relatório Completo**: [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md) - Análise comparativa completa de 4 algoritmos + 3 versões XGBoost

### Abordagem Testada: Comparação de 4 Algoritmos
- ✅ **XGBoost v2.1.0** (⭐ **RECOMENDADO**): PR-AUC 0.8772, Precision 86.60%, 13 FP
- ✅ **XGBoost v2.0.0** (Grid Search): PR-AUC 0.8847, Precision 85.42%, 14 FP  
- ✅ **XGBoost v1.1.0** (Baseline): PR-AUC 0.8719, Precision 72.27%, 33 FP
- ❌ **Decision Tree**: PR-AUC 0.7680, Precision 24.24%, 250 FP (rejeitado)
- ❌ **LightGBM**: PR-AUC 0.0372, Precision 3.36%, 2,475 FP (falha catastrófica)
- ❌ **SVM RBF**: PR-AUC 0.5326, Precision 19.50%, 339 FP (muito lento: 15 min)
---

## 📊 Análise Exploratória (EDA)
### Insights Principais

**Desbalanceamento**:
- Fraudes: 492 (0.172%)
- Legítimas: 284.315 (99.828%)
- Ratio: 1:578

**Amount (Valores)**:
- Fraudes: Mediana $9.25, Média $122.21
- 73.6% das fraudes têm Amount < $100
- Nenhuma fraude tem Amount > $15k

**Correlações (Top 3)**:
- V17: -0.326 (mais forte!)
- V14: -0.302
- V12: -0.261

**Temporal**:
- Fraudes concentradas em certos horários
- Taxa de fraude varia 10x ao longo do dia

📊 **Gráficos**: `docs/images/*.png`  
📋 **Relatório Completo**: [`docs/EDA_REPORT.md`](docs/EDA_REPORT.md) - Análise detalhada com decisões técnicas e justificativas

---

## 🗄️ PostgreSQL Schema

```sql
-- Tabelas do Pipeline
raw_transactions         (284,807 linhas) - CSV original
cleaned_transactions     (284,804 linhas) - Sem outliers
imputed_transactions     (284,804 linhas) - Sem missing
normalized_transactions  (284,804 linhas) - Features normalizadas
engineered_transactions  (284,804 linhas) - Features novas
train_features          (~227,843 linhas) - 80% treino
test_features           (~56,961 linhas)  - 20% teste

-- Tabelas de Metadados
metrics_history          - Métricas de modelos ao longo do tempo
trained_models           - Registro de modelos treinados
data_splits              - Histórico de splits train/test
```

---

## 🎯 Status do Projeto

### ✅ Concluído
- [x] Setup PostgreSQL (Docker Compose)
- [x] Conexão SQLAlchemy testada
- [x] EDA completo com OOP (5 classes)
- [x] Pipeline Step 01: Load Raw Data
- [x] Pipeline Step 02: Remove Outliers
- [x] Pipeline Step 03: Handle Missing Values
- [x] Documentação (plan_main.md, plan_kafka.md, changelog.md)

### 🔄 Em Progresso
- [ ] Revisão do codigo

### 📋 Próximos Passos
- [ ] Dashboard Flask
- [ ] Kafka Streaming (opcional)
- [ ] Vídeo explicativo

---

## 📚 Documentação

- **[Plan Main](docs/plan_main.md)**: Plano completo do MVP
- **[Plan Kafka](docs/plan_kafka.md)**: Enhancement com streaming (opcional)
- **[Changelog](docs/changelog.md)**: Registro de mudanças
- **[Decisões Técnicas](docs/DECISOES_TECNICAS.md)**: Justificativas metodológicas
- **[Pipeline README](src/data_processing/README.md)**: Documentação do pipeline

---

## 🔗 Referências

- **Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Airflow**: https://airflow.apache.org/
- **Scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/
- **PostgreSQL**: https://www.postgresql.org/

---

## 👥 Autores

**Victor Lucas Santos de Oliveira** - [LinkedIn](https://www.linkedin.com/in/vlso/)

**Adrianny Lelis da Silva** - [LinkedIn](https://www.linkedin.com/in/adriannylelis/)

---
  