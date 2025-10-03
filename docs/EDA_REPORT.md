# 📊 EDA Report - Fraud Detection Pipeline

**Data**: 2025-10-02  
**Pipeline**: `src/ml/processing/` (Arquitetura MVC-ML otimizada)  
**Dataset**: creditcard.csv (284,807 transações, 492 fraudes)

---

## 🎯 Objetivo da EDA

Entender padrões de fraude e características do dataset para tomar **decisões técnicas fundamentadas** no pipeline de dados.

---

## 📈 1. Análise de Desbalanceamento de Classes

### Distribuição
```
Classe 0 (Legítima): 284,315 (99.828%)
Classe 1 (Fraude):       492 ( 0.172%)
Ratio: 1:578
```

### 🎯 Decisão Técnica
**NÃO usar SMOTE ou Undersampling**

**Justificativa**:
- ❌ **SMOTE**: Infla dataset artificialmente, risco de overfitting
- ❌ **Undersampling**: Perde 99.6% dos dados legítimos

**Solução Adotada**:
- ✅ `class_weight='balanced'` (Decision Tree, SVM)
- ✅ `scale_pos_weight=577` (XGBoost)
- ✅ **Threshold Tuning** pós-treino (0.2-0.3 em vez de 0.5)
- ✅ Priorizar **Recall** > Precision (melhor bloquear legítima que deixar fraude passar)
- ❌ **Regressão Logística removida**: Sensível a outliers (assume linearidade e distribuição normal)

---

## 💰 2. Análise de Amount (Valores de Transação)

### Estatísticas Gerais
```
Métrica          Legítimas    Fraudes
Média            $88.29       $122.21
Mediana          $22.00       $9.25
Q1 (25%)         $5.65        $1.00
Q3 (75%)         $77.17       $105.89
Máximo           $25,691.16   $2,125.87
```

### 📊 Distribuição de Fraudes por Faixa
```
Amount < $100:      362 fraudes (73.6%)
$100 - $500:         95 fraudes (19.3%)
$500 - $1000:        26 fraudes ( 5.3%)
$1000 - $15k:         9 fraudes ( 1.8%)
> $15k:               0 fraudes ( 0.0%)
```

### 🔍 Análise de Outliers (IQR Method)

**Método Tukey (IQR)**:
```
Q1 = $5.60
Q3 = $77.16
IQR = $71.56
Lower Bound = Q1 - 1.5*IQR = -$101.75
Upper Bound = Q3 + 1.5*IQR = $184.51

Outliers detectados: 31,904 transações (11.2%)
  - Outliers baixos (< -$101.75): 0
  - Outliers altos (> $184.51): 31,904
```

**Fraudes em Outliers**:
```
Total de fraudes: 492
Fraudes em outliers: 91 (18.5%)
Fraudes preservadas se remover: 401 (81.5%)
```

### 🎯 DECISÃO CRÍTICA: NÃO REMOVER OUTLIERS!

**Análise**:
- ✅ Dataset tem apenas **492 fraudes** (já extremamente desbalanceado)
- ✅ **91 fraudes (18.5%)** estão em outliers (Amount > $184.51)
- ✅ Estas são fraudes de **ALTO VALOR** (padrão diferente, mas válido)
- ✅ Remover = perder capacidade de detectar fraudes > $180
- ✅ IQR é método científico, mas **contexto importa mais que fórmula**

**Justificativa Estatística**:
1. **Cada fraude é PRECIOSA**: Dataset pequeno, informação rara
2. **Fraudes de alto valor existem**: Não são erros, são padrões reais
3. **Perda inaceitável**: 18.5% das fraudes = 91 exemplos perdidos
4. **Alternativas melhores**: RobustScaler + modelos não-lineares

**Solução Adotada**:
- ✅ **Manter TODAS as transações** (incluindo outliers)
- ✅ **Step 04**: Usar `RobustScaler` para Amount (resistente a outliers)
- ✅ **Modelos não-lineares**: XGBoost e Decision Tree lidam bem com outliers
- ✅ **Threshold tuning**: Otimizar para capturar fraudes raras (valores extremos)

**Storytelling**:
> "Analisamos outliers usando IQR (método Tukey) e identificamos 31.904 transações extremas. No entanto, 91 fraudes (18.5% do total) estão nesta faixa. Como o dataset já tem apenas 492 fraudes, decidimos MANTER todos os dados. Cada fraude carrega informação preciosa sobre padrões de comportamento fraudulento. Nossa solução: usar RobustScaler (resistente a outliers) e modelos não-lineares (XGBoost) que capturam relações complexas sem precisar descartar dados valiosos."

---

## 🔗 3. Análise de Correlações com Target (Class)

### Métodos Utilizados
1. **Pearson** (correlação linear)
2. **Spearman** (correlação monotônica)
3. **Mutual Information** (relações não-lineares gerais)

### Top 10 Features (Ordenadas por Correlação Média)

| Rank | Feature | Pearson (Linear) | Spearman (Monotônica) | Mutual Info (Geral) | Média |
|------|---------|------------------|-----------------------|---------------------|-------|
| 1    | V14     | 0.3025           | 0.0646                | 0.0081              | 0.1251|
| 2    | V17     | 0.3265           | 0.0424                | 0.0060              | 0.1250|
| 3    | V12     | 0.2606           | 0.0629                | 0.0076              | 0.1104|
| 4    | V10     | 0.2169           | 0.0596                | 0.0075              | 0.0947|
| 5    | V16     | 0.1965           | 0.0499                | 0.0061              | 0.0842|
| 6    | V11     | 0.1549           | 0.0601                | 0.0068              | 0.0740|
| 7    | V3      | 0.1930           | 0.0382                | 0.0047              | 0.0786|
| 8    | V7      | 0.1873           | 0.0395                | 0.0048              | 0.0772|
| 9    | V18     | 0.1115           | 0.0303                | 0.0038              | 0.0485|
| 10   | V4      | 0.1334           | 0.0343                | 0.0040              | 0.0572|

### Bottom 10 Features (Candidatos a Análise)

| Rank | Feature | Pearson | Spearman | Mutual Info | Média  |
|------|---------|---------|----------|-------------|--------|
| 26   | V13     | 0.0046  | 0.0037   | 0.0004      | 0.0029 |
| 27   | V15     | 0.0042  | 0.0028   | 0.0003      | 0.0024 |
| 28   | V22     | 0.0074  | 0.0065   | 0.0007      | 0.0049 |
| 29   | Amount  | 0.0056  | 0.0083   | 0.0014      | 0.0051 |
| 30   | Time    | 0.0123  | 0.0117   | 0.0019      | 0.0086 |

### 🎯 Decisão: Remover V13,V15, V22 com Baixa Correlação Linear

**Por quê?**

#### ❌ Correlação Linear Baixa ≠ Feature Inútil!

| Cenário | Correlação Linear | Utilidade no ML |
|---------|-------------------|-----------------|
| Relação não-linear | Baixa (0.005) | ✅ **ALTA!** (XGBoost captura) |
| Interações com outras features | Baixa individual | ✅ **ALTA!** (Amount × Time) |
| Importância por faixa | Baixa no geral | ✅ **MÉDIA** (fraudes >$500 têm padrão) |
| Redundância com PCA | Baixa | ❌ **BAIXA** (V1-V28 já capturam) |

#### Exemplo: Amount e Time

**Amount**:
- Correlação Pearson: **0.0056** (quase zero!)
- Correlação Spearman: **0.0083** (ainda baixíssima)
- Mutual Info: **0.0014** (quase nenhuma relação)

**MAS**:
- ✅ 73.6% das fraudes têm Amount < $100 (padrão claro!)
- ✅ Feature engineering: `Amount_Bin` (categorização) pode revelar padrões
- ✅ Interação com Time: `Amount × Time_Period` (fraudes altas à noite?)
- ✅ XGBoost pode descobrir splits não-lineares

**Time**:
- Correlação linear baixa (0.0123)
- **MAS**: Taxa de fraude varia **10x** ao longo do dia!
- Feature engineering: `Time_Period` (manhã/tarde/noite/madrugada)

### 💡 Estratégias Corretas

1. **Feature Engineering**: 
   - `Time_Period` (binning temporal)
   - `Amount_Bin` (categorização de valores)
   - `Amount_Log` (log-transform para normalizar distribuição)
   - Estatísticas V-features (Mean, Std, Min, Max, Range)

2. **Modelos Não-Lineares**: 
   - XGBoost, Random Forest capturam padrões complexos
   - Decision Tree faz splits automáticos (não precisa de correlação linear)

3. **Feature Importance**: 
   - Deixar o modelo decidir (não descartar a priori)
   - Usar `feature_importances_` após treino

4. **Threshold Tuning**: 
   - Otimizar decisão final (0.2-0.3 em vez de 0.5)
   - Priorizar Recall (capturar fraudes mesmo com baixa confiança)

---

## ⏰ 4. Análise Temporal

### Distribuição de Transações ao Longo do Tempo
- Dataset cobre **2 dias** (172.800 segundos)
- Transações concentradas em **horários comerciais**
- Gaps noturnos (períodos sem transações)

### Taxa de Fraude ao Longo do Tempo
```
Horário (aproximado)    Taxa de Fraude
00:00 - 06:00           ~1.2% (PICO!)
06:00 - 12:00           ~0.1%
12:00 - 18:00           ~0.1%
18:00 - 24:00           ~0.3%
```

### 🎯 Insight Crítico
**Taxa de fraude é 10x maior à madrugada!**

**Implicações**:
- ✅ Feature engineering: `Time_Period` (categorizar por período do dia)
- ✅ Modelo pode ajustar threshold dinamicamente (horário)
- ✅ Em produção: sistema pode ser mais agressivo à noite

---

## 📊 5. Pipeline de Dados - Decisões Técnicas (ATUALIZADO - OTIMIZADO)

### Step 01: Load Raw Data ✅
- **Input**: `data/creditcard.csv`
- **Output**: PostgreSQL `raw_transactions` (284,807 linhas)
- **Validação**: Schema (31 colunas), integridade
- **Performance**: ~20s (usando PostgreSQL COPY otimizado)
- **Função**: `src/ml/processing/loader.py::save_to_postgresql()` (com use_copy=True)

### Step 02: Análise de Outliers (SEM Remoção) ✅
- **Entrada**: `raw_transactions`
- **Saída**: **Metadata apenas** (dados NÃO duplicados!)
- **Função**: `src/ml/processing/cleaning.py::identify_outliers()`
- **Método**: IQR (Tukey)
- **Outliers detectados**: 31,904 (11.2%)
- **Fraudes em outliers**: 91 (18.5% do total)
- **DECISÃO**: ❌ **NÃO REMOVER** (manter todas as fraudes, senão pode não detectar fraudes de alto valor)
- **Solução**: RobustScaler no Step 04
- **Performance**: ~2.4s
- **Impacto nos Modelos**:
  - ✅ **Decision Tree**: Robusto a outliers (splits baseados em ranking)
  - ✅ **SVM**: RobustScaler normaliza, kernel RBF captura não-linearidades
  - ✅ **XGBoost**: Extremamente robusto (árvores + regularização)
  - ❌ **Regressão Logística REMOVIDA**: Assume linearidade, muito sensível a outliers

### Step 03: Handle Missing Values ✅
- **Entrada**: `raw_transactions`
- **Saída**: **Metadata apenas** (dados não tem missing values!)
- **Função**: `src/ml/processing/validation.py::validate_no_missing_values()`
- **Resultado**: Nenhum missing value encontrado (dataset 100% completo)
- **Performance**: ~2.4s
- **Paralelização**: Roda em paralelo com Step 02 (ThreadPoolExecutor)
- **Estratégias** (caso houvesse missing):
  - Class: Remover linha (crítico)
  - Time: Imputar com mediana
  - Amount: Imputar com mediana por classe
  - V1-V28: Imputar com mediana

### Step 04: Normalize Features ✅ **[NOVO - OTIMIZADO]**
- **Entrada**: `raw_transactions` (Steps 02-03 não modificaram dados)
- **Saída**: `normalized_transactions` + `models/scalers.pkl`
- **Função**: `src/ml/processing/normalization.py::fit_and_transform_features()`
- **Scalers**: 
  - **RobustScaler** para Amount (resistente a outliers - usa mediana e IQR)
  - **StandardScaler** para Time (distribuição normal)
  - **V1-V28**: Mantidos (já são PCA, normalizados)
- **Performance**: ~16.5s (otimizado com PostgreSQL COPY - antes era 90s!)
- **Justificativa RobustScaler**: 
  - Usa mediana e IQR (não média/desvio)
  - Não afetado por outliers
  - Preserva 100% dos dados
  - Ideal para fraudes de alto valor

### Step 05: Feature Engineering ✅ **[NOVO]**
- **Entrada**: `normalized_transactions`
- **Saída**: `engineered_transactions` (284,807 linhas × 40 colunas)
- **Função**: `src/ml/processing/feature_engineering.py::engineer_all_features()`
- **Performance**: ~20.6s (com PostgreSQL COPY otimizado)
- **Features Criadas** (9 novas):
  1. **Time_Period**: Categorização temporal (4 bins)
     - Madrugada (0-6h), Manhã (6-12h), Tarde (12-18h), Noite (18-24h)
     - Captura padrão: fraudes 10x mais frequentes à madrugada
  
  2. **Amount_Log**: Log transformation de Amount
     - Normaliza distribuição assimétrica
     - Reduz impacto de valores extremos
  
  3. **Amount_Bin**: Categorização de valores (4 bins)
     - <$10, $10-100, $100-500, >$500
     - Captura padrão: 73.6% fraudes < $100
  
  4. **V_Features_Mean**: Média de V1-V28
     - Feature agregada para capturar tendência central
  
  5. **V_Features_Std**: Desvio padrão de V1-V28
     - Captura variabilidade entre features PCA
  
  6. **V_Features_Min**: Mínimo de V1-V28
     - Captura valores extremos negativos
  
  7. **V_Features_Max**: Máximo de V1-V28
     - Captura valores extremos positivos
  
  8. **V_Features_Range**: Range (Max - Min) de V1-V28
     - Captura amplitude de variação
  
  9. **(Opcional) V-Interactions**: Interações entre top features
     - V17×V14, V17×V12, V14×V12
     - Captura relações não-lineares

- **Decisão de Design**:
  - Não criar interações por padrão (explosão combinatória)
  - Deixar modelos não-lineares (XGBoost) descobrirem
  - Manter simplicidade e interpretabilidade

### Step 06: Train/Test Split ✅ **[NOVO]**
- **Entrada**: `engineered_transactions` (284,807 linhas × 40 colunas)
- **Saída**: `train_data` (227,845 linhas) + `test_data` (56,962 linhas)
- **Função**: `src/ml/processing/splitters.py::stratified_train_test_split()`
- **Performance**: ~22.2s (com PostgreSQL COPY para ambas tabelas)
- **Método**: **StratifiedShuffleSplit** (mantém proporção de classes)
- **Split**: 80/20
- **Validações**:
  - ✅ **Zero data leakage**: Índices únicos (sem overlap)
  - ✅ **Proporções preservadas**: 
    - Train: 394 fraudes (0.173%)
    - Test: 98 fraudes (0.172%)
    - Diferença: 0.0009% (praticamente idêntico!)
  - ✅ **random_state=42**: Reprodutibilidade garantida
- **Armazenamento**: 
  - PostgreSQL `train_data` e `test_data` tables
  - Rastreabilidade completa (metadata com contagens)

---

## 🚀 6. Otimizações de Performance Implementadas

### PostgreSQL COPY (81.6% mais rápido)
- **Antes**: `pandas.to_sql()` - 90s para Step 04
- **Depois**: PostgreSQL COPY - 16.5s para Step 04
- **Ganho**: 73.5s economizados (81.6% redução!)
- **Implementação**: `src/ml/processing/loader.py::save_to_postgresql()`
- **Fallback**: to_sql() automático se COPY falhar

### Paralelização Steps 02-03 (18.6% mais rápido)
- **Antes**: Sequencial - 4.79s (2.43s + 2.36s)
- **Depois**: Paralelo - 3.90s (max(2.43s, 2.36s))
- **Ganho**: 0.89s economizados
- **Implementação**: ThreadPoolExecutor em `data_pipeline.py`
- **Segurança**: Ambos read-only, sem race conditions

### Metadata-Only para Steps sem Alteração
- **Steps 02-03**: Salvam apenas metadata, NÃO duplicam dados
- **Benefício**: 
  - Zero overhead de I/O desnecessário
  - Rastreabilidade mantida (tabela `pipeline_metadata`)
  - Dados permanecem em `raw_transactions`

### Performance Total do Pipeline
- **Antes otimização**: ~130s (Steps 01-04)
- **Depois otimização**: ~62s (Steps 01-06 completo!)
- **Ganho Total**: **52% mais rápido** 🚀

---

## 🔍 8. Feature Selection Strategy (Config-Driven)

### Por que Correlação Linear Baixa ≠ Feature Inútil?

Durante a EDA, identificamos features com correlação Pearson/Spearman extremamente baixa:
- V13: 0.0046 (Pearson), 0.0037 (Spearman)
- V15: 0.0042 (Pearson), 0.0028 (Spearman)
- V22: 0.0074 (Pearson), 0.0065 (Spearman)

**Decisão**: NÃO descartar automaticamente!

### Razões para Manter Features com Baixa Correlação Linear

1. **Relações Não-Lineares**: XGBoost e Decision Tree capturam padrões que correlação linear não detecta
2. **Interações**: Feature A × Feature B pode ser discriminativa mesmo que A e B individualmente não sejam
3. **Importância por Faixa**: Feature pode ser importante apenas em valores extremos (outliers)
4. **Contexto de Negócio**: Cada fraude é preciosa (apenas 492 no dataset)

### Abordagem Implementada: Análise Automatizada + Decisão Manual

#### Step 05.5: Feature Selection Analysis
**Objetivo**: Informar decisão (NÃO automatizar remoção)

**Análises Realizadas**:
1. **Pearson Correlation**: Relações lineares
2. **Spearman Correlation**: Relações monotônicas
3. **Mutual Information**: Relações não-lineares gerais
4. **VIF (Variance Inflation Factor)**: Multicolinearidade entre features
5. **Correlation Matrix**: Redundância (features com correlação > 0.95)

**Output**: `reports/feature_selection_analysis.json`

```json
{
  "total_features": 40,
  "candidates_for_removal": {
    "low_correlation_all_methods": ["V13", "V15", "V22"],
    "count": 3,
    "features_detail": {
      "V13": {
        "pearson": 0.0046,
        "spearman": 0.0037,
        "mutual_info": 0.0004
      }
    }
  },
  "multicollinearity": {
    "high_vif_features": [],
    "redundant_pairs": []
  },
  "recommendation": {
    "action": "Review and update configs.py",
    "suggested_removals": ["V13", "V15", "V22"]
  }
}
```

#### Step 06: Apply Feature Selection (Config-Driven)
**Objetivo**: Aplicar decisões manuais automaticamente

**Configuração**:
```python
# src/ml/models/configs.py
@dataclass
class FeatureSelectionConfig:
    excluded_features: List[str] = field(default_factory=lambda: [
        # 'V13',  # Baixa correlação (decisão: 2025-10-02)
        # 'V15',  # Baixa correlação (decisão: 2025-10-02)
        # 'V22',  # Baixa correlação (decisão: 2025-10-02)
    ])
    
    model_version: str = "1.0.0"  # Incrementar ao alterar lista
```

**Workflow**:
1. Executar pipeline → gera relatório JSON
2. Revisar relatório + considerar contexto de negócio
3. Editar `configs.py` (descomentar features a remover)
4. Incrementar `model_version`
5. Re-executar pipeline → Step 06 aplica remoção
6. Treinar modelos → comparar performance
7. Se melhor: commit configs; Se pior: reverter

### Vantagens da Abordagem Config-Driven

| Aspecto | Vantagem |
|---------|----------|
| **Rastreabilidade** | Git versiona decisões |
| **Reprodutibilidade** | Mesma config = mesmo resultado |
| **A/B Testing** | Fácil comentar/descomentar features |
| **Context-Aware** | Considera contexto de negócio |
| **Educativo** | Relatório ensina sobre feature importance |
| **Produção-Ready** | Abordagem usada em empresas reais |

### Quando Remover Features?

✅ **Candidatos Fortes**:
- Baixos em TODOS os 3 métodos (Pearson, Spearman, MI)
- VIF > 10 (multicolinearidade alta)
- Correlação > 0.95 com outra feature (redundância)
- Análise de negócio confirma irrelevância

❌ **Manter**:
- Baixa correlação linear MAS alta MI (não-linear!)
- Baixa correlação individual MAS participa de interações
- Contexto de negócio indica relevância (ex: Amount, Time)
- Dataset pequeno (cada feature pode ajudar)

### Versionamento de Modelos

**Pickle para Scalers**:
```python
# models/scalers_v1.1.0.pkl (após feature selection)
# Versionamento manual controlado por configs.py
```

---

## 🎯 9. Estratégia de Modelagem - MVP Pragmático

### Fase 1: Treinamento Inicial (39 Features)

**Decisão**: Começar com **39 features** (Time removido, Amount mantido)

**Justificativa**:
- ✅ **Pragmatismo**: Deixar modelos decidirem quais features importam
- ✅ **Feature Engineering Completo**: 9 novas features criadas
- ✅ **Dimensionalidade Saudável**: 39 features é aceitável para XGBoost/Decision Tree
- ✅ **A/B Testing**: Fácil comparar com/sem Amount posteriormente

**Features Removidas** (1):
- **Time** (cru): Apenas offset temporal no dataset, sem significado em produção

**Features Mantidas** (39):
1. **V1-V28** (28 features): PCA transformations (confidenciais)
2. **Amount Features** (3):
   - Amount (cru) - MANTIDO para testar se ajuda além de Amount_Log
   - Amount_Log - Normaliza distribuição assimétrica
   - Amount_Bin - Categorização por quartil (0-3)
3. **Time Features** (2):
   - Time_Hour - Hora do dia (0-47h)
   - Time_Period_of_Day - Período (0-3: madrugada/manhã/tarde/noite)
4. **V-Statistics** (5 features):
   - V_Features_Mean, Std, Min, Max, Range
5. **Target** (1): Class (0=legítima, 1=fraude)

### Fase 2: Modelos Robustos (3 Algoritmos)

**Decisão**: Treinar 3 modelos resistentes a outliers (Regressão Logística REMOVIDA)

| Modelo | Vantagens | Desvantagens | Uso |
|--------|-----------|--------------|-----|
| **Decision Tree** | ✅ Splits baseados em ranking<br>✅ Robusto a outliers<br>✅ Interpretável | ❌ Overfitting sem poda | ✅ **Baseline** |
| **SVM RBF** | ✅ RobustScaler pre-processing<br>✅ Kernel não-linear<br>✅ Margin-based | ❌ Lento para grandes datasets | ✅ **Produção** |
| **XGBoost** | ✅ Ensemble robusto<br>✅ Feature importance nativo<br>✅ Lida com outliers | ❌ Hiperparâmetros complexos | ✅ **Melhor Performance** |
| ~~Regressão Logística~~ | ❌ Assume linearidade<br>❌ Sensível a outliers (11.2%!)<br>❌ Assume distribuição normal | - | ❌ **REMOVIDO** |

**Configurações**:
```python
# Decision Tree
DecisionTreeClassifier(
    class_weight='balanced',    # Trata desbalanceamento 1:578
    max_depth=15,               # Evita overfitting
    min_samples_split=50,       # Mínimo para split
    random_state=42
)

# SVM RBF
SVC(
    kernel='rbf',               # Kernel não-linear
    class_weight='balanced',    # Trata desbalanceamento
    C=1.0,                      # Regularização
    gamma='scale',              # Auto-ajuste
    probability=True,           # Para threshold tuning
    random_state=42
)

# XGBoost
XGBClassifier(
    scale_pos_weight=577,       # Ratio 1:578 (desbalanceamento)
    max_depth=6,                # Profundidade árvore
    learning_rate=0.1,          # Taxa aprendizado
    n_estimators=100,           # Número de árvores
    subsample=0.8,              # Amostragem para robustez
    colsample_bytree=0.8,       # Feature sampling
    random_state=42
)
```

### Fase 3: Validação Cruzada Estratificada

**Método**: StratifiedKFold (k=5)

**Por quê?**:
- ✅ **Mantém proporção** 0.172% em cada fold
- ✅ **Evita folds sem fraudes** (crítico com 492 fraudes!)
- ✅ **Validação robusta** contra overfitting
- ✅ **Comparação justa** entre modelos

**Métricas**:
```python
# Ordem de prioridade (desbalanceamento extremo)
1. PR-AUC (Precision-Recall AUC)       # Métrica principal
2. Recall (Sensitivity)                # Capturar fraudes > precisão
3. F1-Score                            # Balanço geral
4. Precision                           # Evitar alarmes falsos
5. ROC-AUC                             # Métrica secundária
```

### Fase 4: Grid Search e Threshold Tuning

**Grid Search (PR-AUC scoring)**:
```python
# Decision Tree
param_grid = {
    'max_depth': [10, 15, 20, 25],
    'min_samples_split': [20, 50, 100],
    'min_samples_leaf': [10, 20, 50]
}

# SVM
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto', 0.001, 0.01]
}

# XGBoost
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0]
}
```

**Threshold Tuning**:
```python
# Otimizar threshold (não usar 0.5 padrão)
thresholds = np.arange(0.1, 0.9, 0.05)
best_threshold = optimize_for_recall(y_val, y_proba, thresholds)
# Tipicamente: 0.2-0.3 (priorizar Recall)
```

### Fase 5: Feature Importance Pós-Treino ⭐

**Objetivo**: Identificar features realmente inúteis (não achismo!)

**Método**:
```python
# XGBoost fornece feature importance nativo
importances = xgb_model.feature_importances_
feature_ranking = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# Identificar features com importance ≈ 0
useless_features = feature_ranking[feature_ranking['importance'] < 0.001]['feature'].tolist()
```

**Análise Esperada**:
- **Amount vs Amount_Log**: Qual o modelo realmente usa?
- **V13, V15, V22**: Correlação baixa, mas útil em não-linearidade?
- **V-Statistics**: Agregações ajudam?
- **Time_Hour vs Time_Period_of_Day**: Qual mais discriminativa?

### Fase 6: Refinamento (Se Necessário)

**A/B Testing**:
1. **Treinar com 39 features** (configuração atual)
2. **Extrair feature importance** (XGBoost)
3. **Remover features inúteis** (importance < 0.001)
4. **Re-treinar modelos**
5. **Comparar métricas**:
   - Se PR-AUC melhora ou mantém: ✅ Aceitar remoção
   - Se PR-AUC piora: ❌ Reverter (features tinham valor!)

**Exemplo**:
```python
# Configuração 1: 39 features
model_v1 = train_xgboost(X_39_features, y)
# PR-AUC: 0.85, Recall: 0.82

# Configuração 2: Remover Amount (testar se Amount_Log basta)
X_38_features = X_39_features.drop('Amount', axis=1)
model_v2 = train_xgboost(X_38_features, y)
# PR-AUC: 0.86, Recall: 0.83 (MELHOROU!)

# Decisão: Atualizar configs.py, incrementar model_version para 1.2.0
```

### Fase 7: Seleção do Melhor Modelo

**Critérios de Decisão**:
1. **PR-AUC** (peso 50%): Métrica principal para desbalanceamento
2. **Recall** (peso 30%): Prioridade é capturar fraudes
3. **F1-Score** (peso 20%): Balanço geral
4. **Tempo de Inferência**: Considerar em produção
5. **Interpretabilidade**: Decision Tree > XGBoost > SVM

**Comparação Estatística**:
- McNemar's test para comparar erros
- Bootstrapping para confidence intervals
- Cross-validation consistency (variance entre folds)

### Fase 8: Persistência e Versionamento

**Salvamento**:
```python
# Melhor modelo
joblib.dump(best_model, 'models/best_model.pkl')

# Todos os modelos (comparação)
joblib.dump(tree_model, 'models/tree_model_v1.1.0.pkl')
joblib.dump(svm_model, 'models/svm_model_v1.1.0.pkl')
joblib.dump(xgb_model, 'models/xgboost_model_v1.1.0.pkl')

# Scalers (já salvos)
# models/scalers_v1.1.0.pkl
```

**Metadados no PostgreSQL**:
```sql
INSERT INTO trained_models (
    model_name, algorithm, version, feature_count,
    pr_auc, recall, f1_score, threshold, is_active
) VALUES (
    'xgboost_v1.1.0', 'XGBoost', '1.1.0', 39,
    0.86, 0.83, 0.78, 0.25, TRUE
);
```

---

## 📋 10. Roadmap de Implementação

### ✅ Etapa 1-3: Concluídas
- EDA completa
- Outlier analysis (preservar todos)
- Missing values (zero encontrados)
- Feature engineering (9 novas features)
- Feature selection pragmática (Time removido)

### ⏳ Etapa 4: Treinamento de Modelos (PRÓXIMO)
**Scripts a criar**:
1. `training/01_train_models.py` - Treinamento dos 3 modelos
2. `training/02_hyperparameter_tuning.py` - Grid Search com PR-AUC
3. `training/03_threshold_tuning.py` - Otimização de threshold
4. `training/04_feature_importance.py` - Análise pós-treino

**Outputs esperados**:
- `models/tree_model_v1.1.0.pkl`
- `models/svm_model_v1.1.0.pkl`
- `models/xgboost_model_v1.1.0.pkl`
- `reports/model_comparison.json`
- `reports/feature_importance.json`

### ⏳ Etapa 5: Refinamento (Se Necessário)
- A/B testing (39 features vs subset)
- Remoção de features inúteis (baseado em importance)
- Re-treinamento e comparação
- Decisão final versionada em configs.py

### ⏳ Etapa 6: Dashboard Flask (MVP)
- Interface interativa
- Upload de transação
- Classificação em tempo real
- Visualização de métricas

---

## 🎓 Aprendizados Chave

1. **Outliers ≠ Erros**: 18.5% das fraudes em outliers (crítico manter!)
2. **Correlação Linear ≠ Utilidade**: XGBoost captura não-linearidades
3. **Feature Importance Pós-Treino**: Deixar modelo decidir > intuição humana
4. **Pragmatismo > Perfeição**: 39 features é saudável, começar simples
5. **A/B Testing**: Git versiona decisões, fácil comparar configurações
6. **Threshold Tuning**: 0.2-0.3 melhor que 0.5 para fraudes raras
7. **RobustScaler**: Resistente a outliers (mediana/IQR)
8. **StratifiedKFold**: Essencial para desbalanceamento extremo
9. **PR-AUC > ROC-AUC**: Métrica certa para classes desbalanceadas
10. **Modelos Robustos**: Decision Tree, SVM RBF, XGBoost (Regressão Logística removida)

**Scalers (Pickle + Versionamento)**:
```python
# MVP/POC (atual)
models/scalers.pkl  # Versão ativa

# Produção (recomendado)
models/scalers_v1.0.0.pkl  # Sem feature selection
models/scalers_v1.1.0.pkl  # Com V13, V15, V22 removidos

# Carregar versão específica
from src.ml.models.configs import config
scaler_path = config.paths.get_versioned_scaler_path('1.1.0')
scalers = joblib.load(scaler_path)
```

**Por que Pickle?**
- ✅ Preserva objeto completo (métodos + estado)
- ✅ Rápido (serialização nativa Python)
- ✅ Integração perfeita com scikit-learn
- ✅ Tamanho pequeno (~100 bytes para RobustScaler)

**Alternativas para Produção**:
- **MLflow**: Versionamento automático + tracking completo
- **ONNX**: Cross-platform (C++, Java, etc.)
- **Redis**: Cache distribuído para alta disponibilidade

---

## 🎓 Storytelling para Apresentação

### Decisões Técnicas Principais

1. **Desbalanceamento**: 
   > "Com apenas 492 fraudes em 284.807 transações (0.172%), optamos por NÃO usar SMOTE ou undersampling. Em vez disso, aplicamos `class_weight='balanced'` nos modelos e threshold tuning pós-treino para otimizar recall."

2. **Outliers**: 
   > "Análise IQR identificou 31.904 outliers (11.2%), mas 91 fraudes (18.5%) estão nesta faixa. Decidimos manter TODOS os dados, pois cada fraude é preciosa. Solução: RobustScaler para normalização resistente a outliers."

3. **Seleção de Modelos (3 Modelos Robustos)**:
   > "Removemos Regressão Logística pois assume linearidade e é sensível a outliers. Mantivemos apenas modelos robustos: **Decision Tree** (splits por ranking, não afetado por valores extremos), **SVM com RBF kernel** (normalizado com RobustScaler, captura não-linearidades), e **XGBoost** (árvores + regularização, extremamente robusto). Todos lidam bem com outliers e desbalanceamento via `class_weight='balanced'` e `scale_pos_weight`."

4. **Features com Baixa Correlação**: 
   > "Amount e Time têm correlação linear quase zero com fraude, mas análise revelou padrões não-lineares: 73.6% das fraudes < $100 e taxa de fraude 10x maior à madrugada. Feature engineering (Time_Period, Amount_Bin, estatísticas V-features) e modelos não-lineares (XGBoost) capturam essas relações complexas."

5. **Otimização de Performance**:
   > "Implementamos PostgreSQL COPY para bulk inserts (81.6% mais rápido) e paralelização de steps independentes (ThreadPoolExecutor). Pipeline completo passou de ~130s para ~62s, um ganho de 52%. Metadata tracking evita duplicação de dados quando não há alteração."

6. **Feature Engineering Estratégico**:
   > "Criamos 9 novas features baseadas na EDA: Time_Period captura padrão temporal (fraudes 10x maiores à madrugada), Amount_Bin categoriza valores (73.6% fraudes < $100), estatísticas V-features capturam tendência central e variabilidade. Total: 31 → 40 features sem explosão combinatória."

7. **Feature Selection Config-Driven**:
   > "Implementamos análise automatizada (Step 05.5) que gera relatório JSON com Pearson, Spearman, Mutual Information, VIF e redundância. Desenvolvedor revisa relatório, considera contexto de negócio, e atualiza `configs.py` manualmente. Step 06 aplica remoção automaticamente. Abordagem rastreável (Git) e reprodutível, permitindo A/B testing fácil."

---

## 📚 Referências Técnicas

- **IQR (Tukey's Method)**: Identificação científica de outliers
- **Pearson Correlation**: Relações lineares
- **Spearman Correlation**: Relações monotônicas
- **Mutual Information**: Relações não-lineares gerais
- **RobustScaler**: Normalização resistente a outliers (usa mediana e IQR)
- **class_weight='balanced'**: Ajuste automático de pesos por classe
- **Threshold Tuning**: Otimização de ponto de decisão (Precision-Recall trade-off)
- **PostgreSQL COPY**: Bulk insert nativo (70-80% mais rápido que to_sql)
- **ThreadPoolExecutor**: Paralelização de tarefas I/O-bound

---

**Conclusão**: EDA revelou que o dataset é extremamente desbalanceado (0.172% fraudes), tem outliers importantes (18.5% das fraudes), e features com baixa correlação linear mas alta utilidade não-linear. Decisões foram baseadas em análise estatística rigorosa e contexto do problema (detecção de fraude = recall crítico). Pipeline otimizado processa 284k transações em ~62s com rastreabilidade completa em PostgreSQL. Feature selection implementada com análise automatizada (Step 05.5) e decisão manual config-driven (Step 06), permitindo versionamento e A/B testing.

---

**Autor**: Fraud Detection Team  
**Data**: 2025-10-02  
**Versão**: 4.0.0 (Pipeline otimizado + Steps 04-07 + Feature Selection)
