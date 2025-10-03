# 📋 Exemplos de Transações - Dataset Fraud Detection

Este documento contém exemplos reais de transações do dataset para testes de predição.

---

## 🚨 Transação FRAUDULENTA (Classe = 1)

### JSON Completo (33 features - modelo v2.1.0)

```json
{
  "V1": -1.27124419171437,
  "V2": 2.46267526851135,
  "V3": -2.85139500331783,
  "V4": 2.3244800653478,
  "V5": -1.3722448981369,
  "V6": -0.948195686538643,
  "V7": -3.06523436172054,
  "V9": -2.26877058844813,
  "V10": -4.88114292689057,
  "V11": 2.25514748870463,
  "V12": -4.68638689759229,
  "V14": -6.17428834800643,
  "V15": 0.594379608016446,
  "V16": -4.84969238709652,
  "V17": -6.53652073527011,
  "V18": -3.11909388163881,
  "V19": 1.71549441975915,
  "V20": 0.560478075726644,
  "V21": 0.652941051330455,
  "V22": 0.0819309763507574,
  "V24": -0.523582159233306,
  "V25": 0.224228161862968,
  "V26": 0.756334522703558,
  "V27": 0.632800477330469,
  "V28": 0.250187092757197,
  "Time_Hours": -0.0002211483237206061,
  "Amount_Log": -0.36711945323032036,
  "Amount_Bin": 0,
  "V_Mean": -1.0156700379890384,
  "V_Std": 2.5953238729448946,
  "V_Min": -6.53652073527011,
  "V_Max": 2.46267526851135,
  "V_Range": 8.99919600378146
}
```

### Características desta transação:
- **V17 muito negativo**: -6.54 (indicador forte de fraude)
- **V14 muito negativo**: -6.17 (padrão suspeito)
- **V12 muito negativo**: -4.69 (anomalia)
- **V_Range alto**: 8.99 (alta variabilidade nas features)
- **Amount_Log negativo**: Valor baixo (fraudes tendem a ser valores estratégicos)

---

## ✅ Transação LEGÍTIMA (Classe = 0)

### JSON Completo (33 features - modelo v2.1.0)

```json
{
  "V1": -0.674466064578314,
  "V2": 1.40810501967799,
  "V3": -1.11062205357093,
  "V4": -1.32836577843066,
  "V5": 1.38899603254837,
  "V6": -1.30843906707795,
  "V7": 1.88587890268717,
  "V9": 0.311652212453101,
  "V10": 0.65075700363522,
  "V11": -0.857784661547805,
  "V12": -0.229961445775592,
  "V14": 0.266371326329879,
  "V15": -0.0465441684754424,
  "V16": -0.741398089749789,
  "V17": -0.605616644106022,
  "V18": -0.39256818789208,
  "V19": -0.162648311024695,
  "V20": 0.394321820843914,
  "V21": 0.0800842396026648,
  "V22": 0.810033595602455,
  "V24": 0.707899237446867,
  "V25": -0.13583702273753,
  "V26": 0.0451021964988772,
  "V27": 0.533837219064273,
  "V28": 0.291319252625364,
  "Time_Hours": 0.00038574688759463,
  "Amount_Log": 0.01387658428614615,
  "Amount_Bin": 0,
  "V_Mean": 0.0050617629472185215,
  "V_Std": 0.8002154093262184,
  "V_Min": -1.32836577843066,
  "V_Max": 1.88587890268717,
  "V_Range": 3.21424468111783
}
```

### Características desta transação:
- **Valores dentro da normalidade**: Nenhuma feature extremamente negativa
- **V17 moderado**: -0.61 (não suspeito)
- **V14 positivo**: 0.27 (comportamento normal)
- **V_Range baixo**: 3.21 (baixa variabilidade, padrão consistente)
- **V_Std baixo**: 0.80 (features estáveis)

---

## 🧪 Como Testar com o CLI

### Testar transação fraudulenta:
```bash
# Via JSON inline
python main.py predict --json '{"V1": -1.27, "V2": 2.46, "V3": -2.85, "V4": 2.32, "V5": -1.37, "V6": -0.95, "V7": -3.07, "V9": -2.27, "V10": -4.88, "V11": 2.26, "V12": -4.69, "V14": -6.17, "V15": 0.59, "V16": -4.85, "V17": -6.54, "V18": -3.12, "V19": 1.72, "V20": 0.56, "V21": 0.65, "V22": 0.08, "V24": -0.52, "V25": 0.22, "V26": 0.76, "V27": 0.63, "V28": 0.25, "Time_Hours": -0.0002, "Amount_Log": -0.37, "Amount_Bin": 0, "V_Mean": -1.02, "V_Std": 2.60, "V_Min": -6.54, "V_Max": 2.46, "V_Range": 9.0}'

# Resultado esperado:
# 🚨 Transação 0: FRAUDE (>85% probabilidade)
```

### Testar transação legítima:
```bash
python main.py predict --json '{"V1": -0.67, "V2": 1.41, "V3": -1.11, "V4": -1.33, "V5": 1.39, "V6": -1.31, "V7": 1.89, "V9": 0.31, "V10": 0.65, "V11": -0.86, "V12": -0.23, "V14": 0.27, "V15": -0.05, "V16": -0.74, "V17": -0.61, "V18": -0.39, "V19": -0.16, "V20": 0.39, "V21": 0.08, "V22": 0.81, "V24": 0.71, "V25": -0.14, "V26": 0.05, "V27": 0.53, "V28": 0.29, "Time_Hours": 0.0004, "Amount_Log": 0.01, "Amount_Bin": 0, "V_Mean": 0.005, "V_Std": 0.80, "V_Min": -1.33, "V_Max": 1.89, "V_Range": 3.21}'

# Resultado esperado:
# ✅ Transação 0: LEGÍTIMA (<10% probabilidade)
```

---

## 📊 Diferenças-Chave entre Fraude vs Legítima

| Feature | Fraude | Legítima | Diferença |
|---------|--------|----------|-----------|
| **V17** | -6.54 | -0.61 | **10.7x mais negativo** |
| **V14** | -6.17 | 0.27 | **Extremamente negativo** |
| **V12** | -4.69 | -0.23 | **20.4x mais negativo** |
| **V_Range** | 9.00 | 3.21 | **2.8x maior variabilidade** |
| **V_Std** | 2.60 | 0.80 | **3.2x maior desvio padrão** |
| **V10** | -4.88 | 0.65 | **Anomalia significativa** |

**Insight**: Fraudes tendem a ter **features V extremamente negativas** (V10, V12, V14, V17) e **alta variabilidade** (V_Range, V_Std).

---

## 🔍 Features Mais Importantes para Detecção

Baseado no XGBoost v2.1.0:

1. **V17** - Indicador mais forte (fraudes < -5.0)
2. **V14** - Segundo indicador (fraudes < -4.0)
3. **V12** - Terceiro indicador (fraudes < -3.0)
4. **V10** - Quarto indicador (fraudes < -3.0)
5. **V_Range** - Variabilidade geral (fraudes > 7.0)
6. **V4** - Padrão de gasto (fraudes > 2.0)
7. **V11** - Padrão temporal (fraudes > 2.0)

---

## 📁 Arquivos JSON de Exemplo

### `examples/fraud_transaction.json`
```json
{
  "V1": -1.27, "V2": 2.46, "V3": -2.85, "V4": 2.32, "V5": -1.37,
  "V6": -0.95, "V7": -3.07, "V9": -2.27, "V10": -4.88, "V11": 2.26,
  "V12": -4.69, "V14": -6.17, "V15": 0.59, "V16": -4.85, "V17": -6.54,
  "V18": -3.12, "V19": 1.72, "V20": 0.56, "V21": 0.65, "V22": 0.08,
  "V24": -0.52, "V25": 0.22, "V26": 0.76, "V27": 0.63, "V28": 0.25,
  "Time_Hours": -0.0002, "Amount_Log": -0.37, "Amount_Bin": 0,
  "V_Mean": -1.02, "V_Std": 2.60, "V_Min": -6.54, "V_Max": 2.46, "V_Range": 9.0
}
```

### `examples/legitimate_transaction.json`
```json
{
  "V1": -0.67, "V2": 1.41, "V3": -1.11, "V4": -1.33, "V5": 1.39,
  "V6": -1.31, "V7": 1.89, "V9": 0.31, "V10": 0.65, "V11": -0.86,
  "V12": -0.23, "V14": 0.27, "V15": -0.05, "V16": -0.74, "V17": -0.61,
  "V18": -0.39, "V19": -0.16, "V20": 0.39, "V21": 0.08, "V22": 0.81,
  "V24": 0.71, "V25": -0.14, "V26": 0.05, "V27": 0.53, "V28": 0.29,
  "Time_Hours": 0.0004, "Amount_Log": 0.01, "Amount_Bin": 0,
  "V_Mean": 0.005, "V_Std": 0.80, "V_Min": -1.33, "V_Max": 1.89, "V_Range": 3.21
}
```

---

## 📝 Notas Técnicas

### Features Removidas (v2.1.0)
- **Time**: Offset temporal sem significado real
- **Time_Period_of_Day**: Derivada de Time
- **V8, V23**: Baixa correlação com Class (<0.01)
- **Amount**: Amount_Log é mais informativo
- **V13**: Redundante (alta correlação com V17)

### Pré-processamento
- **RobustScaler**: Aplicado em Time_Hours e Amount_Log
- **PCA Features (V1-V28)**: Já normalizadas no dataset original
- **Engineered Features**: Time_Hours, Amount_Log, Amount_Bin, V-statistics

---

**Modelo**: XGBoost v2.1.0  
**PR-AUC**: 0.8772  
**Precision**: 86.60%  
**False Positives**: 13 (em 56,962 transações)  
**Dataset**: Kaggle Credit Card Fraud Detection
