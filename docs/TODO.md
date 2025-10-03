# 📊 Plano: Dashboard Flask para Monitorar Detecção de Fraudes
## Aplicação Web Minimalista com Real-Time Detection

---

## 🎯 Visão Geral

**Objetivo**: Interface web minimalista para simular transações e visualizar detecções de fraude em tempo real.

**Stack Tecnológica**:
- **Backend**: Flask 3.0+ (REST API)
- **Frontend**: HTML5 + CSS3 + Vanilla JavaScript (sem frameworks pesados)
- **Gráficos**: Chart.js (leve, 60KB)
- **Real-Time**: Server-Sent Events (SSE) ou WebSocket
- **Persistência**: PostgreSQL (tabelas já existentes)

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (SPA)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Dashboard  │  │  Simulator   │  │   History    │         │
│  │   (KPIs)     │  │  (Generate)  │  │  (Table)     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                    │
│                     REST API (JSON)                             │
│                            │                                    │
├────────────────────────────┼────────────────────────────────────┤
│                     FLASK BACKEND                               │
│  ┌─────────────────────────┴────────────────────────┐          │
│  │              routes/api.py                        │          │
│  │  - POST /api/simulate    (gera transação)        │          │
│  │  - POST /api/predict     (classifica)            │          │
│  │  - GET  /api/stats       (KPIs agregados)        │          │
│  │  - GET  /api/history     (últimas detecções)     │          │
│  │  - GET  /api/stream      (SSE real-time)         │          │
│  └──────────────────────────────────────────────────┘          │
│         │                  │                  │                 │
│  ┌──────▼──────┐  ┌────────▼────────┐  ┌─────▼──────┐         │
│  │   Model     │  │   PostgreSQL    │  │  Generator │         │
│  │ xgboost.pkl │  │  transactions   │  │ (synthetic)│         │
│  └─────────────┘  │ classification  │  └────────────┘         │
│                   │   results       │                          │
│                   └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Checklist de Implementação

### **Fase 1: Backend Flask API** 🔧

#### 1.1 Setup Flask App
- [ ] Criar `app/` na raiz do projeto
- [ ] Criar `app/__init__.py` (factory pattern)
- [ ] Criar `app/config.py` (configurações)
- [ ] Criar `requirements-webapp.txt` (dependências extras)
- [ ] Criar `run.py` (entry point)

#### 1.2 Gerador de Transações Sintéticas
- [ ] Criar `app/services/transaction_generator.py`
- [ ] Implementar `generate_legitimate()`: Baseado em estatísticas do dataset
- [ ] Implementar `generate_fraudulent()`: Baseado em padrões de fraude (V17, V14, V12 alterados)
- [ ] Adicionar variabilidade (noise) realista
- [ ] Validar features geradas (33 features do modelo v2.1.0)

#### 1.3 API Endpoints
- [ ] Criar `app/routes/api.py`
- [ ] **POST /api/simulate**: 
  - [ ] Parâmetro: `{"type": "fraud" | "legitimate"}`
  - [ ] Retorna: transação + predição + probabilidade
  - [ ] Salva em PostgreSQL: `transactions` + `classification_results`
- [ ] **POST /api/predict**:
  - [ ] Recebe transação manual (JSON com 33 features)
  - [ ] Classifica com modelo XGBoost v2.1.0
  - [ ] Retorna: predição + probabilidade + confidence
- [ ] **GET /api/stats**:
  - [ ] KPIs: Total transações, fraudes detectadas, taxa de fraude
  - [ ] Métricas: Precision, Recall, F1 (últimas 1000 transações)
  - [ ] Performance: Tempo médio de predição
- [ ] **GET /api/history?limit=50**:
  - [ ] Últimas N classificações (ORDER BY timestamp DESC)
  - [ ] Dados: timestamp, amount, prediction, probability, true_label (se disponível)
- [ ] **GET /api/stream** (opcional):
  - [ ] Server-Sent Events para updates em tempo real
  - [ ] Envia nova classificação assim que ocorre

#### 1.4 Model Loader
- [ ] Criar `app/services/model_service.py`
- [ ] Implementar singleton pattern para carregar modelo uma vez
- [ ] Carregar `models/xgboost_v2.1.0.pkl` e `models/scalers.pkl`
- [ ] Implementar `predict(features)` com validação de features
- [ ] Cache de predições (opcional, Redis)

#### 1.5 Database Integration
- [ ] Criar `app/models/transaction.py` (SQLAlchemy)
- [ ] Tabela `transactions`: Transações simuladas/reais
- [ ] Tabela `classification_results`: Predições do modelo
- [ ] Implementar queries de estatísticas agregadas
- [ ] Adicionar índices para performance (timestamp, prediction)

---

### **Fase 2: Frontend Dashboard** 🎨

#### 2.1 Estrutura HTML
- [ ] Criar `app/static/` e `app/templates/`
- [ ] `templates/index.html`: Single Page Application (SPA)
- [ ] Layout responsivo com CSS Grid/Flexbox
- [ ] 3 seções principais:
  - [ ] **Header**: Logo + título + modelo info (v2.1.0)
  - [ ] **Simulator**: Botões de gerar transação
  - [ ] **Dashboard**: KPIs + gráficos
  - [ ] **History**: Tabela de últimas detecções

#### 2.2 Estilos Minimalistas
- [ ] Criar `static/css/style.css`
- [ ] Paleta de cores:
  - [ ] `--fraud-red`: #E74C3C (fraudes)
  - [ ] `--legit-green`: #27AE60 (legítimas)
  - [ ] `--bg-dark`: #2C3E50 (fundo)
  - [ ] `--text-light`: #ECF0F1 (texto)
- [ ] Tipografia: Inter ou Roboto (Google Fonts)
- [ ] Animações suaves (CSS transitions)
- [ ] Dark theme por padrão

#### 2.3 Simulator (Geração de Transações)
- [ ] Criar `static/js/simulator.js`
- [ ] Botão "🎲 Generate Legitimate Transaction"
- [ ] Botão "🚨 Generate Fraudulent Transaction"
- [ ] Loading state durante predição
- [ ] Toast notification com resultado
- [ ] Atualizar dashboard automaticamente

#### 2.4 KPIs Dashboard
- [ ] Criar `static/js/dashboard.js`
- [ ] **Cards de Métricas** (4 principais):
  - [ ] Total Transações (hoje + total)
  - [ ] Fraudes Detectadas (% do total)
  - [ ] Precision do Modelo (últimas 1000)
  - [ ] Tempo Médio de Resposta (ms)
- [ ] Auto-refresh a cada 5 segundos

#### 2.5 Visualização Real-Time: Activity Feed com Simulador

##### **Activity Feed com Animações + Transaction Simulator** ⭐
- [ ] **Transaction Simulator** (topo do dashboard)
  - [ ] Botão: "🟢 Generate Legitimate Transaction"
  - [ ] Botão: "🔴 Generate Fraudulent Transaction"
  - [ ] Feedback visual: "💫 Generating transaction..." com fade-out (1.5s)
  - [ ] Após geração: novo card entra no feed com slide-in animation
  
- [ ] **Stream de Transações** (Activity Feed abaixo do simulador)
  - [ ] Cards animados entrando da direita (slideInRight 0.5s)
  - [ ] Cada card: timestamp relativo, amount, predição, probabilidade
  - [ ] Cor de fundo baseada em probabilidade (gradiente verde → amarelo → vermelho)
  - [ ] Pulso/glow em fraudes detectadas (pulse 2s infinite)
  - [ ] Botão "Details ▼" para expandir/colapsar informações extras
  - [ ] Limite: 50 cards visíveis, remove os mais antigos automaticamente
  - [ ] Filtros: "All" / "Frauds Only" / "High Risk (>70%)"


#### 2.6 History Table
- [ ] Criar `static/js/history.js`
- [ ] Tabela com últimas 50 classificações
- [ ] Colunas:
  - [ ] Timestamp
  - [ ] Amount (formatado $)
  - [ ] Prediction (badge colorido)
  - [ ] Probability (%)
  - [ ] Confidence (Alta/Média/Baixa)
- [ ] Ordenação por timestamp (DESC)
- [ ] Paginação (scroll infinito ou botão "Load More")
- [ ] Filtros: Todas / Fraudes / Legítimas

---

### **Fase 3: Real-Time Updates** ⚡

#### 3.1 Server-Sent Events (SSE)
- [ ] Implementar `/api/stream` endpoint
- [ ] Frontend conecta ao stream via `EventSource`
- [ ] Enviar evento quando nova transação é classificada
- [ ] Atualizar dashboard sem refresh
- [ ] Adicionar nova linha na tabela de histórico
- [ ] Animar adição de novo item

---

### **Fase 4: Features Avançadas** 🚀

#### 4.2 Explicabilidade (SHAP)
- [ ] Instalar `shap` library
- [ ] Endpoint `GET /api/explain/{transaction_id}`
- [ ] Calcular SHAP values para transação específica
- [ ] Visualizar features que mais contribuíram
- [ ] Gráfico waterfall no frontend

---


---

## 📊 KPIs Principais do Dashboard

### 1. **Operational Metrics** (Cards Superiores)
```
┌────────────────┬────────────────┬────────────────┬────────────────┐
│ Total Trans.   │ Fraudes Det.   │ Precision      │ Avg Response   │
│   1,234        │   42 (3.4%)    │    86.60%      │    12ms        │
└────────────────┴────────────────┴────────────────┴────────────────┘
```

### 2. **Model Performance** (Gráficos)
- **Confusion Matrix** (últimas 1000 transações)
  - True Positives (TP): Fraudes corretamente detectadas
  - True Negatives (TN): Legítimas corretamente classificadas
  - False Positives (FP): Alarmes falsos (custo operacional)
  - False Negatives (FN): Fraudes não detectadas (custo financeiro)

- **Precision/Recall/F1** (trend ao longo do tempo)
  - Gráfico de linha mostrando evolução

### 3. **Business Impact** (Cards Laterais)
```
💰 Fraudes Evitadas (estimado):  $12,450
🚨 Falsos Positivos (custo):     $260
📈 Taxa de Detecção:             81.63%
⚡ Uptime do Sistema:            99.9%
```

### 4. **Real-Time Activity** (Últimas 50)
- Tabela com scroll
- Highlight em fraudes detectadas
- Timestamp relativo ("2 min ago")

---

## 🎨 Wireframe Minimalista

```
╔══════════════════════════════════════════════════════════════════╗
║  🚨 Fraud Detection System  |  Model: XGBoost v2.1.0  |  [Docs]  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📊 DASHBOARD                                                    ║
║  ┌────────────┬────────────┬────────────┬────────────┐          ║
║  │ Total      │ Frauds     │ Precision  │ Latency    │          ║
║  │ 1,234      │ 42 (3.4%)  │ 86.60%     │ 12ms       │          ║
║  └────────────┴────────────┴────────────┴────────────┘          ║
║                                                                  ║
║  🎲 TRANSACTION SIMULATOR                                        ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │  [🟢 Generate Legitimate]  [🔴 Generate Fraudulent]      │   ║
║  │                                                           │   ║
║  │  Last Prediction: ✅ LEGITIMATE (Prob: 2.3%)             │   ║
║  │  Amount: $45.67  |  Features: 33  |  Time: 8ms          │   ║
║  └──────────────────────────────────────────────────────────┘   ║
║                                                                  ║
║  📈 REAL-TIME CHARTS                                             ║
║  ┌─────────────────────────┬─────────────────────────┐          ║
║  │  Time Series            │                         │          ║
║  │  (Last 100 trans.)      │                         │          ║
║  │  [Chart.js Line]        │       
║  │                         │                         │          ║
║  │                         │   TP: 80  | FN: 18      │          ║
║  │                         │   FP: 13  | TN: 56,851  │          ║
║  └─────────────────────────┴─────────────────────────┘          ║
║                                                                  ║
║  📋 DETECTION HISTORY (Last 50)                                  ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │ Time      │ Amount  │ Prediction  │ Prob    │ Confidence │   ║
║  ├──────────────────────────────────────────────────────────┤   ║
║  │ 14:32:15  │ $234.50 │ 🚨 FRAUD   │ 94.2%   │ High       │   ║
║  │ 14:31:58  │ $12.30  │ ✅ LEGIT   │ 1.8%    │ High       │   ║
║  │ 14:31:42  │ $89.00  │ ✅ LEGIT   │ 3.5%    │ High       │   ║
║  │ ...       │ ...     │ ...        │ ...     │ ...        │   ║
║  └──────────────────────────────────────────────────────────┘   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 🛠️ Stack de Tecnologias

### Backend
```python
# requirements-webapp.txt
Flask==3.0.0
Flask-CORS==4.0.0
Flask-SocketIO==5.3.5  # (opcional para WebSocket)
gunicorn==21.2.0
pandas==2.1.4
numpy==1.26.2
xgboost==2.0.3
scikit-learn==1.3.2
SQLAlchemy==2.0.23
psycopg2-binary==2.9.9
python-dotenv==1.0.0
```

### Frontend
```html
<!-- Bibliotecas CDN -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
```

### Estrutura de Arquivos
```
app/
├── __init__.py           # Flask app factory
├── config.py             # Configurações (dev/prod)
├── run.py                # Entry point
│
├── routes/
│   ├── __init__.py
│   ├── api.py            # REST endpoints
│   └── views.py          # HTML routes
│
├── services/
│   ├── __init__.py
│   ├── model_service.py  # Model loader + predict
│   └── transaction_generator.py  # Synthetic data
│
├── models/
│   ├── __init__.py
│   └── transaction.py    # SQLAlchemy models
│
├── static/
│   ├── css/
│   │   └── style.css     # Estilos minimalistas
│   ├── js/
│   │   ├── dashboard.js  # KPIs e gráficos
│   │   ├── simulator.js  # Geração de transações
│   │   └── history.js    # Tabela de histórico
│   └── img/
│       └── logo.png
│
└── templates/
    └── index.html        # SPA principal
```

---

## 🚀 Comandos de Execução

### Desenvolvimento
```bash
# Instalar dependências
pip install -r requirements-webapp.txt

# Variáveis de ambiente
export FLASK_APP=app
export FLASK_ENV=development
export DATABASE_URL=postgresql://fraud_user:fraud_pass_dev@localhost:5432/fraud_detection

# Rodar Flask dev server
python app/run.py

# Acessar: http://localhost:5000
```

### Produção (Docker)
```bash
# Build e start
docker-compose up -d

# Acessar: http://localhost:80
```

---

## 📝 Notas de Implementação

### Prioridade 1 (MVP - 2 dias)
1. ✅ Backend: Endpoints básicos (simulate, predict, stats)
2. ✅ Frontend: HTML + CSS minimalista
3. ✅ Simulator: Botões de gerar transação
4. ✅ Dashboard: KPIs básicos (cards)
5. ✅ History: Tabela de últimas 50

### Prioridade 2 (Funcional - 1 dia)
2. ✅ Real-time: SSE para auto-refresh
3. ✅ Generator: Lógica realista de transações sintéticas

### Prioridade 3 (Extras - se sobrar tempo)
1. ⏳ SHAP explicabilidade  

---

## ✅ Critérios de Sucesso

### Funcional
- [ ] Gerar transação legítima/fraudulenta em 1 clique
- [ ] Classificação em < 50ms (p95)
- [ ] Dashboard atualiza automaticamente (5s)
- [ ] 4 KPIs principais visíveis
- [ ] 2 gráficos interativos (Chart.js)
- [ ] Histórico com últimas 50 transações

### Técnico
- [ ] API REST com 5 endpoints funcionais
- [ ] PostgreSQL persiste todas as classificações
- [ ] Modelo carregado uma vez (singleton)
- [ ] Frontend sem dependências pesadas (< 500KB)
- [ ] Responsivo (funciona em mobile)

### UX
- [ ] Interface minimalista (tema dark)
- [ ] Feedback visual imediato (loading, toast)
- [ ] Animações suaves (< 300ms)
- [ ] Cores intuitivas (verde/vermelho)

---

