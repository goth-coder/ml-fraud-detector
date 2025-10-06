# 🚨 Credit Card Fraud Detection System

![Machine Learning Fraud Detection System](docs/images/dashboard_screenshot.png)

A complete **Machine Learning fraud detection system** featuring a modular pipeline, PostgreSQL persistence, an interactive CLI, and a **real-time web dashboard**.
Developed as part of the **FIAP Tech Challenge – Phase 3**.

---

## 🚀 Quick Demo

**Want to see it in action?** Run the dashboard in just 3 steps:

```bash
# 1. Start the Flask server
python run.py

# 2. Open in your browser
# http://127.0.0.1:5000

# 3. Test the detector!
#    → Click “LEGÍTIMA” or “FRAUDULENTA”
#    → Click “EXECUTAR SIMULAÇÃO”
#    → Watch real-time results:
#       ✅ Model Prediction (classified by XGBoost)
#       🎯 Ground Truth (actual transaction type)
#       📊 Fraud probability (0–100%)
#       🔍 Confidence level (HIGH / MODERATE / LOW)
#       ⚡ Inference latency (measured in real-time)
```

**🎯 What you’ll see:**

* **Simulation Panel** – trigger legitimate or fraudulent transactions in one click
* **Real-Time Stats** – total transactions, detected frauds, recall rate, latency
* **Full History** – all classifications with prediction vs actual label

> 💡 **Tip:** Try simulating multiple fraudulent transactions. The model will realistically miss some (~15% error), demonstrating that it’s **not overfitted**.

---

## 🎯 Objective

Develop a fraud detection system using an **optimized XGBoost** model with:

* ✅ Handling of **highly imbalanced data** (1:578 fraud ratio)
* ✅ Modular **MVC-ML Pipeline** (processing → training → inference)
* ✅ Robust validation with **StratifiedKFold**
* ✅ **Automated Grid Search** with hyperparameter versioning
* ✅ **PostgreSQL** (data) + **Pickle** (models) persistence
* ✅ **Complete CLI** with 4 operational modes
* ✅ **Flask REST API** backend for real-time simulation
* ✅ **Interactive Web Dashboard** for live monitoring and insights

---

## 🏗️ Project Architecture

### Modular MVC-ML + Service Layer

The system follows a **layered modular architecture** for clear separation of concerns:

* **Model (M)**: `src/models/` – SQLAlchemy ORM + ML configs
* **View (V)**: `src/services/frontend/` – HTML templates + static assets
* **Controller (C)**: `src/services/backend/` – Flask routes + logic
* **ML Pipeline**: `src/ml/` – data processing and training
* **Services**: `src/services/` – infrastructure components (DB, ML, Frontend, Backend)

```
ml-fraud-detector/
├── data/                        # 📊 Datasets and configs
│   ├── archive/                 # Older hyperparameter versions
│   ├── examples/                # Sample transactions
│   ├── creditcard.csv           # Original dataset (284,807 rows)
│   └── xgboost_hyperparameters.json
│
├── database/                    # 🗄️ PostgreSQL config
│   ├── migrations/              # SQL migrations
│   ├── docker-compose.yml       # PostgreSQL 15 setup
│   └── schema.sql               # Schema (7 pipeline + 2 webapp tables)
│
├── docs/                        # 📚 Documentation
│   ├── images/                  # Charts and screenshots
│   ├── API_ENDPOINTS.md         # REST API docs
│   ├── DATA_ARCHITECTURE.md     # PostgreSQL + Pickle architecture
│   ├── DECISOES_TECNICAS.md     # ML optimizations
│   ├── EDA_REPORT.md            # Exploratory data analysis
│   ├── MODEL_SELECTION.md       # Model comparison
│   └── TRANSACTION_EXAMPLES.md  # Examples
│
├── models/                      # 🤖 Trained ML models
│   ├── archive/                 # Older versions
│   ├── scalers.pkl              # RobustScaler + StandardScaler
│   └── xgboost_v2.1.0.pkl       # ⭐ Production model
│
├── reports/                     # 📈 Generated analysis
│   └── feature_selection_analysis.json
│
├── src/                         # 💻 Source code
│   ├── ml/                      # 🧠 ML pipeline
│   ├── models/                  # 📦 SQLAlchemy ORM models
│   ├── services/                # 🔌 Service layer
│   └── __init__.py
│
├── main.py                      # 🎯 CLI entry point
├── run.py                       # 🚀 Flask app entry
├── requirements.txt              # Dependencies
└── README.md
```

---

### Data Processing Pipeline (7 Steps)

```
CSV (raw) → [01] Load → [02] Outlier Analysis → [03] Missing Values
         → [04] Normalize → [05] Feature Engineering → [06] Feature Selection
         → [07] Train/Test Split → ✅ Ready for training
```

---

### Persistence Architecture

**PostgreSQL Tables**

* `raw_transactions`, `cleaned_transactions`, … `test_features`
* `classification_results`, `simulated_transactions`

**Pickle Files**

* `models/scalers.pkl` – RobustScaler + StandardScaler
* `models/xgboost_v2.1.0.pkl` – production model

**JSON Configs**

* `xgboost_hyperparameters.json` – active hyperparameters
* `archive/` – previous versions (auto-timestamped)

**Hybrid Design Rationale**

* **PostgreSQL** → traceability + analytics
* **Pickle** → fast model loading
* **JSON** → version control for reproducibility

---

### Technology Stack

* **Python 3.13**, **Pandas**, **XGBoost 3.0.5**
* **Flask 3.0.3**, **SQLAlchemy**, **PostgreSQL 15**
* **Docker Compose**, **ThreadPoolExecutor**, **COPY optimization**

---

## 📊 Dataset

* **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Transactions:** 284,807 (2 days, September 2013)
* **Features:** 30 (PCA-transformed + Time, Amount)
* **Target:** Class (0=legit, 1=fraud)
* **Imbalance:** 492 frauds (0.172%) vs 284,315 legitimate

---

## 🚀 Quick Start

### 1. Requirements

* Python ≥3.13
* PostgreSQL ≥15
* 2GB+ RAM

### 2. Installation

```bash
git clone <repo-url>
cd ml-fraud-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or faster:

```bash
uv pip install -e .
```

### 3. Setup PostgreSQL

**Option A: Docker (recommended)**

```bash
cd database
docker-compose up -d
```

**Option B: Local setup**

```bash
createdb fraud_detection
createuser fraud_user --password fraud_pass_dev
psql -U fraud_user -d fraud_detection -f database/schema.sql
```

### 4. Run the Pipeline

```bash
python main.py pipeline
```

---

## 🖥️ CLI Interface (`main.py`)

Supports 4 main modes:

### 1️⃣ `pipeline`

Runs full data processing pipeline (Steps 01–07).

### 2️⃣ `train`

Trains XGBoost with current JSON hyperparameters.

### 3️⃣ `tune`

Performs automated Grid Search and updates JSON.

### 4️⃣ `predict`

Runs inference on CSV or JSON input.

---

## 🌐 Flask REST API

### Endpoints

| Method | Endpoint        | Description                         |
| ------ | --------------- | ----------------------------------- |
| `POST` | `/api/simulate` | Generate and classify a transaction |
| `GET`  | `/api/stats`    | Get aggregated stats                |
| `GET`  | `/api/history`  | Retrieve prediction history         |
| `GET`  | `/health`       | Health check                        |

Full API reference: [`docs/API_ENDPOINTS.md`](docs/API_ENDPOINTS.md)

---

## 🎨 Web Dashboard

Interactive web dashboard for simulation and monitoring:

* Run real-time simulations
* Track recall, latency, and fraud rate
* Review full prediction history

---

## 📚 Documentation

* **Technical Reports:** EDA, model selection, and data architecture
* **Plans:** MVP roadmap, Kafka scalability, changelog
* **Architecture Docs:** `src/ml/README.md`

---

## 🔧 Key Configs

`src/ml/models/configs.py`:

```python
xgboost_params = {
    'colsample_bytree': 0.7,
    'learning_rate': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'n_estimators': 100,
    'scale_pos_weight': 577,
    'subsample': 0.7,
    'eval_metric': 'aucpr'
}
```

---

## 📊 Performance Metrics

### Pipeline Optimizations (52% faster ⚡)

| Step           | Before | After | Gain   | Optimization    |
| -------------- | ------ | ----- | ------ | --------------- |
| Normalize      | 90s    | 16.5s | +81.6% | PostgreSQL COPY |
| Pipeline Total | 130s   | 62s   | +52%   | Parallelization |

### XGBoost Results

| Metric    | v1.1.0 | v2.0.0 | v2.1.0 ⭐   |
| --------- | ------ | ------ | ---------- |
| PR-AUC    | 0.8719 | 0.8847 | **0.8772**     |
| Precision | 72.27% | 85.42% | 86.60% |
| Recall    | 87.76% | 83.67% | 81.63%     |
| F1-Score  | 79.26% | 84.54% | 84.04%     |

---

## 🗄️ PostgreSQL Schema

Includes:

* Pipeline tables (raw → processed)
* Webapp tables (`classification_results`, `simulated_transactions`) 

---

## 🚀 Status

✅ Completed
📋 Next Step: Optional Kafka streaming for scalability
 
## 🔄 Future Scalability

The modular ML pipeline architecture is designed for easy **Apache Airflow integration**:

* **Step-based structure** (01-07) maps directly to Airflow DAGs
* **Service layer separation** enables distributed task execution  
* **JSON configs + PostgreSQL** provide stateful orchestration support
* **CLI modes** (`train`, `tune`, `pipeline`) can become Airflow operators

This allows seamless migration from local execution to **scheduled, distributed ML workflows** without code restructuring.

> *In large-scale scenarios, Apache Kafka can enable distributed ingestion, parallel consumers, and real-time reprocessing.*
 
---

## 🔗 References

* **Dataset:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Docs:** [XGBoost](https://xgboost.readthedocs.io/), [Flask](https://flask.palletsprojects.com/), [PostgreSQL](https://www.postgresql.org/)

---

## 👥 Authors

**Victor Lucas Santos de Oliveira** – [LinkedIn](https://www.linkedin.com/in/vlso/)
**Adrianny Lelis da Silva** – [LinkedIn](https://www.linkedin.com/in/adriannylelis/)

--- 