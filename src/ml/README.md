# 🏗️ MVC-ML Architecture – Fraud Detection System 

## 📁 Project Structure

```
src/ml/
├── __init__.py
├── processing/           # Reusable functions (OPTIMIZED!)
│   ├── __init__.py
│   ├── loader.py                  # OPTIMIZED: PostgreSQL COPY (81.6% faster)
│   ├── validation.py              # Schema and data integrity validation
│   ├── cleaning.py                # Outlier analysis (preserves all data)
│   ├── normalization.py           # RobustScaler + StandardScaler ✅
│   ├── feature_engineering.py     # 9 new features (Time_Period, Amount_Bin, etc.) ✅
│   ├── feature_selection.py       # Automated analysis (Pearson, Spearman, MI, VIF) ✅
│   ├── splitters.py               # Stratified train/test split ✅
│   └── metadata.py                # Pipeline metadata tracking
│
├── pipelines/            # Scripts chaining functions  
│   ├── __init__.py
│   └── data_pipeline.py           # Complete pipeline (Steps 01–07) ✅
│                                   # - PostgreSQL COPY for bulk inserts
│                                   # - Parallel Steps 02–03 (ThreadPoolExecutor)
│                                   # - Metadata-only mode for non-mutating steps
│                                   # - Feature Selection Analysis (Step 05.5)
│                                   # - Config-driven feature removal (Step 06)
│
├── models/               # Configurations and schemas
│   ├── __init__.py
│   └── configs.py                 # Centralized configuration (dataclasses)
│                                   # - Loads hyperparameters from data/xgboost_hyperparameters.json
│                                   # - Defines feature selection (excluded_features)
│                                   # - Config-driven approach (no hardcoded values)
│
├── training/             # Training scripts
│   ├── __init__.py
│   ├── train.py                   # Training using hyperparameters from JSON
│   ├── tune.py                    # Grid Search + automatic JSON update
│   └── archive/                   # Deprecated scripts (history)
```

---

## 🎯 Architecture Principles

### 1. **Separation of Concerns**

* **Processing** – Pure, reusable, testable functions
* **Pipelines** – Orchestrate processing steps
* **Models** – Configs and schemas loaded from JSON
* **Training** – Model training and tuning

### 2. **Modularity**

* Each function in `processing/` does **exactly one thing**
* Easy to test in isolation
* Simple to compose into different pipelines

### 3. **Centralized Configuration + Versioned JSON**

* Hyperparameters in `data/xgboost_hyperparameters.json` (versioned)
* Feature selection in `models/configs.py` (hardcoded but documented) 
* Easy to change without touching code
* Automatic archival of previous versions in `data/archive/`

### 4. **Production-Ready**

* No hardcoded comparison values
* Dynamic baselines loaded from `data/archive/`
* Automatic versioning of models and hyperparameters

### 5. **Optimized Performance** ⚡

* PostgreSQL COPY for bulk inserts (70–80% faster)
* Parallelization of independent steps (ThreadPoolExecutor)
* Metadata tracking prevents data duplication
* Full pipeline runtime: **~62 s** (previously ~130 s) → **52% faster!**

---

## 🚀 Usage

### Run the Full Pipeline

```bash
# Direct execution
python -m src.ml.pipelines.data_pipeline
```

### Import Functions in Your Scripts

```python
# Load data (OPTIMIZED with COPY!)
from src.ml.processing.loader import load_from_postgresql, save_to_postgresql

# Normalize (RobustScaler for outliers)
from src.ml.processing.normalization import fit_and_transform_features

# Feature engineering (9 new features)
from src.ml.processing.feature_engineering import engineer_all_features

# Stratified train/test split
from src.ml.processing.splitters import stratified_train_test_split

# Use configuration
from src.ml.models.configs import config
print(config.database.connection_string)
```

---

## 📊 Data Pipeline (7 Steps) – COMPLETE ✅

### Step 01 – Load Raw Data ✅

* **Input:** `data/creditcard.csv`
* **Output:** PostgreSQL `raw_transactions` (284,807 rows)
* **Functions:** `processing/loader.py::load_csv_to_dataframe()` + `save_to_postgresql()`
* **Performance:** ~20 s (optimized COPY)

### Step 02 – Outlier Analysis ✅

* **Input:** `raw_transactions`
* **Output:** **Metadata only** (no data duplication)
* **Function:** `processing/cleaning.py::identify_outliers()`
* **Decision:** Preserve 100% of data (18.5% frauds among outliers)
* **Performance:** ~2.4 s
* **Parallelization:** Runs concurrently with Step 03 ⚡

### Step 03 – Handle Missing ✅

* **Input:** `raw_transactions`
* **Output:** **Metadata only**
* **Function:** `processing/validation.py::validate_no_missing_values()`
* **Result:** No missing values found (dataset fully complete)
* **Performance:** ~2.4 s
* **Parallelization:** Runs concurrently with Step 02 ⚡

### Step 04 – Normalize ✅

* **Input:** `raw_transactions`
* **Output:** `normalized_transactions` + `models/scalers.pkl`
* **Function:** `processing/normalization.py::fit_and_transform_features()`
* **Scalers:** RobustScaler (Amount), StandardScaler (Time)
* **Performance:** ~16.5 s (was ~90 s) → **81.6% faster!** 🔥
* **Optimization:** PostgreSQL COPY instead of `to_sql()`
* **Versioning:** Optional – `models/scalers_v1.0.0.pkl`

### Step 05 – Feature Engineering ✅

* **Input:** `normalized_transactions`
* **Output:** `engineered_transactions` (284,807 × 40 cols)
* **Function:** `processing/feature_engineering.py::engineer_all_features()`
* **Features:** Time_Period, Amount_Log, Amount_Bin, V-statistics (Mean/Std/Min/Max/Range)
* **Performance:** ~20.6 s (optimized COPY)

### Step 05.5 – Feature Selection Analysis ✅

* **Input:** `engineered_transactions`
* **Output:** `reports/feature_selection_analysis.json` (JSON report)
* **Function:** `processing/feature_selection.py::analyze_feature_importance()`
* **Analyses:** Pearson, Spearman, Mutual Information, VIF, Correlation Matrix
* **Performance:** ~15 s
* **Action:** Review report and update `configs.py`

### Step 06 – Apply Feature Selection ✅

* **Input:** `engineered_transactions`
* **Output:** `selected_features` (if configured)
* **Function:** `processing/feature_selection.py::apply_feature_selection()`
* **Config-Driven:** Removes features listed in `configs.py::FeatureSelectionConfig.excluded_features`
* **Performance:** ~5 s (or metadata-only if empty)
* **Versioning:** Increment `model_version` when changes occur

### Step 07 – Train/Test Split ✅

* **Input:** `selected_features` or `engineered_transactions`
* **Output:** `train_data` (227 845) + `test_data` (56 962)
* **Function:** `processing/splitters.py::stratified_train_test_split()`
* **Split:** 80/20 stratified (keeps 0.172% ratio)
* **Performance:** ~22.2 s (using COPY for both tables)

---

## 🎯 Feature Selection Strategy – Config-Driven

### Philosophy: Automated Analysis + Manual Decision

**Why not fully automate?**

* ❌ Automatic removal may discard non-linear but useful features
* ❌ Business context matters (e.g. high-value frauds, temporal patterns)
* ❌ Manual A/B testing gives better control

**Solution: Config-Driven Approach**

1. ✅ Step 05.5 – Run automated analysis (Pearson, Spearman, MI, VIF)
2. ✅ JSON Report – `reports/feature_selection_analysis.json`
3. ✅ Developer reviews the report
4. ✅ Update `configs.py` → `excluded_features`
5. ✅ Step 06 – Automatically applies removal on next run

### Development Workflow

*(The step-by-step commands remain unchanged; they are already in English in code.)*

---

### Benefits of This Approach

| Aspect               | Advantage                                           |
| -------------------- | --------------------------------------------------- |
| **Traceability**     | Git tracks feature-removal decisions (`configs.py`) |
| **Reproducibility**  | Same code + same config → same results              |
| **A/B Testing**      | Toggle features easily                              |
| **Business Context** | Human oversight before removal                      |
| **Production-Ready** | Mirrors enterprise config management                |
| **Educational**      | JSON report explains feature importance             |

---

## 🔖 Model Versioning

### Scalers (Pickle)

**MVP/POC (current)**

```python
models/scalers.pkl  # Active version
```

**Production (recommended)**

```python
models/
  scalers_v1.0.0.pkl
  scalers_v1.1.0.pkl  # With feature selection applied
  scalers_v1.2.0.pkl  # With new features added
```

**Enterprise**

```python
import mlflow
with mlflow.start_run():
    mlflow.sklearn.log_model(scaler, "scaler")
    mlflow.log_param("scaler_type", "RobustScaler")
    mlflow.log_param("model_version", "1.1.0")
```

**Why Pickle for Scalers?**

| Method     | Pros                                 | Cons                     | Use                   |
| ---------- | ------------------------------------ | ------------------------ | --------------------- |
| **Pickle** | ✅ Fast, native, keeps object methods | ❌ Python-only / insecure | ✅ MVP/POC             |
| **ONNX**   | ✅ Cross-platform                     | ❌ Complex conversion     | Multi-lang production |
| **MLflow** | ✅ Full version tracking + UI         | ❌ Extra infra needed     | ✅ Enterprise          |
| **SQL**    | ✅ Centralized                        | ❌ Loses object methods   | ❌ Not recommended     |

Decision: Use Pickle + manual versioning (MVP); document MLflow path for future upgrade.

---

## ⚙️ Configurations

All configs live in `src/ml/models/configs.py`,
covering Database, Paths, Data, Pipeline, and Feature Selection.

---

## ⚡ Implemented Optimizations

### 1. PostgreSQL COPY (81.6% faster) 🔥

```python
def save_to_postgresql(df, engine, table_name, use_copy=True):
    """
    Uses native PostgreSQL COPY for bulk inserts.
    Automatically falls back to to_sql() if COPY fails.
    """
```

### 2. Parallel Steps 02-03 (+18.6% faster) ⚡

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    future_02 = executor.submit(run_step_02_outlier_analysis)
    future_03 = executor.submit(run_step_03_handle_missing)
```

### 3. Metadata-Only Mode (ZERO unnecessary I/O)

Steps 02-03 store metadata only (no data duplication).

### 4. Overall Performance

| Metric             | Before     | After     | Gain        |
| ------------------ | ---------- | --------- | ----------- |
| Step 04            | 90 s       | 16.5 s    | **+81.6%**  |
| Steps 02-03        | 4.8 s      | 3.9 s     | **+18.6%**  |
| **Total Pipeline** | **~130 s** | **~62 s** | **+52% 🚀** |

---

## 🧪 Tests

```bash
python test_pipeline.py
python -c "from src.ml.pipelines.data_pipeline import DataPipeline; print('OK')"
python -c "from src.ml.processing.loader import save_to_postgresql; print('OK')"
python -c "from src.ml.processing.feature_engineering import engineer_all_features; print('OK')"
```

---

## 📈 Next Steps

1. ✅ Data Pipeline (01–07) – **Complete**
2. ✅ Performance Optimizations – **Complete**
3. ✅ Feature Selection Analysis – **Complete**
4. ⏳ Training Pipeline – in progress (Decision Tree, SVM, XGBoost)
5. ⏳ Validators – metrics, K-Fold, model comparison
6. ⏳ Flask Dashboard – real-time simulation and visuals
7. ⏳ [Optional] Kafka Enhancement (`plan_kafka.md`)

---

## 📚 References

* **MVC Pattern** – Model-View-Controller
* **ML Pipelines** – Modular and reproducible
* **SOLID Principles** – Single Responsibility, Open/Closed
* **PostgreSQL COPY** – native bulk insert (70–80% faster)
* **ThreadPoolExecutor** – I/O-bound parallelism
* **RobustScaler** – Outlier-resistant normalization

---

**Current Status:** ✅ **Full Data Pipeline Optimized (01–07)**
**Performance:** 52% faster (~62 s vs ~130 s)
**Feature Selection:** Config-driven, 38 final features
**Versioning:** Scalers with version support (`get_versioned_scaler_path`)
**Next:** ⏳ Train ML models (Decision Tree, SVM RBF, XGBoost)

### 🎯 Modeling Strategy

**Phase 1 – Baseline Training (38 features)**
– Decision Tree, SVM RBF, XGBoost
– Stratified K-Fold (k = 5)
– Metrics: PR-AUC, Recall, F1, Precision

**Phase 2 – Feature Importance Analysis**
– Use `feature_importances_` from XGBoost
– Remove low-importance features
– A/B test results

**Phase 3 – Refinement**
– Grid Search (`scoring='average_precision'`)
– Threshold tuning (~0.2–0.3 expected)

**Phase 4 – Flask Dashboard (MVP)**
– Interactive UI with real-time simulation (WebSocket streaming)

---
 