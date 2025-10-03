# 🗄️ Data Architecture Report - Fraud Detection System
## PostgreSQL + Pickle Hybrid Architecture

**Project**: Credit Card Fraud Detection  
**Date**: October 2025  
**Architecture**: Hybrid (PostgreSQL for data, Pickle for models)

---

## 📋 Executive Summary

The fraud detection system uses a **hybrid persistence architecture**:
- **PostgreSQL**: Transaction data, train/test splits, metrics, metadata
- **Pickle**: Trained ML models, scalers, preprocessing artifacts
- **JSON**: Versioned hyperparameters with automatic archiving

This separation enables:
- ✅ **Data Versioning**: Full audit trail of all transactions in database
- ✅ **Model Versioning**: Independent model deployment without data migration
- ✅ **Hyperparameter Versioning**: JSON-based configs with archive system
- ✅ **Fast Inference**: Models loaded once from pickle, data streamed from PostgreSQL
- ✅ **Scalability**: Database handles growing data, models stay lightweight
- ✅ **Production-Ready**: No hardcoded values, dynamic baseline comparisons

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PERSISTENCE LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐         ┌────────────────────────┐  │
│  │   PostgreSQL 15      │         │   Pickle Files         │  │
│  │   (Database)         │         │   (Filesystem)         │  │
│  ├──────────────────────┤         ├────────────────────────┤  │
│  │                      │         │                        │  │
│  │ • Raw Transactions   │         │ • XGBoost Models       │  │
│  │ • Cleaned Data       │         │   - xgboost_v1.1.0.pkl │  │
│  │ • Normalized Data    │         │   - xgboost_v2.0.0.pkl │  │
│  │ • Engineered Features│         │   - xgboost_v2.1.0.pkl │  │
│  │ • Train/Test Splits  │         │                        │  │
│  │ • Predictions        │         │ • Scalers              │  │
│  │ • Metrics History    │         │   - scalers.pkl        │  │
│  │ • Pipeline Metadata  │         │   (RobustScaler)       │  │
│  │                      │         │                        │  │
│  │ Size: ~500 MB        │         │ Size: ~1 MB            │  │
│  │ Growth: Linear       │         │ Growth: Per version    │  │
│  └──────────────────────┘         └────────────────────────┘  │
│           ▲                                   ▲                │
│           │                                   │                │
│           │         ┌────────────────────────┐│                │
│           │         │  JSON Config Files     ││                │
│           │         │  (data/)               ││                │
│           │         ├────────────────────────┤│                │
│           │         │                        ││                │
│           │         │ • xgboost_hyperparams  ││                │
│           │         │   .json (active)       ││                │
│           │         │ • archive/             ││                │
│           │         │   - v2.1.0_20251003    ││                │
│           │         │   - v2.0.0_20251003    ││                │
│           │         │                        ││                │
│           │         │ Size: ~5 KB            ││                │
│           │         │ Growth: Per tune       ││                │
│           │         └────────────────────────┘│                │
│           │                      ▲             │                │
│           ├──────────────────────┼─────────────┤                │
│                          │                                     │
│                ┌─────────▼──────────┐                          │
│                │  Application       │                          │
│                │  (Python Scripts)  │                          │
│                │                    │                          │
│                │ • Data Pipeline    │                          │
│                │ • Training Scripts │                          │
│                │ • Inference CLI    │                          │
│                └────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ PostgreSQL Database Schema

### Database Configuration
- **Engine**: PostgreSQL 15
- **Deployment**: Docker Compose (development)
- **Connection**: SQLAlchemy ORM
- **Optimization**: Connection pooling, COPY for bulk inserts

### Table 1: `raw_transactions`
**Purpose**: Original CSV data (first load)  
**Size**: 284,807 rows × 31 columns  
**Storage**: ~67 MB

```sql
CREATE TABLE raw_transactions (
    Time FLOAT NOT NULL,
    V1 FLOAT NOT NULL,
    V2 FLOAT NOT NULL,
    ...
    V28 FLOAT NOT NULL,
    Amount FLOAT NOT NULL,
    Class INTEGER NOT NULL CHECK (Class IN (0, 1))
);

CREATE INDEX idx_raw_class ON raw_transactions(Class);
CREATE INDEX idx_raw_amount ON raw_transactions(Amount);
```

**Why Store**: 
- ✅ Audit trail of original data
- ✅ Allows pipeline re-runs from scratch
- ✅ Historical reference for data drift detection

---

### Table 2: `normalized_transactions`
**Purpose**: After RobustScaler normalization (Step 04)  
**Size**: 284,807 rows × 31 columns (same as raw)  
**Storage**: ~67 MB

```sql
CREATE TABLE normalized_transactions (
    Time FLOAT NOT NULL,            -- StandardScaler applied
    V1 FLOAT NOT NULL,              -- PCA features (already normalized)
    V2 FLOAT NOT NULL,
    ...
    V28 FLOAT NOT NULL,
    Amount FLOAT NOT NULL,          -- RobustScaler applied (median/IQR)
    Class INTEGER NOT NULL
);
```

**Why Store**:
- ✅ Pre-normalized data ready for feature engineering
- ✅ Avoids re-fitting scalers on every run
- ✅ Consistent scaling across train/test splits

**Optimization**: Uses PostgreSQL COPY (81.6% faster than INSERT)

---

### Table 3: `feature_engineered`
**Purpose**: After feature engineering (Step 05)  
**Size**: 284,807 rows × 40 columns (31 original + 9 new)  
**Storage**: ~90 MB

```sql
CREATE TABLE feature_engineered (
    -- Original 31 features
    Time FLOAT NOT NULL,
    V1 FLOAT NOT NULL,
    ...
    V28 FLOAT NOT NULL,
    Amount FLOAT NOT NULL,
    Class INTEGER NOT NULL,
    
    -- Engineered features (9 new)
    Time_Hour INTEGER,                -- Hour of day (0-23)
    Time_Period_of_Day VARCHAR(20),   -- Morning/Afternoon/Evening/Night
    Amount_Log FLOAT,                 -- log1p(Amount)
    Amount_Bin VARCHAR(20),           -- Low/Medium/High/Very_High
    V_Mean FLOAT,                     -- Mean of V1-V28
    V_Std FLOAT,                      -- Std of V1-V28
    V_Min FLOAT,                      -- Min of V1-V28
    V_Max FLOAT,                      -- Max of V1-V28
    V_Range FLOAT                     -- Max - Min of V1-V28
);
```

**Why Store**:
- ✅ Feature engineering expensive (computed once)
- ✅ Allows feature selection experiments
- ✅ Consistent features across train/test/inference

---

### Table 4: `feature_selected`
**Purpose**: After feature selection (Step 06)  
**Size**: 284,807 rows × 34 columns (40 - 6 excluded)  
**Storage**: ~75 MB

**Excluded Features** (from `excluded_features` in `configs.py`):
- `Time` (raw, redundant with Time_Hour)
- `Time_Period_of_Day` (importance 0.0002)
- `V8`, `V23`, `V13` (low importance)
- `Amount` (raw, redundant with Amount_Log/Amount_Bin)

```sql
CREATE TABLE feature_selected (
    V1 FLOAT NOT NULL,
    V2 FLOAT NOT NULL,
    ...
    V28 FLOAT NOT NULL,  -- Except V8, V23, V13
    Class INTEGER NOT NULL,
    
    Time_Hour INTEGER,
    Amount_Log FLOAT,
    Amount_Bin VARCHAR(20),
    V_Mean FLOAT,
    V_Std FLOAT,
    V_Min FLOAT,
    V_Max FLOAT,
    V_Range FLOAT
);
```

**Why Store**:
- ✅ Final feature set for training
- ✅ Config-driven removal (reproducible)
- ✅ Allows A/B testing different feature sets

---

### Table 5: `train_data`
**Purpose**: Training split (80%)  
**Size**: 227,845 rows × 34 columns  
**Frauds**: 394 (0.173%)  
**Storage**: ~60 MB

```sql
CREATE TABLE train_data (
    V1 FLOAT NOT NULL,
    V2 FLOAT NOT NULL,
    ...
    V28 FLOAT NOT NULL,
    Class INTEGER NOT NULL,
    
    Time_Hour INTEGER,
    Amount_Log FLOAT,
    Amount_Bin VARCHAR(20),
    V_Mean FLOAT,
    V_Std FLOAT,
    V_Min FLOAT,
    V_Max FLOAT,
    V_Range FLOAT
);

CREATE INDEX idx_train_class ON train_data(Class);
```

**Why Store**:
- ✅ Reproducible train/test splits
- ✅ Same split used for v1.1.0, v2.0.0, v2.1.0
- ✅ Allows model comparison on identical data
- ✅ Fast loading for Grid Search (no re-splitting)

---

### Table 6: `test_data`
**Purpose**: Test split (20%)  
**Size**: 56,962 rows × 34 columns  
**Frauds**: 98 (0.172%)  
**Storage**: ~15 MB

```sql
CREATE TABLE test_data (
    -- Same schema as train_data
    V1 FLOAT NOT NULL,
    ...
);

CREATE INDEX idx_test_class ON test_data(Class);
```

**Why Store**:
- ✅ Hold-out set for final evaluation
- ✅ Never used during training/Grid Search
- ✅ Consistent evaluation across model versions

---

### Table 7: `transactions` (Production)
**Purpose**: Inference results storage  
**Size**: Initially empty (grows with usage)  
**Storage**: ~1 KB per transaction

```sql
CREATE TABLE transactions (
    transaction_id VARCHAR(100) PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    amount FLOAT NOT NULL,
    features JSONB NOT NULL,              -- All features as JSON
    predicted_class INTEGER NOT NULL,     -- 0 or 1
    actual_class INTEGER,                 -- NULL if unknown
    fraud_probability FLOAT NOT NULL,     -- Model confidence
    is_correct BOOLEAN,                   -- If actual_class known
    model_version VARCHAR(50),            -- e.g., 'v2.1.0'
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX idx_transactions_predicted ON transactions(predicted_class);
```

**Why Store**:
- ✅ Audit trail of all predictions
- ✅ Model monitoring (accuracy over time)
- ✅ Concept drift detection
- ✅ A/B testing different model versions

---

### Table 8: `metrics_history`
**Purpose**: Aggregated metrics over time  
**Size**: 1 row per time window (e.g., hourly)  
**Storage**: ~100 bytes per row

```sql
CREATE TABLE metrics_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    total_transactions INTEGER NOT NULL,
    frauds_detected INTEGER NOT NULL,
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_negatives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    window_minutes INTEGER DEFAULT 60
);

CREATE INDEX idx_metrics_timestamp ON metrics_history(timestamp DESC);
```

**Why Store**:
- ✅ Performance tracking over time
- ✅ Dashboard data source
- ✅ SLA monitoring (Precision, Recall targets)

---

### Table 9: `pipeline_metadata`
**Purpose**: Track pipeline execution  
**Size**: 1 row per pipeline run  
**Storage**: ~200 bytes per row

```sql
CREATE TABLE pipeline_metadata (
    id SERIAL PRIMARY KEY,
    step_name VARCHAR(100) NOT NULL,
    execution_time FLOAT NOT NULL,
    rows_processed INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'success'
);

CREATE INDEX idx_metadata_created ON pipeline_metadata(created_at DESC);
```

**Why Store**:
- ✅ Pipeline observability
- ✅ Performance monitoring
- ✅ Debugging failed runs

---

## 📦 Pickle Files (Filesystem)

### Location: `models/` directory

### File 1: `xgboost_v1.1.0.pkl`
**Purpose**: Baseline XGBoost model  
**Size**: ~270 KB  
**Created**: October 2025

```python
# Model structure
{
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'scale_pos_weight': 577,
    'n_features_in_': 37
}
```

**Why Pickle**:
- ✅ Fast loading (<1 second)
- ✅ Small file size (KB not MB)
- ✅ Standard scikit-learn format
- ✅ Independent of database

**Usage**:
```python
import pickle
with open('models/xgboost_v1.1.0.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

### File 2: `xgboost_v2.0.0.pkl`
**Purpose**: Grid Search optimized (37 features)  
**Size**: ~350 KB  
**Created**: October 2025

```python
# Model structure
{
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.3,
    'scale_pos_weight': 577,
    'colsample_bytree': 0.7,
    'subsample': 0.8,
    'min_child_weight': 1,
    'n_features_in_': 37
}
```

**Performance**:
- PR-AUC: 0.8847
- Precision: 85.42%
- FP: 14

---

### File 3: `xgboost_v2.1.0.pkl`
**Purpose**: Feature-reduced model (33 features) ⭐ **PRODUCTION**  
**Size**: ~352 KB  
**Created**: October 2025

```python
# Model structure
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.3,
    'scale_pos_weight': 577,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'min_child_weight': 1,
    'n_features_in_': 33  # 4 fewer than v2.0.0
}
```

**Performance**:
- PR-AUC: 0.8772
- Precision: 86.60%
- Recall: 85.71%
- FP: 13
- FN: 14

**Why This Model**:
- ✅ Best Precision/Recall/F1
- ✅ Fewest errors (13 FP, 14 FN)
- ✅ 11% simpler (33 vs 37 features)
- ✅ Faster inference

---

### File 4: `scalers.pkl`
**Purpose**: RobustScaler fitted on training data  
**Size**: ~5 KB  
**Created**: Step 04 of data pipeline

```python
# Scaler structure
{
    'Amount': RobustScaler(
        center_=median,
        scale_=IQR
    ),
    'Time': StandardScaler(
        mean_=mean,
        std_=std
    )
}
```

**Why Pickle**:
- ✅ Must use SAME scaler for train/test/inference
- ✅ Small file size (5 KB)
- ✅ Fast loading
- ✅ Versioned with model

**Usage**:
```python
import pickle
with open('models/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

# Apply to new data
new_data['Amount'] = scalers['Amount'].transform(new_data[['Amount']])
new_data['Time'] = scalers['Time'].transform(new_data[['Time']])
```

---

## 🔄 Data Flow: Training Pipeline

```
Step 01: CSV → raw_transactions (PostgreSQL)
            ↓
Step 02: Outlier analysis (metadata only)
            ↓
Step 03: Missing values check (metadata only)
            ↓
Step 04: Normalize → normalized_transactions + scalers.pkl
            ↓
Step 05: Feature engineering → feature_engineered
            ↓
Step 06: Feature selection → feature_selected (33 features)
            ↓
Step 07: Train/Test split → train_data + test_data
            ↓
Training: train_data + data/xgboost_hyperparameters.json → xgboost_v2.1.0.pkl
            ↓
Evaluation: test_data + xgboost_v2.1.0.pkl → metrics
            ↓
Archive: xgboost_hyperparameters.json → data/archive/v2.1.0_{timestamp}.json
```

**Key Decisions**:
- **Steps 02-03**: Metadata only (no data duplication)
- **Step 04**: PostgreSQL COPY (81.6% faster bulk insert)
- **Steps 02-03**: Parallelized via ThreadPoolExecutor
- **All tables**: Indexed on Class for fast filtering
- **Hyperparameters**: Loaded from JSON (not hardcoded)
- **Versioning**: Automatic archive with timestamps

---

## 🚀 Data Flow: Inference Pipeline

```
New Transaction (CLI)
        ↓
Load data/xgboost_hyperparameters.json (metadata)
        ↓
Load scalers.pkl (once at startup)
        ↓
Load xgboost_v2.1.0.pkl (once at startup)
        ↓
Normalize features (using scalers)
        ↓
Feature engineering (compute 9 features)
        ↓
Feature selection (remove 6 features)
        ↓
Model prediction (XGBoost)
        ↓
Store result → transactions (PostgreSQL)
        ↓
Return prediction
```

**Performance**:
- JSON loading: <1 ms (once)
- Model loading: <1 second (once)
- Inference per transaction: <10 ms
- Database insert: <5 ms (with COPY optimization)

---

## 💾 Storage Summary

### PostgreSQL Database
| Table | Rows | Columns | Size | Purpose |
|-------|------|---------|------|---------|
| raw_transactions | 284,807 | 31 | 67 MB | Original data |
| normalized_transactions | 284,807 | 31 | 67 MB | After scaling |
| feature_engineered | 284,807 | 40 | 90 MB | +9 features |
| feature_selected | 284,807 | 34 | 75 MB | -6 features |
| train_data | 227,845 | 34 | 60 MB | Training set |
| test_data | 56,962 | 34 | 15 MB | Test set |
| transactions | Variable | Varies | ~1 KB/row | Predictions |
| metrics_history | Variable | 11 | ~100 B/row | Aggregates |
| pipeline_metadata | Variable | 6 | ~200 B/row | Tracking |
| **TOTAL** | **~1.4M** | - | **~500 MB** | - |

### Pickle Files
| File | Size | Purpose |
|------|------|---------|
| xgboost_v1.1.0.pkl | 270 KB | Baseline model |
| xgboost_v2.0.0.pkl | 350 KB | Grid Search (37 features) |
| xgboost_v2.1.0.pkl | 352 KB | Production (33 features) ⭐ |
| scalers.pkl | 5 KB | RobustScaler fitted |
| **TOTAL** | **~1 MB** | - |

### JSON Configuration Files
| File | Size | Purpose |
|------|------|---------|
| data/xgboost_hyperparameters.json | ~2 KB | Active hyperparameters (loaded by configs.py) |
| data/archive/v2.1.0_{timestamp}.json | ~2 KB | Archived version (with metadata) |
| data/archive/v2.0.0_{timestamp}.json | ~2 KB | Previous optimized version |
| **TOTAL** | **~5 KB** | - |

**Total Project Storage**: ~501 MB (database) + 1 MB (models) + 5 KB (configs) = **~502 MB**

---

## 🎯 Design Rationale

### Why PostgreSQL for Data?

1. **ACID Compliance**: Transaction safety for production
2. **Query Flexibility**: Complex analytics queries (metrics over time)
3. **Indexing**: Fast lookups on Class, timestamp, prediction
4. **Scalability**: Handles millions of rows efficiently
5. **Audit Trail**: Full history of all data transformations
6. **Integration**: Standard SQL interface for dashboards/tools

### Why JSON for Hyperparameters?

1. **Versioning**: Each tune creates timestamped archive
2. **Human-Readable**: Easy to diff and review changes
3. **Git-Friendly**: Can version control active config
4. **Metadata**: Stores optimization details (CV folds, elapsed time, metrics)
5. **Dynamic Baselines**: tune.py loads previous versions from archive/ automatically
6. **Production-Ready**: No hardcoded comparisons, scales to any number of versions
7. **Portability**: JSON standard across languages/platforms

### Why Pickle for Models?

1. **Versioning**: Each model version is independent file
2. **Fast Loading**: <1 second to load 350 KB model
3. **Portability**: Copy file to deploy model (no database migration)
4. **Simplicity**: Standard Python serialization
5. **Size Efficiency**: KB not MB (vs storing in database)
6. **Model Registry**: Easy to add/remove versions

### Why NOT Store Models in Database?

- ❌ **Slower Loading**: Query + deserialize slower than file load
- ❌ **Versioning Complexity**: Need BLOB columns, version tables
- ❌ **Size**: Database grows unnecessarily
- ❌ **Portability**: Harder to move models between environments
- ✅ **Pickle is Standard**: scikit-learn, XGBoost native support

### Why NOT Use ONNX Instead of Pickle?

**ONNX Considered but NOT Implemented**:
- ⚠️ **Complexity**: Requires onnxmltools, onnxruntime
- ⚠️ **Compatibility**: XGBoost → ONNX conversion can fail
- ⚠️ **Overkill**: Pickle sufficient for Python-only deployment
- ✅ **Future Option**: If cross-language inference needed

**When to Use ONNX**:
- Deploying to non-Python environments (C++, Java)
- Hardware acceleration (GPU inference)
- Cross-framework compatibility

---

## 🔐 Data Security Considerations

### Database
- ✅ **Credentials**: Environment variables only
- ✅ **Connection Pooling**: Reuse connections (SQLAlchemy)
- ✅ **Indexing**: Prevent full table scans
- ⚠️ **Encryption**: Not enabled (development only)

### Pickle Files
- ✅ **Version Control**: `.pkl` files NOT in git (in `.gitignore`)
- ✅ **Binary Format**: Not human-readable
- ⚠️ **Integrity**: No checksum validation (future enhancement)

---

## 📊 Performance Benchmarks

### Database Operations
| Operation | Method | Time | Optimization |
|-----------|--------|------|--------------|
| Bulk Insert (284K rows) | to_sql() | 90s | Baseline |
| Bulk Insert (284K rows) | **COPY** | **16.5s** | **81.6% faster** |
| Train/Test Load (227K rows) | read_sql() | <5s | Indexed |
| Query by Class | SELECT WHERE | <100ms | Indexed |

### Model Operations
| Operation | Time | Notes |
|-----------|------|-------|
| Load xgboost_v2.1.0.pkl | <1s | Once at startup |
| Load scalers.pkl | <100ms | Once at startup |
| Inference (single transaction) | <10ms | After loading |
| Batch inference (1000 transactions) | <500ms | Vectorized |

---

## 🚀 Future Enhancements

### Database
- ⏸️ **Partitioning**: Partition `transactions` by timestamp (monthly)
- ⏸️ **Replication**: Master-slave for read scaling
- ⏸️ **Materialized Views**: Pre-aggregate metrics for dashboard
- ⏸️ **TimescaleDB**: Time-series optimization for metrics_history

### Models
- ⏸️ **ONNX Export**: For cross-platform deployment
- ⏸️ **Model Registry**: MLflow for version tracking
- ⏸️ **Checksum Validation**: Ensure model integrity
- ⏸️ **Compression**: Gzip pickle files (50% size reduction)

### Monitoring
- ⏸️ **Prometheus Metrics**: Export database stats
- ⏸️ **Grafana Dashboard**: Real-time monitoring
- ⏸️ **Alerting**: Slack notifications for concept drift

---

## 📚 References

### Database Files
- `database/schema.sql` - Complete SQL schema
- `database/docker-compose.yml` - PostgreSQL 15 setup
- `src/services/database/connection.py` - SQLAlchemy configuration

### Model Files
- `models/xgboost_v1.1.0.pkl` - Baseline
- `models/xgboost_v2.0.0.pkl` - Grid Search (37 features)
- `models/xgboost_v2.1.0.pkl` - Production (33 features) ⭐
- `models/scalers.pkl` - RobustScaler

### Configuration
- `src/ml/models/configs.py` - Database, paths, feature exclusion
- `.env` (not in git) - Database credentials

### Documentation
- `docs/PERFORMANCE_OPTIMIZATION.md` - PostgreSQL COPY optimization
- `docs/MODEL_SELECTION_FINAL.md` - Model comparison
- `docs/plan_main.md` - Complete pipeline architecture

---

**Report Generated**: October 2025  
**Status**: ✅ Complete  
**Architecture**: PostgreSQL (data) + Pickle (models) Hybrid
