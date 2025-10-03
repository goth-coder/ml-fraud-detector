-- ============================================================================
-- SCHEMA SQL - Sistema de Detecção de Fraude em Cartão de Crédito
-- Tech Challenge Fase 3 - FIAP
-- ============================================================================

-- ============================================================================
-- 1. TRANSAÇÕES HISTÓRICAS
-- ============================================================================
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(100) PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    amount FLOAT NOT NULL,
    features JSONB NOT NULL,  -- Todas as features V1-V28 + Amount + Time
    predicted_class INTEGER NOT NULL CHECK (predicted_class IN (0, 1)),
    actual_class INTEGER NOT NULL CHECK (actual_class IN (0, 1)),
    fraud_probability FLOAT NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    is_correct BOOLEAN NOT NULL,
    model_version VARCHAR(50) DEFAULT 'v1.0',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Índices para otimização de queries
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_predicted_class ON transactions(predicted_class);
CREATE INDEX IF NOT EXISTS idx_transactions_actual_class ON transactions(actual_class);
CREATE INDEX IF NOT EXISTS idx_transactions_is_correct ON transactions(is_correct);

COMMENT ON TABLE transactions IS 'Histórico completo de transações processadas pelo sistema com predições';
COMMENT ON COLUMN transactions.features IS 'JSON com todas as features: V1-V28, Amount, Time';
COMMENT ON COLUMN transactions.predicted_class IS '0 = Legítima, 1 = Fraude';
COMMENT ON COLUMN transactions.fraud_probability IS 'Probabilidade de fraude (0.0 a 1.0)';

-- ============================================================================
-- 2. MÉTRICAS AGREGADAS (TIME-SERIES)
-- ============================================================================
CREATE TABLE IF NOT EXISTS metrics_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    total_transactions INTEGER NOT NULL CHECK (total_transactions >= 0),
    frauds_detected INTEGER NOT NULL CHECK (frauds_detected >= 0),
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_negatives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,
    precision FLOAT CHECK (precision BETWEEN 0 AND 1),
    recall FLOAT CHECK (recall BETWEEN 0 AND 1),
    f1_score FLOAT CHECK (f1_score BETWEEN 0 AND 1),
    window_minutes INTEGER DEFAULT 60  -- Janela de agregação (ex: última hora)
);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics_history(timestamp DESC);

COMMENT ON TABLE metrics_history IS 'Snapshots de métricas agregadas por janela temporal (salvas via APScheduler)';

-- ============================================================================
-- 3. METADADOS DE MODELOS TREINADOS
-- ============================================================================
CREATE TABLE IF NOT EXISTS trained_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL CHECK (
        model_type IN ('LogisticRegression', 'DecisionTree', 'SVM', 'XGBoost')
    ),
    hyperparameters JSONB NOT NULL,  -- Grid Search results
    pr_auc FLOAT NOT NULL CHECK (pr_auc BETWEEN 0 AND 1),
    roc_auc FLOAT NOT NULL CHECK (roc_auc BETWEEN 0 AND 1),
    precision FLOAT CHECK (precision BETWEEN 0 AND 1),
    recall FLOAT CHECK (recall BETWEEN 0 AND 1),
    f1_score FLOAT CHECK (f1_score BETWEEN 0 AND 1),
    best_threshold FLOAT CHECK (best_threshold BETWEEN 0 AND 1),
    training_duration_seconds FLOAT,
    is_active BOOLEAN DEFAULT FALSE,  -- Apenas 1 modelo ativo por vez
    trained_at TIMESTAMP DEFAULT NOW(),
    file_path VARCHAR(255)  -- Caminho do arquivo .pkl
);

CREATE INDEX IF NOT EXISTS idx_models_is_active ON trained_models(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_models_pr_auc ON trained_models(pr_auc DESC);

COMMENT ON TABLE trained_models IS 'Registro de todos os modelos treinados com hiperparâmetros e métricas';
COMMENT ON COLUMN trained_models.is_active IS 'Apenas 1 modelo deve estar ativo (usado no dashboard)';

-- ============================================================================
-- 4. DATASET SPLITS (TRAIN/TEST) - Reprodutibilidade
-- ============================================================================
CREATE TABLE IF NOT EXISTS processed_train_features (
    id SERIAL PRIMARY KEY,
    features JSONB NOT NULL,  -- V1-V28, Amount, Time (escalados)
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS processed_train_target (
    id SERIAL PRIMARY KEY,
    feature_id INTEGER REFERENCES processed_train_features(id) ON DELETE CASCADE,
    target_class INTEGER NOT NULL CHECK (target_class IN (0, 1)),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS processed_test_features (
    id SERIAL PRIMARY KEY,
    features JSONB NOT NULL,
    actual_class INTEGER NOT NULL CHECK (actual_class IN (0, 1)),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_test_features_actual_class ON processed_test_features(actual_class);

COMMENT ON TABLE processed_train_features IS 'Features do conjunto de treino (80% stratified)';
COMMENT ON TABLE processed_train_target IS 'Target do conjunto de treino';
COMMENT ON TABLE processed_test_features IS 'Features + target do conjunto de teste (20% stratified)';

-- ============================================================================
-- 5. VIEWS ÚTEIS PARA DASHBOARD
-- ============================================================================

-- View: Estatísticas em tempo real
CREATE OR REPLACE VIEW v_realtime_stats AS
SELECT 
    COUNT(*) as total_transactions,
    SUM(CASE WHEN predicted_class = 1 THEN 1 ELSE 0 END) as frauds_detected,
    SUM(CASE WHEN is_correct = TRUE THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(
        AVG(CASE WHEN predicted_class = 1 THEN 
            CASE WHEN actual_class = 1 THEN 1.0 ELSE 0.0 END 
        END)::numeric, 
    4) as precision,
    ROUND(
        AVG(CASE WHEN actual_class = 1 THEN 
            CASE WHEN predicted_class = 1 THEN 1.0 ELSE 0.0 END 
        END)::numeric, 
    4) as recall
FROM transactions
WHERE timestamp > NOW() - INTERVAL '1 hour';

-- View: Modelo ativo atual
CREATE OR REPLACE VIEW v_active_model AS
SELECT 
    model_name,
    model_type,
    pr_auc,
    roc_auc,
    best_threshold,
    trained_at,
    file_path
FROM trained_models
WHERE is_active = TRUE
LIMIT 1;

-- View: Performance por modelo
CREATE OR REPLACE VIEW v_model_comparison AS
SELECT 
    model_name,
    model_type,
    pr_auc,
    roc_auc,
    recall,
    precision,
    f1_score,
    trained_at
FROM trained_models
ORDER BY pr_auc DESC;

-- ============================================================================
-- 6. METADATA DE PROCESSAMENTO (PIPELINE STEPS)
-- ============================================================================
CREATE TABLE IF NOT EXISTS pipeline_metadata (
    id SERIAL PRIMARY KEY,
    step_number INTEGER NOT NULL CHECK (step_number BETWEEN 1 AND 6),
    step_name VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    rows_processed INTEGER NOT NULL,
    rows_output INTEGER NOT NULL,
    data_modified BOOLEAN NOT NULL,  -- Se dados foram alterados ou só analisados
    metadata JSONB,  -- Estatísticas do step (ex: outliers detectados, missing values, etc)
    duration_seconds FLOAT,
    status VARCHAR(20) DEFAULT 'success' CHECK (status IN ('success', 'failed', 'skipped'))
);

CREATE INDEX IF NOT EXISTS idx_pipeline_metadata_step ON pipeline_metadata(step_number, timestamp DESC);

COMMENT ON TABLE pipeline_metadata IS 'Histórico de execução do pipeline de processamento';
COMMENT ON COLUMN pipeline_metadata.data_modified IS 'TRUE = dados alterados, FALSE = apenas análise/metadata';
COMMENT ON COLUMN pipeline_metadata.metadata IS 'JSON com estatísticas do step (outliers, missing values, scalers, etc)';

-- ============================================================================
-- 7. FUNÇÕES AUXILIARES
-- ============================================================================

-- Função: Garantir apenas 1 modelo ativo
CREATE OR REPLACE FUNCTION fn_set_active_model()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_active = TRUE THEN
        -- Desativar todos os outros modelos
        UPDATE trained_models 
        SET is_active = FALSE 
        WHERE id != NEW.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Aplicar função ao atualizar modelo
CREATE TRIGGER trg_set_active_model
BEFORE UPDATE OF is_active ON trained_models
FOR EACH ROW
EXECUTE FUNCTION fn_set_active_model();

-- ============================================================================
-- 7. SEED DATA (OPCIONAL - para testes)
-- ============================================================================
COMMENT ON DATABASE fraud_detection IS 'Sistema de Detecção de Fraude - Tech Challenge Fase 3';

-- Mensagem de sucesso
DO $$
BEGIN
    RAISE NOTICE '✅ Schema criado com sucesso!';
    RAISE NOTICE '📊 Tabelas: transactions, metrics_history, trained_models, processed_*';
    RAISE NOTICE '👁️ Views: v_realtime_stats, v_active_model, v_model_comparison';
    RAISE NOTICE '🔧 Triggers: trg_set_active_model (garante apenas 1 modelo ativo)';
END $$;
