-- Migração: Criar tabelas para Webapp de Detecção de Fraudes
-- Data: 2025-10-06
-- Descrição: Adiciona tabelas classification_results e simulated_transactions

-- Tabela de resultados de classificação
CREATE TABLE IF NOT EXISTS classification_results (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(20) NOT NULL,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    fraud_probability FLOAT NOT NULL,
    transaction_features JSONB NOT NULL,
    source VARCHAR(20) DEFAULT 'webapp',
    
    -- Índices para performance
    CONSTRAINT check_probability CHECK (fraud_probability >= 0 AND fraud_probability <= 1)
);

-- Índices para queries do dashboard
CREATE INDEX IF NOT EXISTS idx_classification_predicted_at 
    ON classification_results(predicted_at DESC);
    
CREATE INDEX IF NOT EXISTS idx_classification_is_fraud 
    ON classification_results(is_fraud);
    
CREATE INDEX IF NOT EXISTS idx_classification_model_version 
    ON classification_results(model_version);

-- Tabela de transações simuladas
CREATE TABLE IF NOT EXISTS simulated_transactions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    transaction_type VARCHAR(20) NOT NULL,
    features JSONB NOT NULL,
    classification_id INTEGER,
    
    -- Foreign key para resultado de classificação
    CONSTRAINT fk_classification 
        FOREIGN KEY (classification_id) 
        REFERENCES classification_results(id)
        ON DELETE SET NULL,
    
    -- Validação de tipo
    CONSTRAINT check_transaction_type 
        CHECK (transaction_type IN ('legitimate', 'fraud'))
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_simulated_created_at 
    ON simulated_transactions(created_at DESC);
    
CREATE INDEX IF NOT EXISTS idx_simulated_type 
    ON simulated_transactions(transaction_type);

-- Comentários para documentação
COMMENT ON TABLE classification_results IS 'Resultados de classificação do webapp (histórico de predições)';
COMMENT ON TABLE simulated_transactions IS 'Transações simuladas pelo dashboard';

COMMENT ON COLUMN classification_results.transaction_features IS 'JSON com 33 features da transação';
COMMENT ON COLUMN classification_results.fraud_probability IS 'Probabilidade de fraude (0.0 a 1.0)';
COMMENT ON COLUMN classification_results.source IS 'Origem da predição: webapp, api, batch';

COMMENT ON COLUMN simulated_transactions.features IS 'JSON com features completas da transação';
COMMENT ON COLUMN simulated_transactions.classification_id IS 'Link opcional com classification_results';

-- Verificação
SELECT 'Tabelas criadas com sucesso!' AS status;
