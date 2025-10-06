/*
 * JavaScript para Dashboard de Detecção de Fraudes 
 * 
 * Mudanças principais:
 * 1. Resultado agora é notificação toast (não card fixo)
 * 2. Confiança calculada pela distância de 50% (decision boundary)
 * 3. Estatísticas apenas na sidebar (removido duplicação inline)
 */

// ============================================
// CONFIGURAÇÃO
// ============================================

const API_BASE_URL = 'http://127.0.0.1:5000/api';

// ============================================
// FUNÇÕES AUXILIARES
// ============================================

/**
 * Mostra um toast de notificação com resultado da classificação
 * @param {string} message - Mensagem (HTML aceito)
 * @param {string} type - Tipo do toast ('success', 'danger', 'warning')
 */
function showToast(message, type = 'info') {
    const toastElement = document.getElementById('notification-toast');
    const toastMessage = document.getElementById('toast-message');
    const toastHeader = toastElement.querySelector('.toast-header');
    
    // Define a mensagem (aceita HTML)
    toastMessage.innerHTML = message;
    
    // Define cor do header baseado no tipo
    toastHeader.className = `toast-header bg-${type} text-white`;
    
    // Cria instância do toast do Bootstrap
    const toast = new bootstrap.Toast(toastElement, {
        delay: 10000 // 10 segundos para ler resultado completo (ground truth + predição)
    });
    
    // Mostra o toast
    toast.show();
}

/**
 * Formata data/hora para exibição com timezone
 * @param {string} isoString - Data em formato ISO
 * @returns {string} Data formatada (dd/mm/yyyy hh:mm UTC±HH:MM)
 */
function formatDateTime(isoString) {
    const date = new Date(isoString);
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    
    // Pega timezone offset (ex: -03:00 para BRT)
    const tzOffset = -date.getTimezoneOffset();
    const tzHours = String(Math.floor(Math.abs(tzOffset) / 60)).padStart(2, '0');
    const tzMinutes = String(Math.abs(tzOffset) % 60).padStart(2, '0');
    const tzSign = tzOffset >= 0 ? '+' : '-';
    const tz = `UTC${tzSign}${tzHours}:${tzMinutes}`;
    
    return `${day}/${month}/${year} ${hours}:${minutes} ${tz}`;
}

// ============================================
// LÓGICA PRINCIPAL - SIMULAÇÃO
// ============================================

/**
 * Executa simulação de transação
 * Chama a API, exibe resultado como toast e atualiza dados
 */
async function simulateTransaction() {
    const btn = document.getElementById('btn-simulate');
    
    // Desabilita botão durante requisição (evita duplo-clique)
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processando...';
    
    try {
        // Faz requisição HTTP POST para /api/simulate
        const response = await fetch(`${API_BASE_URL}/simulate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                transaction_type: selectedType  // 'legitimate' ou 'fraud'
            })
        });
        
        // Verifica se a resposta foi bem-sucedida
        if (!response.ok) {
            throw new Error('Erro na resposta da API');
        }
        
        // Converte resposta JSON
        const data = await response.json();
        
        // Exibe resultado como notificação
        displayResult(data);
        
        // Atualiza histórico e estatísticas
        loadHistory();
        loadStats();
        
    } catch (error) {
        console.error('Erro ao simular transação:', error);
        showToast('❌ Erro ao processar simulação. Tente novamente.', 'danger');
    } finally {
        // Reabilita botão
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-bolt me-2"></i>EXECUTAR SIMULAÇÃO';
    }
}

/**
 * Exibe resultado como notificação toast
 * 
 * LÓGICA:
 * - Predição: O que o modelo classificou (> 50% = fraude)
 * - Ground Truth: O que a transação realmente é (transaction_type)
 * - Mostra AMBOS para o usuário ver se acertou/errou
 * 
 * CONFIANÇA RESTRITA:
 * - ALTA: prob < 10% OU prob > 90% (muito longe do threshold)
 * - MODERADA: 10-20% OU 80-90% (relativamente confiável)
 * - BAIXA: 20-80% (próximo ao threshold, incerto)
 */
function displayResult(data) {
    const predictedFraud = data.is_fraud;
    const groundTruth = (data.transaction_type === 'fraud');
    const percentage = (data.fraud_probability * 100).toFixed(1);
    const prob = data.fraud_probability * 100;
    
    let confidence = 'BAIXA';
    let confidenceEmoji = '⚠️';
    if (prob < 10 || prob > 90) {
        confidence = 'ALTA';
        confidenceEmoji = '✅';
    } else if ((prob >= 10 && prob < 20) || (prob > 80 && prob <= 90)) {
        confidence = 'MODERADA';
        confidenceEmoji = '⚡';
    }
    
    const isCorrect = (predictedFraud === groundTruth);
    const resultEmoji = isCorrect ? '✅' : '❌';
    const resultLabel = isCorrect ? 'ACERTO' : (groundTruth ? 'FALSO NEGATIVO' : 'FALSO POSITIVO');
    
    let message = '';
    let toastType = '';
    
    if (predictedFraud) {
        toastType = 'danger';
        message = `
            <div class="d-flex align-items-center mb-2">
                <h5 class="mb-0 text-white">🚨 <strong>FRAUDE DETECTADA!</strong></h5>
            </div>
            <div class="mb-2 text-white">
                <strong>ID:</strong> #${data.classification_id}<br>
                <strong>Probabilidade:</strong> ${percentage}%<br>
                <strong>Confiança:</strong> ${confidenceEmoji} ${confidence}<br>
                <strong>Ground Truth:</strong> ${groundTruth ? '🚨 Fraude Real' : '✅ Legítima Real'}<br>
                <strong>Resultado:</strong> ${resultEmoji} ${resultLabel}
            </div>
            <small class="text-white-50">Modelo: ${data.model_version}</small>
        `;
    } else {
        toastType = 'success';
        message = `
            <div class="d-flex align-items-center mb-2">
                <h5 class="mb-0 text-white">✅ <strong>Transação Legítima</strong></h5>
            </div>
            <div class="mb-2 text-white">
                <strong>ID:</strong> #${data.classification_id}<br>
                <strong>Probabilidade de fraude:</strong> ${percentage}%<br>
                <strong>Confiança:</strong> ${confidenceEmoji} ${confidence}<br>
                <strong>Ground Truth:</strong> ${groundTruth ? '🚨 Fraude Real' : '✅ Legítima Real'}<br>
                <strong>Resultado:</strong> ${resultEmoji} ${resultLabel}
            </div>
            <small class="text-white-50">Modelo: ${data.model_version}</small>
        `;
    }
    
    showToast(message, toastType);
}

// ============================================
// LÓGICA - ESTATÍSTICAS
// ============================================

/**
 * Carrega estatísticas das últimas 24h
 * Atualiza apenas o painel da sidebar (não há mais inline)
 */
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const data = await response.json();
        
        const stats = data.stats;
        
        document.getElementById('stat-total-sidebar').textContent = stats.total;
        document.getElementById('stat-fraud-sidebar').textContent = stats.fraud_count;
        document.getElementById('stat-percentage-sidebar').textContent = `${stats.fraud_percentage.toFixed(1)}%`;
        document.getElementById('stat-recall-sidebar').textContent = `${stats.recall.toFixed(1)}%`;
        document.getElementById('stat-latency-sidebar').textContent = `${stats.avg_latency_ms.toFixed(0)}ms`;
        
        const percentageBar = document.getElementById('stat-percentage-bar');
        percentageBar.style.width = `${stats.fraud_percentage}%`;
        
    } catch (error) {
        console.error('Erro ao carregar estatísticas:', error);
        document.getElementById('stat-total-sidebar').textContent = '0';
        document.getElementById('stat-fraud-sidebar').textContent = '0';
        document.getElementById('stat-percentage-sidebar').textContent = '0%';
        document.getElementById('stat-recall-sidebar').textContent = '0.0%';
        document.getElementById('stat-latency-sidebar').textContent = '0ms';
    }
}

// ============================================
// LÓGICA - HISTÓRICO
// ============================================

/**
 * Carrega histórico de classificações
 * Busca últimas 10 classificações e popula tabela
 */
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/history?limit=10`);
        const data = await response.json();
        
        // A API retorna { success: true, history: [...], count: N }
        const history = data.history || [];
        
        const tbody = document.getElementById('history-table-body');
        
        // Se não há dados
        if (history.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center text-muted">
                        Nenhuma classificação encontrada
                    </td>
                </tr>
            `;
            return;
        }
        
        tbody.innerHTML = history.map(item => {
            const predictedFraud = item.fraud_probability > 0.5;
            
            return `
            <tr>
                <td>#${item.id}</td>
                <td>${formatDateTime(item.predicted_at)}</td>
                <td>
                    ${predictedFraud
                        ? '<span class="badge bg-danger">🚨 Fraude</span>' 
                        : '<span class="badge bg-success">✅ Legítima</span>'}
                </td>
                <td>
                    <span class="badge bg-secondary">${(item.fraud_probability * 100).toFixed(1)}%</span>
                </td>
                <td>
                    <span class="text-muted">$${item.amount.toFixed(2)}</span>
                </td>
                <td>
                    <span class="badge ${item.confidence === 'ALTA' ? 'bg-success' : item.confidence === 'MODERADA' ? 'bg-warning' : 'bg-secondary'}">${item.confidence}</span>
                </td>
                <td>
                    <small class="text-muted">${item.model_version}</small>
                </td>
            </tr>
        `;}).join('');
        
    } catch (error) {
        console.error('Erro ao carregar histórico:', error);
        document.getElementById('history-table-body').innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-danger">
                    <i class="fas fa-exclamation-triangle"></i> Erro ao carregar dados
                </td>
            </tr>
        `;
    }
}

// ============================================
// LÓGICA - HEALTH CHECK
// ============================================

/**
 * Verifica status da API
 * Atualiza badge no navbar
 */
async function checkHealth() {
    try {
        const response = await fetch('http://127.0.0.1:5000/health');
        const health = await response.json();
        
        const badge = document.getElementById('status-badge');
        if (health.status === 'healthy') {
            badge.className = 'badge bg-success';
            badge.innerHTML = '<i class="fas fa-check-circle"></i> API Online';
        } else {
            badge.className = 'badge bg-warning';
            badge.innerHTML = '<i class="fas fa-exclamation-triangle"></i> API com problemas';
        }
    } catch (error) {
        console.error('Erro no health check:', error);
        const badge = document.getElementById('status-badge');
        badge.className = 'badge bg-danger';
        badge.innerHTML = '<i class="fas fa-times-circle"></i> API Offline';
    }
}

// ============================================
// LÓGICA - TOGGLE SWITCH
// ============================================

let selectedType = 'legitimate'; // Estado do toggle

/**
 * Atualiza visual do toggle switch
 */
function updateToggleVisual() {
    const options = document.querySelectorAll('.toggle-option');
    const toggleInput = document.getElementById('transaction-toggle');
    
    options.forEach(option => {
        option.classList.remove('active');
    });
    
    if (selectedType === 'legitimate') {
        options[0].classList.add('active');
        toggleInput.checked = false;
    } else {
        options[1].classList.add('active');
        toggleInput.checked = true;
    }
}

// ============================================
// LÓGICA - LIMPAR HISTÓRICO
// ============================================

/**
 * Limpa todo o histórico de classificações
 * Pede confirmação antes de executar
 */
async function clearHistory() {
    if (!confirm('⚠️ Tem certeza que deseja limpar TODO o histórico?\n\nEsta ação NÃO pode ser desfeita!')) {
        return;
    }
    
    const btn = document.getElementById('btn-clear-history');
    
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Limpando...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/clear-history`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Error clearing history');
        }
        
        const data = await response.json();
        
        showToast(`✅ History cleared successfully! ${data.deleted_count} registros removidos.`, 'success');
        
        loadHistory();
        loadStats();
        
    } catch (error) {
        console.error('Error clearing history:', error);
        showToast('❌ Error clearing history. Tente novamente.', 'danger');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-trash-alt"></i> Limpar Histórico';
    }
}

// ============================================
// INICIALIZAÇÃO
// ============================================

/**
 * Executa quando a página carregar completamente
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Dashboard inicializado');
    
    // Carrega dados iniciais
    loadHistory();
    loadStats();
    checkHealth();
    
    // Health check periódico (a cada 30 segundos)
    setInterval(checkHealth, 30000);
    
    // Event Listeners
    document.getElementById('btn-simulate').addEventListener('click', simulateTransaction);
    document.getElementById('btn-refresh-history').addEventListener('click', loadHistory);
    document.getElementById('btn-clear-history').addEventListener('click', clearHistory);
    
    // Toggle switch event listeners
    const toggleInput = document.getElementById('transaction-toggle');
    const toggleOptions = document.querySelectorAll('.toggle-option');
    
    toggleInput.addEventListener('change', function() {
        selectedType = this.checked ? 'fraud' : 'legitimate';
        updateToggleVisual();
    });
    
    toggleOptions.forEach(option => {
        option.addEventListener('click', function() {
            selectedType = this.dataset.type;
            updateToggleVisual();
        });
    });
    
    // Inicializa visual do toggle
    updateToggleVisual();
});
