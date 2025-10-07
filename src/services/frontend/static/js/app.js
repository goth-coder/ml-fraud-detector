/*
 * JavaScript for Fraud Detection Dashboard
 * 
 * Key changes:
 * 1. Result is now a toast notification (not a fixed card)
 * 2. Confidence calculated by distance from 50% (decision boundary)
 * 3. Stats only in the sidebar (removed inline duplication)
 */

// ============================================
// CONFIGURATION
// ============================================

const API_BASE_URL = 'http://127.0.0.1:5000/api';

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * Show a toast notification with classification result
 * @param {string} message - Message (HTML allowed)
 * @param {string} type - Toast type ('success', 'danger', 'warning')
 */
function showToast(message, type = 'info') {
    const toastElement = document.getElementById('notification-toast');
    const toastMessage = document.getElementById('toast-message');
    const toastHeader = toastElement.querySelector('.toast-header');
    
    // Set message (HTML allowed)
    toastMessage.innerHTML = message;
    
    // Set header color based on type
    toastHeader.className = `toast-header bg-${type} text-white`;
    
    // Create Bootstrap toast instance
    const toast = new bootstrap.Toast(toastElement, {
        delay: 10000 // 10 seconds to read full result (ground truth + prediction)
    });
    
    // Show toast
    toast.show();
}

/**
 * Format date/time for display with timezone
 * @param {string} isoString - Date in ISO format
 * @returns {string} Formatted date (mm/dd/yyyy hh:mm UTC±HH:MM)
 */
function formatDateTime(isoString) {
    const date = new Date(isoString);
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const year = date.getFullYear();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    
    // Get timezone offset (e.g., -03:00 for BRT)
    const tzOffset = -date.getTimezoneOffset();
    const tzHours = String(Math.floor(Math.abs(tzOffset) / 60)).padStart(2, '0');
    const tzMinutes = String(Math.abs(tzOffset) % 60).padStart(2, '0');
    const tzSign = tzOffset >= 0 ? '+' : '-';
    const tz = `UTC${tzSign}${tzHours}:${tzMinutes}`;
    
    return `${month}/${day}/${year} ${hours}:${minutes} ${tz}`;
}

// ============================================
// MAIN LOGIC - SIMULATION
// ============================================

/**
 * Execute transaction simulation
 * Calls the API, shows result as toast, and updates data
 */
async function simulateTransaction() {
    const btn = document.getElementById('btn-simulate');
    
    // Disable button during request (avoid double click)
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
    
    try {
        // Make HTTP POST request to /api/simulate
        const response = await fetch(`${API_BASE_URL}/simulate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                transaction_type: selectedType  // 'legitimate' or 'fraud'
            })
        });
        
        // Check if response was successful
        if (!response.ok) {
            throw new Error('API response error');
        }
        
        // Parse JSON response
        const data = await response.json();
        
        // Show result as notification
        displayResult(data);
        
        // Refresh history and stats
        loadHistory();
        loadStats();
        
    } catch (error) {
        console.error('Error simulating transaction:', error);
        showToast('❌ Error processing simulation. Please try again.', 'danger');
    } finally {
        // Re-enable button
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-bolt me-2"></i>RUN SIMULATION';
    }
}

/**
 * Show result as toast notification
 * 
 * LOGIC:
 * - Prediction: What the model classified (> 50% = fraud)
 * - Ground Truth: What the transaction really is (transaction_type)
 * - Show BOTH so the user sees hits/misses
 * 
 * RESTRICTED CONFIDENCE:
 * - HIGH: prob < 10% OR prob > 90% (far from threshold)
 * - MODERATE: 10-20% OR 80-90% (reasonably confident)
 * - LOW: 20-80% (near threshold, uncertain)
 */
function displayResult(data) {
    const predictedFraud = data.is_fraud;
    const groundTruth = (data.transaction_type === 'fraud');
    const percentage = (data.fraud_probability * 100).toFixed(1);
    const prob = data.fraud_probability * 100;
    
    let confidence = 'LOW';
    let confidenceEmoji = '⚠️';
    if (prob < 10 || prob > 90) {
        confidence = 'HIGH';
        confidenceEmoji = '✅';
    } else if ((prob >= 10 && prob < 20) || (prob > 80 && prob <= 90)) {
        confidence = 'MODERATE';
        confidenceEmoji = '⚡';
    }
    
    const isCorrect = (predictedFraud === groundTruth);
    const resultEmoji = isCorrect ? '✅' : '❌';
    const resultLabel = isCorrect ? 'HIT' : (groundTruth ? 'FALSE NEGATIVE' : 'FALSE POSITIVE');
    
    let message = '';
    let toastType = '';
    
    if (predictedFraud) {
        toastType = 'danger';
        message = `
            <div class="d-flex align-items-center mb-2">
                <h5 class="mb-0 text-white">🚨 <strong>FRAUD DETECTED!</strong></h5>
            </div>
            <div class="mb-2 text-white">
                <strong>ID:</strong> #${data.classification_id}<br>
                <strong>Probability:</strong> ${percentage}%<br>
                <strong>Confidence:</strong> ${confidenceEmoji} ${confidence}<br>
                <strong>Ground Truth:</strong> ${groundTruth ? '🚨 Actual Fraud' : '✅ Actual Legitimate'}<br>
                <strong>Result:</strong> ${resultEmoji} ${resultLabel}
            </div>
            <small class="text-white-50">Modelo: ${data.model_version}</small>
        `;
    } else {
        toastType = 'success';
        message = `
            <div class="d-flex align-items-center mb-2">
                <h5 class="mb-0 text-white">✅ <strong>Legitimate Transaction</strong></h5>
            </div>
            <div class="mb-2 text-white">
                <strong>ID:</strong> #${data.classification_id}<br>
                <strong>Fraud probability:</strong> ${percentage}%<br>
                <strong>Confidence:</strong> ${confidenceEmoji} ${confidence}<br>
                <strong>Ground Truth:</strong> ${groundTruth ? '🚨 Actual Fraud' : '✅ Actual Legitimate'}<br>
                <strong>Result:</strong> ${resultEmoji} ${resultLabel}
            </div>
            <small class="text-white-50">Modelo: ${data.model_version}</small>
        `;
    }
    
    showToast(message, toastType);
}

// ============================================
// LOGIC - STATS
// ============================================

/**
 * Load statistics for the last 24h
 * Updates only the sidebar panel (no inline anymore)
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
        console.error('Error loading statistics:', error);
        document.getElementById('stat-total-sidebar').textContent = '0';
        document.getElementById('stat-fraud-sidebar').textContent = '0';
        document.getElementById('stat-percentage-sidebar').textContent = '0%';
        document.getElementById('stat-recall-sidebar').textContent = '0.0%';
        document.getElementById('stat-latency-sidebar').textContent = '0ms';
    }
}

// ============================================
// LOGIC - HISTORY
// ============================================

/**
 * Load classification history
 * Fetch last 10 classifications and populate table
 */
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/history?limit=10`);
        const data = await response.json();
        
    // API returns { success: true, history: [...], count: N }
        const history = data.history || [];
        
        const tbody = document.getElementById('history-table-body');
        
        // If no data
        if (history.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center text-muted">
                        No classifications found
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
                        ? '<span class="badge bg-danger">🚨 Fraud</span>' 
                        : '<span class="badge bg-success">✅ Legitimate</span>'}
                </td>
                <td>
                    <span class="badge bg-secondary">${(item.fraud_probability * 100).toFixed(1)}%</span>
                </td>
                <td>
                    <span class="text-muted">$${item.amount.toFixed(2)}</span>
                </td>
                <td>
                    <span class="badge ${item.confidence === 'HIGH' ? 'bg-success' : item.confidence === 'MODERATE' ? 'bg-warning' : 'bg-secondary'}">${item.confidence}</span>
                </td>
                <td>
                    <small class="text-muted">${item.model_version}</small>
                </td>
            </tr>
        `;}).join('');
        
    } catch (error) {
        console.error('Error loading history:', error);
        document.getElementById('history-table-body').innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-danger">
                    <i class="fas fa-exclamation-triangle"></i> Error loading data
                </td>
            </tr>
        `;
    }
}

// ============================================
// LOGIC - HEALTH CHECK
// ============================================

/**
 * Check API status
 * Update navbar badge
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
            badge.innerHTML = '<i class="fas fa-exclamation-triangle"></i> API issues';
        }
    } catch (error) {
        console.error('Health check error:', error);
        const badge = document.getElementById('status-badge');
        badge.className = 'badge bg-danger';
        badge.innerHTML = '<i class="fas fa-times-circle"></i> API Offline';
    }
}

// ============================================
// LOGIC - TOGGLE SWITCH
// ============================================

let selectedType = 'legitimate'; // Toggle state

/**
 * Update toggle switch visuals
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
// LOGIC - CLEAR HISTORY
// ============================================

/**
 * Clear the entire classification history
 * Ask for confirmation before executing
 */
async function clearHistory() {
    if (!confirm('⚠️ Are you sure you want to CLEAR ALL history?\n\nThis action CANNOT be undone!')) {
        return;
    }
    
    const btn = document.getElementById('btn-clear-history');
    
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Clearing...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/clear-history`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Error clearing history');
        }
        
        const data = await response.json();
        
    showToast(`✅ History cleared successfully! ${data.deleted_count} records removed.`, 'success');
        
        loadHistory();
        loadStats();
        
    } catch (error) {
        console.error('Error clearing history:', error);
        showToast('❌ Error clearing history. Please try again.', 'danger');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-trash-alt"></i> Clear History';
    }
}

// ============================================
// INITIALIZATION
// ============================================

/**
 * Runs when the page has fully loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Dashboard initialized');
    
    // Load initial data
    loadHistory();
    loadStats();
    checkHealth();
    
    // Periodic health check (every 30 seconds)
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
    
    // Initialize toggle visuals
    updateToggleVisual();
});
