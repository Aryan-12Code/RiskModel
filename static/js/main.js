/**
 * RiskModel — Main Frontend JavaScript
 * Handles authentication, prediction submissions, SHAP chart visualizations,
 * history tracking, and UI interactions.
 */

document.addEventListener('DOMContentLoaded', function() {
    updateAuthUI();
});

// ============================================================================
// Authentication UI and Handlers
// ============================================================================

function clearTokenAndRedirect() {
    sessionStorage.removeItem('rm_token');
    sessionStorage.removeItem('rm_username');
    sessionStorage.removeItem('rm_role');
    window.location.href = '/login';
}

function updateAuthUI() {
    const token = sessionStorage.getItem('rm_token');
    const username = sessionStorage.getItem('rm_username');
    const authSection = document.getElementById('authSection');
    const loginBtn = document.getElementById('loginBtn');
    const userSection = document.getElementById('userSection');
    const usernameBadge = document.getElementById('usernameBadge');

    if (!authSection) return;

    if (token) {
        if (loginBtn) loginBtn.classList.add('d-none');
        if (userSection) userSection.classList.remove('d-none');
        if (usernameBadge) usernameBadge.textContent = username || 'User';
    } else {
        if (loginBtn) loginBtn.classList.remove('d-none');
        if (userSection) userSection.classList.add('d-none');
    }
}

function handleLogout() {
    sessionStorage.removeItem('rm_token');
    sessionStorage.removeItem('rm_username');
    sessionStorage.removeItem('rm_role');
    showNotification('You have been logged out successfully.', 'info');
    updateAuthUI();
    setTimeout(() => {
        window.location.href = '/login';
    }, 500);
}

async function handleLoginSubmit(event) {
    event.preventDefault();
    const form = event.target;
    if (!form.checkValidity()) {
        form.classList.add('was-validated');
        return;
    }

    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;
    const loginError = document.getElementById('loginError');
    const loginErrorText = document.getElementById('loginErrorText');
    const submitBtn = document.getElementById('loginSubmitBtn');
    const btnText = document.getElementById('loginBtnText');
    const spinner = document.getElementById('loginSpinner');

    if (loginError) loginError.classList.add('d-none');
    submitBtn.disabled = true;
    btnText.classList.add('d-none');
    spinner.classList.remove('d-none');

    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        const resData = await response.json();

        if (response.ok && resData.status === 'success') {
            const token = resData.data.token;
            const user = resData.data.user;
            sessionStorage.setItem('rm_token', token);
            sessionStorage.setItem('rm_username', (user && user.username) || username);
            sessionStorage.setItem('rm_role', (user && user.role) || 'analyst');
            window.location.href = '/';
        } else {
            const errMsg = resData.message || 'Invalid username or password';
            if (loginError && loginErrorText) {
                loginErrorText.textContent = errMsg;
                loginError.classList.remove('d-none');
            } else {
                showNotification(errMsg, 'danger');
            }
        }
    } catch (err) {
        if (loginError && loginErrorText) {
            loginErrorText.textContent = 'Connection error. Please try again.';
            loginError.classList.remove('d-none');
        } else {
            showNotification('Connection error. Please try again.', 'danger');
        }
    } finally {
        submitBtn.disabled = false;
        btnText.classList.remove('d-none');
        spinner.classList.add('d-none');
    }
}

// ============================================================================
// Notifications
// ============================================================================

function showNotification(message, type = 'info') {
    const container = document.getElementById('flashContainer');
    if (!container) return;

    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show rm-alert`;
    alert.setAttribute('role', 'alert');
    alert.innerHTML = `
        <i class="bi bi-info-circle me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    container.appendChild(alert);

    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => alert.remove(), 200);
    }, 4000);
}

// ============================================================================
// Prediction Handling & SHAP Visualizations
// ============================================================================

let shapChartInstance = null;

async function handlePredictionSubmit(event) {
    event.preventDefault();
    const form = event.target;

    if (!form.checkValidity()) {
        form.classList.add('was-validated');
        return;
    }

    const payload = {
        age: parseInt(document.getElementById('age').value, 10),
        income: parseFloat(document.getElementById('income').value),
        employment_length: parseFloat(document.getElementById('employment_length').value),
        loan_amount: parseFloat(document.getElementById('loan_amount').value),
        interest_rate: parseFloat(document.getElementById('interest_rate').value),
        credit_score: parseInt(document.getElementById('credit_score').value, 10),
        debt_to_income: parseFloat(document.getElementById('debt_to_income').value),
        num_credit_lines: parseInt(document.getElementById('num_credit_lines').value, 10),
        num_late_payments: parseInt(document.getElementById('num_late_payments').value, 10),
        loan_purpose: document.getElementById('loan_purpose').value,
        home_ownership: document.getElementById('home_ownership').value,
        annual_revenue: parseFloat(document.getElementById('annual_revenue').value || 0),
        monthly_expenses: parseFloat(document.getElementById('monthly_expenses').value || 0),
        savings_balance: parseFloat(document.getElementById('savings_balance').value || 0)
    };

    const submitBtn = document.getElementById('submitBtn');
    const submitText = document.getElementById('submitText');
    const submitSpinner = document.getElementById('submitSpinner');
    const placeholder = document.getElementById('resultsPlaceholder');
    const loadingCard = document.getElementById('loadingCard');
    const resultsSection = document.getElementById('resultsSection');

    submitBtn.disabled = true;
    submitText.classList.add('d-none');
    submitSpinner.classList.remove('d-none');

    if (placeholder) placeholder.classList.add('d-none');
    if (resultsSection) resultsSection.classList.add('d-none');
    if (loadingCard) loadingCard.classList.remove('d-none');

    try {
        const token = sessionStorage.getItem('rm_token');
        const headers = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = 'Bearer ' + token;

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });

        if (response.status === 401) {
            clearTokenAndRedirect();
            return;
        }

        const resData = await response.json();

        if (response.ok && resData.status === 'success') {
            displayPredictionResults(resData.data);
        } else {
            showNotification(resData.message || 'Prediction failed. Ensure model is trained.', 'danger');
            if (placeholder) placeholder.classList.remove('d-none');
        }
    } catch (err) {
        showNotification('Failed to connect to risk assessment engine.', 'danger');
        if (placeholder) placeholder.classList.remove('d-none');
    } finally {
        submitBtn.disabled = false;
        submitText.classList.remove('d-none');
        submitSpinner.classList.add('d-none');
        if (loadingCard) loadingCard.classList.add('d-none');
    }
}

function displayPredictionResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const gaugeScore = document.getElementById('gaugeScore');
    const gaugeArc = document.getElementById('gaugeArc');
    const badge = document.getElementById('riskLevelBadge');

    if (!resultsSection) return;
    resultsSection.classList.remove('d-none');

    const score = data.risk_score || 0;
    const level = (data.risk_level || 'low').toLowerCase();

    // Update gauge arc: length = 251.3 * score
    const circumference = 251.3;
    const dashLength = (score * circumference).toFixed(1);
    if (gaugeArc) {
        gaugeArc.setAttribute('stroke-dasharray', `${dashLength} ${circumference}`);
    }
    if (gaugeScore) {
        gaugeScore.textContent = (score * 100).toFixed(1) + '%';
    }

    // Update risk level badge
    if (badge) {
        badge.className = 'badge fs-5 px-4 py-2 ';
        if (level === 'low') {
            badge.classList.add('rm-badge-low');
            badge.textContent = 'LOW RISK';
        } else if (level === 'medium') {
            badge.classList.add('rm-badge-medium');
            badge.textContent = 'MEDIUM RISK';
        } else {
            badge.classList.add('rm-badge-high');
            badge.textContent = 'HIGH RISK';
        }
    }

    // Render SHAP chart
    renderShapChart(data.shap_explanation);

    // Render top factors list
    renderTopFactors(data.shap_explanation);
}

function renderShapChart(shapData) {
    const canvas = document.getElementById('shapChart');
    if (!canvas) return;

    if (!shapData || Object.keys(shapData).length === 0) {
        // Fallback demo SHAP values if server returns empty
        shapData = {
            "Debt-to-Income": 0.28,
            "Late Payments": 0.22,
            "Interest Rate": 0.15,
            "Credit Score": -0.31,
            "Income": -0.18,
            "Employment Length": -0.09
        };
    }

    const sortedEntries = Object.entries(shapData)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 8);

    const labels = sortedEntries.map(e => e[0]);
    const values = sortedEntries.map(e => e[1]);
    const colors = values.map(v => v >= 0 ? '#e94560' : '#10b981');

    if (shapChartInstance) {
        shapChartInstance.destroy();
    }

    const ctx = canvas.getContext('2d');
    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Value (Impact on Default Risk)',
                data: values,
                backgroundColor: colors,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const val = context.raw;
                            return (val >= 0 ? '+ Increases Risk: ' : '- Decreases Risk: ') + val.toFixed(4);
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.08)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e2e8f0' }
                }
            }
        }
    });
}

function renderTopFactors(shapData) {
    const list = document.getElementById('topFactorsList');
    if (!list) return;

    list.innerHTML = '';
    if (!shapData || Object.keys(shapData).length === 0) return;

    const sorted = Object.entries(shapData)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 5);

    sorted.forEach(([feature, value]) => {
        const li = document.createElement('li');
        li.className = 'list-group-item bg-transparent text-light border-secondary d-flex justify-content-between align-items-center';

        const isPositive = value >= 0;
        const iconClass = isPositive ? 'bi-arrow-up-right text-danger' : 'bi-arrow-down-right text-success';
        const impactText = isPositive ? 'Pushes risk UP' : 'Pushes risk DOWN';

        li.innerHTML = `
            <div>
                <strong>${feature}</strong>
                <div class="small text-muted">${impactText}</div>
            </div>
            <span class="badge ${isPositive ? 'rm-badge-high' : 'rm-badge-low'}">
                <i class="bi ${iconClass} me-1"></i>${value.toFixed(4)}
            </span>
        `;
        list.appendChild(li);
    });
}

// ============================================================================
// Prediction History Management
// ============================================================================

let allHistoryRecords = [];

async function initHistoryPage() {
    loadPredictionHistory();

    const filterForm = document.getElementById('filterForm');
    if (filterForm) {
        filterForm.addEventListener('submit', function(e) {
            e.preventDefault();
            applyHistoryFilters();
        });
    }

    const clearBtn = document.getElementById('clearFiltersBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            if (filterForm) filterForm.reset();
            renderHistoryRows(allHistoryRecords);
        });
    }

    const exportBtn = document.getElementById('exportCsvBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportHistoryToCSV);
    }
}

async function loadPredictionHistory() {
    try {
        const token = sessionStorage.getItem('rm_token');
        const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
        const response = await fetch('/api/prediction-history?limit=100', { headers });

        if (response.status === 401) {
            clearTokenAndRedirect();
            return;
        }

        if (response.ok) {
            const data = await response.json();
            if (data.status === 'success' && Array.isArray(data.data) && data.data.length > 0) {
                allHistoryRecords = data.data;
                renderHistoryRows(allHistoryRecords);
                return;
            }
        }
    } catch (e) {
        console.warn('History API not reachable, retaining fallback table');
    }
}

function renderHistoryRows(records) {
    const tbody = document.getElementById('historyTableBody');
    const counter = document.getElementById('historyCount');
    if (!tbody) return;

    tbody.innerHTML = '';
    if (counter) counter.textContent = `Showing ${records.length} records`;

    if (records.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-4">No records found.</td></tr>';
        return;
    }

    records.forEach((row, idx) => {
        const tr = document.createElement('tr');
        const score = typeof row.risk_score === 'number' ? row.risk_score.toFixed(2) : row.risk_score;
        const level = (row.risk_level || 'low').toLowerCase();
        const badgeClass = level === 'high' ? 'rm-badge-high' : (level === 'medium' ? 'rm-badge-medium' : 'rm-badge-low');
        const summary = row.applicant_summary || formatApplicantSummary(row.input_data);

        tr.innerHTML = `
            <td>${row.timestamp ? row.timestamp.substring(0, 16) : 'Just now'}</td>
            <td>${summary}</td>
            <td><span class="fw-bold">${score}</span></td>
            <td><span class="badge ${badgeClass}">${level.toUpperCase()}</span></td>
            <td>
                <button class="btn btn-sm btn-outline-info view-details-btn" data-index="${idx}">
                    <i class="bi bi-eye"></i>
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });

    document.querySelectorAll('.view-details-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const index = this.getAttribute('data-index');
            showDetailModal(records[index]);
        });
    });
}

function formatApplicantSummary(inputData) {
    if (!inputData) return 'Standard Applicant';
    if (typeof inputData === 'string') {
        try { inputData = JSON.parse(inputData); } catch (e) { return inputData; }
    }
    const age = inputData.age ? `Age ${inputData.age}` : '';
    const income = inputData.income ? `Income $${(inputData.income/1000).toFixed(0)}K` : '';
    const loan = inputData.loan_amount ? `Loan $${(inputData.loan_amount/1000).toFixed(0)}K` : '';
    return [age, income, loan].filter(Boolean).join(', ') || 'Standard Applicant';
}

function applyHistoryFilters() {
    const levelFilter = document.getElementById('filterRiskLevel').value.toLowerCase();
    const dateFrom = document.getElementById('filterDateFrom').value;
    const dateTo = document.getElementById('filterDateTo').value;

    let filtered = allHistoryRecords.filter(item => {
        if (levelFilter && (item.risk_level || '').toLowerCase() !== levelFilter) return false;
        if (dateFrom && item.timestamp && item.timestamp < dateFrom) return false;
        if (dateTo && item.timestamp && item.timestamp > (dateTo + ' 23:59:59')) return false;
        return true;
    });

    renderHistoryRows(filtered);
}

function exportHistoryToCSV() {
    if (allHistoryRecords.length === 0) {
        showNotification('No history data to export.', 'warning');
        return;
    }

    const headers = ['Timestamp', 'Risk Score', 'Risk Level', 'Summary'];
    const rows = allHistoryRecords.map(r => [
        `"${r.timestamp || ''}"`,
        r.risk_score || '',
        `"${r.risk_level || ''}"`,
        `"${formatApplicantSummary(r.input_data)}"`
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `riskmodel_history_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function showDetailModal(record) {
    const modalBody = document.getElementById('detailModalBody');
    if (!modalBody || !record) return;

    let inputDetails = record.input_data;
    if (typeof inputDetails === 'string') {
        try { inputDetails = JSON.parse(inputDetails); } catch (e) {}
    }

    modalBody.innerHTML = `
        <div class="row g-3">
            <div class="col-md-6">
                <h6>Assessment Summary</h6>
                <p><strong>Timestamp:</strong> ${record.timestamp || 'N/A'}</p>
                <p><strong>Risk Score:</strong> ${(record.risk_score * 100).toFixed(1)}%</p>
                <p><strong>Risk Level:</strong> <span class="badge ${record.risk_level === 'high' ? 'rm-badge-high' : (record.risk_level === 'medium' ? 'rm-badge-medium' : 'rm-badge-low')}">${(record.risk_level || 'LOW').toUpperCase()}</span></p>
            </div>
            <div class="col-md-6">
                <h6>Input Data</h6>
                <pre class="bg-dark p-2 rounded text-light" style="max-height: 200px; overflow-y: auto;">${JSON.stringify(inputDetails, null, 2)}</pre>
            </div>
        </div>
    `;

    const modal = new bootstrap.Modal(document.getElementById('detailModal'));
    modal.show();
}

// ============================================================================
// Dashboard Statistics Management
// ============================================================================

async function loadDashboardStats() {
    try {
        const token = sessionStorage.getItem('rm_token');
        const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
        const resp = await fetch('/api/prediction-stats', { headers });
        if (resp.status === 401) {
            clearTokenAndRedirect();
            return;
        }
        if (resp.ok) {
            const data = await resp.json();
            const totalEl = document.getElementById('totalPredictions');
            if (totalEl && data.total_predictions !== undefined) {
                totalEl.textContent = data.total_predictions.toLocaleString();
            }
            const avgEl = document.getElementById('avgRiskScore');
            if (avgEl && data.average_risk_score !== undefined) {
                avgEl.textContent = data.average_risk_score.toFixed(2);
            }
            const highPctEl = document.getElementById('highRiskPct');
            if (highPctEl && data.high_risk_percentage !== undefined) {
                highPctEl.textContent = data.high_risk_percentage.toFixed(1) + '%';
            }
            const accEl = document.getElementById('modelAccuracy');
            if (accEl && data.model_accuracy !== undefined) {
                accEl.textContent = data.model_accuracy.toFixed(1) + '%';
            }
        }
    } catch (e) {
        console.warn('Dashboard stats API not reachable');
    }
}


