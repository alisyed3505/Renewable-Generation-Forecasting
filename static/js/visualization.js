const API_URL = 'http://localhost:8000/api/data';
let currentPage = 1;
let currentData = [];
let chartInstance = null;

// Columns to display in table
const TABLE_COLS = [
    'time_idx', 'power_normed', 'hour_of_day',
    'SolarRadiationGlobalAt0', 'TemperatureAt0'
];

async function fetchData(page) {
    try {
        const res = await fetch(`${API_URL}?page=${page}&limit=50`);
        const result = await res.json();

        currentData = result.data;
        currentPage = result.page;

        renderTable(currentData);
        updatePageInfo(result.page, result.total_pages);

        if (document.getElementById('chart-view').style.display !== 'none') {
            updateChart();
        }

    } catch (e) {
        console.error("Failed to fetch data", e);
    }
}

function renderTable(data) {
    const thead = document.getElementById('table-header');
    const tbody = document.getElementById('table-body');

    // Set headers if empty
    if (thead.innerHTML.trim() === '') {
        // Use keys from first row or default list
        const keys = data.length > 0 ? Object.keys(data[0]) : TABLE_COLS;
        thead.innerHTML = keys.map(k => `<th>${k}</th>`).join('');
    }

    tbody.innerHTML = data.map(row => {
        return `<tr>${Object.values(row).map(val => `<td>${val}</td>`).join('')}</tr>`;
    }).join('');
}

function updatePageInfo(page, total) {
    document.getElementById('page-info').innerText = `Page ${page} of ${total}`;
}

function changePage(delta) {
    const newPage = currentPage + delta;
    if (newPage > 0) {
        fetchData(newPage);
    }
}

function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.tab[onclick="switchTab('${tabName}')"]`).classList.add('active');

    if (tabName === 'table') {
        document.getElementById('table-view').classList.remove('hidden');
        document.getElementById('chart-view').classList.add('hidden');
    } else {
        document.getElementById('table-view').classList.add('hidden');
        document.getElementById('chart-view').classList.remove('hidden');
        updateChart();
    }
}

function updateChart() {
    const ctx = document.getElementById('dataChart').getContext('2d');
    const type = document.getElementById('chart-type').value;
    const yAxisKey = document.getElementById('y-axis-select').value;

    if (chartInstance) {
        chartInstance.destroy();
    }

    // Prepare data
    // For simplicity, we use index as X-axis for now, or construct a label
    const labels = currentData.map((_, i) => `Point ${i + 1}`);
    const dataPoints = currentData.map(d => d[yAxisKey]);

    const config = {
        type: type,
        data: {
            labels: labels,
            datasets: [{
                label: yAxisKey,
                data: dataPoints,
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.5)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { display: false } // Hide x labels for cleanliness if too many
                }
            },
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            }
        }
    };

    chartInstance = new Chart(ctx, config);
}

// Init
document.addEventListener('DOMContentLoaded', () => {
    // Create js dir if not exists? No, browser handles it.
    // Just fetch data
    fetchData(1);
});
