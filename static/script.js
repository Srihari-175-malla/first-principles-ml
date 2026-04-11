let currentBm = null;

document.addEventListener('DOMContentLoaded', () => {
  runValidationBenchmarks();
});

function toggleTheme() {
  document.body.classList.toggle('light-theme');
}

function switchNavTab(tabId) {
  document.querySelectorAll('.view-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

  document.getElementById(tabId).classList.add('active');

  if (tabId === 'bm-view') document.getElementById('tab-btn-bm').classList.add('active');
  if (tabId === 'grad-view') document.getElementById('tab-btn-grad').classList.add('active');
  if (tabId === 'pca-view') {
    document.getElementById('tab-btn-pca').classList.add('active');
    runPCAReconstruction();
  }
}

async function runValidationBenchmarks() {
  const loader = document.getElementById('bm-loader');
  loader.style.display = 'block';

  try {
    const resp = await fetch('/api/benchmarks');
    const data = await resp.json();
    currentBm = data.validation_benchmarks;
    renderBenchmarkTable(currentBm);
  } catch (e) {
    console.error('Benchmark error:', e);
  } finally {
    loader.style.display = 'none';
  }
}

function renderBenchmarkTable(bm) {
  const tbody = document.querySelector('#bm-table tbody');
  if (!bm) return;

  tbody.innerHTML = bm.map(row => `
    <tr>
      <td><strong>${row.algorithm}</strong></td>
      <td>${row.custom_metric_name}</td>
      <td><code>${row.custom_score}${row.custom_metric_name === 'Accuracy' || row.custom_metric_name.includes('%') ? '%' : ''}</code></td>
      <td><code>${row.sklearn_score}${row.custom_metric_name === 'Accuracy' || row.custom_metric_name.includes('%') ? '%' : ''}</code></td>
      <td><span style="color:${row.gap_percent <= 1.0 ? '#10b981' : '#f59e0b'}; font-weight:700;">${row.gap_percent}%</span></td>
      <td><span style="background:rgba(16,185,129,0.2); border:1px solid #10b981; color:#10b981; padding:0.25rem 0.6rem; border-radius:6px; font-weight:600;">${row.status}</span></td>
    </tr>
  `).join('');
}

async function runGradientCheck() {
  const hDim = parseInt(document.getElementById('hidden-dim-input').value);
  const box = document.getElementById('grad-result-box');
  box.style.display = 'block';

  try {
    const resp = await fetch('/api/gradient_check', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hidden_dim: hDim, n_samples: 20, learning_rate: 0.05 })
    });

    const data = await resp.json();
    document.getElementById('grad-diff-val').innerText = `${data.relative_gradient_diff.toExponential(4)}`;
    document.getElementById('grad-status-lbl').innerHTML = data.passed ?
      `<span style="color:#10b981; font-weight:700;">${data.status}</span>` :
      `<span style="color:#ef4444; font-weight:700;">${data.status}</span>`;
  } catch (e) {
    console.error('Gradient check error:', e);
  }
}

async function runPCAReconstruction() {
  const k = parseInt(document.getElementById('pca-comp-input').value);

  try {
    const resp = await fetch('/api/pca_reconstruct', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_components: k })
    });

    const data = await resp.json();
    document.getElementById('stat-pca-var').innerText = `${data.explained_variance_percent}%`;
    document.getElementById('stat-pca-err').innerText = `${data.reconstruction_error_mse}`;
  } catch (e) {
    console.error('PCA reconstruction error:', e);
  }
}
