import { api } from './api.js';
import { escapeHtml, formatSize, formatDate } from './utils.js';
import { getPipelineLabel, pipelinePillClass, getAnalysisLabel } from './pipelineStatus.js';

const statTotalVideos = document.getElementById('statTotalVideos');
const statAnalyses = document.getElementById('statAnalyses');
const statLastAnalysis = document.getElementById('statLastAnalysis');
const statSuccessRate = document.getElementById('statSuccessRate');
const recentTableBody = document.getElementById('recentTableBody');
const analysesTableBody = document.getElementById('dashboardAnalysesBody');
const formatList = document.getElementById('formatList');
const activityChart = document.getElementById('activityChart');

function renderRecent(items) {
  if (!items.length) {
    recentTableBody.innerHTML =
      '<tr><td colspan="6" class="table-empty">Nenhum vídeo processado. <a href="#" data-goto="converter" style="color:var(--green)">Importar vídeo</a></td></tr>';
    recentTableBody.querySelector('[data-goto]')?.addEventListener('click', (e) => {
      e.preventDefault();
      document.querySelector('[data-view="converter"]')?.click();
    });
    return;
  }

  recentTableBody.innerHTML = items
    .map((item) => {
      const status = item.pipeline_status || 'converted';
      return `
    <tr>
      <td><span class="cell-name" title="${escapeHtml(item.video_name)}">${escapeHtml(item.video_name)}</span></td>
      <td><span class="cell-format">${escapeHtml(item.original_format)}</span></td>
      <td>${formatSize(item.file_size)}</td>
      <td>${formatDate(item.converted_at)}</td>
      <td><span class="status-pill status-pill--${pipelinePillClass(status)}">${getPipelineLabel(status)}</span></td>
      <td><a href="/api/history/${item.id}/download" class="btn btn-ghost" download>Baixar</a></td>
    </tr>`;
    })
    .join('');
}

function renderRecentAnalyses(items) {
  if (!analysesTableBody) return;
  if (!items.length) {
    analysesTableBody.innerHTML =
      '<tr><td colspan="3" class="table-empty">Nenhuma análise realizada</td></tr>';
    return;
  }

  analysesTableBody.innerHTML = items
    .map(
      (a) => `
    <tr>
      <td><span class="cell-name">${escapeHtml(a.video_name)}</span></td>
      <td>${formatDate(a.created_at)}</td>
      <td><span class="status-pill status-pill--${a.status === 'completed' ? 'analysis-complete' : a.status === 'analyzing' ? 'analyzing' : a.status === 'error' ? 'error' : 'converted'}">${getAnalysisLabel(a.status)}</span></td>
    </tr>`
    )
    .join('');
}

function renderFormats(formats) {
  if (!formats.length) {
    formatList.innerHTML = '<li class="format-empty">Nenhum formato registrado</li>';
    return;
  }
  formatList.innerHTML = formats
    .map(
      (f) => `
    <li>
      <span>${escapeHtml(f.original_format)}</span>
      <span class="format-count">${f.count}</span>
    </li>`
    )
    .join('');
}

function renderActivity(activity) {
  if (!activity.length) {
    activityChart.innerHTML = '<p class="activity-empty">Sem atividade nos últimos 7 dias</p>';
    return;
  }
  const max = Math.max(...activity.map((a) => a.count), 1);
  activityChart.innerHTML = activity
    .map((a) => {
      const day = a.day?.slice(5) || '';
      const h = Math.max((a.count / max) * 100, 8);
      return `
      <div class="activity-bar-wrap">
        <div class="activity-bar" style="height:${h}%"></div>
        <span class="activity-bar-label">${day}</span>
      </div>`;
    })
    .join('');
}

async function loadDashboard() {
  try {
    const data = await api('/api/dashboard/stats');

    statTotalVideos.textContent = data.totalVideos;
    statAnalyses.textContent = data.totalAnalyses;
    statSuccessRate.textContent = `${data.successRate}%`;
    statLastAnalysis.textContent = data.lastAnalysisAt
      ? formatDate(data.lastAnalysisAt)
      : '—';

    renderRecent(data.recent || []);
    renderRecentAnalyses(data.recentAnalyses || []);
    renderFormats(data.formatBreakdown || []);
    renderActivity(data.activity || []);
  } catch (err) {
    recentTableBody.innerHTML = `<tr><td colspan="6" class="table-empty">${escapeHtml(err.message)}</td></tr>`;
  }
}

window.addEventListener('dashboard:load', loadDashboard);
window.addEventListener('dashboard:refresh', loadDashboard);
loadDashboard();
