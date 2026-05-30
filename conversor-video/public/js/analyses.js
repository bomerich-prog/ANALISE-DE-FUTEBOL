import { listAnalyses } from './analysisClient.js';
import { escapeHtml, formatDate } from './utils.js';
import { getAnalysisLabel, pipelinePillClass, getPipelineLabel } from './pipelineStatus.js';
import { ANALYSIS_STATUS } from './pipelineStatus.js';

const analysesTableBody = document.getElementById('analysesTableBody');
const analysesEmpty = document.getElementById('analysesEmpty');
const analysesLoading = document.getElementById('analysesLoading');
const analysesTable = document.getElementById('analysesTable');
const analysesSearch = document.getElementById('analysesSearch');

let allAnalyses = [];

function analysisStatusClass(status) {
  const map = {
    [ANALYSIS_STATUS.PENDING]: 'sent',
    [ANALYSIS_STATUS.ANALYZING]: 'analyzing',
    [ANALYSIS_STATUS.COMPLETED]: 'analysis-complete',
    [ANALYSIS_STATUS.ERROR]: 'error',
  };
  return map[status] || 'sent';
}

function renderAnalyses(items) {
  analysesTableBody.innerHTML = '';

  if (!items.length) {
    analysesTable.classList.add('hidden');
    analysesEmpty.classList.remove('hidden');
    analysesEmpty.querySelector('p').textContent = allAnalyses.length
      ? 'Nenhum resultado para a busca.'
      : 'Nenhuma análise realizada ainda.';
    return;
  }

  analysesEmpty.classList.add('hidden');
  analysesTable.classList.remove('hidden');

  items.forEach((item) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="cell-name" title="${escapeHtml(item.video_name)}">${escapeHtml(item.video_name)}</span></td>
      <td>${formatDate(item.created_at)}</td>
      <td><span class="status-pill status-pill--${analysisStatusClass(item.status)}">${getAnalysisLabel(item.status)}</span></td>
      <td><span class="cell-format">${escapeHtml(item.original_format || '—')}</span></td>
    `;
    analysesTableBody.appendChild(tr);
  });
}

function applyFilter() {
  const q = analysesSearch?.value.trim().toLowerCase() || '';
  const filtered = q
    ? allAnalyses.filter((a) => a.video_name.toLowerCase().includes(q))
    : allAnalyses;
  renderAnalyses(filtered);
}

async function loadAnalysesPage() {
  analysesLoading?.classList.remove('hidden');
  analysesEmpty?.classList.add('hidden');
  analysesTable?.classList.add('hidden');

  try {
    allAnalyses = await listAnalyses();
    applyFilter();
  } catch (err) {
    analysesTableBody.innerHTML = `<tr><td colspan="4" class="table-empty">${escapeHtml(err.message)}</td></tr>`;
    analysesTable.classList.remove('hidden');
  } finally {
    analysesLoading?.classList.add('hidden');
  }
}

analysesSearch?.addEventListener('input', applyFilter);
window.addEventListener('analyses:load', loadAnalysesPage);

export { getPipelineLabel, pipelinePillClass };
