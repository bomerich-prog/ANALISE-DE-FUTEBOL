import { api } from './api.js';
import { startAnalysis } from './analysisClient.js';
import { escapeHtml, formatSize, formatDate } from './utils.js';
import {
  PIPELINE_STATUS,
  getPipelineLabel,
  pipelinePillClass,
  getAnalysisLabel,
} from './pipelineStatus.js';

const historyTable = document.getElementById('historyTable');
const historyTableBody = document.getElementById('historyTableBody');
const historyEmpty = document.getElementById('historyEmpty');
const historyLoading = document.getElementById('historyLoading');
const historySearch = document.getElementById('historySearch');
const historySort = document.getElementById('historySort');

let allItems = [];

function canAnalyze(item) {
  const s = item.pipeline_status;
  return s === PIPELINE_STATUS.CONVERTED || s === PIPELINE_STATUS.ERROR;
}

function displayStatus(item) {
  return getPipelineLabel(item.pipeline_status || PIPELINE_STATUS.CONVERTED);
}

function sortItems(items, sortKey) {
  const list = [...items];
  switch (sortKey) {
    case 'date-asc':
      return list.sort((a, b) => a.converted_at.localeCompare(b.converted_at));
    case 'name-asc':
      return list.sort((a, b) => a.video_name.localeCompare(b.video_name));
    case 'name-desc':
      return list.sort((a, b) => b.video_name.localeCompare(a.video_name));
    case 'size-asc':
      return list.sort((a, b) => a.file_size - b.file_size);
    case 'size-desc':
      return list.sort((a, b) => b.file_size - a.file_size);
    default:
      return list.sort((a, b) => b.converted_at.localeCompare(a.converted_at));
  }
}

function filterItems(items, query) {
  if (!query.trim()) return items;
  const q = query.trim().toLowerCase();
  return items.filter((i) => i.video_name.toLowerCase().includes(q));
}

async function analyzeFromHistory(videoId, btn) {
  btn.disabled = true;
  btn.textContent = 'Analisando...';
  try {
    await startAnalysis(videoId, {
      onProgress: () => {},
      onComplete: () => {},
      onError: (msg) => {
        throw new Error(msg);
      },
    });
    await loadHistory();
    window.dispatchEvent(new CustomEvent('dashboard:refresh'));
    window.dispatchEvent(new CustomEvent('analyses:load'));
  } catch (err) {
    alert(err.message);
    btn.disabled = false;
    btn.textContent = 'Analisar Vídeo';
  }
}

function renderTable(items) {
  historyTableBody.innerHTML = '';

  if (!items.length) {
    historyTable.classList.add('hidden');
    historyEmpty.classList.remove('hidden');
    historyEmpty.querySelector('p').textContent = allItems.length
      ? 'Nenhum resultado para a busca.'
      : 'Nenhum vídeo na biblioteca.';
    return;
  }

  historyEmpty.classList.add('hidden');
  historyTable.classList.remove('hidden');

  items.forEach((item) => {
    const status = item.pipeline_status || PIPELINE_STATUS.CONVERTED;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="cell-name" title="${escapeHtml(item.video_name)}">${escapeHtml(item.video_name)}</span></td>
      <td><span class="cell-format">${escapeHtml(item.original_format)}</span></td>
      <td>${formatSize(item.file_size)}</td>
      <td>${formatDate(item.converted_at)}</td>
      <td><span class="status-pill status-pill--${pipelinePillClass(status)}">${displayStatus(item)}</span></td>
      <td>
        <div class="table-actions">
          ${canAnalyze(item) ? `<button type="button" class="btn btn-primary" style="width:auto;padding:0.4rem 0.85rem;font-size:0.8rem" data-analyze="${item.id}">Analisar Vídeo</button>` : ''}
          <a href="/api/history/${item.id}/download" class="btn btn-secondary" style="width:auto;padding:0.4rem 0.85rem;font-size:0.8rem" download>Baixar</a>
          <button type="button" class="btn btn-ghost" data-delete="${item.id}">Excluir</button>
        </div>
      </td>
    `;

    tr.querySelector('[data-analyze]')?.addEventListener('click', (e) => {
      analyzeFromHistory(item.id, e.target);
    });

    tr.querySelector('[data-delete]')?.addEventListener('click', async () => {
      if (!confirm('Remover este vídeo da biblioteca?')) return;
      try {
        await api(`/api/history/${item.id}`, { method: 'DELETE' });
        allItems = allItems.filter((i) => i.id !== item.id);
        applyFilters();
        window.dispatchEvent(new CustomEvent('dashboard:refresh'));
      } catch (err) {
        alert(err.message);
      }
    });

    historyTableBody.appendChild(tr);
  });
}

function applyFilters() {
  renderTable(filterItems(sortItems(allItems, historySort.value), historySearch.value));
}

async function loadHistory() {
  historyLoading.classList.remove('hidden');
  historyEmpty.classList.add('hidden');
  historyTable.classList.add('hidden');

  try {
    const data = await api('/api/history');
    allItems = data.history || [];
    applyFilters();
  } catch (err) {
    historyTableBody.innerHTML = `<tr><td colspan="6" class="table-empty">${escapeHtml(err.message)}</td></tr>`;
    historyTable.classList.remove('hidden');
  } finally {
    historyLoading.classList.add('hidden');
  }
}

historySearch?.addEventListener('input', applyFilters);
historySort?.addEventListener('change', applyFilters);
window.addEventListener('history:load', loadHistory);
