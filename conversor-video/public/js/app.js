import { FFmpeg } from 'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/esm/index.js';
import { fetchFile, toBlobURL } from 'https://cdn.jsdelivr.net/npm/@ffmpeg/util@0.12.1/dist/esm/index.js';
import { escapeHtml, formatSize } from './utils.js';
import { startAnalysis } from './analysisClient.js';
import {
  PIPELINE_STATUS,
  PIPELINE_STEPS,
  getPipelineLabel,
  pipelinePillClass,
} from './pipelineStatus.js';

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const videoList = document.getElementById('videoList');
const videoCards = document.getElementById('videoCards');
const queueCount = document.getElementById('queueCount');
const errorBanner = document.getElementById('errorBanner');
const errorMessage = document.getElementById('errorMessage');
const errorClose = document.getElementById('errorClose');

let ffmpeg = null;
let ffmpegLoading = false;
const videos = new Map();

function showError(message) {
  errorMessage.textContent = message;
  errorBanner.classList.remove('hidden');
}

function hideError() {
  errorBanner.classList.add('hidden');
}

errorClose?.addEventListener('click', hideError);

function updateQueueCount() {
  if (queueCount) queueCount.textContent = `${videos.size} ${videos.size === 1 ? 'item' : 'itens'}`;
}

function isMp4(file) {
  const ext = file.name.split('.').pop()?.toLowerCase();
  return ext === 'mp4' || file.type === 'video/mp4';
}

function getOriginalFormat(file) {
  const ext = file.name.split('.').pop()?.toLowerCase();
  if (ext) return ext.toUpperCase();
  if (file.type) return file.type.split('/')[1] || 'desconhecido';
  return 'desconhecido';
}

async function loadFFmpeg() {
  if (ffmpeg) return ffmpeg;
  if (ffmpegLoading) {
    while (ffmpegLoading) await new Promise((r) => setTimeout(r, 100));
    return ffmpeg;
  }
  ffmpegLoading = true;
  try {
    ffmpeg = new FFmpeg();
    const baseURL = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core-st@0.12.6/dist/esm';
    await ffmpeg.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
    });
    return ffmpeg;
  } catch {
    ffmpeg = null;
    throw new Error('Não foi possível carregar o módulo de conversão.');
  } finally {
    ffmpegLoading = false;
  }
}

async function saveToHistory(blob, meta) {
  const formData = new FormData();
  formData.append('video', blob, meta.outputName);
  formData.append('videoName', meta.videoName);
  formData.append('originalFormat', meta.originalFormat);
  const res = await fetch('/api/history', { method: 'POST', credentials: 'include', body: formData });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || 'Erro ao salvar na biblioteca.');
  return data;
}

function renderPipelineSteps(currentStatus) {
  const order = PIPELINE_STEPS;
  let activeIdx = order.indexOf(currentStatus);
  if (currentStatus === PIPELINE_STATUS.ERROR) {
    activeIdx = order.indexOf(PIPELINE_STATUS.CONVERTED);
  }
  if (currentStatus === PIPELINE_STATUS.ANALYSIS_COMPLETE) {
    activeIdx = order.length;
  }

  const steps = order
    .map((step, i) => {
      let cls = 'pipeline-step';
      if (currentStatus === PIPELINE_STATUS.ANALYSIS_COMPLETE) {
        cls += ' pipeline-step--done';
      } else if (currentStatus === PIPELINE_STATUS.ERROR) {
        cls += i <= order.indexOf(PIPELINE_STATUS.CONVERTED) ? ' pipeline-step--done' : ' pipeline-step--pending';
      } else if (i < activeIdx) {
        cls += ' pipeline-step--done';
      } else if (i === activeIdx) {
        cls += ' pipeline-step--active';
      } else {
        cls += ' pipeline-step--pending';
      }
      return `<span class="${cls}">${getPipelineLabel(step)}</span>`;
    })
    .join('<span class="pipeline-connector"></span>');

  return `<div class="pipeline-steps">${steps}</div>`;
}

function setPipelineStatus(cardId, status) {
  const card = document.querySelector(`[data-id="${cardId}"]`);
  if (!card) return;

  const pill = card.querySelector('.status-pill');
  if (pill) {
    pill.className = `status-pill status-pill--${pipelinePillClass(status)}`;
    pill.textContent = getPipelineLabel(status);
  }

  const stepsEl = card.querySelector('.pipeline-steps-wrap');
  if (stepsEl) stepsEl.innerHTML = renderPipelineSteps(status);

  videos.get(cardId).pipelineStatus = status;
}

function createVideoCard(id, file) {
  const card = document.createElement('div');
  card.className = 'process-card';
  card.dataset.id = id;

  card.innerHTML = `
    <div class="process-card-header">
      <div>
        <div class="process-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</div>
        <div class="process-meta">${formatSize(file.size)} · ${escapeHtml(getOriginalFormat(file))}</div>
      </div>
      <span class="status-pill status-pill--sent">${getPipelineLabel(PIPELINE_STATUS.SENT)}</span>
    </div>
    <div class="pipeline-steps-wrap">${renderPipelineSteps(PIPELINE_STATUS.SENT)}</div>
    <div class="progress-track"><div class="progress-bar-fill" id="progress-${id}"></div></div>
    <div class="progress-label" id="progressText-${id}">Aguardando processamento...</div>
    <div class="process-error hidden" id="cardError-${id}"></div>
    <div class="process-success hidden" id="cardSuccess-${id}"></div>
    <div class="analysis-section hidden" id="analysisSection-${id}">
      <div class="analysis-section-header">
        <span class="analysis-section-title">Etapa de análise</span>
      </div>
      <div class="progress-track progress-track--analysis"><div class="progress-bar-fill progress-bar-fill--analysis" id="analysisProgress-${id}"></div></div>
      <div class="progress-label" id="analysisProgressText-${id}"></div>
    </div>
    <video class="video-preview hidden" id="preview-${id}" controls></video>
    <div class="process-actions hidden" id="actions-${id}">
      <button type="button" class="btn btn-primary" style="width:auto" data-action="analyze">Analisar Vídeo</button>
      <button type="button" class="btn btn-secondary" style="width:auto" data-action="download">Baixar MP4</button>
      <button type="button" class="btn btn-ghost" data-action="remove">Remover</button>
    </div>
  `;

  videoCards.appendChild(card);
  card.querySelector('[data-action="download"]')?.addEventListener('click', () => downloadVideo(id));
  card.querySelector('[data-action="remove"]')?.addEventListener('click', () => removeVideo(id));
  card.querySelector('[data-action="analyze"]')?.addEventListener('click', () => runAnalysis(id));

  return card;
}

function updateProgress(id, percent, text) {
  document.getElementById(`progress-${id}`)?.style && (document.getElementById(`progress-${id}`).style.width = `${percent}%`);
  const label = document.getElementById(`progressText-${id}`);
  if (label) label.textContent = text;
}

function updateAnalysisProgress(id, percent, text) {
  document.getElementById(`analysisProgress-${id}`)?.style && (document.getElementById(`analysisProgress-${id}`).style.width = `${percent}%`);
  const label = document.getElementById(`analysisProgressText-${id}`);
  if (label) label.textContent = text;
}

function hideConversionProgress(id) {
  document.querySelector(`[data-id="${id}"] .progress-track:not(.progress-track--analysis)`)?.classList.add('hidden');
  document.getElementById(`progressText-${id}`)?.classList.add('hidden');
}

function showCardError(id, message) {
  hideConversionProgress(id);
  setPipelineStatus(id, PIPELINE_STATUS.ERROR);
  const el = document.getElementById(`cardError-${id}`);
  if (el) {
    el.textContent = message;
    el.classList.remove('hidden');
  }
  document.getElementById(`actions-${id}`)?.classList.remove('hidden');
  document.querySelector(`[data-id="${id}"] [data-action="analyze"]`)?.classList.add('hidden');
  document.querySelector(`[data-id="${id}"] [data-action="download"]`)?.classList.add('hidden');
}

async function runAnalysis(cardId) {
  const video = videos.get(cardId);
  if (!video?.videoId || video.analyzing) return;

  const analyzeBtn = document.querySelector(`[data-id="${cardId}"] [data-action="analyze"]`);
  if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analisando...';
  }

  video.analyzing = true;
  setPipelineStatus(cardId, PIPELINE_STATUS.ANALYZING);

  const section = document.getElementById(`analysisSection-${cardId}`);
  section?.classList.remove('hidden');
  updateAnalysisProgress(cardId, 0, 'Iniciando análise...');

  try {
    await startAnalysis(video.videoId, {
      onProgress: (pct, msg) => {
        updateAnalysisProgress(cardId, pct, msg);
        setPipelineStatus(cardId, PIPELINE_STATUS.ANALYZING);
      },
      onComplete: (analysis) => {
        setPipelineStatus(cardId, PIPELINE_STATUS.ANALYSIS_COMPLETE);
        updateAnalysisProgress(cardId, 100, 'Análise concluída com sucesso.');
        const success = document.getElementById(`cardSuccess-${cardId}`);
        if (success) {
          success.textContent = 'Análise concluída com sucesso. Resultados disponíveis no histórico de análises.';
          success.classList.remove('hidden');
        }
        if (analyzeBtn) {
          analyzeBtn.classList.add('hidden');
        }
        video.analyzing = false;
        window.dispatchEvent(new CustomEvent('analyses:load'));
      },
      onError: (msg) => {
        setPipelineStatus(cardId, PIPELINE_STATUS.ERROR);
        showError(`Erro na análise: ${msg}`);
        if (analyzeBtn) {
          analyzeBtn.disabled = false;
          analyzeBtn.textContent = 'Analisar Vídeo';
        }
        video.analyzing = false;
      },
    });
  } catch (err) {
    setPipelineStatus(cardId, PIPELINE_STATUS.CONVERTED);
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Analisar Vídeo';
    }
    video.analyzing = false;
    showError(err.message);
  }
}

window.runVideoAnalysis = runAnalysis;

async function processVideo(id, file) {
  hideError();
  setPipelineStatus(id, PIPELINE_STATUS.SENT);
  updateProgress(id, 5, 'Arquivo recebido...');

  const originalFormat = getOriginalFormat(file);
  const baseName = file.name.replace(/\.[^.]+$/, '');
  const outputName = `${baseName}.mp4`;
  let resultBlob;
  let resultName = outputName;

  if (isMp4(file)) {
    setPipelineStatus(id, PIPELINE_STATUS.CONVERTING);
    updateProgress(id, 40, 'Validando MP4...');
    resultBlob = file;
    resultName = file.name;
    videos.set(id, { ...videos.get(id), blob: file, blobUrl: URL.createObjectURL(file), outputName: resultName });
  } else {
    try {
      setPipelineStatus(id, PIPELINE_STATUS.CONVERTING);
      updateProgress(id, 8, 'Inicializando conversão...');
      const ff = await loadFFmpeg();
      const inputExt = file.name.split('.').pop()?.toLowerCase() || 'video';
      const inputName = `input.${inputExt}`;

      ff.on('progress', ({ progress, time }) => {
        const pct = Math.min(Math.round(progress * 100), 99);
        const timeSec = time ? Math.round(time / 1000000) : 0;
        updateProgress(id, pct, `Convertendo... ${pct}%${timeSec ? ` · ${timeSec}s` : ''}`);
      });

      await ff.writeFile(inputName, await fetchFile(file));
      updateProgress(id, 15, 'Normalizando para MP4...');

      const exitCode = await ff.exec([
        '-i', inputName,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart',
        'output.mp4',
      ]);

      if (exitCode !== 0) throw new Error('Conversão falhou. Formato não suportado.');

      const data = await ff.readFile('output.mp4');
      resultBlob = new Blob([data.buffer], { type: 'video/mp4' });
      const blobUrl = URL.createObjectURL(resultBlob);
      videos.set(id, { ...videos.get(id), blob: resultBlob, blobUrl, outputName: resultName });
      await ff.deleteFile(inputName).catch(() => {});
      await ff.deleteFile('output.mp4').catch(() => {});
      updateProgress(id, 100, 'Conversão concluída');
    } catch (err) {
      showCardError(id, err.message);
      showError(`Falha: "${file.name}": ${err.message}`);
      return;
    }
  }

  try {
    setPipelineStatus(id, PIPELINE_STATUS.CONVERTING);
    updateProgress(id, 90, 'Enviando para biblioteca...');
    const saved = await saveToHistory(resultBlob, { videoName: baseName, originalFormat, outputName: resultName });
    const videoId = saved.item?.id;

    videos.set(id, {
      ...videos.get(id),
      videoId,
      pipelineStatus: PIPELINE_STATUS.CONVERTED,
    });

    hideConversionProgress(id);
    setPipelineStatus(id, PIPELINE_STATUS.CONVERTED);

    const preview = document.getElementById(`preview-${id}`);
    const blobUrl = videos.get(id).blobUrl || URL.createObjectURL(resultBlob);
    if (preview) {
      preview.src = blobUrl;
      preview.classList.remove('hidden');
    }

    const success = document.getElementById(`cardSuccess-${id}`);
    if (success) {
      success.textContent = 'Vídeo convertido e salvo. Inicie a análise para continuar o pipeline.';
      success.classList.remove('hidden');
    }

    document.getElementById(`actions-${id}`)?.classList.remove('hidden');
    window.dispatchEvent(new CustomEvent('dashboard:refresh'));
  } catch (err) {
    setPipelineStatus(id, PIPELINE_STATUS.ERROR);
    showError(`Convertido localmente, mas falha ao salvar: ${err.message}`);
  }
}

function downloadVideo(id) {
  const video = videos.get(id);
  if (!video?.blobUrl) return;
  const a = document.createElement('a');
  a.href = video.blobUrl;
  a.download = video.outputName;
  a.click();
}

function removeVideo(id) {
  const video = videos.get(id);
  if (video?.blobUrl) URL.revokeObjectURL(video.blobUrl);
  videos.delete(id);
  document.querySelector(`[data-id="${id}"]`)?.remove();
  updateQueueCount();
  if (videos.size === 0) videoList.classList.add('hidden');
}

async function handleFiles(fileList) {
  const files = Array.from(fileList).filter(
    (f) => f.type.startsWith('video/') || /\.(mp4|webm|mkv|avi|mov|wmv|flv|m4v|ogv|3gp|mpeg|mpg)$/i.test(f.name)
  );
  if (!files.length) {
    showError('Nenhum arquivo de vídeo válido.');
    return;
  }
  videoList.classList.remove('hidden');
  for (const file of files) {
    const id = crypto.randomUUID();
    videos.set(id, { pipelineStatus: PIPELINE_STATUS.SENT });
    createVideoCard(id, file);
    updateQueueCount();
    processVideo(id, file);
  }
}

uploadZone?.addEventListener('click', (e) => {
  if (e.target.closest('.btn-upload-trigger')) return;
  fileInput.click();
});
uploadZone?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput?.addEventListener('change', (e) => {
  if (e.target.files.length) handleFiles(e.target.files);
  fileInput.value = '';
});
uploadZone?.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone?.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone?.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});

export { runAnalysis };
