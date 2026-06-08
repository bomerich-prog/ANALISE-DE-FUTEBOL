import {
  startAnalysis,
  getAnalysis,
  getLatestAnalysisForVideo,
} from './analysisClient.js';
import { ANALYSIS_STATUS, getAnalysisLabel } from './pipelineStatus.js';

const POLL_INTERVAL_MS = 500;

const modal = document.getElementById('analysisVisualization');
const videoNameEl = document.getElementById('analysisVideoName');
const statusTextEl = document.getElementById('analysisStatusText');
const statusMessageEl = document.getElementById('analysisStatusMessage');
const progressEl = document.getElementById('analysisViewerProgress');
const progressTextEl = document.getElementById('analysisViewerProgressText');
const playersEl = document.getElementById('analysisPlayersCount');
const ballEl = document.getElementById('analysisBallFound');
const framesEl = document.getElementById('analysisFramesProcessed');
const processedVideoEl = document.getElementById('analysisProcessedVideo');
const liveFrameEl = document.getElementById('analysisLiveFrame');
const placeholderEl = document.getElementById('analysisVideoPlaceholder');

let pollTimer = null;
let activeAnalysisId = null;

function setText(el, value) {
  if (el) el.textContent = value;
}

function openModal() {
  modal?.classList.remove('hidden');
}

function closeModal() {
  modal?.classList.add('hidden');
  stopPolling();
}

function stopPolling() {
  if (pollTimer) window.clearTimeout(pollTimer);
  pollTimer = null;
}

function hideLiveFrame() {
  if (!liveFrameEl) return;
  liveFrameEl.removeAttribute('src');
  liveFrameEl.classList.add('hidden');
}

function resetViewer() {
  activeAnalysisId = null;
  stopPolling();
  setText(videoNameEl, 'Aguardando vídeo...');
  setText(statusTextEl, 'Aguardando');
  setText(statusMessageEl, 'Pronto para iniciar YOLO.');
  setText(progressTextEl, '0%');
  setText(playersEl, '0');
  setText(ballEl, 'Não');
  setText(framesEl, '0');
  if (progressEl) progressEl.style.width = '0%';
  if (processedVideoEl) {
    processedVideoEl.removeAttribute('src');
    processedVideoEl.load();
    processedVideoEl.classList.add('hidden');
  }
  hideLiveFrame();
  placeholderEl?.classList.remove('hidden');
}

function getMetrics(analysis) {
  const metrics = analysis?.result_json?.metrics || {};
  const progressStats = analysis?.progress_stats || {};
  return {
    frames_processed: progressStats.frames_processed ?? metrics.frames_processed ?? 0,
    players_detected: progressStats.players_detected ?? metrics.players_detected ?? 0,
    ball_detected: progressStats.ball_detected ?? metrics.ball_detected ?? false,
    debug_frame_url:
      progressStats.debug_frame_url ?? analysis?.result_json?.artifacts?.debug_live_frame_url ?? null,
  };
}

function showProcessedVideo(analysis) {
  const url = analysis?.result_json?.artifacts?.processed_video_url;
  if (!url || !processedVideoEl) return;

  if (processedVideoEl.getAttribute('src') !== url) {
    processedVideoEl.src = url;
    processedVideoEl.load();
  }
  placeholderEl?.classList.add('hidden');
  hideLiveFrame();
  processedVideoEl.classList.remove('hidden');
}

function showLiveFrame(metrics) {
  if (!metrics.debug_frame_url || !liveFrameEl || !processedVideoEl?.classList.contains('hidden')) return;

  if (liveFrameEl.getAttribute('src') !== metrics.debug_frame_url) {
    liveFrameEl.src = metrics.debug_frame_url;
  }
  placeholderEl?.classList.add('hidden');
  liveFrameEl.classList.remove('hidden');
}

function renderAnalysis(analysis) {
  if (!analysis) return;

  activeAnalysisId = analysis.id;
  const pct = Math.max(0, Math.min(100, analysis.progress ?? 0));
  const metrics = getMetrics(analysis);

  setText(videoNameEl, analysis.video_name || `Vídeo #${analysis.video_id}`);
  setText(statusTextEl, getAnalysisLabel(analysis.status));
  setText(statusMessageEl, analysis.message || 'Análise visual YOLO em execução.');
  setText(progressTextEl, `${pct}%`);
  setText(playersEl, String(metrics.players_detected));
  setText(ballEl, metrics.ball_detected ? 'Sim' : 'Não');
  setText(framesEl, String(metrics.frames_processed));
  if (progressEl) progressEl.style.width = `${pct}%`;

  if (analysis.status === ANALYSIS_STATUS.COMPLETED) {
    showProcessedVideo(analysis);
  } else {
    showLiveFrame(metrics);
  }
}

async function pollAnalysis(analysisId) {
  stopPolling();
  try {
    const analysis = await getAnalysis(analysisId);
    renderAnalysis(analysis);

    if (analysis.status === ANALYSIS_STATUS.ANALYZING || analysis.status === ANALYSIS_STATUS.PENDING) {
      pollTimer = window.setTimeout(() => pollAnalysis(analysisId), POLL_INTERVAL_MS);
      return;
    }

    window.dispatchEvent(new CustomEvent('analysis:finished', { detail: { analysis } }));
  } catch (err) {
    setText(statusTextEl, 'Erro');
    setText(statusMessageEl, err.message);
  }
}

async function openAnalysisVisualization({ analysisId, videoId } = {}) {
  resetViewer();
  openModal();

  if (analysisId) {
    await pollAnalysis(analysisId);
    return;
  }

  if (videoId) {
    const { video, analysis } = await getLatestAnalysisForVideo(videoId);
    setText(videoNameEl, video?.video_name || `Vídeo #${videoId}`);
    if (analysis) {
      renderAnalysis(analysis);
      await pollAnalysis(analysis.id);
    } else {
      setText(statusMessageEl, 'Nenhuma análise encontrada para este vídeo.');
    }
  }
}

async function startAnalysisInVisualization(videoId) {
  resetViewer();
  openModal();

  try {
    await startAnalysis(videoId, {
      onStarted: (analysis) => {
        activeAnalysisId = analysis.id;
        renderAnalysis({
          ...analysis,
          video_id: videoId,
          status: ANALYSIS_STATUS.ANALYZING,
          progress_stats: { frames_processed: 0, players_detected: 0, ball_detected: false },
        });
      },
      onProgress: (pct, msg, analysis) => {
        renderAnalysis(analysis || {
          id: activeAnalysisId,
          video_id: videoId,
          status: ANALYSIS_STATUS.ANALYZING,
          progress: pct,
          message: msg,
        });
      },
      onComplete: (analysis) => {
        renderAnalysis(analysis);
        window.dispatchEvent(new CustomEvent('analysis:finished', { detail: { analysis } }));
      },
      onError: (msg) => {
        setText(statusTextEl, 'Erro');
        setText(statusMessageEl, msg);
      },
    });
  } catch (err) {
    setText(statusTextEl, 'Erro');
    setText(statusMessageEl, err.message);
  }
}

document.querySelectorAll('[data-analysis-close]').forEach((el) => {
  el.addEventListener('click', closeModal);
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !modal?.classList.contains('hidden')) {
    closeModal();
  }
});

window.addEventListener('analysis-visualization:open', (event) => {
  openAnalysisVisualization(event.detail);
});

export { openAnalysisVisualization, startAnalysisInVisualization };
