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
const possessionEl = document.getElementById('analysisPossessionTeam');
const bluePassesEl = document.getElementById('analysisBluePasses');
const redPassesEl = document.getElementById('analysisRedPasses');
const turnoversEl = document.getElementById('analysisTurnovers');
const ballEl = document.getElementById('analysisBallFound');
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

function userFacingMessage(status, rawMessage) {
  if (status === ANALYSIS_STATUS.ERROR) {
    return rawMessage || 'Não foi possível concluir a análise.';
  }
  if (status === ANALYSIS_STATUS.COMPLETED) return 'Análise concluída.';
  if (status === ANALYSIS_STATUS.ANALYZING || status === ANALYSIS_STATUS.PENDING) {
    return 'Analisando a partida...';
  }
  return 'Aguardando análise.';
}

function possessionLabel(team) {
  if (team === 'blue' || team === 0 || team === '0' || team === 'Time Azul') return 'Time Azul';
  if (team === 'red' || team === 1 || team === '1' || team === 'Time Vermelho') return 'Time Vermelho';
  return 'Aguardando análise';
}

function getEvents(analysis) {
  const progressEvents = analysis?.progress_stats?.events || analysis?.events || {};
  const resultEvents = analysis?.result_json?.events || {};
  const metrics = analysis?.result_json?.metrics || {};
  const progressStats = analysis?.progress_stats || {};

  return {
    current_possession_team:
      progressEvents.current_possession_team ?? resultEvents.current_possession_team ?? null,
    blue_passes: progressEvents.blue_passes ?? resultEvents.blue_passes ?? 0,
    red_passes: progressEvents.red_passes ?? resultEvents.red_passes ?? 0,
    turnovers: progressEvents.turnovers ?? resultEvents.turnovers ?? 0,
    ball_detected: Boolean(
      progressEvents.ball_detected ??
        resultEvents.ball_detected ??
        progressStats.ball_detected ??
        metrics.ball_detected
    ),
    debug_frame_url:
      progressStats.debug_frame_url ?? analysis?.result_json?.artifacts?.debug_live_frame_url ?? null,
  };
}

function resetViewer() {
  activeAnalysisId = null;
  stopPolling();
  setText(videoNameEl, 'Aguardando vídeo...');
  setText(statusTextEl, 'Aguardando');
  setText(statusMessageEl, 'Aguardando análise.');
  setText(progressTextEl, '0%');
  setText(possessionEl, 'Aguardando análise');
  setText(bluePassesEl, '0');
  setText(redPassesEl, '0');
  setText(turnoversEl, '0');
  setText(ballEl, 'Não detectada');
  if (progressEl) progressEl.style.width = '0%';
  if (processedVideoEl) {
    processedVideoEl.removeAttribute('src');
    processedVideoEl.load();
    processedVideoEl.classList.add('hidden');
  }
  hideLiveFrame();
  placeholderEl?.classList.remove('hidden');
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

function showLiveFrame(events) {
  if (!events.debug_frame_url || !liveFrameEl || !processedVideoEl?.classList.contains('hidden')) return;

  if (liveFrameEl.getAttribute('src') !== events.debug_frame_url) {
    liveFrameEl.src = events.debug_frame_url;
  }
  placeholderEl?.classList.add('hidden');
  liveFrameEl.classList.remove('hidden');
}

function renderMatchEvents(events) {
  setText(possessionEl, possessionLabel(events.current_possession_team));
  setText(bluePassesEl, String(events.blue_passes ?? 0));
  setText(redPassesEl, String(events.red_passes ?? 0));
  setText(turnoversEl, String(events.turnovers ?? 0));
  setText(ballEl, events.ball_detected ? 'Detectada' : 'Não detectada');
}

function renderAnalysis(analysis) {
  if (!analysis) return;

  activeAnalysisId = analysis.id;
  const pct = Math.max(0, Math.min(100, analysis.progress ?? 0));
  const events = getEvents(analysis);

  setText(videoNameEl, analysis.video_name || `Vídeo #${analysis.video_id}`);
  setText(statusTextEl, getAnalysisLabel(analysis.status));
  setText(statusMessageEl, userFacingMessage(analysis.status, analysis.message));
  setText(progressTextEl, `${pct}%`);
  renderMatchEvents(events);
  if (progressEl) progressEl.style.width = `${pct}%`;

  if (analysis.status === ANALYSIS_STATUS.COMPLETED) {
    showProcessedVideo(analysis);
  } else {
    showLiveFrame(events);
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
          progress_stats: {
            frames_processed: 0,
            players_detected: 0,
            ball_detected: false,
            events: analysis.events || {
              current_possession_team: null,
              blue_passes: 0,
              red_passes: 0,
              turnovers: 0,
              ball_detected: false,
            },
          },
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
