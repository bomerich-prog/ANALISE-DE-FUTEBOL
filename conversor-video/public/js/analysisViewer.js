import {
  startAnalysis,
  getAnalysis,
  getLatestAnalysisForVideo,
} from './analysisClient.js';
import { countPassesByTeam } from './passCounts.js';
import {
  readPossessionByTeam,
  formatPossessionPercent,
  formatPossessionDetail,
} from './possessionCounts.js';
import {
  readBallHeatmap,
  drawBallHeatmap,
  hottestCell,
  formatHottestZone,
  formatHottestDetail,
} from './ballHeatmap.js';
import { ANALYSIS_STATUS, getAnalysisLabel } from './pipelineStatus.js';

const POLL_INTERVAL_MS = 500;

function byId(id) {
  if (typeof document === 'undefined') return null;
  return document.getElementById(id);
}

const modal = byId('analysisVisualization');
const videoNameEl = byId('analysisVideoName');
const statusTextEl = byId('analysisStatusText');
const statusMessageEl = byId('analysisStatusMessage');
const progressEl = byId('analysisViewerProgress');
const progressTextEl = byId('analysisViewerProgressText');
const possessionEl = byId('analysisPossessionTeam');
const bluePassesEl = byId('analysisBluePasses');
const redPassesEl = byId('analysisRedPasses');
const turnoversEl = byId('analysisTurnovers');
const ballEl = byId('analysisBallFound');
const possessionTeam0PercentEl = byId('possessionTeam0Percent');
const possessionTeam1PercentEl = byId('possessionTeam1Percent');
const possessionNonePercentEl = byId('possessionNonePercent');
const possessionTeam0DetailEl = byId('possessionTeam0Detail');
const possessionTeam1DetailEl = byId('possessionTeam1Detail');
const possessionNoneDetailEl = byId('possessionNoneDetail');
const possessionBarTeam0El = byId('possessionBarTeam0');
const possessionBarTeam1El = byId('possessionBarTeam1');
const possessionBarNoneEl = byId('possessionBarNone');
const ballHeatmapCanvasEl = byId('ballHeatmapCanvas');
const ballHeatmapEmptyEl = byId('ballHeatmapEmpty');
const ballHeatmapObservedEl = byId('ballHeatmapObserved');
const ballHeatmapHotZoneEl = byId('ballHeatmapHotZone');
const ballHeatmapHotDetailEl = byId('ballHeatmapHotDetail');
const ballHeatmapNoFieldEl = byId('ballHeatmapNoField');
const processedVideoEl = byId('analysisProcessedVideo');
const liveFrameEl = byId('analysisLiveFrame');
const placeholderEl = byId('analysisVideoPlaceholder');

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
  if (team === 'Team 0' || team === 0 || team === '0') return 'Team 0';
  if (team === 'Team 1' || team === 1 || team === '1') return 'Team 1';
  return 'Aguardando análise';
}

function backendPasses(analysis) {
  const live = analysis?.progress_stats?.passes;
  if (live && typeof live === 'object') return live;
  const saved = analysis?.result_json?.passes;
  if (saved && typeof saved === 'object') return saved;
  return null;
}

function readTurnovers(passes) {
  const number = Number(passes?.turnovers);
  if (!Number.isFinite(number) || number < 0) return 0;
  return Math.round(number);
}

export function getEvents(analysis) {
  const progressEvents = analysis?.progress_stats?.events || analysis?.events || {};
  const resultEvents = analysis?.result_json?.events || {};
  const metrics = analysis?.result_json?.metrics || {};
  const progressStats = analysis?.progress_stats || {};
  const passes = countPassesByTeam(backendPasses(analysis));

  return {
    current_possessor_track_id:
      progressEvents.current_possessor_track_id ?? resultEvents.current_possessor_track_id ?? null,
    current_possession_team:
      progressEvents.current_possession_team ?? resultEvents.current_possession_team ?? null,
    blue_passes: passes.team0,
    red_passes: passes.team1,
    turnovers: readTurnovers(backendPasses(analysis)),
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
  renderPossessionByTeam(null);
  renderBallHeatmap(null);
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

function setBarWidth(el, percent) {
  if (!el) return;
  const value = Number(percent);
  const width = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
  el.style.width = `${width}%`;
}

function renderPossessionByTeam(analysis) {
  const stats = readPossessionByTeam(analysis);
  setText(possessionTeam0PercentEl, formatPossessionPercent(stats.percent_team_0));
  setText(possessionTeam1PercentEl, formatPossessionPercent(stats.percent_team_1));
  setText(possessionNonePercentEl, formatPossessionPercent(stats.percent_without_possessor));
  setText(
    possessionTeam0DetailEl,
    formatPossessionDetail(stats.frames_team_0, stats.seconds_team_0)
  );
  setText(
    possessionTeam1DetailEl,
    formatPossessionDetail(stats.frames_team_1, stats.seconds_team_1)
  );
  setText(
    possessionNoneDetailEl,
    formatPossessionDetail(stats.frames_without_possessor, stats.seconds_without_possessor)
  );
  setBarWidth(possessionBarTeam0El, stats.percent_team_0);
  setBarWidth(possessionBarTeam1El, stats.percent_team_1);
  setBarWidth(possessionBarNoneEl, stats.percent_without_possessor);
}

function renderBallHeatmap(analysis) {
  const heatmap = readBallHeatmap(analysis);
  if (ballHeatmapCanvasEl) {
    drawBallHeatmap(
      ballHeatmapCanvasEl.getContext('2d'),
      ballHeatmapCanvasEl.width,
      ballHeatmapCanvasEl.height,
      heatmap
    );
  }
  ballHeatmapEmptyEl?.classList.toggle('hidden', heatmap.observed_points > 0);
  const hottest = hottestCell(heatmap);
  setText(ballHeatmapObservedEl, String(heatmap.observed_points));
  setText(ballHeatmapHotZoneEl, formatHottestZone(hottest));
  setText(ballHeatmapHotDetailEl, formatHottestDetail(hottest));
  setText(
    ballHeatmapNoFieldEl,
    String(heatmap.frames_without_field_point + heatmap.frames_outside_field)
  );
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
  renderPossessionByTeam(analysis);
  renderBallHeatmap(analysis);
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
              current_possessor_track_id: null,
              current_possession_team: null,
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

if (typeof document !== 'undefined') {
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
}

export { openAnalysisVisualization, startAnalysisInVisualization, renderAnalysis };
