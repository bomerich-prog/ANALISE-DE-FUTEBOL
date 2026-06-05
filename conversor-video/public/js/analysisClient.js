import { api } from './api.js';
import { ANALYSIS_STATUS } from './pipelineStatus.js';

const POLL_INTERVAL_MS = 400;

/**
 * Cliente de análise — comunica com a API e faz polling de progresso.
 * A lógica real roda no servidor via AnalysisService + Provider.
 */
export async function startAnalysis(videoId, { onStarted, onProgress, onComplete, onError }) {
  const start = await api('/api/analyses/start', {
    method: 'POST',
    body: JSON.stringify({ videoId }),
  });

  const analysisId = start.id;
  onStarted?.(start);

  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const { analysis } = await api(`/api/analyses/${analysisId}`);
        onProgress?.(analysis.progress ?? 0, analysis.message ?? '', analysis);

        if (analysis.status === ANALYSIS_STATUS.COMPLETED) {
          onComplete?.(analysis);
          window.dispatchEvent(new CustomEvent('dashboard:refresh'));
          resolve(analysis);
          return;
        }

        if (analysis.status === ANALYSIS_STATUS.ERROR) {
          const msg = analysis.result_json?.error || 'Erro na análise.';
          onError?.(msg);
          reject(new Error(msg));
          return;
        }

        setTimeout(poll, POLL_INTERVAL_MS);
      } catch (err) {
        onError?.(err.message);
        reject(err);
      }
    };

    setTimeout(poll, POLL_INTERVAL_MS);
  });
}

export async function listAnalyses() {
  const data = await api('/api/analyses');
  return data.analyses || [];
}

export async function getAnalysis(analysisId) {
  const data = await api(`/api/analyses/${analysisId}`);
  return data.analysis;
}

export async function getLatestAnalysisForVideo(videoId) {
  return api(`/api/analyses/video/${videoId}/latest`);
}
