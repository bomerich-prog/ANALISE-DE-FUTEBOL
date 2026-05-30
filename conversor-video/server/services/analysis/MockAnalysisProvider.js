const { AnalysisProvider, updateAnalysisProgress } = require('./AnalysisProvider');

const MOCK_STEPS = [
  { progress: 10, message: 'Carregando frames do vídeo...', delay: 600 },
  { progress: 30, message: 'Detectando jogadores (simulado)...', delay: 800 },
  { progress: 55, message: 'Rastreando movimentação (simulado)...', delay: 900 },
  { progress: 75, message: 'Extraindo métricas táticas (simulado)...', delay: 700 },
  { progress: 95, message: 'Consolidando resultados...', delay: 600 },
];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Provedor mock — simula pipeline de visão computacional.
 * Trocar por YoloAnalysisProvider / OpenCVAnalysisProvider sem alterar rotas.
 */
class MockAnalysisProvider extends AnalysisProvider {
  async analyze(analysisId, videoId, userId, onProgress) {
    for (const step of MOCK_STEPS) {
      updateAnalysisProgress(analysisId, step.progress, step.message);
      onProgress?.(step.progress, step.message);
      await sleep(step.delay);
    }

    return {
      version: 'mock-1.0',
      provider: 'MockAnalysisProvider',
      message: 'Análise simulada concluída. Módulos OpenCV/YOLO serão integrados futuramente.',
      video_id: videoId,
      metrics: {
        frames_processed: 0,
        players_detected: 0,
        ball_tracks: 0,
        events_extracted: 0,
      },
      tracks: {},
      events: {},
    };
  }
}

module.exports = MockAnalysisProvider;
