const { db } = require('../../db');
const MockAnalysisProvider = require('./MockAnalysisProvider');
const progressStore = require('./progressStore');
const {
  setAnalysisStatus,
  setVideoPipelineStatus,
  PIPELINE_STATUS,
  ANALYSIS_STATUS,
} = require('./AnalysisProvider');

/**
 * Orquestrador de análises — ponto único de entrada para rotas e jobs.
 * Alterar getProvider() para registrar provedores reais de CV/ML.
 */
class AnalysisService {
  constructor() {
    this.provider = this.getProvider();
    this.running = new Set();
  }

  getProvider() {
    const type = process.env.ANALYSIS_PROVIDER || 'mock';
    switch (type) {
      case 'mock':
      default:
        return new MockAnalysisProvider();
    }
  }

  /**
   * Inicia análise assíncrona para um vídeo convertido.
   */
  async startAnalysis(userId, videoId) {
    const video = db
      .prepare('SELECT * FROM conversions WHERE id = ? AND user_id = ?')
      .get(videoId, userId);

    if (!video) {
      throw new Error('Vídeo não encontrado.');
    }

    if (video.pipeline_status === PIPELINE_STATUS.ANALYZING) {
      throw new Error('Este vídeo já está em análise.');
    }

    const existing = db
      .prepare(
        `SELECT id FROM analyses
         WHERE video_id = ? AND user_id = ? AND status = ?`
      )
      .get(videoId, userId, ANALYSIS_STATUS.ANALYZING);

    if (existing) {
      throw new Error('Já existe uma análise em andamento para este vídeo.');
    }

    const insert = db
      .prepare(
        `INSERT INTO analyses (user_id, video_id, status, result_json)
         VALUES (?, ?, ?, ?)`
      )
      .run(userId, videoId, ANALYSIS_STATUS.ANALYZING, '{}');

    const analysisId = insert.lastInsertRowid;

    setVideoPipelineStatus(videoId, userId, PIPELINE_STATUS.ANALYZING);
    progressStore.set(analysisId, { progress: 0, message: 'Iniciando análise...' });

    this.runInBackground(analysisId, videoId, userId);

    return {
      id: analysisId,
      video_id: videoId,
      status: ANALYSIS_STATUS.ANALYZING,
      progress: 0,
      message: 'Iniciando análise...',
    };
  }

  runInBackground(analysisId, videoId, userId) {
    if (this.running.has(analysisId)) return;
    this.running.add(analysisId);

    this.provider
      .analyze(analysisId, videoId, userId, (progress, message) => {
        progressStore.set(analysisId, { progress, message });
      })
      .then((result) => {
        setAnalysisStatus(analysisId, ANALYSIS_STATUS.COMPLETED, result);
        setVideoPipelineStatus(videoId, userId, PIPELINE_STATUS.ANALYSIS_COMPLETE);
        progressStore.set(analysisId, { progress: 100, message: 'Análise concluída com sucesso.' });
      })
      .catch((err) => {
        console.error('Erro na análise:', err);
        setAnalysisStatus(analysisId, ANALYSIS_STATUS.ERROR, {
          error: err.message || 'Erro na análise.',
        });
        setVideoPipelineStatus(videoId, userId, PIPELINE_STATUS.ERROR);
        progressStore.set(analysisId, { progress: 0, message: err.message });
      })
      .finally(() => {
        this.running.delete(analysisId);
      });
  }

  getAnalysis(userId, analysisId) {
    const row = db
      .prepare(
        `SELECT a.*, c.video_name
         FROM analyses a
         JOIN conversions c ON c.id = a.video_id
         WHERE a.id = ? AND a.user_id = ?`
      )
      .get(analysisId, userId);

    if (!row) return null;

    const progress = progressStore.get(analysisId);

    return {
      id: row.id,
      video_id: row.video_id,
      video_name: row.video_name,
      status: row.status,
      created_at: row.created_at,
      updated_at: row.updated_at,
      result_json: row.result_json ? JSON.parse(row.result_json) : {},
      progress: progress.progress,
      message: progress.message,
    };
  }

  listAnalyses(userId) {
    return db
      .prepare(
        `SELECT a.id, a.video_id, a.status, a.created_at, a.updated_at,
                c.video_name, c.original_format, c.file_size
         FROM analyses a
         JOIN conversions c ON c.id = a.video_id
         WHERE a.user_id = ?
         ORDER BY a.created_at DESC`
      )
      .all(userId);
  }
}

module.exports = new AnalysisService();
