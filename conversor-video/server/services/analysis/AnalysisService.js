const { db } = require('../../db');
const MockAnalysisProvider = require('./MockAnalysisProvider');
const YoloAnalysisProvider = require('./yolo/YoloAnalysisProvider');
const progressStore = require('./progressStore');
const {
  setAnalysisStatus,
  setVideoPipelineStatus,
  PIPELINE_STATUS,
  ANALYSIS_STATUS,
} = require('./AnalysisProvider');
const {
  createPlaceholderMatchEvents,
  normalizeMatchEvents,
  toUserFacingAnalysisMessage,
} = require('./matchEvents');
const { passProgressFromPayload } = require('./yolo/passProgress');

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
    const type = process.env.ANALYSIS_PROVIDER || 'yolo';
    switch (type) {
      case 'mock':
        return new MockAnalysisProvider();
      case 'yolo':
      case 'ultralytics':
        return new YoloAnalysisProvider();
      default:
        throw new Error(`Provider de análise inválido: ${type}`);
    }
  }

  getProgressStatsFromResult(result) {
    const metrics = result?.metrics || {};
    const stats = {
      frames_processed: metrics.frames_processed || 0,
      players_detected: metrics.players_detected || 0,
      ball_detected: Boolean(metrics.ball_detected),
      events: normalizeMatchEvents(result?.events),
    };
    const passes = passProgressFromPayload({ passes: result?.passes });
    if (passes) stats.passes = passes;
    if (result?.artifacts?.debug_live_frame_url) {
      stats.debug_frame_url = result.artifacts.debug_live_frame_url;
    }
    return stats;
  }

  sanitizeAnalysisResult(resultJson, analysisId) {
    if (!resultJson || typeof resultJson !== 'object') return {};
    const result = JSON.parse(JSON.stringify(resultJson));
    if (result.artifacts?.processed_video_path) {
      delete result.artifacts.processed_video_path;
      result.artifacts.processed_video_url = `/api/analyses/${analysisId}/video`;
    }
    if (result.artifacts?.debug_live_frame_path) {
      delete result.artifacts.debug_live_frame_path;
      result.artifacts.debug_live_frame_url = `/api/analyses/${analysisId}/debug-frame`;
    }
    result.events = normalizeMatchEvents(result.events);
    return result;
  }

  getAnalysisVideoPath(userId, analysisId) {
    const row = db
      .prepare(
        `SELECT a.result_json
         FROM analyses a
         WHERE a.id = ? AND a.user_id = ?`
      )
      .get(analysisId, userId);

    if (!row?.result_json) return null;

    const result = JSON.parse(row.result_json);
    return result?.artifacts?.processed_video_path || null;
  }

  getAnalysisDebugFrame(userId, analysisId) {
    const row = db
      .prepare(
        `SELECT id, result_json
         FROM analyses
         WHERE id = ? AND user_id = ?`
      )
      .get(analysisId, userId);

    if (!row) return null;

    const progress = progressStore.get(analysisId);
    if (progress.debug_frame?.data) {
      return progress.debug_frame;
    }

    const result = row.result_json ? JSON.parse(row.result_json) : {};
    const legacyPath = result?.artifacts?.debug_live_frame_path;
    if (legacyPath) {
      return {
        contentType: 'image/jpeg',
        path: legacyPath,
      };
    }

    return null;
  }

  getVideoForAnalysis(userId, videoId) {
    return db
      .prepare(
        `SELECT id, video_name, pipeline_status
         FROM conversions
         WHERE id = ? AND user_id = ?`
      )
      .get(videoId, userId);
  }

  getLatestAnalysisForVideo(userId, videoId) {
    const row = db
      .prepare(
        `SELECT a.*, c.video_name
         FROM analyses a
         JOIN conversions c ON c.id = a.video_id
         WHERE a.video_id = ? AND a.user_id = ?
         ORDER BY a.created_at DESC
         LIMIT 1`
      )
      .get(videoId, userId);

    return this.mapAnalysisRow(row);
  }

  mapAnalysisRow(row) {
    if (!row) return null;

    const progress = progressStore.get(row.id);
    const resultJson = row.result_json ? JSON.parse(row.result_json) : {};
    const progressStats = progress.progress_stats || this.getProgressStatsFromResult(resultJson);
    progressStats.events = normalizeMatchEvents(progressStats.events || resultJson.events);
    const resolvedProgress =
      progress.message || progress.progress
        ? progress.progress
        : row.status === ANALYSIS_STATUS.COMPLETED
          ? 100
          : 0;
    const resolvedMessage = toUserFacingAnalysisMessage(
      row.status,
      progress.message || resultJson.error
    );

    return {
      id: row.id,
      video_id: row.video_id,
      video_name: row.video_name,
      status: row.status,
      created_at: row.created_at,
      updated_at: row.updated_at,
      result_json: this.sanitizeAnalysisResult(resultJson, row.id),
      progress: resolvedProgress,
      message: resolvedMessage,
      progress_stats: progressStats,
      events: progressStats.events,
    };
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
    const initialEvents = createPlaceholderMatchEvents();
    progressStore.set(analysisId, {
      progress: 0,
      message: 'Analisando a partida...',
      progress_stats: {
        frames_processed: 0,
        players_detected: 0,
        ball_detected: false,
        events: initialEvents,
      },
    });

    this.runInBackground(analysisId, videoId, userId);

    return {
      id: analysisId,
      video_id: videoId,
      video_name: video.video_name,
      status: ANALYSIS_STATUS.ANALYZING,
      progress: 0,
      message: 'Analisando a partida...',
      progress_stats: {
        frames_processed: 0,
        players_detected: 0,
        ball_detected: false,
        events: initialEvents,
      },
      events: initialEvents,
    };
  }

  runInBackground(analysisId, videoId, userId) {
    if (this.running.has(analysisId)) return;
    this.running.add(analysisId);

    this.provider
      .analyze(analysisId, videoId, userId, (progress, message, details = {}) => {
        progressStore.set(analysisId, { progress, message, ...details });
      })
      .then((result) => {
        setAnalysisStatus(analysisId, ANALYSIS_STATUS.COMPLETED, result);
        setVideoPipelineStatus(videoId, userId, PIPELINE_STATUS.ANALYSIS_COMPLETE);
        progressStore.set(analysisId, {
          progress: 100,
          message: 'Análise concluída com sucesso.',
          progress_stats: this.getProgressStatsFromResult(result),
        });
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

    return this.mapAnalysisRow(row);
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
