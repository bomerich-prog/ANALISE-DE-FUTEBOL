const { db } = require('../../db');
const { PIPELINE_STATUS, ANALYSIS_STATUS } = require('../../constants/pipelineStatus');

/**
 * Contrato para provedores de análise de vídeo.
 * Substituir MockAnalysisProvider por OpenCV/YOLOProvider no futuro.
 *
 * @typedef {Object} AnalysisResult
 * @property {string} version
 * @property {Object} metrics
 * @property {Object} [tracks]
 * @property {Object} [events]
 */

class AnalysisProvider {
  /**
   * @param {number} analysisId
   * @param {number} videoId
   * @param {number} userId
   * @param {(progress: number, message: string) => void} onProgress
   * @returns {Promise<AnalysisResult>}
   */
  // eslint-disable-next-line no-unused-vars
  async analyze(analysisId, videoId, userId, onProgress) {
    throw new Error('Método analyze() deve ser implementado pelo provedor.');
  }
}

function updateAnalysisProgress(analysisId, progress, message) {
  const progressStore = require('./progressStore');
  progressStore.set(analysisId, { progress, message });
}

function setAnalysisStatus(analysisId, status, resultJson = null) {
  if (resultJson !== null) {
    db.prepare(
      `UPDATE analyses SET status = ?, result_json = ?, updated_at = datetime('now') WHERE id = ?`
    ).run(status, JSON.stringify(resultJson), analysisId);
  } else {
    db.prepare(
      `UPDATE analyses SET status = ?, updated_at = datetime('now') WHERE id = ?`
    ).run(status, analysisId);
  }
}

function setVideoPipelineStatus(videoId, userId, status) {
  db.prepare(
    `UPDATE conversions SET pipeline_status = ? WHERE id = ? AND user_id = ?`
  ).run(status, videoId, userId);
}

module.exports = {
  AnalysisProvider,
  updateAnalysisProgress,
  setAnalysisStatus,
  setVideoPipelineStatus,
  PIPELINE_STATUS,
  ANALYSIS_STATUS,
};
