/** Armazena progresso em memória (mock). Futuro: Redis ou fila de jobs. */
const store = new Map();

module.exports = {
  set(analysisId, data) {
    const previous = store.get(analysisId) || {};
    store.set(analysisId, {
      ...previous,
      ...data,
      progress_stats: {
        ...(previous.progress_stats || {}),
        ...(data.progress_stats || {}),
      },
      updatedAt: Date.now(),
    });
  },
  get(analysisId) {
    return store.get(analysisId) || { progress: 0, message: '' };
  },
  delete(analysisId) {
    store.delete(analysisId);
  },
};
