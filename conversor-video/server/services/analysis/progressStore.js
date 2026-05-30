/** Armazena progresso em memória (mock). Futuro: Redis ou fila de jobs. */
const store = new Map();

module.exports = {
  set(analysisId, data) {
    store.set(analysisId, { ...data, updatedAt: Date.now() });
  },
  get(analysisId) {
    return store.get(analysisId) || { progress: 0, message: '' };
  },
  delete(analysisId) {
    store.delete(analysisId);
  },
};
