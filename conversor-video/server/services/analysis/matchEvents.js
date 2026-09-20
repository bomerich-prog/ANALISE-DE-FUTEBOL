/**
 * Contrato de eventos da partida enviado ao frontend.
 * Nesta etapa apenas a posse é preenchida pelo worker.
 *
 * @typedef {Object} MatchEvents
 * @property {number|null} current_possessor_track_id
 * @property {string|null} current_possession_team  'blue' | 'red' | null
 */

function normalizePossessionTeam(value) {
  if (value === 'blue' || value === 'red') return value;
  if (value === 0 || value === '0') return 'blue';
  if (value === 1 || value === '1') return 'red';
  if (value === 'Time Azul') return 'blue';
  if (value === 'Time Vermelho') return 'red';
  return null;
}

function normalizePossessorTrackId(value) {
  if (value === undefined || value === null || value === '') return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) return null;
  return parsed;
}

function createPlaceholderMatchEvents(overrides = {}) {
  return {
    current_possessor_track_id: normalizePossessorTrackId(overrides.current_possessor_track_id),
    current_possession_team: normalizePossessionTeam(overrides.current_possession_team),
  };
}

function normalizeMatchEvents(raw, fallbacks = {}) {
  const source = raw && typeof raw === 'object' ? raw : {};
  return createPlaceholderMatchEvents({
    current_possessor_track_id:
      source.current_possessor_track_id ?? fallbacks.current_possessor_track_id,
    current_possession_team: source.current_possession_team ?? fallbacks.current_possession_team,
  });
}

function toUserFacingAnalysisMessage(status, rawMessage) {
  if (status === 'error') {
    return rawMessage || 'Não foi possível concluir a análise.';
  }
  if (status === 'completed') {
    return 'Análise concluída.';
  }
  if (status === 'analyzing' || status === 'pending') {
    return 'Analisando a partida...';
  }
  return 'Aguardando análise.';
}

module.exports = {
  createPlaceholderMatchEvents,
  normalizeMatchEvents,
  toUserFacingAnalysisMessage,
};
