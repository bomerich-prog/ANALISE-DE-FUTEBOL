/**
 * Contrato de eventos da partida enviado ao frontend.
 * Os valores reais serão preenchidos quando a lógica de posse/passes existir.
 * Este módulo apenas normaliza o payload — não calcula eventos.
 *
 * @typedef {Object} MatchEvents
 * @property {string|null} current_possession_team  'blue' | 'red' | null
 * @property {number} blue_passes
 * @property {number} red_passes
 * @property {number} turnovers
 * @property {boolean} ball_detected
 */

function toOptionalCount(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizePossessionTeam(value) {
  if (value === 'blue' || value === 'red') return value;
  if (value === 0 || value === '0') return 'blue';
  if (value === 1 || value === '1') return 'red';
  if (value === 'Time Azul') return 'blue';
  if (value === 'Time Vermelho') return 'red';
  return null;
}

function createPlaceholderMatchEvents(overrides = {}) {
  return {
    current_possession_team: normalizePossessionTeam(overrides.current_possession_team),
    blue_passes: toOptionalCount(overrides.blue_passes, 0),
    red_passes: toOptionalCount(overrides.red_passes, 0),
    turnovers: toOptionalCount(overrides.turnovers, 0),
    ball_detected: Boolean(overrides.ball_detected),
  };
}

function normalizeMatchEvents(raw, fallbacks = {}) {
  const source = raw && typeof raw === 'object' ? raw : {};
  return createPlaceholderMatchEvents({
    current_possession_team: source.current_possession_team ?? fallbacks.current_possession_team,
    blue_passes: source.blue_passes ?? fallbacks.blue_passes,
    red_passes: source.red_passes ?? fallbacks.red_passes,
    turnovers: source.turnovers ?? fallbacks.turnovers,
    ball_detected: source.ball_detected ?? fallbacks.ball_detected,
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
