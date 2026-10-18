/** Copy the possession totals already accumulated for the live progress payload. */
function possessionProgressFromPayload(payload) {
  const stats = payload && payload.possession_by_team;
  if (!stats || typeof stats !== 'object') return null;
  const numeric = (value) => {
    const number = Number(value);
    return Number.isFinite(number) ? number : 0;
  };
  return {
    frames_total: numeric(stats.frames_total),
    frames_team_0: numeric(stats.frames_team_0),
    frames_team_1: numeric(stats.frames_team_1),
    frames_without_possessor: numeric(stats.frames_without_possessor),
    seconds_team_0: numeric(stats.seconds_team_0),
    seconds_team_1: numeric(stats.seconds_team_1),
    seconds_without_possessor: numeric(stats.seconds_without_possessor),
    percent_team_0: numeric(stats.percent_team_0),
    percent_team_1: numeric(stats.percent_team_1),
    percent_without_possessor: numeric(stats.percent_without_possessor),
  };
}

module.exports = { possessionProgressFromPayload };
