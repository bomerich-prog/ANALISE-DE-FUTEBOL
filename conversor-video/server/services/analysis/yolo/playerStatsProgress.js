/** Copy the per-track totals the worker already accumulated. */

function numeric(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function whole(value) {
  const number = numeric(value);
  return number > 0 ? Math.round(number) : 0;
}

function optionalMean(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function playerStatsFromPayload(payload) {
  const stats = payload && payload.player_stats;
  if (!stats || typeof stats !== 'object' || !Array.isArray(stats.tracks)) return null;
  const tracks = [];
  for (const raw of stats.tracks) {
    if (!raw || typeof raw !== 'object') continue;
    const trackId = Number(raw.track_id);
    if (!Number.isInteger(trackId) || trackId < 0) continue;
    const team = raw.team === 'Team 0' || raw.team === 'Team 1' ? raw.team : null;
    const percent = Number(raw.percent_of_team);
    tracks.push({
      track_id: trackId,
      team,
      frames_seen: whole(raw.frames_seen),
      frames_with_ball: whole(raw.frames_with_ball),
      possession_spells: whole(raw.possession_spells),
      seconds_with_ball: numeric(raw.seconds_with_ball),
      passes_made: whole(raw.passes_made),
      passes_received: whole(raw.passes_received),
      percent_of_team: team && Number.isFinite(percent) ? percent : null,
      position_samples: whole(raw.position_samples),
      mean_x_cm: optionalMean(raw.mean_x_cm),
      mean_y_cm: optionalMean(raw.mean_y_cm),
    });
  }
  return { tracks };
}

module.exports = { playerStatsFromPayload };
