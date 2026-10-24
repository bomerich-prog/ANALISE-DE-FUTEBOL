/** Read the per-track totals the pipeline already accumulated. */

import { formatPossessionSeconds } from './possessionCounts.js';

export function emptyPlayerStats() {
  return { tracks: [] };
}

function numeric(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function whole(value) {
  const number = numeric(value);
  return number > 0 ? Math.round(number) : 0;
}

function normalizeTrack(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const trackId = Number(raw.track_id);
  if (!Number.isInteger(trackId) || trackId < 0) return null;
  const team = raw.team === 'Team 0' || raw.team === 'Team 1' ? raw.team : null;
  const percent = Number(raw.percent_of_team);
  const meanX = Number(raw.mean_x_cm);
  const meanY = Number(raw.mean_y_cm);
  return {
    track_id: trackId,
    team,
    frames_seen: whole(raw.frames_seen),
    frames_with_ball: whole(raw.frames_with_ball),
    possession_spells: whole(raw.possession_spells),
    seconds_with_ball: Math.max(0, numeric(raw.seconds_with_ball)),
    passes_made: whole(raw.passes_made),
    passes_received: whole(raw.passes_received),
    percent_of_team: team && Number.isFinite(percent) ? percent : null,
    position_samples: whole(raw.position_samples),
    mean_x_cm: Number.isFinite(meanX) ? meanX : null,
    mean_y_cm: Number.isFinite(meanY) ? meanY : null,
  };
}

export function readPlayerStats(analysis) {
  const live = analysis?.progress_stats?.player_stats;
  const saved = analysis?.result_json?.player_stats;
  for (const source of [live, saved]) {
    if (source && typeof source === 'object' && Array.isArray(source.tracks)) {
      return { tracks: source.tracks.map(normalizeTrack).filter(Boolean) };
    }
  }
  return emptyPlayerStats();
}

/** Passes made, then time on the ball, then how long the track was visible. */
export function sortPlayerTracks(tracks) {
  return [...tracks].sort((a, b) => {
    if (b.passes_made !== a.passes_made) return b.passes_made - a.passes_made;
    if (b.frames_with_ball !== a.frames_with_ball) return b.frames_with_ball - a.frames_with_ball;
    if (b.frames_seen !== a.frames_seen) return b.frames_seen - a.frames_seen;
    return a.track_id - b.track_id;
  });
}

export function formatPlayerSummary(tracks) {
  const list = Array.isArray(tracks) ? tracks : [];
  if (!list.length) return 'Nenhuma trilha ainda.';
  const withBall = list.filter((track) => track.frames_with_ball > 0).length;
  const withPass = list.filter((track) => track.passes_made > 0).length;
  return `${list.length} trilhas · ${withBall} com posse · ${withPass} com passe. Ordenado por passes feitos e tempo de posse.`;
}

export function formatPlayerTeam(team) {
  if (team === 'Team 0' || team === 'Team 1') return team;
  return 'Sem time';
}

export function formatPlayerPossession(seconds) {
  return formatPossessionSeconds(seconds);
}

export function formatPlayerPercent(percent) {
  if (percent == null || percent === '') return '—';
  const number = Number(percent);
  if (!Number.isFinite(number)) return '—';
  return `${number.toFixed(1)}%`;
}

export function formatMeanPosition(track) {
  if (!track || track.mean_x_cm == null || track.mean_y_cm == null) return 'Sem coordenada';
  const length = (track.mean_x_cm / 100).toFixed(1);
  const width = (track.mean_y_cm / 100).toFixed(1);
  return `${length} m × ${width} m`;
}

export function toggleOpenTrack(openIds, trackId) {
  const next = new Set(openIds);
  if (next.has(trackId)) next.delete(trackId);
  else next.add(trackId);
  return next;
}
