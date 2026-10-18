/** Read the possession split the pipeline already accumulated. */

export function emptyPossessionByTeam() {
  return {
    frames_total: 0,
    frames_team_0: 0,
    frames_team_1: 0,
    frames_without_possessor: 0,
    seconds_team_0: 0,
    seconds_team_1: 0,
    seconds_without_possessor: 0,
    percent_team_0: 0,
    percent_team_1: 0,
    percent_without_possessor: 0,
  };
}

function numeric(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

export function readPossessionByTeam(analysis) {
  const live = analysis?.progress_stats?.possession_by_team;
  const saved = analysis?.result_json?.possession_by_team;
  const source = live && typeof live === 'object'
    ? live
    : saved && typeof saved === 'object'
      ? saved
      : null;
  if (!source) return emptyPossessionByTeam();
  return {
    frames_total: numeric(source.frames_total),
    frames_team_0: numeric(source.frames_team_0),
    frames_team_1: numeric(source.frames_team_1),
    frames_without_possessor: numeric(source.frames_without_possessor),
    seconds_team_0: numeric(source.seconds_team_0),
    seconds_team_1: numeric(source.seconds_team_1),
    seconds_without_possessor: numeric(source.seconds_without_possessor),
    percent_team_0: numeric(source.percent_team_0),
    percent_team_1: numeric(source.percent_team_1),
    percent_without_possessor: numeric(source.percent_without_possessor),
  };
}

export function formatPossessionPercent(percent) {
  return `${numeric(percent).toFixed(1)}%`;
}

export function formatPossessionSeconds(seconds) {
  const value = Math.max(0, numeric(seconds));
  if (value < 60) return `${value.toFixed(1)} s`;
  const total = Math.round(value);
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return `${minutes}:${String(rest).padStart(2, '0')}`;
}

export function formatPossessionDetail(frames, seconds) {
  return `${Math.round(numeric(frames))} frames · ${formatPossessionSeconds(seconds)}`;
}
