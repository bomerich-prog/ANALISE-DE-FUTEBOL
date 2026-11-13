/** Draw the visible-third radar. Points outside that third are omitted. */

export const FIELD_LENGTH_CM = 12000;
export const FIELD_WIDTH_CM = 7000;
export const VISIBLE_THIRD_CM = 3500;
export const EDGE_TOLERANCE_CM = 100;

const GRASS = '#306e2a';
const GRASS_DIM = '#122a10';
const TEAM_COLOR = { 0: '#2f7ae8', 1: '#dc4444' };

function finite(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function positive(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

export function emptyPitchRadar() {
  return {
    calibrated: false,
    goal_end: null,
    visible_third_cm: VISIBLE_THIRD_CM,
    field_length_cm: FIELD_LENGTH_CM,
    field_width_cm: FIELD_WIDTH_CM,
    players: [],
    ball: null,
  };
}

export function inCalibratedThird(
  x,
  y,
  goalEnd,
  length = FIELD_LENGTH_CM,
  width = FIELD_WIDTH_CM,
  third = VISIBLE_THIRD_CM,
) {
  if (goalEnd !== 'low_x' && goalEnd !== 'high_x') return false;
  if (x === null || y === null) return false;
  if (y < -EDGE_TOLERANCE_CM || y > width + EDGE_TOLERANCE_CM) return false;
  if (goalEnd === 'pitch') {
    return x >= -EDGE_TOLERANCE_CM && x <= length + EDGE_TOLERANCE_CM;
  }
  if (goalEnd === 'centre') {
    const middle = length / 2;
    return x >= middle - third - EDGE_TOLERANCE_CM && x <= middle + third + EDGE_TOLERANCE_CM;
  }
  if (goalEnd === 'low_x') return x >= -EDGE_TOLERANCE_CM && x <= third + EDGE_TOLERANCE_CM;
  return x >= length - third - EDGE_TOLERANCE_CM && x <= length + EDGE_TOLERANCE_CM;
}

export function normalizePitchRadar(source) {
  const empty = emptyPitchRadar();
  if (!source || typeof source !== 'object') return empty;
  const goalEnd = source.goal_end === 'high_x' || source.goal_end === 'low_x' || source.goal_end === 'centre' || source.goal_end === 'pitch'
    ? source.goal_end
    : null;
  const calibrated = Boolean(source.calibrated) && goalEnd !== null;
  const length = positive(source.field_length_cm, FIELD_LENGTH_CM);
  const width = positive(source.field_width_cm, FIELD_WIDTH_CM);
  const third = positive(source.visible_third_cm, VISIBLE_THIRD_CM);
  const players = [];
  if (calibrated && Array.isArray(source.players)) {
    for (const person of source.players) {
      if (!person || typeof person !== 'object') continue;
      const x = finite(person.map_x);
      const y = finite(person.map_y);
      if (!inCalibratedThird(x, y, goalEnd, length, width, third)) continue;
      let team = finite(person.team_id);
      team = team === 0 || team === 1 ? team : null;
      players.push({
        map_x: x,
        map_y: y,
        team_id: team,
        role: person.role === 'goalkeeper' ? 'goalkeeper' : 'player',
        held: person.held === true,
      });
      if (players.length >= 32) break;
    }
  }
  let ball = null;
  if (calibrated && source.ball && typeof source.ball === 'object') {
    const x = finite(source.ball.map_x);
    const y = finite(source.ball.map_y);
    if (inCalibratedThird(x, y, goalEnd, length, width, third)) ball = { map_x: x, map_y: y };
  }
  return {
    calibrated,
    goal_end: calibrated ? goalEnd : null,
    visible_third_cm: third,
    field_length_cm: length,
    field_width_cm: width,
    players,
    ball,
  };
}

export function readPitchRadar(analysis) {
  const live = analysis?.progress_stats?.pitch_radar;
  const saved = analysis?.result_json?.pitch_radar;
  if (live && typeof live === 'object') return normalizePitchRadar(live);
  if (saved && typeof saved === 'object') return normalizePitchRadar(saved);
  return emptyPitchRadar();
}

function fieldPixel(x, y, width, height, length, pitchWidth) {
  return [(x / length) * width, (y / pitchWidth) * height];
}

function strokeTemplate(ctx, width, height) {
  const length = FIELD_LENGTH_CM;
  const pitchWidth = FIELD_WIDTH_CM;
  const sx = width / length;
  const sy = height / pitchWidth;
  const midY = pitchWidth / 2;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(1, 1, width - 2, height - 2);
  ctx.beginPath();
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(width / 2, height / 2, 915 * sx, 915 * sy, 0, 0, Math.PI * 2);
  ctx.stroke();
  for (const [boxLength, boxWidth] of [
    [2015, 4100],
    [550, 1832],
  ]) {
    const top = (midY - boxWidth / 2) * sy;
    const boxHeight = boxWidth * sy;
    ctx.strokeRect(0, top, boxLength * sx, boxHeight);
    ctx.strokeRect(width - boxLength * sx, top, boxLength * sx, boxHeight);
  }
}

export function drawPitchRadar(ctx, width, height, radar) {
  if (!ctx) return 0;
  const map = normalizePitchRadar(radar);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = GRASS;
  ctx.fillRect(0, 0, width, height);
  const third = (width * map.visible_third_cm) / map.field_length_cm;
  ctx.fillStyle = GRASS_DIM;
  if (map.goal_end === 'pitch') {
    // The shot is already locked. The whole template stays bright.
  } else if (map.goal_end === 'low_x') ctx.fillRect(third, 0, width - third, height);
  else if (map.goal_end === 'high_x') ctx.fillRect(0, 0, width - third, height);
  else if (map.goal_end === 'centre') {
    const band = (width * map.visible_third_cm) / map.field_length_cm;
    const middle = width / 2;
    ctx.fillRect(0, 0, Math.max(0, middle - band), height);
    ctx.fillRect(Math.min(width, middle + band), 0, Math.max(0, width - (middle + band)), height);
  } else ctx.fillRect(0, 0, width, height);
  strokeTemplate(ctx, width, height);
  if (map.goal_end === 'low_x' || map.goal_end === 'high_x' || map.goal_end === 'centre') {
    const band = map.goal_end === 'centre' ? third : third;
    const x = map.goal_end === 'high_x' ? width - band : map.goal_end === 'centre' ? width / 2 - band : 0;
    const span = map.goal_end === 'centre' ? band * 2 : band;
    ctx.save();
    ctx.strokeStyle = '#ffdc00';
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 1, 1, Math.max(0, span - 2), height - 2);
    ctx.restore();
  }
  const radius = Math.max(5, width / 70);
  let dots = 0;
  map.players.forEach((person) => {
    const [x, y] = fieldPixel(person.map_x, person.map_y, width, height, map.field_length_cm, map.field_width_cm);
    ctx.beginPath();
    ctx.arc(x, y, person.held ? radius + 1.5 : radius, 0, Math.PI * 2);
    if (person.held) {
      ctx.strokeStyle = TEAM_COLOR[person.team_id] || '#b4b4b4';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      ctx.fillStyle = TEAM_COLOR[person.team_id] || '#b4b4b4';
      ctx.fill();
      ctx.lineWidth = person.role === 'goalkeeper' ? 2.5 : 1.5;
      ctx.strokeStyle = person.role === 'goalkeeper' ? '#ffc107' : '#ffffff';
      ctx.stroke();
    }
    dots += 1;
  });
  if (map.ball) {
    const [x, y] = fieldPixel(map.ball.map_x, map.ball.map_y, width, height, map.field_length_cm, map.field_width_cm);
    ctx.beginPath();
    ctx.fillStyle = '#ffffff';
    ctx.arc(x, y, Math.max(4, radius - 1), 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ff8c00';
    ctx.stroke();
    dots += 1;
  }
  return dots;
}
