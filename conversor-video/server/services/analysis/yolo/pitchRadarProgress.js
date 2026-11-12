/** Copy the visible-third radar. Points outside that third are dropped. */

const FIELD_LENGTH_CM = 12000;
const FIELD_WIDTH_CM = 7000;
const VISIBLE_THIRD_CM = 3500;
const EDGE_TOLERANCE_CM = 100;

function finite(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function positive(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function inThird(x, y, goalEnd, length, width, third) {
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

function pitchRadarFromPayload(payload) {
  const radar = payload && payload.pitch_radar;
  if (!radar || typeof radar !== 'object') return null;
  const goalEnd = radar.goal_end === 'high_x' || radar.goal_end === 'low_x' || radar.goal_end === 'centre' || radar.goal_end === 'pitch'
    ? radar.goal_end
    : null;
  const calibrated = Boolean(radar.calibrated) && goalEnd !== null;
  const length = positive(radar.field_length_cm, FIELD_LENGTH_CM);
  const width = positive(radar.field_width_cm, FIELD_WIDTH_CM);
  const third = positive(radar.visible_third_cm, VISIBLE_THIRD_CM);
  const players = [];
  if (calibrated && Array.isArray(radar.players)) {
    for (const person of radar.players) {
      if (!person || typeof person !== 'object') continue;
      const x = finite(person.map_x);
      const y = finite(person.map_y);
      if (!inThird(x, y, goalEnd, length, width, third)) continue;
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
  if (calibrated && radar.ball && typeof radar.ball === 'object') {
    const x = finite(radar.ball.map_x);
    const y = finite(radar.ball.map_y);
    if (inThird(x, y, goalEnd, length, width, third)) ball = { map_x: x, map_y: y };
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

module.exports = { pitchRadarFromPayload };
