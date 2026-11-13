/**
 * The 2D panel draws only points inside the calibrated third.
 */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { visibleThirdRect } from '../public/js/ballHeatmap.js';
import {
  drawPitchRadar,
  inCalibratedThird,
  normalizePitchRadar,
  readPitchRadar,
} from '../public/js/pitchRadar.js';

const require = createRequire(import.meta.url);
const { pitchRadarFromPayload } = require('../server/services/analysis/yolo/pitchRadarProgress.js');

const sample = {
  calibrated: true,
  goal_end: 'low_x',
  visible_third_cm: 3500,
  field_length_cm: 12000,
  field_width_cm: 7000,
  players: [
    { map_x: 1000, map_y: 3500, team_id: 0, role: 'player' },
    { map_x: 6000, map_y: 3500, team_id: 1, role: 'player' },
    { map_x: 400, map_y: 3400, team_id: 0, role: 'goalkeeper' },
  ],
  ball: { map_x: 1500, map_y: 3000 },
};

const forwarded = pitchRadarFromPayload({ pitch_radar: sample });
assert.equal(forwarded.players.length, 2, 'the bridge drops the midfield player');
assert.equal(forwarded.players[1].role, 'goalkeeper', 'the keeper stays');
assert.equal(forwarded.ball.map_x, 1500, 'the ball in the third stays');
assert.equal(pitchRadarFromPayload({}), null, 'a payload without a radar is left alone');

const cleared = pitchRadarFromPayload({
  pitch_radar: { calibrated: false, goal_end: null, players: sample.players, ball: sample.ball },
});
assert.equal(cleared.calibrated, false, 'a refused frame stays on the channel');
assert.equal(cleared.players.length, 0, 'a refused frame keeps no dots');
assert.equal(cleared.ball, null, 'a refused frame keeps no ball');

const live = readPitchRadar({
  progress_stats: { pitch_radar: sample },
  result_json: { pitch_radar: { ...sample, goal_end: 'high_x', players: [] } },
});
assert.equal(live.players.length, 2, 'the live sample wins');
assert.equal(readPitchRadar({}).calibrated, false, 'the panel starts uncalibrated');
assert.equal(inCalibratedThird(11000, 3500, 'high_x'), true, 'the far box is inside its third');
assert.equal(inCalibratedThird(4000, 3500, 'low_x'), false, '40 m is already outside the near third');

const normalized = normalizePitchRadar(sample);
assert.equal(normalized.players.length, 2, 'the panel drops the same midfield player');

const calls = [];
const fakeContext = new Proxy(
  {},
  {
    get(target, property) {
      if (property in target) return target[property];
      if (property === 'canvas') return {};
      return (...args) => {
        calls.push([property, ...args]);
        return fakeContext;
      };
    },
    set(target, property, value) {
      target[property] = value;
      return true;
    },
  }
);
const dots = drawPitchRadar(fakeContext, 720, 420, sample);
assert.equal(dots, 3, 'two players and the ball are drawn');
assert.ok(calls.some(([name]) => name === 'arc'), 'dots are circles');
assert.equal(drawPitchRadar(fakeContext, 720, 420, cleared), 0, 'a refused frame draws the field without dots');

const near = visibleThirdRect(1200, 'low_x');
const far = visibleThirdRect(1200, 'high_x');
assert.equal(near.x, 0, 'the near third starts at the goal line');
assert.equal(near.width, 350, '35 m of a 120 m pitch is 350 px in a 1200 px canvas');
assert.equal(far.x, 850, 'the far third starts at 85 m');
assert.equal(visibleThirdRect(1200, null), null, 'no goal means no highlight');

console.log('pitch radar panel tests passed');
