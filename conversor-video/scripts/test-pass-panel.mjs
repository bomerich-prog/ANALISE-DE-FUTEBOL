#!/usr/bin/env node
/**
 * The match panel reads pass events the backend already stored.
 * It does not score passes itself.
 */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { getEvents } from '../public/js/analysisViewer.js';
import { countPassesByTeam } from '../public/js/passCounts.js';

const require = createRequire(import.meta.url);
const { passProgressFromPayload } = require('../server/services/analysis/yolo/passProgress.js');

const screenRecordingPasses = {
  total_passes: 5,
  events: [
    { type: 'pass', from_track_id: 15, to_track_id: 11, team: 'Team 1', frame: 60 },
    { type: 'pass', from_track_id: 11, to_track_id: 41, team: 'Team 1', frame: 227 },
    { type: 'pass', from_track_id: 41, to_track_id: 47, team: 'Team 1', frame: 294 },
    { type: 'pass', from_track_id: 47, to_track_id: 61, team: 'Team 1', frame: 334 },
    { type: 'pass', from_track_id: 61, to_track_id: 74, team: 'Team 1', frame: 368 },
  ],
};

const counted = countPassesByTeam(screenRecordingPasses);
assert.equal(counted.team0, 0, 'Team 0 passes in the screen recording');
assert.equal(counted.team1, 5, 'Team 1 passes in the screen recording');
assert.equal(counted.team0 + counted.team1, screenRecordingPasses.total_passes, 'every event is on one team');

const live = getEvents({
  progress_stats: {
    passes: screenRecordingPasses,
    events: { current_possessor_track_id: 11, current_possession_team: 'Team 1' },
  },
});
assert.equal(live.blue_passes, 0, 'Team 0 counter while the analysis is running');
assert.equal(live.red_passes, 5, 'Team 1 counter while the analysis is running');
assert.equal(live.turnovers, 0, 'same-team passes do not add a turnover');
assert.equal(live.current_possession_team, 'Team 1', 'possession still comes from the progress payload');

const finished = getEvents({
  result_json: { passes: screenRecordingPasses, events: {} },
});
assert.equal(finished.blue_passes, 0, 'Team 0 counter after the analysis is stored');
assert.equal(finished.red_passes, 5, 'Team 1 counter after the analysis is stored');

const newerLive = getEvents({
  progress_stats: {
    passes: {
      total_passes: 1,
      events: [{ team: 'Team 0' }],
    },
  },
  result_json: { passes: screenRecordingPasses },
});
assert.equal(newerLive.blue_passes, 1, 'a live progress update replaces the saved total');
assert.equal(newerLive.red_passes, 0, 'the saved Team 1 total is not mixed in');

const empty = getEvents({});
assert.equal(empty.blue_passes, 0, 'missing passes stay at zero for Team 0');
assert.equal(empty.red_passes, 0, 'missing passes stay at zero for Team 1');
assert.equal(empty.turnovers, 0, 'the turnover card starts at zero');

const beforeTurnover = getEvents({
  progress_stats: {
    passes: { total_passes: 0, events: [], turnovers: 0 },
    events: { current_possession_team: 'Team 1' },
  },
});
assert.equal(beforeTurnover.turnovers, 0, 'no turnover while the same team still has the ball');

const otherTeamConfirmed = getEvents({
  progress_stats: {
    passes: { total_passes: 0, events: [], turnovers: 1 },
    events: { current_possession_team: 'Team 0' },
  },
});
assert.equal(otherTeamConfirmed.turnovers, 1, 'the count rises when the other team is confirmed');

const dispute = getEvents({
  progress_stats: {
    passes: { total_passes: 0, events: [], turnovers: 0, disputes: 1 },
    events: { current_possession_team: 'Team 0' },
  },
});
assert.equal(dispute.turnovers, 0, 'a dispute is not a turnover');

const finishedTurnovers = getEvents({
  result_json: {
    passes: { total_passes: 4, events: [], turnovers: 2 },
    events: {},
  },
});
assert.equal(finishedTurnovers.turnovers, 2, 'the stored total stays on the card');

const forwarded = passProgressFromPayload({
  passes: {
    total_passes: 2,
    events: [{ team: 'Team 0' }, { team: 'Team 1' }],
    raw_candidate_passes: 9,
  },
});
assert.deepEqual(forwarded, {
  total_passes: 2,
  events: [{ team: 'Team 0' }, { team: 'Team 1' }],
  turnovers: 0,
});

const forwardedTurnovers = passProgressFromPayload({
  passes: { total_passes: 0, events: [], turnovers: 3, disputes: 1 },
});
assert.equal(forwardedTurnovers.turnovers, 3, 'live progress keeps the turnover total');
assert.equal(forwardedTurnovers.disputes, undefined, 'a dispute is not forwarded as a turnover');
assert.equal(passProgressFromPayload({ events: {} }), null, 'possession events are not pass counts');

const split = countPassesByTeam(forwarded);
assert.equal(split.team0, 1);
assert.equal(split.team1, 1);

console.log('pass panel tests passed');
