#!/usr/bin/env node
/**
 * The players tab reads per-track totals and keeps a clicked row open.
 */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import {
  readPlayerStats,
  sortPlayerTracks,
  formatPlayerSummary,
  formatPlayerTeam,
  formatPlayerPossession,
  formatPlayerPercent,
  formatMeanPosition,
  toggleOpenTrack,
} from '../public/js/playerStats.js';

const require = createRequire(import.meta.url);
const { playerStatsFromPayload } = require('../server/services/analysis/yolo/playerStatsProgress.js');

const saved = {
  tracks: [
    {
      track_id: 15,
      team: 'Team 1',
      frames_seen: 20,
      frames_with_ball: 4,
      possession_spells: 1,
      seconds_with_ball: 0.4,
      passes_made: 0,
      passes_received: 1,
      percent_of_team: 40,
      position_samples: 20,
      mean_x_cm: 2000,
      mean_y_cm: 3500,
    },
    {
      track_id: 11,
      team: 'Team 1',
      frames_seen: 10,
      frames_with_ball: 6,
      possession_spells: 2,
      seconds_with_ball: 0.6,
      passes_made: 2,
      passes_received: 1,
      percent_of_team: 60,
      position_samples: 8,
      mean_x_cm: 4500,
      mean_y_cm: 2800,
    },
  ],
};

const fromResult = readPlayerStats({ result_json: { player_stats: saved } });
assert.equal(fromResult.tracks.length, 2);
assert.equal(fromResult.tracks[0].passes_made, 0);

const live = readPlayerStats({
  progress_stats: {
    player_stats: {
      tracks: [{ ...saved.tracks[1], passes_made: 3, seconds_with_ball: 0.8 }],
    },
  },
  result_json: { player_stats: saved },
});
assert.equal(live.tracks.length, 1, 'a live update replaces the saved table');
assert.equal(live.tracks[0].passes_made, 3);

const empty = readPlayerStats({});
assert.deepEqual(empty.tracks, []);
assert.equal(formatPlayerSummary(empty.tracks), 'Nenhuma trilha ainda.');

const ranked = sortPlayerTracks(saved.tracks);
assert.deepEqual(ranked.map((track) => track.track_id), [11, 15]);
assert.equal(
  formatPlayerSummary(ranked),
  '2 trilhas · 2 com posse · 1 com passe. Ordenado por passes feitos e tempo de posse.',
);
assert.equal(formatPlayerTeam(null), 'Sem time');
assert.equal(formatPlayerTeam('Team 0'), 'Team 0');
assert.equal(formatPlayerPossession(0.6), '0.6 s');
assert.equal(formatPlayerPercent(60), '60.0%');
assert.equal(formatPlayerPercent(null), '—');
assert.equal(formatMeanPosition(ranked[0]), '45.0 m × 28.0 m');
assert.equal(formatMeanPosition({ mean_x_cm: null, mean_y_cm: null }), 'Sem coordenada');

let open = toggleOpenTrack(new Set(), 11);
assert.equal(open.has(11), true);
open = toggleOpenTrack(open, 11);
assert.equal(open.has(11), false);

const forwarded = playerStatsFromPayload({
  player_stats: {
    tracks: [
      ...saved.tracks,
      { track_id: -1, team: 'Team 0', frames_seen: 4 },
      { track_id: 4, team: 'blue', frames_seen: 3, frames_with_ball: 1, percent_of_team: 100 },
    ],
  },
});
assert.equal(forwarded.tracks.length, 3);
assert.equal(forwarded.tracks.find((track) => track.track_id === 4).team, null);
assert.equal(forwarded.tracks.find((track) => track.track_id === 4).percent_of_team, null);
assert.equal(playerStatsFromPayload({ passes: saved }), null);

console.log('player panel tests passed');
