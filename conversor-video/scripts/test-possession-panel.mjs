#!/usr/bin/env node
/**
 * The possession card reads the split the pipeline already accumulated.
 */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import {
  readPossessionByTeam,
  formatPossessionDetail,
  formatPossessionPercent,
  formatPossessionSeconds,
} from '../public/js/possessionCounts.js';

const require = createRequire(import.meta.url);
const { possessionProgressFromPayload } = require('../server/services/analysis/yolo/possessionProgress.js');

const saved = {
  frames_total: 20,
  frames_team_0: 10,
  frames_team_1: 5,
  frames_without_possessor: 5,
  seconds_team_0: 1,
  seconds_team_1: 0.5,
  seconds_without_possessor: 0.5,
  percent_team_0: 50,
  percent_team_1: 25,
  percent_without_possessor: 25,
};

const fromResult = readPossessionByTeam({ result_json: { possession_by_team: saved } });
assert.equal(fromResult.percent_team_0, 50);
assert.equal(fromResult.frames_team_1, 5);
assert.equal(fromResult.frames_without_possessor, 5);

const live = readPossessionByTeam({
  progress_stats: {
    possession_by_team: { ...saved, frames_team_0: 4, percent_team_0: 40 },
  },
  result_json: { possession_by_team: saved },
});
assert.equal(live.frames_team_0, 4, 'a live update replaces the saved split');
assert.equal(live.percent_team_0, 40);

const empty = readPossessionByTeam({});
assert.equal(empty.frames_total, 0);
assert.equal(empty.percent_without_possessor, 0);
assert.equal(formatPossessionPercent(empty.percent_team_0), '0.0%');
assert.equal(formatPossessionDetail(0, 0), '0 frames · 0.0 s');
assert.equal(formatPossessionSeconds(75), '1:15');

const forwarded = possessionProgressFromPayload({
  possession_by_team: { ...saved, raw_candidate_passes: 9 },
});
assert.deepEqual(forwarded, {
  frames_total: 20,
  frames_team_0: 10,
  frames_team_1: 5,
  frames_without_possessor: 5,
  seconds_team_0: 1,
  seconds_team_1: 0.5,
  seconds_without_possessor: 0.5,
  percent_team_0: 50,
  percent_team_1: 25,
  percent_without_possessor: 25,
});
assert.equal(possessionProgressFromPayload({ passes: saved }), null);

console.log('possession panel tests passed');
