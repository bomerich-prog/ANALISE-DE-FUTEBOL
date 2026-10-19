#!/usr/bin/env node
/**
 * The heatmap panel draws the grid the worker sent. It does not locate the ball itself.
 */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import {
  readBallHeatmap,
  cellIntensity,
  heatColor,
  hottestCell,
  formatHottestZone,
  formatHottestDetail,
  drawBallHeatmap,
} from '../public/js/ballHeatmap.js';

const require = createRequire(import.meta.url);
const { ballHeatmapFromPayload } = require('../server/services/analysis/yolo/ballHeatmapProgress.js');

function grid(columns, rows, fill = 0) {
  return Array.from({ length: rows }, () => Array(columns).fill(fill));
}

function workerHeatmap(cells) {
  const flat = cells.flat();
  return {
    columns: cells[0].length,
    rows: cells.length,
    field_length_cm: 12000,
    field_width_cm: 7000,
    cells,
    observed_points: flat.reduce((sum, value) => sum + value, 0),
    max_cell_count: Math.max(...flat),
    frames_total: 40,
    frames_observed_ball: 30,
    frames_without_field_point: 3,
    frames_outside_field: 1,
  };
}

const cells = grid(12, 8);
cells[4][6] = 12;
cells[3][5] = 3;
const live = workerHeatmap(cells);

const forwarded = ballHeatmapFromPayload({ ball_heatmap: live });
assert.deepEqual(forwarded.cells, cells, 'the bridge keeps every cell');
assert.equal(forwarded.observed_points, 15, 'observed total survives');
assert.equal(forwarded.frames_without_field_point, 3, 'missing homography count survives');
assert.equal(ballHeatmapFromPayload({}), null, 'no heatmap is not an empty grid');
assert.equal(
  ballHeatmapFromPayload({ ball_heatmap: { ...live, cells: grid(12, 7) } }),
  null,
  'a grid with the wrong row count is dropped'
);
assert.equal(
  ballHeatmapFromPayload({ ball_heatmap: { ...live, cells: [[1, 2]] } }),
  null,
  'a grid with the wrong shape is dropped'
);

const empty = readBallHeatmap({});
assert.equal(empty.observed_points, 0, 'the panel starts empty');
assert.equal(empty.cells.length, 8, 'empty grid still has 8 rows');
assert.equal(hottestCell(empty), null, 'no zone without a ball');
assert.equal(formatHottestZone(null), '—', 'zone placeholder');

const savedCells = grid(12, 8);
savedCells[0][0] = 1;
const preferred = readBallHeatmap({
  progress_stats: { ball_heatmap: live },
  result_json: { ball_heatmap: workerHeatmap(savedCells) },
});
assert.equal(preferred.cells[4][6], 12, 'a live update wins over the stored result');

const finished = readBallHeatmap({ result_json: { ball_heatmap: live } });
assert.equal(finished.observed_points, 15, 'the stored result stays visible after the analysis');

const hottest = hottestCell(finished);
assert.equal(hottest.row, 4, 'busiest row');
assert.equal(hottest.column, 6, 'busiest column');
assert.equal(hottest.share, 80, '12 of 15 balls');
assert.equal(formatHottestZone(hottest), '60–70 m', 'busiest zone along the pitch');
assert.equal(formatHottestDetail(hottest), '80.0% das bolas · largura 35–43,8 m', 'zone detail');

assert.equal(cellIntensity(0, 12), 0, 'an empty cell is not painted');
assert.equal(cellIntensity(12, 12), 1, 'the busiest cell is full intensity');
assert.equal(cellIntensity(3, 12), 0.5, 'square root keeps lighter cells visible');
assert.equal(heatColor(0), 'rgba(0, 0, 0, 0)', 'zero is transparent');
assert.equal(heatColor(1), 'rgba(229, 57, 53, 0.85)', 'the busiest cell is red');

const calls = [];
const fakeContext = new Proxy(
  {},
  {
    get(target, property) {
      if (property in target) return target[property];
      return (...args) => calls.push([property, ...args]);
    },
    set(target, property, value) {
      target[property] = value;
      return true;
    },
  }
);
const painted = drawBallHeatmap(fakeContext, 720, 420, finished);
assert.equal(painted, 2, 'only the two used cells are painted');
assert.ok(calls.some(([name]) => name === 'strokeRect'), 'pitch lines are drawn');
assert.equal(drawBallHeatmap(fakeContext, 720, 420, empty), 0, 'nothing is painted before a ball');

console.log('ball heatmap panel tests passed');
