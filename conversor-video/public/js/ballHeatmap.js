/** Read and draw the observed-ball grid the pipeline already accumulated. */

const DEFAULT_COLUMNS = 12;
const DEFAULT_ROWS = 8;
const DEFAULT_LENGTH_CM = 12000;
const DEFAULT_WIDTH_CM = 7000;
const PENALTY_BOX_LENGTH_CM = 2015;
const PENALTY_BOX_WIDTH_CM = 4100;
const GOAL_BOX_LENGTH_CM = 550;
const GOAL_BOX_WIDTH_CM = 1832;
const CENTRE_CIRCLE_RADIUS_CM = 915;

const LOW_COLOR = [255, 235, 59];
const MID_COLOR = [255, 152, 0];
const HIGH_COLOR = [229, 57, 53];

function count(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.round(number) : 0;
}

function positive(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

export function emptyBallHeatmap() {
  return {
    columns: DEFAULT_COLUMNS,
    rows: DEFAULT_ROWS,
    field_length_cm: DEFAULT_LENGTH_CM,
    field_width_cm: DEFAULT_WIDTH_CM,
    cells: Array.from({ length: DEFAULT_ROWS }, () => Array(DEFAULT_COLUMNS).fill(0)),
    observed_points: 0,
    max_cell_count: 0,
    frames_total: 0,
    frames_observed_ball: 0,
    frames_without_field_point: 0,
    frames_outside_field: 0,
  };
}

function normalize(source) {
  const columns = count(source.columns);
  const rows = count(source.rows);
  if (!columns || !rows || !Array.isArray(source.cells) || source.cells.length !== rows) return null;
  const cells = [];
  let max = 0;
  let total = 0;
  for (const row of source.cells) {
    if (!Array.isArray(row) || row.length !== columns) return null;
    const clean = row.map(count);
    for (const value of clean) {
      total += value;
      if (value > max) max = value;
    }
    cells.push(clean);
  }
  return {
    columns,
    rows,
    field_length_cm: positive(source.field_length_cm, DEFAULT_LENGTH_CM),
    field_width_cm: positive(source.field_width_cm, DEFAULT_WIDTH_CM),
    cells,
    observed_points: total,
    max_cell_count: max,
    frames_total: count(source.frames_total),
    frames_observed_ball: count(source.frames_observed_ball),
    frames_without_field_point: count(source.frames_without_field_point),
    frames_outside_field: count(source.frames_outside_field),
  };
}

export function readBallHeatmap(analysis) {
  const live = analysis?.progress_stats?.ball_heatmap;
  const saved = analysis?.result_json?.ball_heatmap;
  for (const source of [live, saved]) {
    if (source && typeof source === 'object') {
      const heatmap = normalize(source);
      if (heatmap) return heatmap;
    }
  }
  return emptyBallHeatmap();
}

/** Square root keeps a lightly used zone visible next to the busiest one. */
export function cellIntensity(value, max) {
  const cell = count(value);
  const top = count(max);
  if (!cell || !top) return 0;
  return Math.sqrt(Math.min(1, cell / top));
}

function mix(a, b, t) {
  return a.map((channel, index) => Math.round(channel + (b[index] - channel) * t));
}

export function heatColor(intensity) {
  const t = Math.max(0, Math.min(1, Number(intensity) || 0));
  if (t === 0) return 'rgba(0, 0, 0, 0)';
  const rgb = t < 0.5 ? mix(LOW_COLOR, MID_COLOR, t / 0.5) : mix(MID_COLOR, HIGH_COLOR, (t - 0.5) / 0.5);
  const alpha = (0.25 + 0.6 * t).toFixed(2);
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

export function hottestCell(heatmap) {
  if (!heatmap?.observed_points) return null;
  let best = null;
  heatmap.cells.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      if (value > 0 && (!best || value > best.count)) {
        best = { row: rowIndex, column: columnIndex, count: value };
      }
    });
  });
  if (!best) return null;
  const cellLength = heatmap.field_length_cm / heatmap.columns / 100;
  const cellWidth = heatmap.field_width_cm / heatmap.rows / 100;
  return {
    ...best,
    share: (best.count / heatmap.observed_points) * 100,
    length_from_m: best.column * cellLength,
    length_to_m: (best.column + 1) * cellLength,
    width_from_m: best.row * cellWidth,
    width_to_m: (best.row + 1) * cellWidth,
  };
}

function meters(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1).replace('.', ',');
}

export function formatHottestZone(cell) {
  if (!cell) return '—';
  return `${meters(cell.length_from_m)}–${meters(cell.length_to_m)} m`;
}

export function formatHottestDetail(cell) {
  if (!cell) return 'Aguardando bola observada';
  return `${cell.share.toFixed(1)}% das bolas · largura ${meters(cell.width_from_m)}–${meters(cell.width_to_m)} m`;
}

function drawPitchLines(ctx, width, height, heatmap) {
  const sx = width / heatmap.field_length_cm;
  const sy = height / heatmap.field_width_cm;
  const midY = heatmap.field_width_cm / 2;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, width - 2, height - 2);

  ctx.beginPath();
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.stroke();

  ctx.beginPath();
  ctx.ellipse(width / 2, height / 2, CENTRE_CIRCLE_RADIUS_CM * sx, CENTRE_CIRCLE_RADIUS_CM * sy, 0, 0, Math.PI * 2);
  ctx.stroke();

  for (const [boxLength, boxWidth] of [
    [PENALTY_BOX_LENGTH_CM, PENALTY_BOX_WIDTH_CM],
    [GOAL_BOX_LENGTH_CM, GOAL_BOX_WIDTH_CM],
  ]) {
    const top = (midY - boxWidth / 2) * sy;
    const boxHeight = boxWidth * sy;
    ctx.strokeRect(0, top, boxLength * sx, boxHeight);
    ctx.strokeRect(width - boxLength * sx, top, boxLength * sx, boxHeight);
  }
}

export function drawBallHeatmap(ctx, width, height, heatmap) {
  if (!ctx) return 0;
  const map = heatmap || emptyBallHeatmap();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#1f5f2c';
  ctx.fillRect(0, 0, width, height);

  const stripe = width / 12;
  ctx.fillStyle = 'rgba(255, 255, 255, 0.035)';
  for (let index = 0; index < 12; index += 2) {
    ctx.fillRect(index * stripe, 0, stripe, height);
  }

  const cellWidth = width / map.columns;
  const cellHeight = height / map.rows;
  let painted = 0;
  map.cells.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      const intensity = cellIntensity(value, map.max_cell_count);
      if (!intensity) return;
      ctx.fillStyle = heatColor(intensity);
      ctx.fillRect(columnIndex * cellWidth, rowIndex * cellHeight, cellWidth, cellHeight);
      painted += 1;
    });
  });

  drawPitchLines(ctx, width, height, map);
  return painted;
}

export function drawVisibleThird(ctx, width, height, goalEnd) {
  if (!ctx || (goalEnd !== 'high_x' && goalEnd !== 'low_x')) return;
  const third = width / 3;
  const x = goalEnd === 'high_x' ? width - third : 0;
  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
  ctx.setLineDash([7, 5]);
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 2, 2, third - 4, height - 4);
  ctx.restore();
}

export function drawPlayerHeat(ctx, width, height, heatmap) {
  if (!ctx || !heatmap?.observed_points) return 0;
  const cellWidth = width / heatmap.columns;
  const cellHeight = height / heatmap.rows;
  let painted = 0;
  heatmap.cells.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      const intensity = cellIntensity(value, heatmap.max_cell_count);
      if (!intensity) return;
      ctx.strokeStyle = `rgba(120, 190, 255, ${0.35 + 0.55 * intensity})`;
      ctx.lineWidth = 2;
      ctx.strokeRect(
        columnIndex * cellWidth + 2,
        rowIndex * cellHeight + 2,
        cellWidth - 4,
        cellHeight - 4
      );
      painted += 1;
    });
  });
  return painted;
}
