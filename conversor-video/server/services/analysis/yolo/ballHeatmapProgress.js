/** Copy the observed-ball grid the worker already accumulated. */

function count(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.round(number) : 0;
}

function ballHeatmapFromPayload(payload) {
  const heatmap = payload && payload.ball_heatmap;
  if (!heatmap || typeof heatmap !== 'object') return null;
  const columns = count(heatmap.columns);
  const rows = count(heatmap.rows);
  if (!columns || !rows || !Array.isArray(heatmap.cells) || heatmap.cells.length !== rows) {
    return null;
  }
  const cells = [];
  for (const row of heatmap.cells) {
    if (!Array.isArray(row) || row.length !== columns) return null;
    cells.push(row.map(count));
  }
  const fieldLength = Number(heatmap.field_length_cm);
  const fieldWidth = Number(heatmap.field_width_cm);
  return {
    columns,
    rows,
    field_length_cm: Number.isFinite(fieldLength) && fieldLength > 0 ? fieldLength : 12000,
    field_width_cm: Number.isFinite(fieldWidth) && fieldWidth > 0 ? fieldWidth : 7000,
    cells,
    observed_points: count(heatmap.observed_points),
    max_cell_count: count(heatmap.max_cell_count),
    frames_total: count(heatmap.frames_total),
    frames_observed_ball: count(heatmap.frames_observed_ball),
    frames_without_field_point: count(heatmap.frames_without_field_point),
    frames_outside_field: count(heatmap.frames_outside_field),
  };
}

module.exports = { ballHeatmapFromPayload };
