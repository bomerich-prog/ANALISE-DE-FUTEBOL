"""Ball heatmap on the 2D radar pitch, one metre per cell.

Each frame with an observed ball adds the same radar point that draws the ball
on the 2D map (105 x 68 m, the camera fixed on the halfway line). Predicted and
remembered positions never count. When the ball is lost for a short while and
seen again, the frames in between get estimated balls on the straight line
between the two sightings, evenly spaced, as if the ball rolled at a constant
speed. Both ends are in metres on the pitch, so a camera pan in between does
not bend the line. The estimate is rebuilt from the sightings each time the
report is read, so it is exact at the end of the analysis.
"""

from __future__ import annotations

import math

from pitch_radar import RADAR_LENGTH_CM, RADAR_WIDTH_CM
from roboflow_soccer_config import SoccerPitchConfiguration

BALL_HEATMAP_CELL_CM = 100.0
BALL_HEATMAP_COLUMNS = int(round(RADAR_LENGTH_CM / BALL_HEATMAP_CELL_CM))
BALL_HEATMAP_ROWS = int(round(RADAR_WIDTH_CM / BALL_HEATMAP_CELL_CM))
# Map jitter puts a ball on the touchline a little outside.
BALL_HEATMAP_EDGE_TOLERANCE_CM = 100.0
# A gap up to this many frames is filled with estimated balls. A longer one
# may hide a dribble or a ball that went and came back.
BALL_HEATMAP_MAX_GAP_FRAMES = 45
# The straight line is only drawn if the ball could have covered it:
# 120 cm per frame is 36 m/s at 30 fps, faster than a hard shot.
BALL_HEATMAP_MAX_STEP_CM = 120.0
_NOT_OBSERVED_SOURCES = {"predicted", "remembered", "lost"}
FIELD_GRID_COLUMNS = 12
FIELD_GRID_ROWS = 8


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


class BallHeatmapAccumulator:
    def __init__(
        self,
        cell_cm=BALL_HEATMAP_CELL_CM,
        field_length_cm=RADAR_LENGTH_CM,
        field_width_cm=RADAR_WIDTH_CM,
        edge_tolerance_cm=BALL_HEATMAP_EDGE_TOLERANCE_CM,
        max_gap_frames=BALL_HEATMAP_MAX_GAP_FRAMES,
        max_step_cm=BALL_HEATMAP_MAX_STEP_CM,
    ):
        self.field_length_cm = float(field_length_cm)
        self.field_width_cm = float(field_width_cm)
        self.columns = int(round(self.field_length_cm / float(cell_cm)))
        self.rows = int(round(self.field_width_cm / float(cell_cm)))
        self.edge_tolerance_cm = float(edge_tolerance_cm)
        self.max_gap_frames = int(max_gap_frames)
        self.max_step_cm = float(max_step_cm)
        self.sightings = []
        self.frames_total = 0
        self.frames_observed_ball = 0
        self.frames_without_field_point = 0
        self.frames_outside_field = 0

    def observe(self, frame_number, ball_detections, radar_point):
        """One frame. radar_point is the ball on the 2D map in radar centimetres, or None."""
        self.frames_total += 1
        ball = ball_detections[0] if ball_detections else None
        if not ball or ball.get("source") in _NOT_OBSERVED_SOURCES:
            return
        self.frames_observed_ball += 1
        x = y = None
        if radar_point is not None:
            x = _finite(radar_point[0])
            y = _finite(radar_point[1])
        if x is None or y is None:
            self.frames_without_field_point += 1
            return
        point = self._on_pitch(x, y)
        if point is None:
            self.frames_outside_field += 1
            return
        self.sightings.append((int(frame_number), point[0], point[1]))

    def _on_pitch(self, x, y):
        tolerance = self.edge_tolerance_cm
        if not (-tolerance <= x <= self.field_length_cm + tolerance):
            return None
        if not (-tolerance <= y <= self.field_width_cm + tolerance):
            return None
        return min(max(x, 0.0), self.field_length_cm), min(max(y, 0.0), self.field_width_cm)

    def estimated_points(self):
        """Balls on the straight line between two sightings with a short gap between them."""
        points = []
        for (frame_a, xa, ya), (frame_b, xb, yb) in zip(self.sightings, self.sightings[1:]):
            gap = frame_b - frame_a
            if gap < 2 or gap > self.max_gap_frames:
                continue
            if math.hypot(xb - xa, yb - ya) > self.max_step_cm * gap:
                continue
            for step in range(1, gap):
                t = step / float(gap)
                points.append((frame_a + step, xa + (xb - xa) * t, ya + (yb - ya) * t))
        return points

    def _cell(self, x, y):
        column = min(self.columns - 1, int(x / self.field_length_cm * self.columns))
        row = min(self.rows - 1, int(y / self.field_width_cm * self.rows))
        return row, column

    def as_dict(self):
        cells = [[0] * self.columns for _ in range(self.rows)]
        estimated = self.estimated_points()
        for _frame, x, y in list(self.sightings) + estimated:
            row, column = self._cell(x, y)
            cells[row][column] += 1
        total = len(self.sightings) + len(estimated)
        return {
            "coordinates": "radar",
            "columns": self.columns,
            "rows": self.rows,
            "field_length_cm": self.field_length_cm,
            "field_width_cm": self.field_width_cm,
            "cells": cells,
            "observed_points": total,
            "seen_points": len(self.sightings),
            "estimated_points": len(estimated),
            "max_cell_count": max((max(row) for row in cells), default=0),
            "max_gap_frames": self.max_gap_frames,
            "frames_total": self.frames_total,
            "frames_observed_ball": self.frames_observed_ball,
            "frames_without_field_point": self.frames_without_field_point,
            "frames_outside_field": self.frames_outside_field,
        }


def _heatmap_point(detection):
    """Map coordinates win. Event field coordinates are only a fallback."""
    map_x = _finite(detection.get("map_x"))
    map_y = _finite(detection.get("map_y"))
    if map_x is not None and map_y is not None:
        return map_x, map_y
    return _finite(detection.get("field_x")), _finite(detection.get("field_y"))


class FieldGridHeatmap:
    """Counts on a fixed 12 x 8 grid of the 120 x 70 m field. The player heat of the visible third uses it."""

    def __init__(
        self,
        columns=FIELD_GRID_COLUMNS,
        rows=FIELD_GRID_ROWS,
        pitch=None,
        edge_tolerance_cm=BALL_HEATMAP_EDGE_TOLERANCE_CM,
    ):
        pitch = pitch or SoccerPitchConfiguration()
        self.columns = int(columns)
        self.rows = int(rows)
        self.field_length_cm = float(pitch.length)
        self.field_width_cm = float(pitch.width)
        self.edge_tolerance_cm = float(edge_tolerance_cm)
        self.cells = [[0] * self.columns for _ in range(self.rows)]
        self.frames_total = 0
        self.frames_observed_ball = 0
        self.frames_without_field_point = 0
        self.frames_outside_field = 0
        self.observed_points = 0
        self.max_cell_count = 0

    def count_point(self, field_x, field_y):
        cell = self._cell_for(field_x, field_y)
        if cell is None:
            return False
        row, column = cell
        self.cells[row][column] += 1
        self.observed_points += 1
        if self.cells[row][column] > self.max_cell_count:
            self.max_cell_count = self.cells[row][column]
        return True

    def observe(self, ball_detections):
        self.frames_total += 1
        ball = ball_detections[0] if ball_detections else None
        if not ball or ball.get("source") in _NOT_OBSERVED_SOURCES:
            return
        self.frames_observed_ball += 1
        field_x, field_y = _heatmap_point(ball)
        if field_x is None or field_y is None:
            self.frames_without_field_point += 1
            return
        if not self.count_point(field_x, field_y):
            self.frames_outside_field += 1

    def _cell_for(self, field_x, field_y):
        tolerance = self.edge_tolerance_cm
        if not (-tolerance <= field_x <= self.field_length_cm + tolerance):
            return None
        if not (-tolerance <= field_y <= self.field_width_cm + tolerance):
            return None
        x = min(max(field_x, 0.0), self.field_length_cm)
        y = min(max(field_y, 0.0), self.field_width_cm)
        column = min(self.columns - 1, int(x / self.field_length_cm * self.columns))
        row = min(self.rows - 1, int(y / self.field_width_cm * self.rows))
        return row, column

    def as_dict(self):
        return {
            "columns": self.columns,
            "rows": self.rows,
            "field_length_cm": self.field_length_cm,
            "field_width_cm": self.field_width_cm,
            "cells": [list(row) for row in self.cells],
            "observed_points": self.observed_points,
            "max_cell_count": self.max_cell_count,
            "frames_total": self.frames_total,
            "frames_observed_ball": self.frames_observed_ball,
            "frames_without_field_point": self.frames_without_field_point,
            "frames_outside_field": self.frames_outside_field,
        }
