"""Count observed ball positions on a fixed pitch grid.

Only the detection the frame actually saw is counted. Predicted and remembered
positions never reach this grid. Field coordinates come from the homography
that already ran for the frame, so this adds one bounds check and one sum.
"""

from __future__ import annotations

import math

from roboflow_soccer_config import SoccerPitchConfiguration

BALL_HEATMAP_COLUMNS = 12
BALL_HEATMAP_ROWS = 8
# Homography jitter puts a ball on the touchline a few centimetres outside.
BALL_HEATMAP_EDGE_TOLERANCE_CM = 100.0
_NOT_OBSERVED_SOURCES = {"predicted", "remembered", "lost"}


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
        columns=BALL_HEATMAP_COLUMNS,
        rows=BALL_HEATMAP_ROWS,
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

    def observe(self, ball_detections):
        self.frames_total += 1
        ball = ball_detections[0] if ball_detections else None
        if not ball or ball.get("source") in _NOT_OBSERVED_SOURCES:
            return
        self.frames_observed_ball += 1
        field_x = _finite(ball.get("field_x"))
        field_y = _finite(ball.get("field_y"))
        if field_x is None or field_y is None:
            self.frames_without_field_point += 1
            return
        cell = self._cell_for(field_x, field_y)
        if cell is None:
            self.frames_outside_field += 1
            return
        row, column = cell
        self.cells[row][column] += 1
        self.observed_points += 1
        if self.cells[row][column] > self.max_cell_count:
            self.max_cell_count = self.cells[row][column]

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
