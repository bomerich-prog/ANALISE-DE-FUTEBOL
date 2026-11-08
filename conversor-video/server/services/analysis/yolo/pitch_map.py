"""Project the visible penalty third, and nothing past it.

The broadcast camera in this clip shows one penalty area. The fit uses the
painted 16.5 m line and the 9.15 m arc, which fixes the scale at the penalty
spot 5.5 m behind that line. Coordinates are published only inside the 35 m
third of the goal that calibration is looking at. The other half of the pitch
stays empty. Possession and passes never read this matrix.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from pitch_homography import (
    _painted_arc_distance,
    _clean_paint,
    fit_broadcast_penalty,
    line_support,
)
from roboflow_soccer_config import SoccerPitchConfiguration
from roboflow_view import ViewTransformer

VISIBLE_THIRD_CM = 3500.0
EDGE_TOLERANCE_CM = 100.0
HOLD_FRAMES = 15
SCALE_JUMP_RATIO = 1.35
SAME_SHOT_SCALE_RATIO = 1.08
MIN_LINE_SUPPORT = 0.40


def scale_is_stable(previous, new, ratio=SCALE_JUMP_RATIO):
    """True when this frame's centimetres-per-pixel stays near the last one."""
    if new is None or new <= 0:
        return False
    if previous is None:
        return True
    if previous <= 0:
        return False
    return (previous / ratio) <= new <= (previous * ratio)


def visible_bounds(positive_x, pitch=None):
    """Field rectangle, in centimetres, of the goal third the camera can see."""
    pitch = pitch or SoccerPitchConfiguration()
    length = float(pitch.length)
    width = float(pitch.width)
    third = min(VISIBLE_THIRD_CM, length)
    if positive_x:
        return length - third, length, 0.0, width
    return 0.0, third, 0.0, width


def _penalty_polylines(positive_x, pitch):
    """Box samples, the two 16.5 m sides, and the arc outside the box."""
    length = float(pitch.length)
    width = float(pitch.width)
    goal = length if positive_x else 0.0
    front = length - 1650.0 if positive_x else 1650.0
    y0 = (width - 4032.0) / 2.0
    y1 = (width + 4032.0) / 2.0
    corners = [(goal, y0), (front, y0), (front, y1), (goal, y1), (goal, y0)]
    box = []
    for (x1, y1_), (x2, y2_) in zip(corners, corners[1:]):
        for step in np.linspace(0, 1, 16, endpoint=False):
            box.append((x1 + (x2 - x1) * step, y1_ + (y2_ - y1_) * step))
    sides = []
    for side_y in (y0, y1):
        for step in np.linspace(0.08, 0.92, 12):
            sides.append((goal + (front - goal) * step, side_y))
    spot = length - 1100.0 if positive_x else 1100.0
    sign = -1.0 if positive_x else 1.0
    arc = []
    for degrees in np.linspace(-50, 50, 18):
        angle = np.deg2rad(degrees)
        x = spot + sign * 915.0 * np.cos(angle)
        y = width / 2.0 + 915.0 * np.sin(angle)
        outside = x < front if positive_x else x > front
        if outside:
            arc.append((x, y))
    return box, sides, arc


def _project_field(matrix, points):
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        return None
    projected = cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), inverse).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None
    return projected


def box_matches_visible_paint(matrix, positive_x, grass_mask, white_mask=None, pitch=None):
    """The 16.5 m box has to stay on the grass, and one side has to meet the paint.

    A matrix can close on the penalty line and still draw the box through the
    stands. Without a grass picture there is nothing to reject, so unit tests
    that only draw the line and the arc stay accepted.
    """
    if grass_mask is None or matrix is None or positive_x is None:
        return True
    pitch = pitch or SoccerPitchConfiguration()
    box, sides, _arc = _penalty_polylines(bool(positive_x), pitch)
    projected_box = _project_field(matrix, box)
    projected_sides = _project_field(matrix, sides)
    if projected_box is None or projected_sides is None:
        return False
    height, width = grass_mask.shape[:2]
    inside = 0
    crowd = 0
    on_grass = 0
    for x, y in projected_box:
        if not (0 <= x < width and 0 <= y < height):
            continue
        inside += 1
        if y < height * 0.18:
            crowd += 1
        if grass_mask[int(y), int(x)] > 0:
            on_grass += 1
    if inside < 24:
        return False
    if crowd / inside > 0.04:
        return False
    if on_grass / inside < 0.90:
        return False
    if white_mask is None:
        return True
    distance = cv2.distanceTransform(cv2.bitwise_not(white_mask), cv2.DIST_L2, 3)
    side_hits = [0, 0]
    side_seen = [0, 0]
    for index, (x, y) in enumerate(projected_sides):
        if not (0 <= x < width and 0 <= y < height):
            continue
        if grass_mask[int(y), int(x)] == 0:
            continue
        slot = 0 if index < 12 else 1
        side_seen[slot] += 1
        if distance[int(y), int(x)] <= 6:
            side_hits[slot] += 1
    visible = [side_hits[slot] / side_seen[slot] for slot in (0, 1) if side_seen[slot] >= 6]
    return bool(visible) and max(visible) >= 0.35


def inside_visible(x, y, positive_x, pitch=None, tolerance=EDGE_TOLERANCE_CM):
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    x0, x1, y0, y1 = visible_bounds(positive_x, pitch)
    return (x0 - tolerance) <= float(x) <= (x1 + tolerance) and (y0 - tolerance) <= float(y) <= (y1 + tolerance)


def draw_visible_penalty(frame, matrix, positive_x, color=(0, 220, 255)):
    """Draw the 16.5 m box and the arc. The rest of the template stays off."""
    if frame is None or matrix is None:
        return
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        return
    pitch = SoccerPitchConfiguration()
    length = float(pitch.length)
    goal = length if positive_x else 0.0
    front = length - 1650.0 if positive_x else 1650.0
    y0 = (float(pitch.width) - 4032.0) / 2.0
    y1 = (float(pitch.width) + 4032.0) / 2.0
    box = np.array([[goal, y0], [front, y0], [front, y1], [goal, y1]], dtype=np.float32)
    spot = length - 1100.0 if positive_x else 1100.0
    arc = []
    sign = -1.0 if positive_x else 1.0
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 24):
        x = spot + sign * 915.0 * np.cos(angle)
        y = float(pitch.width) / 2.0 + 915.0 * np.sin(angle)
        outside = x < front if positive_x else x > front
        if outside:
            arc.append((x, y))
    _polyline(frame, inverse, box, color, closed=True)
    if len(arc) >= 2:
        _polyline(frame, inverse, np.array(arc, dtype=np.float32), color, closed=False)


def _polyline(frame, inverse, field_points, color, closed):
    projected = cv2.perspectiveTransform(field_points.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    height, width = frame.shape[:2]
    if not np.isfinite(projected).all():
        return
    if np.max(np.abs(projected)) > max(width, height) * 4:
        return
    cv2.polylines(frame, [np.round(projected).astype(np.int32)], closed, color, 2, cv2.LINE_AA)


class VisiblePitchMap:
    """One penalty homography per shot, dropped as soon as the paint disagrees."""

    def __init__(self, pitch=None, hold_frames=HOLD_FRAMES):
        self.pitch = pitch or SoccerPitchConfiguration()
        self.hold_frames = int(hold_frames)
        self.matrix = None
        self.transformer = None
        self.positive_x = None
        self.scale = None
        self.hold_left = 0
        self.generation = 0
        self.accepted_frames = 0
        self.held_frames = 0
        self.rejected_frames = 0
        self.reasons = Counter()

    def update(self, white_line_mask, frame=None, grass_mask=None):
        report = fit_broadcast_penalty(white_line_mask, self.pitch, frame)
        if report is not None and not scale_is_stable(self.scale, report["cm_per_px"]):
            self.reasons["scale_jump"] += 1
            report = None
        if report is not None and not box_matches_visible_paint(
            report["matrix"], report["positive_x"], grass_mask, white_line_mask, self.pitch
        ):
            self.reasons["off_pitch"] += 1
            report = None
        if report is not None:
            return self._accept(report)
        if self._still_on_paint(white_line_mask, grass_mask):
            self.hold_left -= 1
            self.held_frames += 1
            self.reasons["held"] += 1
            return self._decision(True, True, "held")
        self._drop()
        self.rejected_frames += 1
        self.reasons["no_visible_penalty"] += 1
        return self._decision(False, False, "no_visible_penalty")

    def project(self, points):
        """Field centimetres inside the visible third, or None past it."""
        if points is None or self.transformer is None or self.positive_x is None:
            return []
        array = np.asarray(points, dtype=np.float32)
        if array.size == 0:
            return []
        transformed = self.transformer.transform_points(array.reshape(-1, 2))
        mapped = []
        for x, y in transformed:
            if inside_visible(float(x), float(y), self.positive_x, self.pitch):
                mapped.append((float(x), float(y)))
            else:
                mapped.append(None)
        return mapped

    def metrics(self):
        goal_end = None
        if self.positive_x is True:
            goal_end = "high_x"
        elif self.positive_x is False:
            goal_end = "low_x"
        return {
            "method": "penalty line and arc, 16.5 m deep, published only inside 35 m of that goal",
            "feeds_events": False,
            "frames_with_map": self.accepted_frames + self.held_frames,
            "frames_without_map": self.rejected_frames,
            "accepted_frames": self.accepted_frames,
            "held_frames": self.held_frames,
            "visible_third_cm": VISIBLE_THIRD_CM,
            "penalty_depth_cm": 1650.0,
            "penalty_width_cm": 4032.0,
            "goal_end": goal_end,
            "cm_per_px": None if self.scale is None else round(float(self.scale), 4),
            "reasons": dict(self.reasons),
        }

    def _accept(self, report):
        scale = float(report["cm_per_px"])
        same_shot = self.matrix is not None and scale_is_stable(self.scale, scale, SAME_SHOT_SCALE_RATIO)
        if not same_shot:
            self.generation += 1
        self.matrix = report["matrix"]
        self.transformer = ViewTransformer.from_matrix(self.matrix)
        self.positive_x = bool(report["positive_x"])
        self.scale = scale
        self.hold_left = self.hold_frames
        self.accepted_frames += 1
        self.reasons["accepted"] += 1
        return self._decision(True, False, "accepted")

    def _still_on_paint(self, white_line_mask, grass_mask=None):
        if self.matrix is None or self.hold_left <= 0 or self.positive_x is None:
            return False
        if not box_matches_visible_paint(self.matrix, self.positive_x, grass_mask, white_line_mask, self.pitch):
            return False
        support = line_support(self.matrix, white_line_mask, self.pitch)
        if support is not None and support >= MIN_LINE_SUPPORT:
            return True
        paint = _clean_paint(white_line_mask)
        if paint is None:
            return False
        distance = _painted_arc_distance(self.matrix, paint, self.positive_x)
        return distance is not None and distance <= 8.0

    def _drop(self):
        self.matrix = None
        self.transformer = None
        self.positive_x = None
        self.scale = None
        self.hold_left = 0

    def _decision(self, accepted, held, reason):
        return {
            "accepted": accepted,
            "held": held,
            "reason": reason,
            "matrix": self.matrix,
            "positive_x": self.positive_x,
            "generation": self.generation if accepted else None,
            "goal_end": "high_x" if self.positive_x else "low_x" if self.positive_x is False else None,
        }
