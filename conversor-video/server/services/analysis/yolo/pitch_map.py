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
)
from roboflow_soccer_config import SoccerPitchConfiguration
from roboflow_view import ViewTransformer

VISIBLE_THIRD_CM = 3500.0
EDGE_TOLERANCE_CM = 100.0
HOLD_FRAMES = 15
SCALE_JUMP_RATIO = 1.35
SAME_SHOT_SCALE_RATIO = 1.08
MIN_FRONT_WHITE = 0.50
ARC_MAX_DISTANCE_PX = 8.0
ARC_P90_DISTANCE_PX = 8.0
PROBE_HALF_WIDTH_CM = 1200.0
PROBE_STEP_CM = 50.0
MAX_GOAL_DEPTH_CM = 1650.0
MAX_ARC_DEPTH_CM = 500.0
CROWD_BAND = 0.18


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


def _project_field(matrix, points):
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        return None
    projected = cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), inverse).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None
    return projected


def _front_x(positive_x, pitch):
    length = float(pitch.length)
    return length - 1650.0 if positive_x else 1650.0


def _goal_sign(positive_x):
    return 1.0 if positive_x else -1.0


def _project_one(matrix, point):
    projected = _project_field(matrix, [point])
    if projected is None:
        return float("nan"), float("nan")
    return float(projected[0][0]), float(projected[0][1])


def _arc_distances(matrix, paint, positive_x):
    """Median and 90th-percentile distance, in pixels, from the arc to the paint."""
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        return None, None
    length = 12000.0
    front = length - 1650.0 if positive_x else 1650.0
    spot = length - 1100.0 if positive_x else 1100.0
    samples = []
    for angle in np.linspace(0, 2 * np.pi, 48):
        x = spot + 915.0 * np.cos(angle)
        y = 3500.0 + 915.0 * np.sin(angle)
        outside = x < front if positive_x else x > front
        if outside:
            samples.append((x, y))
    projected = cv2.perspectiveTransform(np.asarray(samples, np.float32).reshape(-1, 1, 2), inverse).reshape(-1, 2)
    height, width = paint.shape[:2]
    distance = cv2.distanceTransform(cv2.bitwise_not(paint), cv2.DIST_L2, 3)
    inside = []
    for x, y in projected:
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height:
            inside.append((float(x), float(y), float(distance[int(y), int(x)])))
    if len(inside) < 10:
        return None, None
    xs = [point[0] for point in inside]
    ys = [point[1] for point in inside]
    if max(xs) - min(xs) < 70 and max(ys) - min(ys) < 70:
        return None, None
    values = [point[2] for point in inside]
    return float(np.median(values)), float(np.percentile(values, 90))


def _classify_pixel(x, y, grass_mask):
    height, width = grass_mask.shape[:2]
    if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height):
        return "off"
    if y < height * CROWD_BAND:
        return "crowd"
    if grass_mask[int(y), int(x)] > 0:
        return "grass"
    return "board"


def paint_patch(matrix, positive_x, grass_mask, white_mask=None, pitch=None):
    """Depth, in centimetres, that stays on the grass past the 16.5 m line.

    The goal line of the full box may sit in the stands. That does not reject
    the frame. None means the 16.5 m line or the arc missed the paint, so the
    frame publishes nothing. Without a grass picture the penalty depth is kept
    for tests that only draw the line and the arc.
    """
    if matrix is None or positive_x is None:
        return None
    pitch = pitch or SoccerPitchConfiguration()
    if grass_mask is None:
        return {
            "toward_goal_cm": MAX_GOAL_DEPTH_CM,
            "toward_arc_cm": MAX_ARC_DEPTH_CM,
            "checked": False,
        }
    front = _front_x(bool(positive_x), pitch)
    width = float(pitch.width)
    y0 = (width - 4032.0) / 2.0
    y1 = (width + 4032.0) / 2.0
    front_line = [(front, y0 + (y1 - y0) * step) for step in np.linspace(0, 1, 24)]
    projected = _project_field(matrix, front_line)
    if projected is None:
        return None
    inside = []
    for x, y in projected:
        kind = _classify_pixel(float(x), float(y), grass_mask)
        if kind == "off":
            continue
        inside.append((float(x), float(y), kind))
    if len(inside) < 8:
        return None
    if sum(kind == "crowd" for _x, _y, kind in inside) / len(inside) > 0.05:
        return None
    on_grass = [(x, y) for x, y, kind in inside if kind == "grass"]
    if len(on_grass) / len(inside) < 0.90:
        return None
    if white_mask is not None:
        distance = cv2.distanceTransform(cv2.bitwise_not(white_mask), cv2.DIST_L2, 3)
        white_hits = sum(distance[int(y), int(x)] <= 6 for x, y in on_grass)
        if white_hits / len(on_grass) < MIN_FRONT_WHITE:
            return None
        paint = _clean_paint(white_mask)
        if paint is None:
            return None
        arc_distance, arc_p90 = _arc_distances(matrix, paint, bool(positive_x))
        if arc_distance is None or arc_distance > ARC_MAX_DISTANCE_PX or arc_p90 > ARC_P90_DISTANCE_PX:
            return None
    sign = _goal_sign(bool(positive_x))
    centers = (
        width / 2.0 - PROBE_HALF_WIDTH_CM,
        width / 2.0,
        width / 2.0 + PROBE_HALF_WIDTH_CM,
    )

    def depth_holds(cm, direction):
        x = front + direction * cm
        kinds = [_classify_pixel(*_project_one(matrix, (x, center)), grass_mask) for center in centers]
        if "crowd" in kinds or "board" in kinds:
            return False
        return kinds[1] == "grass"

    def walk(direction, limit):
        reached = 0.0
        cm = 0.0
        while cm <= limit + 1e-6:
            if not depth_holds(cm, direction):
                break
            reached = cm
            cm += PROBE_STEP_CM
        return reached

    toward_goal = walk(sign, MAX_GOAL_DEPTH_CM)
    toward_arc = walk(-sign, MAX_ARC_DEPTH_CM)
    if toward_goal <= 0.0 and toward_arc <= 0.0:
        return None
    return {
        "toward_goal_cm": float(toward_goal),
        "toward_arc_cm": float(toward_arc),
        "checked": True,
    }


def box_matches_visible_paint(matrix, positive_x, grass_mask, white_mask=None, pitch=None):
    """True when the 16.5 m line and the arc support a grass patch."""
    return paint_patch(matrix, positive_x, grass_mask, white_mask, pitch) is not None


def inside_measured(x, y, positive_x, toward_goal_cm, toward_arc_cm, pitch=None, tolerance=EDGE_TOLERANCE_CM):
    """True when the point stays inside the depth the grass probe verified."""
    if toward_goal_cm is None or toward_arc_cm is None:
        return True
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    pitch = pitch or SoccerPitchConfiguration()
    front = _front_x(bool(positive_x), pitch)
    delta = (float(x) - front) * _goal_sign(bool(positive_x))
    if delta >= 0.0:
        return delta <= float(toward_goal_cm) + tolerance
    return -delta <= float(toward_arc_cm) + tolerance


def inside_visible(x, y, positive_x, pitch=None, tolerance=EDGE_TOLERANCE_CM):
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    x0, x1, y0, y1 = visible_bounds(positive_x, pitch)
    return (x0 - tolerance) <= float(x) <= (x1 + tolerance) and (y0 - tolerance) <= float(y) <= (y1 + tolerance)


def draw_visible_penalty(
    frame,
    matrix,
    positive_x,
    color=(0, 220, 255),
    toward_goal_cm=None,
    toward_arc_cm=None,
):
    """Draw the 16.5 m line, the arc, and only the depth that stays out of the stands."""
    if frame is None or matrix is None:
        return
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        return
    pitch = SoccerPitchConfiguration()
    front = _front_x(bool(positive_x), pitch)
    y0 = (float(pitch.width) - 4032.0) / 2.0
    y1 = (float(pitch.width) + 4032.0) / 2.0
    goal_depth = MAX_GOAL_DEPTH_CM if toward_goal_cm is None else float(toward_goal_cm)
    sign = _goal_sign(bool(positive_x))
    front_line = [(front, y0 + (y1 - y0) * step) for step in np.linspace(0, 1, 24)]
    near_side = [(front + sign * goal_depth * step, y1) for step in np.linspace(0, 1, 16)]
    far_side = [(front + sign * goal_depth * step, y0) for step in np.linspace(0, 1, 16)]
    spot = float(pitch.length) - 1100.0 if positive_x else 1100.0
    arc_sign = -1.0 if positive_x else 1.0
    arc_limit = MAX_ARC_DEPTH_CM if toward_arc_cm is None else float(toward_arc_cm)
    arc = []
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 24):
        x = spot + arc_sign * 915.0 * np.cos(angle)
        y = float(pitch.width) / 2.0 + 915.0 * np.sin(angle)
        outside = (front - x) if positive_x else (x - front)
        if 0.0 < outside <= arc_limit + 50.0:
            arc.append((x, y))
    _polyline(frame, inverse, np.array(front_line, dtype=np.float32), color)
    _polyline(frame, inverse, np.array(near_side, dtype=np.float32), color)
    _polyline(frame, inverse, np.array(far_side, dtype=np.float32), color)
    if len(arc) >= 2:
        _polyline(frame, inverse, np.array(arc, dtype=np.float32), color)


def _polyline(frame, inverse, field_points, color):
    projected = cv2.perspectiveTransform(field_points.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    height, width = frame.shape[:2]
    if not np.isfinite(projected).all():
        return
    if np.max(np.abs(projected)) > max(width, height) * 4:
        return
    run = []
    for x, y in projected:
        safe = 0 <= x < width and CROWD_BAND * height <= y < height
        if safe:
            run.append((int(round(x)), int(round(y))))
            continue
        if len(run) >= 2:
            cv2.polylines(frame, [np.array(run, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        run = []
    if len(run) >= 2:
        cv2.polylines(frame, [np.array(run, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


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
        self.toward_goal_cm = None
        self.toward_arc_cm = None
        self.grass = None
        self.accepted_frames = 0
        self.held_frames = 0
        self.rejected_frames = 0
        self.reasons = Counter()

    def update(self, white_line_mask, frame=None, grass_mask=None):
        report = fit_broadcast_penalty(white_line_mask, self.pitch, frame)
        if report is not None and not scale_is_stable(self.scale, report["cm_per_px"]):
            self.reasons["scale_jump"] += 1
            report = None
        if report is not None and grass_mask is not None:
            patch = paint_patch(
                report["matrix"], report["positive_x"], grass_mask, white_line_mask, self.pitch
            )
            if patch is None:
                self.reasons["off_pitch"] += 1
                report = None
            else:
                report["toward_goal_cm"] = patch["toward_goal_cm"]
                report["toward_arc_cm"] = patch["toward_arc_cm"]
        if report is not None:
            return self._accept(report, grass_mask)
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
        for (image_x, image_y), (x, y) in zip(array.reshape(-1, 2), transformed):
            if not inside_visible(float(x), float(y), self.positive_x, self.pitch):
                mapped.append(None)
                continue
            if not inside_measured(
                float(x),
                float(y),
                self.positive_x,
                self.toward_goal_cm,
                self.toward_arc_cm,
                self.pitch,
            ):
                mapped.append(None)
                continue
            if self.grass is not None and _classify_pixel(float(image_x), float(image_y), self.grass) != "grass":
                mapped.append(None)
                continue
            mapped.append((float(x), float(y)))
        return mapped

    def metrics(self):
        goal_end = None
        if self.positive_x is True:
            goal_end = "high_x"
        elif self.positive_x is False:
            goal_end = "low_x"
        return {
            "method": "penalty line and arc, published only while the projection stays on the grass",
            "toward_goal_cm": self.toward_goal_cm,
            "toward_arc_cm": self.toward_arc_cm,
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

    def _accept(self, report, grass_mask):
        scale = float(report["cm_per_px"])
        same_shot = self.matrix is not None and scale_is_stable(self.scale, scale, SAME_SHOT_SCALE_RATIO)
        if not same_shot:
            self.generation += 1
        self.matrix = report["matrix"]
        self.transformer = ViewTransformer.from_matrix(self.matrix)
        self.positive_x = bool(report["positive_x"])
        self.scale = scale
        self.toward_goal_cm = report.get("toward_goal_cm")
        self.toward_arc_cm = report.get("toward_arc_cm")
        self.grass = grass_mask
        self.hold_left = self.hold_frames
        self.accepted_frames += 1
        self.reasons["accepted"] += 1
        return self._decision(True, False, "accepted")

    def _still_on_paint(self, white_line_mask, grass_mask=None):
        if self.matrix is None or self.hold_left <= 0 or self.positive_x is None:
            return False
        if grass_mask is not None:
            patch = paint_patch(self.matrix, self.positive_x, grass_mask, white_line_mask, self.pitch)
            if patch is None:
                return False
            self.toward_goal_cm = patch["toward_goal_cm"]
            self.toward_arc_cm = patch["toward_arc_cm"]
            self.grass = grass_mask
            return True
        paint = _clean_paint(white_line_mask)
        if paint is None:
            return False
        distance = _painted_arc_distance(self.matrix, paint, self.positive_x)
        return distance is not None and distance <= ARC_MAX_DISTANCE_PX

    def _drop(self):
        self.matrix = None
        self.transformer = None
        self.positive_x = None
        self.scale = None
        self.toward_goal_cm = None
        self.toward_arc_cm = None
        self.grass = None
        self.hold_left = 0

    def _decision(self, accepted, held, reason):
        return {
            "accepted": accepted,
            "held": held,
            "reason": reason,
            "matrix": self.matrix,
            "positive_x": self.positive_x,
            "toward_goal_cm": self.toward_goal_cm,
            "toward_arc_cm": self.toward_arc_cm,
            "generation": self.generation if accepted else None,
            "goal_end": "high_x" if self.positive_x else "low_x" if self.positive_x is False else None,
        }
