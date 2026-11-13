"""Project the visible penalty third, and nothing past it.

The broadcast camera in this clip shows one penalty area. The fit uses the
painted 16.5 m line and the 9.15 m arc, which fixes the scale at the penalty
spot 5.5 m behind that line. Coordinates are published only inside the 35 m
third of the goal that calibration is looking at. The other half of the pitch
stays empty. Possession and passes never read this matrix.

The radar draws that same published point, on the video and in the 2D panel.
A frame without a matrix draws no dots and does not keep the previous ones.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from pitch_homography import (
    _painted_arc_distance,
    _clean_paint,
    centre_marks_still_on_paint,
    fit_broadcast_centre,
    fit_broadcast_penalty,
)
from roboflow_soccer_config import SoccerPitchConfiguration
from roboflow_view import ViewTransformer

VISIBLE_THIRD_CM = 3500.0
MAX_FIELD_SHIFT_CM = 1200.0
RADAR_HOLD_FRAMES = 4
EDGE_TOLERANCE_CM = 100.0
HOLD_FRAMES = 15
SCALE_JUMP_RATIO = 1.35
SAME_SHOT_SCALE_RATIO = 1.08
MIN_FRONT_WHITE = 0.50
ARC_MAX_DISTANCE_PX = 8.0
# The painted arc is an ellipse on a broadcast frame. The belly sits on the
# white line; the last tenth of the samples can sit a few pixels off it.
# A circle drawn on the wrong mark lands further out than this.
ARC_P90_DISTANCE_PX = 16.0
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


def field_shift_cm(old_matrix, new_matrix):
    """How far the same image points move on the pitch, in centimetres.

    A mirrored field moves them by tens of metres. A small camera drift does not.
    """
    samples = np.array(
        [(x, y) for x in (3000.0, 6000.0, 9000.0) for y in (2000.0, 3500.0, 5000.0)],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    try:
        inverse = np.linalg.inv(np.asarray(old_matrix, dtype=np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        return None
    image = cv2.perspectiveTransform(samples, inverse)
    back = cv2.perspectiveTransform(image, np.asarray(new_matrix, dtype=np.float32)).reshape(-1, 2)
    if not np.isfinite(back).all():
        return None
    original = samples.reshape(-1, 2)
    return float(np.median(np.hypot(back[:, 0] - original[:, 0], back[:, 1] - original[:, 1])))


def inside_visible(x, y, positive_x, pitch=None, tolerance=EDGE_TOLERANCE_CM, anchor="penalty"):
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    pitch = pitch or SoccerPitchConfiguration()
    if anchor == "shot":
        return (-tolerance) <= float(x) <= (float(pitch.length) + tolerance) and (
            -tolerance
        ) <= float(y) <= (float(pitch.width) + tolerance)
    if anchor == "centre":
        half = min(VISIBLE_THIRD_CM, float(pitch.length) / 2.0)
        middle = float(pitch.length) / 2.0
        return (middle - half - tolerance) <= float(x) <= (middle + half + tolerance) and (
            -tolerance
        ) <= float(y) <= (float(pitch.width) + tolerance)
    x0, x1, y0, y1 = visible_bounds(positive_x, pitch)
    return (x0 - tolerance) <= float(x) <= (x1 + tolerance) and (y0 - tolerance) <= float(y) <= (y1 + tolerance)


RADAR_MARGIN_PX = 12
RADAR_MAX_WIDTH_PX = 320
_RADAR_DIM = 0.38
_BALL_RADAR_SKIP = {"predicted", "remembered", "lost"}
_TEAM_BGR = {0: (232, 122, 47), 1: (68, 68, 220)}


def _radar_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def radar_inset_box(width, height, margin=RADAR_MARGIN_PX):
    """Bottom-right rectangle for the 2D pitch, or None when the frame is too small."""
    width = int(width)
    height = int(height)
    if width < 80 or height < 80:
        return None
    radar_w = min(RADAR_MAX_WIDTH_PX, max(160, width // 4))
    radar_h = int(round(radar_w * 7000 / 12000))
    if radar_w + margin * 2 > width or radar_h + margin * 2 > height:
        radar_w = width - margin * 2
        radar_h = int(round(radar_w * 7000 / 12000))
    if radar_w < 120 or radar_h < 70 or radar_h + margin > height:
        return None
    return width - radar_w - margin, height - radar_h - margin, radar_w, radar_h


def _radar_person(detection, role, positive_x, pitch, anchor="penalty"):
    x = _radar_number(detection.get("map_x"))
    y = _radar_number(detection.get("map_y"))
    if x is None or y is None or not inside_visible(x, y, positive_x, pitch, anchor=anchor):
        return None
    team = detection.get("team_id")
    try:
        team = int(team)
    except (TypeError, ValueError):
        team = None
    if team not in (0, 1):
        team = None
    return {
        "map_x": round(x, 1),
        "map_y": round(y, 1),
        "team_id": team,
        "role": role,
        "track_id": _radar_track_id(detection.get("track_id")),
        "held": False,
    }


def _radar_track_id(value):
    try:
        track_id = int(value)
    except (TypeError, ValueError):
        return None
    if track_id < 0:
        return None
    return track_id


class RadarTrackMemory:
    """Repeat a track's last map point for a few missed frames.

    This is only the drawing list. It does not move the point, and it never
    writes back onto the detections that possession and the heatmaps read.
    """

    def __init__(self, hold_frames=RADAR_HOLD_FRAMES):
        self.hold_frames = int(hold_frames)
        self.tracks = {}

    def observe(self, people):
        seen = set()
        drawn = []
        for person in people or []:
            track_id = _radar_track_id(person.get("track_id"))
            fresh = {
                "map_x": person["map_x"],
                "map_y": person["map_y"],
                "team_id": person.get("team_id"),
                "role": person.get("role") or "player",
                "track_id": track_id,
                "held": False,
            }
            drawn.append(fresh)
            if track_id is None:
                continue
            seen.add(track_id)
            self.tracks[track_id] = {
                "map_x": fresh["map_x"],
                "map_y": fresh["map_y"],
                "team_id": fresh["team_id"],
                "role": fresh["role"],
                "missing": 0,
            }
        expired = []
        for track_id, saved in self.tracks.items():
            if track_id in seen:
                continue
            saved["missing"] += 1
            if saved["missing"] > self.hold_frames:
                expired.append(track_id)
                continue
            drawn.append(
                {
                    "map_x": saved["map_x"],
                    "map_y": saved["map_y"],
                    "team_id": saved["team_id"],
                    "role": saved["role"],
                    "track_id": track_id,
                    "held": True,
                }
            )
        for track_id in expired:
            del self.tracks[track_id]
        return drawn


def build_radar_snapshot(players, goalkeepers, ball, positive_x, pitch=None, anchor="penalty"):
    """Points the 2D panel may draw. An uncalibrated frame publishes nobody."""
    pitch = pitch or SoccerPitchConfiguration()
    if anchor not in ("centre", "shot", "penalty"):
        anchor = "penalty"
    empty = {
        "calibrated": False,
        "goal_end": None,
        "visible_third_cm": VISIBLE_THIRD_CM,
        "field_length_cm": float(pitch.length),
        "field_width_cm": float(pitch.width),
        "players": [],
        "ball": None,
    }
    if anchor == "penalty" and positive_x is None:
        return empty
    side = bool(positive_x) if positive_x is not None else False
    people = []
    for detection in players or []:
        row = _radar_person(detection, "player", side, pitch, anchor)
        if row is not None:
            people.append(row)
    for detection in goalkeepers or []:
        row = _radar_person(detection, "goalkeeper", side, pitch, anchor)
        if row is not None:
            people.append(row)
    ball_point = None
    if isinstance(ball, dict) and ball.get("source") not in _BALL_RADAR_SKIP:
        x = _radar_number(ball.get("map_x"))
        y = _radar_number(ball.get("map_y"))
        if x is not None and y is not None and inside_visible(x, y, side, pitch, anchor=anchor):
            ball_point = {"map_x": round(x, 1), "map_y": round(y, 1)}
    return {
        "calibrated": True,
        "goal_end": "pitch" if anchor == "shot" else "centre" if anchor == "centre" else ("high_x" if positive_x else "low_x"),
        "visible_third_cm": VISIBLE_THIRD_CM,
        "field_length_cm": float(pitch.length),
        "field_width_cm": float(pitch.width),
        "players": people,
        "ball": ball_point,
    }


def draw_visible_centre(frame, matrix, color=(0, 220, 255)):
    """Draw the centre circle and the halfway line that fixed this frame."""
    if frame is None or matrix is None:
        return
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        return
    circle = [
        (6000.0 + 915.0 * np.cos(angle), 3500.0 + 915.0 * np.sin(angle))
        for angle in np.linspace(0, 2 * np.pi, 48)
    ]
    line = [(6000.0, y) for y in np.linspace(400.0, 6600.0, 24)]
    _polyline(frame, inverse, np.array(circle, dtype=np.float32), color)
    _polyline(frame, inverse, np.array(line, dtype=np.float32), color)


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


def _radar_pixel(x_cm, y_cm, radar_w, radar_h, pitch):
    px = int(round(float(x_cm) / float(pitch.length) * (radar_w - 1)))
    py = int(round(float(y_cm) / float(pitch.width) * (radar_h - 1)))
    return px, py


def _draw_radar_template(radar, pitch):
    height, width = radar.shape[:2]
    color = (230, 235, 230)
    vertices = pitch.vertices
    for start, end in pitch.edges:
        x1, y1 = _radar_pixel(*vertices[start - 1], width, height, pitch)
        x2, y2 = _radar_pixel(*vertices[end - 1], width, height, pitch)
        cv2.line(radar, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    center = _radar_pixel(pitch.length / 2.0, pitch.width / 2.0, width, height, pitch)
    axis_x = max(1, int(round(pitch.centre_circle_radius / float(pitch.length) * (width - 1))))
    axis_y = max(1, int(round(pitch.centre_circle_radius / float(pitch.width) * (height - 1))))
    cv2.ellipse(radar, center, (axis_x, axis_y), 0, 0, 360, color, 1, cv2.LINE_AA)


def _draw_radar_dot(radar, x_cm, y_cm, pitch, color, radius, held=False):
    height, width = radar.shape[:2]
    px, py = _radar_pixel(x_cm, y_cm, width, height, pitch)
    if not (0 <= px < width and 0 <= py < height):
        return
    if held:
        cv2.circle(radar, (px, py), radius + 2, color, 2, cv2.LINE_AA)
        return
    cv2.circle(radar, (px, py), radius, color, -1, cv2.LINE_AA)
    cv2.circle(radar, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)


def compose_pitch_radar(frame, players, ball, positive_x, goalkeepers=None, anchor="penalty", snapshot=None):
    """Paint the calibrated pitch in the corner. Missing coordinates leave no dot."""
    if snapshot is None:
        if anchor not in ("centre", "shot", "penalty"):
            anchor = "penalty"
        if frame is None or (anchor == "penalty" and positive_x is None):
            return None
        snapshot = build_radar_snapshot(players, goalkeepers, ball, positive_x, anchor=anchor)
    if frame is None:
        return None
    box = radar_inset_box(frame.shape[1], frame.shape[0])
    if box is None:
        return None
    if not snapshot["calibrated"]:
        return None
    x0, y0, radar_w, radar_h = box
    pitch = SoccerPitchConfiguration()
    radar = np.zeros((radar_h, radar_w, 3), dtype=np.uint8)
    radar[:] = (42, 110, 48)
    third_px = max(1, int(round(radar_w * VISIBLE_THIRD_CM / float(pitch.length))))
    label = "35 m"
    if snapshot["goal_end"] == "pitch":
        third_x = 0
        third_px = radar_w
        label = ""
    elif snapshot["goal_end"] == "centre":
        middle = float(pitch.length) / 2.0
        left = int(round(radar_w * (middle - VISIBLE_THIRD_CM) / float(pitch.length)))
        right = int(round(radar_w * (middle + VISIBLE_THIRD_CM) / float(pitch.length)))
        left = min(max(left, 0), radar_w - 1)
        right = min(max(right, left + 1), radar_w)
        if left:
            radar[:, :left] = (radar[:, :left].astype(np.float32) * _RADAR_DIM).astype(np.uint8)
        if right < radar_w:
            radar[:, right:] = (radar[:, right:].astype(np.float32) * _RADAR_DIM).astype(np.uint8)
        third_x = left
        third_px = right - left
    elif positive_x:
        radar[:, : radar_w - third_px] = (radar[:, : radar_w - third_px].astype(np.float32) * _RADAR_DIM).astype(np.uint8)
        third_x = radar_w - third_px
    else:
        radar[:, third_px:] = (radar[:, third_px:].astype(np.float32) * _RADAR_DIM).astype(np.uint8)
        third_x = 0
    _draw_radar_template(radar, pitch)
    if label:
        cv2.rectangle(radar, (third_x, 0), (min(radar_w, third_x + third_px) - 1, radar_h - 1), (0, 220, 255), 1)
        cv2.putText(
            radar,
            label,
            (third_x + 4, radar_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    radius = max(3, radar_w // 56)
    for person in snapshot["players"]:
        color = _TEAM_BGR.get(person["team_id"], (180, 180, 180))
        _draw_radar_dot(
            radar,
            person["map_x"],
            person["map_y"],
            pitch,
            color,
            radius,
            held=bool(person.get("held")),
        )
        if person["role"] == "goalkeeper":
            px, py = _radar_pixel(person["map_x"], person["map_y"], radar_w, radar_h, pitch)
            cv2.circle(radar, (px, py), radius + 2, (7, 193, 255), 1, cv2.LINE_AA)
    if snapshot["ball"] is not None:
        ball_radius = max(2, radius - 1)
        _draw_radar_dot(
            radar,
            snapshot["ball"]["map_x"],
            snapshot["ball"]["map_y"],
            pitch,
            (255, 255, 255),
            ball_radius,
        )
        ball_px, ball_py = _radar_pixel(snapshot["ball"]["map_x"], snapshot["ball"]["map_y"], radar_w, radar_h, pitch)
        cv2.circle(radar, (ball_px, ball_py), ball_radius, (0, 140, 255), 1, cv2.LINE_AA)
    cv2.rectangle(radar, (0, 0), (radar_w - 1, radar_h - 1), (255, 255, 255), 1)
    frame[y0 : y0 + radar_h, x0 : x0 + radar_w] = radar
    return snapshot


class VisiblePitchMap:
    """One pitch for the whole shot.

    The first fit that sits on the paint locks which side of the picture is
    which goal. Later fits may nudge that map. A fit that would throw the
    players to the other half is ignored, and a frame with no lines keeps the
    locked map. Players do not change ends between two frames.
    """

    def __init__(self, pitch=None, hold_frames=HOLD_FRAMES):
        self.pitch = pitch or SoccerPitchConfiguration()
        self.hold_frames = int(hold_frames)
        self.matrix = None
        self.transformer = None
        self.positive_x = None
        self.anchor = "penalty"
        self.locked = False
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

    def _screen(self, report, white_line_mask, grass_mask):
        if report is None:
            return None
        if not scale_is_stable(self.scale, report["cm_per_px"]):
            self.reasons["scale_jump"] += 1
            return None
        if grass_mask is not None and report.get("anchor") != "centre":
            patch = paint_patch(
                report["matrix"], report["positive_x"], grass_mask, white_line_mask, self.pitch
            )
            if patch is None:
                self.reasons["off_pitch"] += 1
                return None
            report["toward_goal_cm"] = patch["toward_goal_cm"]
            report["toward_arc_cm"] = patch["toward_arc_cm"]
        return report

    def update(self, white_line_mask, frame=None, grass_mask=None):
        bow = fit_broadcast_centre(white_line_mask, self.pitch, frame)
        shallow = bow if bow is not None and bow.get("anchor") == "penalty" else None
        report = self._screen(shallow, white_line_mask, grass_mask)
        if report is None:
            report = self._screen(
                fit_broadcast_penalty(white_line_mask, self.pitch, frame),
                white_line_mask,
                grass_mask,
            )
        if report is None and bow is not None and bow.get("anchor") == "centre":
            report = self._screen(bow, white_line_mask, grass_mask)
        if report is not None and self.matrix is not None and not self._agrees(report):
            self.reasons["field_jump"] += 1
            report = None
        if report is not None:
            return self._accept(report, grass_mask)
        if self.matrix is not None:
            self.held_frames += 1
            self.reasons["carried"] += 1
            return self._decision(True, True, "carried")
        self._drop()
        self.rejected_frames += 1
        self.reasons["no_visible_penalty"] += 1
        return self._decision(False, False, "no_visible_penalty")

    def project(self, points):
        """Field centimetres inside the visible third, or None past it."""
        if points is None or self.transformer is None:
            return []
        if not self.locked and self.anchor != "centre" and self.positive_x is None:
            return []
        array = np.asarray(points, dtype=np.float32)
        if array.size == 0:
            return []
        transformed = self.transformer.transform_points(array.reshape(-1, 2))
        mapped = []
        for (image_x, image_y), (x, y) in zip(array.reshape(-1, 2), transformed):
            anchor = "shot" if self.locked else self.anchor
            if not inside_visible(float(x), float(y), self.positive_x, self.pitch, anchor=anchor):
                mapped.append(None)
                continue
            if not self.locked and self.anchor != "centre" and not inside_measured(
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
        if self.locked and self.matrix is not None:
            goal_end = "pitch"
        elif self.anchor == "centre" and self.matrix is not None:
            goal_end = "centre"
        elif self.positive_x is True:
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

    def _agrees(self, report):
        if not scale_is_stable(self.scale, report.get("cm_per_px"), 1.25):
            return False
        shift = field_shift_cm(self.matrix, report["matrix"])
        return shift is not None and shift <= MAX_FIELD_SHIFT_CM

    def _accept(self, report, grass_mask):
        scale = float(report["cm_per_px"])
        if not self.locked:
            self.generation += 1
        self.locked = True
        self.matrix = report["matrix"]
        self.transformer = ViewTransformer.from_matrix(self.matrix)
        self.anchor = "centre" if report.get("anchor") == "centre" else "penalty"
        self.positive_x = None if self.anchor == "centre" else bool(report["positive_x"])
        self.scale = scale
        self.toward_goal_cm = report.get("toward_goal_cm")
        self.toward_arc_cm = report.get("toward_arc_cm")
        self.grass = grass_mask
        self.hold_left = self.hold_frames
        self.accepted_frames += 1
        self.reasons["accepted"] += 1
        return self._decision(True, False, "accepted")

    def _still_on_paint(self, white_line_mask, grass_mask=None):
        if self.matrix is None or self.hold_left <= 0:
            return False
        if self.anchor == "centre":
            paint = _clean_paint(white_line_mask)
            if paint is None or not centre_marks_still_on_paint(self.matrix, paint):
                return False
            self.grass = grass_mask
            return True
        if self.positive_x is None:
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
        self.anchor = "penalty"
        self.locked = False
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
            "anchor": "shot" if accepted and self.locked else (self.anchor if accepted else "penalty"),
            "goal_end": (
                "pitch"
                if accepted and self.locked
                else "centre"
                if accepted and self.anchor == "centre"
                else "high_x"
                if self.positive_x
                else "low_x"
                if self.positive_x is False
                else None
            ),
        }
