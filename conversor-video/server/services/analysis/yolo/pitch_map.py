"""Project the visible penalty third, and nothing past it.

The fit uses the painted 16.5 m line and the 9.15 m arc, which fixes the scale
at the penalty spot 5.5 m behind that line. When the two ends of that line are
on the paint, the width follows those corners, so the far half and the near
half can have different pixel lengths. A side is the thin painted ridge that
leaves a corner toward the goal, and a player's width is where a line parallel
to that ridge meets the 16.5 m line. The arc scale is left alone. Coordinates
are published inside the 35 m third of the goal the calibration is looking at.
The grass probe accepts or rejects that calibration. It does not erase a player
or the ball who already land inside the third. A short gap keeps the last good
matrix, then drops it. A new fit that flips the same line, or jumps the scale
while the line stays put, is refused. Possession and passes never read this matrix.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from pitch_homography import (
    _painted_arc_distance,
    _clean_paint,
    fit_broadcast_centre,
    fit_broadcast_penalty,
)
from roboflow_soccer_config import SoccerPitchConfiguration
from roboflow_view import ViewTransformer

VISIBLE_THIRD_CM = 3500.0
# The centre circle fixes the scale. Publish 30 m each side of halfway,
# which is the grass a midfield camera actually shows.
CENTRE_HALF_CM = 3000.0
TEAM_MARKER_BGR = {0: (255, 64, 0), 1: (0, 0, 255)}
BALL_MARKER_BGR = (0, 140, 255)
REFEREE_MARKER_BGR = (0, 255, 255)
NEUTRAL_MARKER_BGR = (180, 180, 180)
EDGE_TOLERANCE_CM = 100.0
HOLD_FRAMES = 15
# While the current arc still sits on the paint, a fit that moves the same
# field points further than this is a rival, not a camera pan.
MAX_FIELD_SHIFT_CM = 1200.0
# The 16.5 m line has to travel this far in the picture before the old end
# is allowed to give way to the other penalty.
LINE_MOVE_PX = 140.0
SANE_CM_MIN = 2.0
SANE_CM_MAX = 25.0
SCALE_JUMP_RATIO = 1.35
SAME_SHOT_SCALE_RATIO = 1.08
MIN_FRONT_WHITE = 0.50
ARC_MAX_DISTANCE_PX = 8.0
# The belly of the arc has to sit on the paint. The last tenth of a broadcast
# ellipse can sit a few pixels off it; a circle on the wrong mark lands further.
ARC_P90_DISTANCE_PX = 14.0
PROBE_HALF_WIDTH_CM = 1200.0
PROBE_STEP_CM = 50.0
MAX_GOAL_DEPTH_CM = 1650.0
MAX_ARC_DEPTH_CM = 500.0
CROWD_BAND = 0.18
# The painted arc meets the 16.5 m line 7.31 m either side of the spot.
# The corners are further out, where that line actually ends.
_ARC_MEET_CM = float(np.sqrt(915.0**2 - 550.0**2))
_BOX_HALF_CM = 2016.0
_CORNER_MIN_CM = 1200.0
_CORNER_MAX_CM = 3000.0
# A stop on the picture edge is the frame cutting the line, not a painted corner.
_CORNER_EDGE_PX = 16.0


def scale_is_stable(previous, new, ratio=SCALE_JUMP_RATIO):
    """True when this frame's centimetres-per-pixel stays near the last one."""
    if new is None or new <= 0:
        return False
    if previous is None:
        return True
    if previous <= 0:
        return False
    return (previous / ratio) <= new <= (previous * ratio)


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


def replacement_verdict(same_side, scale_ok, shift_cm, line_move_px, coherent, sane=True):
    """How a validated fit relates to the matrix already on screen.

    A nudge stays on the same penalty. A camera move is a different line,
    after the old arc has left the paint, and it may cross 12 m. A flip or a
    scale jump that keeps the line in place is refused.
    """
    if not sane:
        return "absurd"
    modest = same_side and scale_ok and shift_cm is not None and shift_cm <= MAX_FIELD_SHIFT_CM
    if modest:
        return "nudge"
    if coherent:
        if not same_side:
            return "mirrored"
        if not scale_ok:
            return "scale_jump"
        return "field_jump"
    if line_move_px is not None and line_move_px >= LINE_MOVE_PX:
        return "camera_move"
    if not same_side:
        return "mirrored"
    return "scale_jump"


def visible_bounds(positive_x, pitch=None, region="penalty"):
    """Field rectangle, in centimetres, of the grass this calibration can see."""
    pitch = pitch or SoccerPitchConfiguration()
    length = float(pitch.length)
    width = float(pitch.width)
    if region == "centre":
        half = min(CENTRE_HALF_CM, length / 2.0)
        return length / 2.0 - half, length / 2.0 + half, 0.0, width
    third = min(VISIBLE_THIRD_CM, length)
    if positive_x:
        return length - third, length, 0.0, width
    return 0.0, third, 0.0, width


def marker_bgr(role, team_id=None):
    """Colour of one dot on the 2D map. Team 0 is blue and team 1 is red."""
    name = str(role or "").strip().lower()
    if name == "ball":
        return BALL_MARKER_BGR
    if name == "referee":
        return REFEREE_MARKER_BGR
    if team_id in TEAM_MARKER_BGR:
        return TEAM_MARKER_BGR[int(team_id)]
    try:
        team = int(team_id)
    except (TypeError, ValueError):
        team = None
    if team in TEAM_MARKER_BGR:
        return TEAM_MARKER_BGR[team]
    return NEUTRAL_MARKER_BGR


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


def front_image_point(matrix, positive_x, pitch=None):
    """Where the middle of the 16.5 m line falls in the picture."""
    if matrix is None or positive_x is None:
        return None
    pitch = pitch or SoccerPitchConfiguration()
    projected = _project_field(matrix, [(_front_x(bool(positive_x), pitch), float(pitch.width) / 2.0)])
    if projected is None:
        return None
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
    """True when a probe point stays inside the depth that accepted the homography.

    Drawing does not call this. A player inside the visible third is published
    even when the probe stopped short of the goal line.
    """
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


def inside_visible(x, y, positive_x, pitch=None, tolerance=EDGE_TOLERANCE_CM, region="penalty"):
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    x0, x1, y0, y1 = visible_bounds(positive_x, pitch, region)
    return (x0 - tolerance) <= float(x) <= (x1 + tolerance) and (y0 - tolerance) <= float(y) <= (y1 + tolerance)


def centre_on_paint(matrix, grass_mask, white_mask, pitch=None):
    """True when the centre circle and the halfway line still sit on the paint."""
    if matrix is None:
        return False
    pitch = pitch or SoccerPitchConfiguration()
    length = float(pitch.length)
    width = float(pitch.width)
    circle = [
        (
            length / 2.0 + 915.0 * np.cos(theta),
            width / 2.0 + 915.0 * np.sin(theta),
        )
        for theta in np.linspace(0, 2 * np.pi, 36, endpoint=False)
    ]
    projected = _project_field(matrix, circle)
    if projected is None or white_mask is None:
        return False
    height, frame_width = white_mask.shape[:2]
    distance = cv2.distanceTransform(cv2.bitwise_not(white_mask), cv2.DIST_L2, 3)
    samples = []
    for x, y in projected:
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < frame_width and 0 <= y < height:
            samples.append(float(distance[int(y), int(x)]))
    if len(samples) < 16 or float(np.median(samples)) > ARC_MAX_DISTANCE_PX:
        return False
    line = [(length / 2.0, 800.0 + 5400.0 * step) for step in np.linspace(0, 1, 16)]
    line_points = _project_field(matrix, line)
    if line_points is None:
        return False
    on_frame = [
        (float(x), float(y))
        for x, y in line_points
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < frame_width and 0 <= y < height
    ]
    if len(on_frame) < 6:
        return False
    white_hits = sum(distance[int(y), int(x)] <= 6 for x, y in on_frame)
    if white_hits / len(on_frame) < 0.35:
        return False
    if grass_mask is None:
        return True
    center = _project_one(matrix, (length / 2.0, width / 2.0))
    return _classify_pixel(center[0], center[1], grass_mask) == "grass"


def _line_frame(matrix, positive_x, pitch):
    """Image foot of the 16.5 m line, plus the width axis and the direction to the goal."""
    front = _front_x(bool(positive_x), pitch)
    sign = _goal_sign(bool(positive_x))
    image = _project_field(
        matrix,
        [(front, float(pitch.width) / 2.0), (front, float(pitch.width) / 2.0 + 100.0), (front + sign * 100.0, float(pitch.width) / 2.0)],
    )
    if image is None:
        return None
    foot = image[0]
    along = image[1] - foot
    perp = image[2] - foot
    along_norm = float(np.linalg.norm(along))
    perp_norm = float(np.linalg.norm(perp))
    if along_norm < 1e-6 or perp_norm < 1e-6:
        return None
    return foot, along / along_norm, perp / perp_norm


def _width_terms(t_low, t_high, center):
    """1D homography along the 16.5 m line. t=0 is the arc, the two t's are the corners.

    Field y is (a t + center) / (c t + 1). Equal pixel halves leave c at 0 and
    this is the old scale. A compressed far side gets a different c.
    """
    if t_low >= -40.0 or t_high <= 40.0:
        return None
    product = 2.0 * float(t_low) * float(t_high)
    if abs(product) < 1e-6:
        return None
    slope = -(_BOX_HALF_CM) / float(t_low)
    curve = -(float(t_low) + float(t_high)) / product
    pole = None if abs(curve) < 1e-8 else -1.0 / curve
    if pole is not None and t_low < pole < t_high:
        return None
    tilt = (center - _BOX_HALF_CM) * curve + slope
    for station in (float(t_low), 0.0, float(t_high)):
        if curve * station + 1.0 <= 0.35:
            return None
    return float(tilt), float(curve)


def field_y_from_line(t_px, terms, center):
    """Field y, in centimetres, of a pixel offset along the 16.5 m line."""
    if terms is None:
        return None
    tilt, curve = terms
    denom = curve * float(t_px) + 1.0
    if denom <= 0.35:
        return None
    return (tilt * float(t_px) + float(center)) / denom


def _line_ends(paint, matrix, positive_x, cm_per_px, pitch=None):
    """Each real end of the 16.5 m line. A pair also carries the width terms.

    One end can be off the picture. That end is absent and the other is kept,
    so a single visible corner can still grow its side. The width terms exist
    only when both ends pass, which is what penalty_corners publishes.
    """
    if paint is None or matrix is None or cm_per_px is None or cm_per_px <= 0:
        return None
    pitch = pitch or SoccerPitchConfiguration()
    frame = _line_frame(matrix, positive_x, pitch)
    if frame is None:
        return None
    foot, along, perp = frame
    ys, xs = np.where(paint > 0)
    if len(xs) < 40:
        return None
    delta = np.stack([xs, ys], axis=1).astype(np.float64) - foot
    tee = delta @ along
    ess = delta @ perp
    step = 8
    stations = np.arange(-520, 521, step)
    lit = np.array(
        [
            int(np.sum((np.abs(tee - station) <= step) & (np.abs(ess) <= 7.0))) >= 12
            for station in stations
        ]
    )
    start = int(round((_ARC_MEET_CM / float(cm_per_px) + 24.0) / step))

    def walk(sign):
        index = len(stations) // 2 + sign * start
        last = None
        gap = 0
        while 0 <= index < len(stations):
            if lit[index]:
                last = float(stations[index])
                gap = 0
            else:
                gap += 1
                if gap > 10:
                    break
            index += sign
        return last

    def stopped(station):
        corner = foot + along * float(station)
        height, width = paint.shape[:2]
        inside = (
            _CORNER_EDGE_PX <= corner[0] < width - _CORNER_EDGE_PX
            and _CORNER_EDGE_PX <= corner[1] < height - _CORNER_EDGE_PX
        )
        if not inside:
            return False
        sign = 1.0 if station > 0.0 else -1.0
        beyond = ((tee - station) * sign > 0.0) & ((tee - station) * sign < 48.0)
        on_line = int(np.sum(beyond & (np.abs(ess) <= 8.0)))
        drifted = int(np.sum(beyond & (np.abs(ess) > 8.0) & (np.abs(ess) <= 24.0)))
        return on_line < 15 and drifted < 30

    def accept(station):
        if station is None:
            return None
        reached = abs(float(station)) * float(cm_per_px)
        if not (_CORNER_MIN_CM <= reached <= _CORNER_MAX_CM) or not stopped(station):
            return None
        return float(station)

    t_low, t_high = accept(walk(-1)), accept(walk(1))
    if t_low is None and t_high is None:
        return None
    center = float(pitch.width) / 2.0
    terms = None
    if t_low is not None and t_high is not None:
        terms = _width_terms(t_low, t_high, center)
        if terms is None:
            return None
    return {
        "t_low": None if t_low is None else float(t_low),
        "t_high": None if t_high is None else float(t_high),
        "terms": terms,
        "foot": foot,
        "along": along,
        "perp": perp,
        "corner_low": None if t_low is None else foot + along * float(t_low),
        "corner_high": None if t_high is None else foot + along * float(t_high),
    }


def penalty_corners(paint, matrix, positive_x, cm_per_px, pitch=None):
    """The two ends of the painted 16.5 m line, or None.

    Each end has to be a real stop, one on each side of the arc, and a
    plausible 20.16 m at the arc's scale. A line that merely drifts out of
    the band is not a corner, and the caller keeps the single scale.
    """
    found = _line_ends(paint, matrix, positive_x, cm_per_px, pitch)
    if found is None or found["terms"] is None:
        return None
    return found


def _gray_image(frame):
    if frame is None:
        return None
    image = np.asarray(frame)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        return None
    return image.astype(np.float32)


def _sample_gray(gray, x, y):
    height, width = gray.shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if not (0 <= xi < width and 0 <= yi < height):
        return None
    return float(gray[yi, xi])


def _leaves_pitch(gray, grass, x, y):
    """True on the stands, the boards, or a pixel that is not grass."""
    height, width = gray.shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if not (1 <= xi < width - 1 and 1 <= yi < height - 1):
        return True
    if yi < height * CROWD_BAND:
        return True
    if grass is not None and int(grass[yi, xi]) == 0:
        return True
    if float(gray[yi, xi]) >= 180.0:
        return True
    above = gray[max(0, yi - 16): max(0, yi - 4), xi]
    if above.size and float(np.mean(above >= 185.0)) > 0.45:
        return True
    return False


def _ridge_contrast(gray, x, y, normal):
    center = _sample_gray(gray, x, y)
    if center is None or center >= 180.0:
        return None
    left = _sample_gray(gray, x + normal[0] * 5.0, y + normal[1] * 5.0)
    right = _sample_gray(gray, x - normal[0] * 5.0, y - normal[1] * 5.0)
    if left is None or right is None:
        return None
    return center - max(left, right)


def _angle_between(first, second):
    cross = float(first[0] * second[1] - first[1] * second[0])
    dot = float(np.dot(first, second))
    return abs(float(np.degrees(np.arctan2(cross, dot))))


def _walk_side(gray, grass, origin, direction, perp, foot):
    """Follow one fixed direction along the thin ridge. The boards stop it."""
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    pos = np.asarray(origin, dtype=np.float64)
    hits = []
    gap = 0
    for _ in range(240):
        found = None
        for offset in range(-14, 15, 2):
            cand = pos + direction * 4.0 + normal * float(offset)
            if _leaves_pitch(gray, grass, cand[0], cand[1]):
                continue
            if abs(float(np.dot(cand - foot, perp))) < 8.0:
                continue
            score = _ridge_contrast(gray, cand[0], cand[1], normal)
            if score is None or score < 6.0:
                continue
            if found is None or score > found[0]:
                found = (score, cand)
        if found is None:
            gap += 1
            if gap > 14:
                break
            pos = pos + direction * 4.0
            continue
        gap = 0
        pos = found[1]
        hits.append(pos.copy())
    if len(hits) < 12:
        return None
    # A late jump onto the board edge leaves the straight ridge. Drop it.
    end = hits[-1]
    for point in reversed(hits):
        offset = float(np.dot(point - origin, normal))
        if abs(offset) <= 12.0:
            end = point
            break
    length = float(np.linalg.norm(end - origin))
    if length < 80.0:
        return None
    height, width = gray.shape[:2]
    border = bool(end[0] <= 8.0 or end[1] <= 8.0 or end[0] >= width - 9.0 or end[1] >= height - 9.0)
    outgoing = (end - origin) / length
    if float(np.dot(outgoing, perp)) < 0.15:
        return None
    return {
        "origin": np.asarray(origin, dtype=np.float64),
        "end": end,
        "direction": outgoing,
        "border": border,
        "hits": len(hits),
    }


def side_from_corner(frame, grass, origin, along, perp, foot):
    """The painted side leaving one corner, toward the goal.

    The side is a thin ridge on the grass, a few degrees off horizontal on a
    broadcast camera. It is not the perpendicular of the 16.5 m line, and it
    stops where the grass stops. The picture edge is only where the ink stops.
    """
    gray = _gray_image(frame)
    if gray is None or origin is None or along is None or perp is None or foot is None:
        return None
    origin = np.asarray(origin, dtype=np.float64)
    along = np.asarray(along, dtype=np.float64)
    perp = np.asarray(perp, dtype=np.float64)
    foot = np.asarray(foot, dtype=np.float64)
    ranked = []
    for degree in range(-180, 180, 2):
        direction = np.array(
            [np.cos(np.radians(degree)), np.sin(np.radians(degree))],
            dtype=np.float64,
        )
        if float(np.dot(direction, perp)) < 0.25:
            continue
        if _angle_between(direction, along) < 18.0:
            continue
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        scores = []
        for distance in (36.0, 56.0, 80.0, 108.0, 140.0, 172.0):
            point = origin + direction * distance
            if _leaves_pitch(gray, grass, point[0], point[1]):
                break
            if abs(float(np.dot(point - foot, perp))) < 10.0:
                continue
            score = _ridge_contrast(gray, point[0], point[1], normal)
            if score is None:
                continue
            scores.append(score)
        if len(scores) < 4:
            continue
        positive = sum(1 for score in scores if score >= 6.0)
        median = float(np.median(scores))
        if positive >= 3 and median >= 4.0:
            ranked.append((positive, median, direction))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = None
    for _positive, _median, direction in ranked[:3]:
        walked = _walk_side(gray, grass, origin, direction, perp, foot)
        if walked is None:
            continue
        if best is None or walked["hits"] > best["hits"]:
            best = walked
    return best


def _ink_block(gray, grass, x, y):
    """Why the yellow ink should pause. A thin white line is paint, not a board."""
    height, width = gray.shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if not (0 <= xi < width and 0 <= yi < height) or yi < height * CROWD_BAND:
        return "board"
    # The ridge walk already stopped at the boards. A player on the grass is a hole.
    if grass is not None and int(grass[yi, xi]) == 0:
        return "hole"
    return "ok"


def _segment_on_pitch(frame, grass, origin, end):
    """The straight ink from the corner, cut where it enters the boards.

    A player standing on the line is a short hole and the ink crosses him.
    The advertising is bright and the ink stops.
    """
    gray = _gray_image(frame)
    if gray is None or origin is None or end is None:
        return None
    origin = np.asarray(origin, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - origin
    length = float(np.linalg.norm(delta))
    if length < 40.0:
        return None
    direction = delta / length
    last = origin.copy()
    hole = 0
    travelled = 3.0
    while travelled <= length + 0.5:
        point = origin + direction * min(travelled, length)
        block = _ink_block(gray, grass, point[0], point[1])
        if block == "board":
            break
        if block == "hole":
            hole += 1
            if hole >= 28:
                break
        else:
            last = point.copy()
            hole = 0
        travelled += 3.0
    if float(np.linalg.norm(last - origin)) < 40.0:
        return None
    return last


def draw_visible_penalty(
    frame,
    matrix,
    positive_x,
    color=(0, 220, 255),
    toward_goal_cm=None,
    toward_arc_cm=None,
    corners=None,
    corner_perp=None,
    side_px=None,
    side_lines=None,
    side_anchor=None,
):
    """Draw the 16.5 m line, the arc, and the sides that stay on the grass."""
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
    lines = [segment for segment in (side_lines or []) if segment is not None and len(segment) == 2]
    if corners is not None and corner_perp is not None and side_px is not None and len(corners) == 2:
        perp = np.asarray(corner_perp, dtype=np.float64)
        if isinstance(side_px, (tuple, list)) and len(side_px) == 2:
            reaches = [max(0.0, float(value)) for value in side_px]
        else:
            reaches = [max(0.0, float(side_px)), max(0.0, float(side_px))]
        _image_segment(frame, corners[0], corners[1], color)
        for corner, reach in zip(corners, reaches):
            covered = any(
                float(np.hypot(segment[0][0] - corner[0], segment[0][1] - corner[1])) < 30.0
                for segment in lines
            )
            if covered:
                continue
            if reach > 1.0:
                _image_segment(frame, corner, np.asarray(corner, dtype=np.float64) + perp * reach, color)
        for start, end in lines:
            _image_segment(frame, start, end, color)
    else:
        _polyline(frame, inverse, np.array(front_line, dtype=np.float32), color)
        # One visible corner replaces its own side. The other side stays as it was.
        if side_anchor != "high":
            _polyline(frame, inverse, np.array(near_side, dtype=np.float32), color)
        if side_anchor != "low":
            _polyline(frame, inverse, np.array(far_side, dtype=np.float32), color)
        for start, end in lines:
            _image_segment(frame, start, end, color)
    if len(arc) >= 2:
        _polyline(frame, inverse, np.array(arc, dtype=np.float32), color)


def draw_visible_centre(frame, matrix, color=(0, 220, 255)):
    """Draw the halfway line and the centre circle, and nothing in the stands."""
    if frame is None or matrix is None:
        return
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        return
    pitch = SoccerPitchConfiguration()
    length = float(pitch.length)
    width = float(pitch.width)
    line = [(length / 2.0, width * step) for step in np.linspace(0, 1, 24)]
    circle = [
        (length / 2.0 + 915.0 * np.cos(theta), width / 2.0 + 915.0 * np.sin(theta))
        for theta in np.linspace(0, 2 * np.pi, 36, endpoint=False)
    ]
    _polyline(frame, inverse, np.array(line, dtype=np.float32), color)
    _polyline(frame, inverse, np.array(circle, dtype=np.float32), color)


def _image_segment(frame, start, end, color):
    """Draw one image segment, stopping where it enters the stands."""
    height, width = frame.shape[:2]
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    run = []
    for step in np.linspace(0.0, 1.0, 16):
        point = start + (end - start) * float(step)
        safe = 0 <= point[0] < width and CROWD_BAND * height <= point[1] < height
        if safe:
            run.append((int(round(point[0])), int(round(point[1]))))
            continue
        if len(run) >= 2:
            cv2.polylines(frame, [np.array(run, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        run = []
    if len(run) >= 2:
        cv2.polylines(frame, [np.array(run, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


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
    """The penalty in frame, replaced when the camera shows a different one.

    A frame whose arc still sits on the paint keeps its matrix. A fit that
    only rescales or mirrors that same line is ignored. A short gap keeps the
    last good matrix, then drops it. The next line that lands on the white
    marks takes over, even if that moves the field by more than 12 m.
    """

    def __init__(self, pitch=None, hold_frames=HOLD_FRAMES):
        self.pitch = pitch or SoccerPitchConfiguration()
        self.hold_frames = int(hold_frames)
        self.matrix = None
        self.transformer = None
        self.positive_x = None
        self.region = None
        self.scale = None
        self.hold_left = 0
        self.generation = 0
        self.toward_goal_cm = None
        self.toward_arc_cm = None
        self.grass = None
        self.corner_low = None
        self.corner_high = None
        self.corner_perp = None
        self.side_px = None
        self.width_terms = None
        self.side_direction = None
        self.side_lines = None
        self.side_anchor = None
        self._width_foot = None
        self._width_along = None
        self._corner_t = None
        self._corner_matrix_id = None
        self.accepted_frames = 0
        self.held_frames = 0
        self.rejected_frames = 0
        self.reasons = Counter()

    def update(self, white_line_mask, frame=None, grass_mask=None):
        report = fit_broadcast_penalty(white_line_mask, self.pitch, frame)
        if report is not None and not (SANE_CM_MIN <= float(report["cm_per_px"]) <= SANE_CM_MAX):
            self.reasons["absurd"] += 1
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
        if report is None:
            centre = fit_broadcast_centre(white_line_mask, self.pitch, frame)
            if centre is not None and not (SANE_CM_MIN <= float(centre["cm_per_px"]) <= SANE_CM_MAX):
                self.reasons["absurd"] += 1
                centre = None
            if centre is not None and centre_on_paint(centre["matrix"], grass_mask, white_line_mask, self.pitch):
                report = centre
            elif centre is not None:
                self.reasons["off_pitch"] += 1
        coherent = self._geometry_holds(white_line_mask, grass_mask)
        fresh_view = False
        if report is not None and self.matrix is not None:
            report_region = report.get("region") or "penalty"
            if report_region != (self.region or "penalty"):
                # A living calibration keeps its region. Once it has left the
                # paint, the other view may take over.
                if coherent:
                    self.reasons["region_held"] += 1
                    report = None
                else:
                    fresh_view = True
                    self.reasons["camera_move"] += 1
            else:
                same_side = bool(report["positive_x"]) == bool(self.positive_x)
                scale_ok = scale_is_stable(self.scale, report["cm_per_px"])
                shift = field_shift_cm(self.matrix, report["matrix"])
                verdict = replacement_verdict(
                    same_side,
                    scale_ok,
                    shift,
                    self._front_move_px(report),
                    coherent,
                )
                if verdict == "camera_move":
                    fresh_view = True
                    self.reasons["camera_move"] += 1
                elif verdict != "nudge":
                    self.reasons[verdict] += 1
                    report = None
        if report is not None:
            return self._publish(
                self._accept(report, grass_mask, fresh_view=fresh_view), white_line_mask, frame
            )
        if coherent:
            self.held_frames += 1
            self.hold_left = self.hold_frames
            self.reasons["held"] += 1
            return self._publish(self._decision(True, True, "held"), white_line_mask, frame)
        # A few bad frames keep the last good matrix. The counter does not refill.
        if self.matrix is not None and self.hold_left > 0:
            self.hold_left -= 1
            self.held_frames += 1
            self.reasons["bridge"] += 1
            return self._publish(self._decision(True, True, "held"), white_line_mask, frame)
        self._drop()
        self.rejected_frames += 1
        self.reasons["no_visible_penalty"] += 1
        return self._decision(False, False, "no_visible_penalty")

    def project(self, points):
        """Field centimetres inside the visible third, or None past it.

        The grass probe already accepted this matrix. A point on the grass
        inside that third is published. The probe depth is not a second cut.
        """
        if points is None or self.transformer is None or self.positive_x is None:
            return []
        array = np.asarray(points, dtype=np.float32)
        if array.size == 0:
            return []
        transformed = self.transformer.transform_points(array.reshape(-1, 2))
        mapped = []
        center = float(self.pitch.width) / 2.0
        for (image_x, image_y), (x, y) in zip(array.reshape(-1, 2), transformed):
            corrected = self._field_y(float(image_x), float(image_y), float(y), center)
            if corrected is not None:
                y = corrected
            if not inside_visible(
                float(x),
                float(y),
                self.positive_x,
                self.pitch,
                region=self.region or "penalty",
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
        if self.region == "centre":
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

    def _accept(self, report, grass_mask, fresh_view=False):
        scale = float(report["cm_per_px"])
        same_shot = (
            not fresh_view
            and self.matrix is not None
            and scale_is_stable(self.scale, scale, SAME_SHOT_SCALE_RATIO)
        )
        if not same_shot:
            self.generation += 1
        self.matrix = report["matrix"]
        self.transformer = ViewTransformer.from_matrix(self.matrix)
        self.positive_x = bool(report["positive_x"])
        self.region = report.get("region") or "penalty"
        self.scale = scale
        self.toward_goal_cm = report.get("toward_goal_cm")
        self.toward_arc_cm = report.get("toward_arc_cm")
        self.grass = grass_mask
        self.hold_left = self.hold_frames
        self.accepted_frames += 1
        self.reasons["accepted"] += 1
        return self._decision(True, False, "accepted")

    def _front_move_px(self, report):
        old = front_image_point(self.matrix, self.positive_x, self.pitch)
        new = front_image_point(report["matrix"], report["positive_x"], self.pitch)
        if old is None or new is None:
            return None
        return float(np.hypot(new[0] - old[0], new[1] - old[1]))

    def _geometry_holds(self, white_line_mask, grass_mask=None):
        """True when the current arc still lands on this frame. The hold counter is ignored."""
        if self.matrix is None or self.positive_x is None:
            return False
        if self.region == "centre":
            return centre_on_paint(self.matrix, grass_mask, white_line_mask, self.pitch)
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

    def _still_on_paint(self, white_line_mask, grass_mask=None):
        if self.hold_left <= 0:
            return False
        return self._geometry_holds(white_line_mask, grass_mask)

    def _drop(self):
        self.matrix = None
        self.transformer = None
        self.positive_x = None
        self.region = None
        self.scale = None
        self.toward_goal_cm = None
        self.toward_arc_cm = None
        self.grass = None
        self.corner_low = None
        self.corner_high = None
        self.corner_perp = None
        self.side_px = None
        self.width_terms = None
        self.side_direction = None
        self.side_lines = None
        self.side_anchor = None
        self._width_foot = None
        self._width_along = None
        self._corner_t = None
        self._corner_matrix_id = None
        self.hold_left = 0

    def _field_y(self, image_x, image_y, affine_y, center):
        """Field y from the side's direction, or from the 16.5 m line alone.

        A line through the player parallel to the painted side meets the 16.5 m
        line at his width. Field x stays the affine depth, so the arc does not move.
        """
        if self.width_terms is None:
            return None
        if (
            self.side_direction is not None
            and self._width_foot is not None
            and self._width_along is not None
        ):
            along = self._width_along
            direction = self.side_direction
            cross = float(along[0] * direction[1] - along[1] * direction[0])
            if abs(cross) >= 0.2:
                rhs = np.array([image_x, image_y], dtype=np.float64) - self._width_foot
                # foot + t * along = player + s * direction
                try:
                    solved = np.linalg.solve(np.column_stack([along, -direction]), rhs)
                except np.linalg.LinAlgError:
                    solved = None
                if solved is not None:
                    corrected = field_y_from_line(float(solved[0]), self.width_terms, center)
                    if corrected is not None:
                        return corrected
        if not self.scale:
            return None
        offset = (float(affine_y) - center) / float(self.scale)
        return field_y_from_line(offset, self.width_terms, center)

    def _grass_reach(self, corner, perp, cap):
        """How far the side stays on the grass, never past the probed depth."""
        if cap <= 0.0 or self.grass is None:
            return cap
        height, width = self.grass.shape[:2]
        last = 0.0
        missed = 0
        travelled = 0.0
        while travelled <= cap:
            point = np.asarray(corner, dtype=np.float64) + perp * travelled
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if not (0 <= x < width and 0 <= y < height):
                break
            if self.grass[y, x] > 0:
                last = travelled
                missed = 0
            else:
                missed += 1
                if missed >= 4 and travelled > 12.0:
                    break
            travelled += 2.0
        return last

    def _apply_corners(self, found):
        self.width_terms = found["terms"]
        self.corner_perp = found["perp"]
        self.corner_low = found["corner_low"]
        self.corner_high = found["corner_high"]
        cap = float(self.toward_goal_cm) / float(self.scale) if self.toward_goal_cm and self.scale else 0.0
        perp = found["perp"]
        self.side_px = (
            self._grass_reach(found["corner_low"], perp, cap),
            self._grass_reach(found["corner_high"], perp, cap),
        )

    def _point_pair(self, start, end):
        if start is None or end is None:
            return None
        return (
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])),
        )

    def _side_segment(self, frame, origin, end):
        clipped = _segment_on_pitch(frame, self.grass, origin, end)
        if clipped is None:
            return None
        return self._point_pair(origin, clipped)

    def _seek_side(self, found, frame, name, both):
        """The side at this corner. A faint line may end past the white mask."""
        origin = found.get("corner_" + name)
        station = found.get("t_" + name)
        if origin is None:
            return None

        def accept(point):
            walked = side_from_corner(
                frame, self.grass, point, found["along"], found["perp"], found["foot"]
            )
            if walked is None:
                return None
            segment = self._side_segment(frame, point, walked["end"])
            if segment is None:
                return None
            return {"name": name, "segment": segment, "walked": walked}

        found_here = accept(origin)
        if found_here is not None or both or station is None or not self.scale:
            return found_here
        # One corner, and the white mask stopped short of it. Step outward
        # along the 16.5 m line and keep the longest painted side.
        sign = -1.0 if name == "low" else 1.0
        height, width = self.grass.shape[:2]
        best = None
        misses = 0
        for extra in range(4, 161, 4):
            probe = float(station) + sign * float(extra)
            reached = abs(probe) * float(self.scale)
            if reached > _CORNER_MAX_CM:
                break
            if reached < _CORNER_MIN_CM:
                continue
            point = found["foot"] + found["along"] * probe
            inside = (
                _CORNER_EDGE_PX <= point[0] < width - _CORNER_EDGE_PX
                and _CORNER_EDGE_PX <= point[1] < height - _CORNER_EDGE_PX
            )
            if not inside:
                break
            found_there = accept(point)
            if found_there is None:
                if best is not None:
                    misses += 1
                    if misses >= 3:
                        break
                continue
            misses = 0
            if best is None or self._segment_length(found_there) > self._segment_length(best):
                best = found_there
        return best

    def _segment_length(self, item):
        start, end = item["segment"]
        return float(np.hypot(end[0] - start[0], end[1] - start[1]))

    def _measure_sides(self, found, frame):
        """Draw each visible side on its paint. Copy the angle only when one is missing."""
        self.side_direction = None
        self.side_lines = None
        self.side_anchor = None
        self._width_foot = None
        self._width_along = None
        if frame is None or self.grass is None:
            return
        both = found.get("corner_low") is not None and found.get("corner_high") is not None
        traced = []
        for name in ("low", "high"):
            item = self._seek_side(found, frame, name, both)
            if item is not None:
                traced.append(item)
        if not traced:
            return
        if both and found.get("terms") is not None:
            closed = [item for item in traced if not item["walked"]["border"]]
            pool = closed or traced
            pool.sort(key=lambda item: item["walked"]["hits"], reverse=True)
            self.side_direction = pool[0]["walked"]["direction"]
            self._width_foot = np.asarray(found["foot"], dtype=np.float64)
            self._width_along = np.asarray(found["along"], dtype=np.float64)
            have = {item["name"] for item in traced}
            segments = [item["segment"] for item in traced]
            for name in ("low", "high"):
                if name in have:
                    continue
                origin = found["corner_" + name]
                far = origin + self.side_direction * 1400.0
                copied = self._side_segment(frame, origin, far)
                if copied is not None:
                    segments.append(copied)
            self.side_lines = segments
            return
        if len(traced) == 1 and not both:
            self.side_anchor = traced[0]["name"]
            self.side_lines = [traced[0]["segment"]]

    def _refresh_corners(self, white_line_mask, frame=None):
        """Measure the corners and the sides. A missed pair keeps the last one of this matrix."""
        self.corner_low = None
        self.corner_high = None
        self.corner_perp = None
        self.side_px = None
        self.width_terms = None
        self.side_direction = None
        self.side_lines = None
        self.side_anchor = None
        self._width_foot = None
        self._width_along = None
        if self.matrix is None or self.positive_x is None or self.region == "centre" or not self.scale:
            self._corner_t = None
            self._corner_matrix_id = None
            return
        paint = _clean_paint(white_line_mask)
        found = _line_ends(paint, self.matrix, self.positive_x, self.scale, self.pitch)
        matrix_id = id(self.matrix)
        if found is not None and found["terms"] is not None:
            self._corner_t = (found["t_low"], found["t_high"])
            self._corner_matrix_id = matrix_id
            self._apply_corners(found)
            self._measure_sides(found, frame)
            return
        if self._corner_t is not None and self._corner_matrix_id == matrix_id:
            geometry = _line_frame(self.matrix, self.positive_x, self.pitch)
            if geometry is not None:
                foot, along, perp = geometry
                t_low, t_high = self._corner_t
                terms = _width_terms(t_low, t_high, float(self.pitch.width) / 2.0)
                if terms is not None:
                    restored = {
                        "terms": terms,
                        "perp": perp,
                        "foot": foot,
                        "along": along,
                        "corner_low": foot + along * t_low,
                        "corner_high": foot + along * t_high,
                    }
                    self._apply_corners(restored)
                    self._measure_sides(restored, frame)
                    return
        if found is not None:
            self._measure_sides(found, frame)

    def _publish(self, decision, white_line_mask, frame=None):
        self._refresh_corners(white_line_mask, frame)
        decision["side_lines"] = self.side_lines
        decision["side_anchor"] = self.side_anchor
        if self.corner_low is None or self.corner_high is None or self.corner_perp is None:
            return decision
        decision["corners"] = (
            (float(self.corner_low[0]), float(self.corner_low[1])),
            (float(self.corner_high[0]), float(self.corner_high[1])),
        )
        decision["corner_perp"] = (float(self.corner_perp[0]), float(self.corner_perp[1]))
        if self.side_px is None:
            decision["side_px"] = None
        elif isinstance(self.side_px, tuple):
            decision["side_px"] = tuple(float(value) for value in self.side_px)
        else:
            decision["side_px"] = float(self.side_px)
        return decision

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
            "region": self.region if accepted else None,
            "goal_end": (
                "centre"
                if self.region == "centre"
                else "high_x"
                if self.positive_x
                else "low_x"
                if self.positive_x is False
                else None
            ),
            "corners": None,
            "corner_perp": None,
            "side_px": None,
            "side_lines": None,
            "side_anchor": None,
        }
