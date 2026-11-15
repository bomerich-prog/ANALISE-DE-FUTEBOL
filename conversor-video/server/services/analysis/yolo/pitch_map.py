"""Project the visible penalty third, and nothing past it.

The fit uses the painted 16.5 m line and the 9.15 m arc, which fixes the scale
at the penalty spot 5.5 m behind that line. When the two ends of that line are
on the paint, the width follows those corners, so the far half and the near
half can have different pixel lengths. A side is the thin painted line that
leaves a corner toward the goal. It ends where it meets the goal line, the
point where three painted lines leave together, or at the picture edge when
the goal line is not in the picture. A side with no paint in the picture takes
the other side's angle. A player's width is where a line parallel to the side
meets the 16.5 m line. The arc scale is left alone.

When the arc fit misses and the matrix on screen has left the paint, an open
camera still shows the box in thin lines. Its four painted corners, or one
corner and the arc when the goal line is off the picture, give a true
perspective, checked point by point against the ink. The arc fit takes over
again as soon as it lands. Coordinates
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
    _penalty_lines,
    _side_matches_image,
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


_INK_STEP = 2.0
_BOX_COST_PX = 3.0
_SIDE_MIN_PX = 100.0
_SIDE_MIN_LIT = 0.8
_SIDE_OPEN_PX = 150.0
_SIDE_OPEN_LIT = 0.9


def line_ink(frame, grass, white_mask=None):
    """Thin bright marks on the grass. A far line is too faint for the white mask.

    The darkest channel is low on grass and high on paint. A top-hat keeps
    what is narrower than a line and drops shirts, boards and mowing stripes.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or grass is None:
        return None
    low = np.min(image, axis=2)
    top = cv2.morphologyEx(
        low, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    near = cv2.dilate(grass, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    ink = ((top >= 14) & (near > 0)).astype(np.uint8) * 255
    core = (cv2.distanceTransform(ink, cv2.DIST_L2, 3) >= 4.0).astype(np.uint8) * 255
    blobs = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ink = cv2.bitwise_and(ink, cv2.bitwise_not(blobs))
    if white_mask is not None:
        ink = cv2.bitwise_or(ink, white_mask)
    ink[: int(ink.shape[0] * CROWD_BAND), :] = 0
    return ink


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def _lit(ink, points):
    """Which sample points sit on the ink, and which are inside the picture."""
    height, width = ink.shape[:2]
    xs = np.rint(points[..., 0]).astype(np.int64)
    ys = np.rint(points[..., 1]).astype(np.int64)
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    lit = np.zeros(xs.shape, dtype=bool)
    lit[inside] = ink[ys[inside], xs[inside]] > 0
    return lit, inside


def _runs(lit, inside, gap):
    """Each ray's painted run from its start. A long hole or the picture edge ends it."""
    count = lit.shape[-1]
    index = np.arange(count)
    lit = lit & inside
    last_lit = np.maximum.accumulate(np.where(lit, index, -1), axis=-1)
    broken = ((index - last_lit) > gap) | (~inside)
    stop = np.where(broken.any(axis=-1), broken.argmax(axis=-1), count)
    before = np.clip(stop - 1, 0, count - 1)
    last = np.take_along_axis(last_lit, before[..., None], axis=-1)[..., 0]
    last = np.where(stop == 0, -1, last)
    total = np.cumsum(lit, axis=-1)
    painted = np.take_along_axis(total, np.clip(last, 0, count - 1)[..., None], axis=-1)[..., 0]
    painted = np.where(last >= 0, painted, 0)
    left_picture = np.take_along_axis(~inside, np.clip(stop, 0, count - 1)[..., None], axis=-1)[..., 0]
    left_picture = left_picture & (stop < count)
    return last, painted, left_picture


def _ray_runs(ink, origins, directions, step=_INK_STEP, reach=900.0, gap_px=36.0, skip=8.0):
    """Painted length and painted share of every ray, origins by directions."""
    distance = np.arange(skip, reach, step)
    points = (
        origins[:, None, None, :]
        + directions[None, :, None, :] * distance[None, None, :, None]
    )
    lit, inside = _lit(ink, points)
    last, painted, left_picture = _runs(lit, inside, int(gap_px / step))
    length = np.where(last >= 0, distance[np.clip(last, 0, len(distance) - 1)], 0.0)
    share = np.where(last >= 0, painted / np.maximum(last + 1, 1), 0.0)
    return length, share, left_picture


def _ray_run(ink, origin, direction, **options):
    length, share, left_picture = _ray_runs(
        ink,
        np.asarray(origin, dtype=np.float64)[None, :],
        np.asarray(direction, dtype=np.float64)[None, :],
        **options,
    )
    return float(length[0, 0]), float(share[0, 0]), bool(left_picture[0, 0])


def _fit_ink_line(ink, origin, direction, start, stop, band):
    """Least squares on the ink beside a ray. The ray itself when there is too little."""
    ys, xs = np.where(ink > 0)
    delta = np.stack([xs, ys], axis=1).astype(np.float64) - origin
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    forward = delta @ direction
    keep = (forward >= start) & (forward <= stop) & (np.abs(delta @ normal) <= band)
    if int(np.count_nonzero(keep)) < 20:
        return origin, direction
    points = (delta[keep] + origin).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    fitted = _unit([float(vx), float(vy)])
    if float(fitted @ direction) < 0.0:
        fitted = -fitted
    return np.array([float(x0), float(y0)], dtype=np.float64), fitted


def _meet(first_point, first_direction, second_point, second_direction):
    """Distance along the first line to where it meets the second, or None."""
    try:
        solved = np.linalg.solve(
            np.column_stack([first_direction, -np.asarray(second_direction)]),
            np.asarray(second_point, dtype=np.float64) - first_point,
        )
    except np.linalg.LinAlgError:
        return None
    return float(solved[0])


def painted_front_line(ink, foot, along, half=650.0, reach=36.0, turn=5.0):
    """The painted 16.5 m line beside the calibrated one.

    A held matrix lags the camera by a few pixels. The sides have to leave the
    paint, so the line is searched again a few pixels and degrees around it.
    """
    ys, xs = np.where(ink > 0)
    delta = np.stack([xs, ys], axis=1).astype(np.float64) - foot
    normal = np.array([-along[1], along[0]], dtype=np.float64)
    wide = reach + half * float(np.tan(np.radians(turn)))
    delta = delta[(np.abs(delta @ along) <= half) & (np.abs(delta @ normal) <= wide)]
    if len(delta) < 80:
        return foot, along
    base = float(np.degrees(np.arctan2(along[1], along[0])))
    best = None
    for turned in np.arange(-turn, turn + 0.01, 0.5):
        angle = np.radians(base + turned)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        across = np.array([-direction[1], direction[0]], dtype=np.float64)
        bins = np.bincount(
            np.clip(np.rint(delta @ across + 200.0).astype(np.int64), 0, 400), minlength=401
        )
        bins = bins[:-2] + bins[1:-1] + bins[2:]
        low = int(200 - reach)
        peak = int(np.argmax(bins[low: int(200 + reach)])) + low
        if best is None or bins[peak] > best[0]:
            best = (int(bins[peak]), direction, across, float(peak + 1 - 200.0))
    votes, direction, across, offset = best
    if votes < 80:
        return foot, along
    keep = np.abs(delta @ across - offset) <= 3.0
    if int(np.count_nonzero(keep)) < 60:
        return foot, along
    points = (delta[keep] + foot).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    fitted = _unit([float(vx), float(vy)])
    if float(fitted @ along) < 0.0:
        fitted = -fitted
    origin = np.array([float(x0), float(y0)], dtype=np.float64)
    return origin + fitted * float((foot - origin) @ fitted), fitted


def _side_fan(along, perp, step):
    """Directions that leave the 16.5 m line toward the goal."""
    fan = []
    for degree in np.arange(-180.0, 180.0, step):
        direction = np.array(
            [np.cos(np.radians(degree)), np.sin(np.radians(degree))], dtype=np.float64
        )
        if float(direction @ perp) < 0.12:
            continue
        if abs(float(direction[0] * along[1] - direction[1] * along[0])) < np.sin(np.radians(14.0)):
            continue
        fan.append(direction)
    return np.array(fan, dtype=np.float64)


def side_from_line(ink, wide, foot, along, perp, sign, start_px):
    """The painted side that leaves the 16.5 m line on one side of the arc.

    Every painted station of the line is a possible corner, and the corner is
    where the longest thin line leaves toward the goal. The touchline never
    touches the 16.5 m line, and a player on it is a short blob.
    """
    height, width = wide.shape[:2]
    distance = np.arange(float(start_px), 1500.0, _INK_STEP)
    points = foot[None, :] + sign * distance[:, None] * along[None, :]
    lit, inside = _lit(wide, points)
    painted = np.cumsum(lit & inside)
    # The corner is on the paint of the 16.5 m line. Past its end the line
    # would only cross the touchline, which is longer than any side.
    last, _count, _left = _runs(lit[None, :], inside[None, :], int(60.0 / _INK_STEP))
    stop = int(last[0]) + int(12.0 / _INK_STEP)
    stations = []
    for index in range(0, min(len(distance), stop + 1), 3):
        point = points[index]
        if not (12.0 <= point[0] < width - 12.0 and 12.0 <= point[1] < height - 12.0):
            break
        if index >= 10 and painted[index] / (index + 1.0) < 0.6:
            continue
        stations.append(sign * float(distance[index]))
    fan = _side_fan(along, perp, 2.0)
    if not stations or len(fan) == 0:
        return None
    stations = np.array(stations, dtype=np.float64)
    length, share, _left = _ray_runs(
        wide, foot[None, :] + stations[:, None] * along[None, :], fan, step=4.0
    )
    score = np.where((length >= _SIDE_MIN_PX) & (share >= _SIDE_MIN_LIT), length * share, 0.0)
    if float(score.max()) <= 0.0:
        return None
    at, toward = np.unravel_index(int(np.argmax(score)), score.shape)
    near = stations[at] + np.arange(-6.0, 6.1, 2.0)
    base = float(np.degrees(np.arctan2(fan[toward][1], fan[toward][0])))
    close = np.array(
        [[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in base + np.arange(-2.5, 2.6, 0.5)],
        dtype=np.float64,
    )
    length, share, _left = _ray_runs(wide, foot[None, :] + near[:, None] * along[None, :], close)
    at, toward = np.unravel_index(int(np.argmax(length * share)), length.shape)
    origin = foot + along * float(near[at])
    point, direction = _fit_ink_line(ink, origin, close[toward], 10.0, float(length[at, toward]), 4.0)
    station = _meet(foot, along, point, direction)
    if station is None:
        return None
    corner = foot + along * station
    run, lit_share, left_picture = _ray_run(wide, corner, direction)
    if run < _SIDE_MIN_PX or lit_share < _SIDE_MIN_LIT:
        return None
    return {
        "corner": corner,
        "station": float(station),
        "direction": direction,
        "length": run,
        "share": lit_share,
        "border": left_picture,
        "junction": None,
    }


def crossing_lines(ink, foot, along, perp, min_offset=40.0):
    """Painted lines that run beside the 16.5 m line on the goal side.

    The goal line is one of them. Perspective turns it a few degrees off the
    16.5 m line, so a small fan of angles is voted and each peak is fitted.
    """
    ys, xs = np.where(ink > 0)
    delta = np.stack([xs, ys], axis=1).astype(np.float64) - foot
    delta = delta[(delta @ perp) > min_offset * 0.5]
    if len(delta) < 50:
        return []
    base = float(np.degrees(np.arctan2(along[1], along[0])))
    peaks = []
    for turned in np.arange(-22.0, 22.1, 1.0):
        angle = np.radians(base + turned)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        across = np.array([-direction[1], direction[0]], dtype=np.float64)
        if float(across @ perp) < 0.0:
            across = -across
        bins = np.bincount(
            np.clip(np.rint((delta @ across) / 2.0).astype(np.int64), 0, 2000), minlength=2001
        )
        for peak in np.argsort(bins)[-6:]:
            if bins[peak] >= 60 and peak * 2.0 >= min_offset:
                peaks.append((int(bins[peak]), float(peak * 2.0), direction, across))
    peaks.sort(key=lambda item: -item[0])
    kept = []
    for votes, offset, direction, across in peaks:
        same = any(
            abs(offset - other[1]) < 14.0 and float(direction @ other[2]) > np.cos(np.radians(6.0))
            for other in kept
        )
        if same:
            continue
        kept.append((votes, offset, direction, across))
        if len(kept) >= 8:
            break
    lines = []
    for _votes, offset, direction, across in kept:
        keep = np.abs(delta @ across - offset) <= 3.0
        if int(np.count_nonzero(keep)) < 40:
            continue
        points = (delta[keep] + foot).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        lines.append((np.array([float(x0), float(y0)], dtype=np.float64), _unit([float(vx), float(vy)])))
    return lines


def side_junction(wide, side, lines):
    """Where the side meets the goal line: three painted lines leave one point.

    The side arrives, and the goal line goes on to both hands. The line of
    the small box is parallel too, but it is not painted where the side
    would cross it. The first painted crossing is the goal line.
    """
    best = None
    for point, direction in lines:
        reach = _meet(side["corner"], side["direction"], point, direction)
        if reach is None or reach < 60.0 or reach > side["length"] + 30.0:
            continue
        meeting = side["corner"] + side["direction"] * reach
        one, _share, _left = _ray_run(wide, meeting, direction, reach=200.0, gap_px=14.0, skip=6.0)
        other, _share, _left = _ray_run(wide, meeting, -direction, reach=200.0, gap_px=14.0, skip=6.0)
        both_hands = min(one, other) >= 30.0 and one + other >= 100.0
        if not both_hands and max(one, other) < 140.0:
            continue
        if best is None or reach < best[0]:
            best = (reach, meeting, direction)
    return None if best is None else (best[1], best[2])


def penalty_sides(frame, grass, foot, along, perp, start_px, white_mask=None, ink=None):
    """The painted sides of the penalty area, each from its corner.

    A side that reaches the goal line ends on that junction. A side whose
    goal line is off the picture runs to the picture edge. A short or broken
    line is not a side.
    """
    if ink is None:
        ink = line_ink(frame, grass, white_mask)
    if ink is None or foot is None or along is None or perp is None:
        return None
    wide = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    foot = np.asarray(foot, dtype=np.float64)
    foot, along = painted_front_line(ink, foot, _unit(along))
    perp = _unit(np.asarray(perp, dtype=np.float64) - along * float(np.dot(perp, along)))
    lines = None
    sides = {}
    for name, sign in (("low", -1.0), ("high", 1.0)):
        side = side_from_line(ink, wide, foot, along, perp, sign, start_px)
        if side is None:
            continue
        if lines is None:
            lines = crossing_lines(ink, foot, along, perp)
        met = side_junction(wide, side, lines)
        side["junction"] = None if met is None else met[0]
        side["goal_line"] = None if met is None else met[1]
        if side["junction"] is None:
            to_edge = side["border"] and side["length"] >= _SIDE_OPEN_PX
            open_run = side["length"] >= _SIDE_OPEN_PX and side["share"] >= _SIDE_OPEN_LIT
            if not (to_edge or open_run):
                continue
        sides[name] = side
    return {"foot": foot, "along": along, "perp": perp, "sides": sides}


def _painted_middle(ink, origin, direction):
    """Middle of the painted stretch of a line, as a point on it."""
    ys, xs = np.where(ink > 0)
    delta = np.stack([xs, ys], axis=1).astype(np.float64) - origin
    across = np.array([-direction[1], direction[0]], dtype=np.float64)
    along = (delta @ direction)[np.abs(delta @ across) <= 3.0]
    if len(along) < 60:
        return None
    low, high = np.percentile(along, [5.0, 95.0])
    return origin + direction * float((low + high) / 2.0)


def _line_end(ink, foot, along, sign, gap_px=40.0):
    """Station where the paint of the 16.5 m line stops inside the picture, or None."""
    height, width = ink.shape[:2]
    wide = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    distance = np.arange(0.0, 1500.0, _INK_STEP)
    points = foot[None, :] + sign * distance[:, None] * along[None, :]
    lit, inside = _lit(wide, points)
    last, painted, left_picture = _runs(lit[None, :], inside[None, :], int(gap_px / _INK_STEP))
    if int(last[0]) < 30 or bool(left_picture[0]):
        return None
    if painted[0] / (last[0] + 1.0) < 0.6:
        return None
    end = points[int(last[0])]
    if not (12.0 <= end[0] < width - 12.0 and 12.0 <= end[1] < height - 12.0):
        return None
    return sign * float(distance[int(last[0])])


def local_scale(matrix, positive_x, pitch=None):
    """Centimetres per pixel along the 16.5 m line, at its middle."""
    pitch = pitch or SoccerPitchConfiguration()
    front = _front_x(bool(positive_x), pitch)
    middle = float(pitch.width) / 2.0
    image = _project_field(matrix, [(front, middle - 200.0), (front, middle + 200.0)])
    if image is None:
        return None
    span = float(np.linalg.norm(image[1] - image[0]))
    return 400.0 / span if span > 1e-6 else None


def _box_template(pitch, positive_x):
    """Field points of the painted marks around the penalty area, with weights."""
    width = float(pitch.width)
    middle = width / 2.0
    front = _front_x(positive_x, pitch)
    goal = float(pitch.length) if positive_x else 0.0
    inward = -_goal_sign(positive_x)
    spot = goal + inward * 1100.0
    small = goal + inward * 550.0
    y0, y1 = middle - _BOX_HALF_CM, middle + _BOX_HALF_CM
    points = []
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 40):
        x = spot + inward * 915.0 * np.cos(angle)
        y = middle + 915.0 * np.sin(angle)
        if (x - front) * inward > 20.0:
            points.append((x, y, 2.0))
    arc_count = len(points)
    for step in np.linspace(0.0, 1.0, 14):
        points.append((front, y0 + (y1 - y0) * step, 1.0))
        points.append((goal, y0 + (y1 - y0) * step, 0.5))
        points.append((front + (goal - front) * step, y1, 1.0))
        points.append((front + (goal - front) * step, y0, 1.0))
        points.append((small, middle - 916.0 + 1832.0 * step, 1.0))
    for step in np.linspace(0.0, 1.0, 6):
        points.append((small + (goal - small) * step, middle - 916.0, 1.0))
        points.append((small + (goal - small) * step, middle + 916.0, 1.0))
    array = np.array(points, dtype=np.float64)
    return array[:, :2], array[:, 2], arc_count


def _template_cost(matrix, template, weights, arc_count, distance):
    """Weighted mean distance, in pixels, from the projected marks to the ink."""
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    homogeneous = np.column_stack([template, np.ones(len(template))]) @ inverse.T
    depth = homogeneous[:, 2]
    if np.any(np.abs(depth) < 1e-9):
        return None
    image = homogeneous[:, :2] / depth[:, None]
    height, width = distance.shape[:2]
    inside = (
        np.isfinite(image).all(axis=1)
        & (image[:, 0] >= 0) & (image[:, 0] < width)
        & (image[:, 1] >= height * CROWD_BAND) & (image[:, 1] < height)
    )
    if int(np.count_nonzero(inside[:arc_count])) < arc_count // 2 or int(np.count_nonzero(inside)) < 40:
        return None
    # A collapsed matrix drops every mark on one painted pixel. The box has to span the picture.
    spread = np.ptp(image[inside], axis=0)
    if float(max(spread[0], spread[1])) < 120.0 or float(min(spread[0], spread[1])) < 20.0:
        return None
    xs = image[inside, 0].astype(np.int64)
    ys = image[inside, 1].astype(np.int64)
    reach = np.minimum(distance[ys, xs], 20.0)
    total = float(np.sum(weights[inside]))
    if total <= 0.0:
        return None
    return float(np.sum(reach * weights[inside]) / total)


def _box_keeps_shape(matrix, positive_x, pitch):
    """False for a matrix that flattens the box: the arc on the line, or no depth."""
    width = float(pitch.width)
    middle = width / 2.0
    front = _front_x(positive_x, pitch)
    goal = float(pitch.length) if positive_x else 0.0
    outward = -_goal_sign(positive_x)
    points = _project_field(
        matrix,
        [
            (front, middle - _BOX_HALF_CM),
            (front, middle + _BOX_HALF_CM),
            (goal, middle - _BOX_HALF_CM),
            (front + outward * _D_DEPTH_CM, middle),
            (front, middle),
        ],
    )
    if points is None:
        return False
    low, high, goal_low, tip, centre = points
    # Same handedness as the arc fit: a mirrored box swaps the near and far touchlines.
    along_x = _project_field(matrix, [(front, middle), (front + 100.0, middle), (front, middle + 100.0)])
    if along_x is None:
        return False
    step_x = along_x[1] - along_x[0]
    step_y = along_x[2] - along_x[0]
    if float(step_x[0] * step_y[1] - step_x[1] * step_y[0]) <= 0.0:
        return False
    line = high - low
    length = float(np.linalg.norm(line))
    if length < 120.0:
        return False
    normal = np.array([-line[1], line[0]]) / length
    tip_out = abs(float((tip - centre) @ normal))
    depth = abs(float((goal_low - low) @ normal))
    return tip_out >= 12.0 and depth >= 25.0


def box_from_sides(measured, grass, ink, pitch=None):
    """Image-to-field matrix from the four painted corners of the penalty area.

    One side that ends on the goal line gives two corners: where it leaves
    the 16.5 m line and where it meets the goal line. The other two lie on
    those same two lines. When they are painted in the picture they are
    read. When the camera cuts them off they are the pair that puts the arc,
    the small box and the rest of the marks back on the paint. The result is
    a true perspective, and the 16.5 m line and the arc still have to pass
    the grass probe.
    """
    pitch = pitch or SoccerPitchConfiguration()
    if measured is None:
        return None
    sides = measured["sides"]
    anchor_name = None
    for name, side in sides.items():
        if side["junction"] is not None and (
            anchor_name is None or side["length"] > sides[anchor_name]["length"]
        ):
            anchor_name = name
    if anchor_name is None:
        return None
    anchor = sides[anchor_name]
    other_name = "high" if anchor_name == "low" else "low"
    sign = 1.0 if anchor_name == "low" else -1.0
    along = np.asarray(measured["along"], dtype=np.float64) * sign
    corner = anchor["corner"]
    meeting = anchor["junction"]
    goal_direction = np.asarray(anchor["goal_line"], dtype=np.float64)
    if float(goal_direction @ along) < 0.0:
        goal_direction = -goal_direction
    depth_px = float(np.linalg.norm(meeting - corner))
    if depth_px < 60.0:
        return None
    other = sides.get(other_name)
    station = _line_end(ink, measured["foot"], measured["along"], sign)
    painted = _painted_reach(ink, corner, along)
    if other is not None:
        fronts = [float((other["corner"] - corner) @ along)]
    elif station is not None:
        fronts = [float((measured["foot"] + measured["along"] * station - corner) @ along)]
    else:
        # The 16.5 m line runs off the picture. The corner is somewhere past the edge.
        fronts = list(np.geomspace(max(painted, 150.0), max(painted, 150.0) * 5.0, 22))
    if fronts[0] < 120.0:
        return None
    if other is not None and other["junction"] is not None:
        backs = [float((other["junction"] - meeting) @ goal_direction)]
    else:
        backs = None
    width = float(pitch.width)
    y_anchor = width / 2.0 - sign * _BOX_HALF_CM
    y_other = width / 2.0 + sign * _BOX_HALF_CM
    distance = cv2.distanceTransform(cv2.bitwise_not(ink), cv2.DIST_L2, 3)
    toward_goal = float((meeting - corner)[0])
    for positive_x in (toward_goal >= 0.0, toward_goal < 0.0):
        front = _front_x(positive_x, pitch)
        goal = float(pitch.length) if positive_x else 0.0
        field = np.array(
            [(front, y_anchor), (goal, y_anchor), (front, y_other), (goal, y_other)], dtype=np.float32
        )
        template, weights, arc_count = _box_template(pitch, positive_x)

        def solve(front_px, back_px):
            image = np.array(
                [corner, meeting, corner + along * front_px, meeting + goal_direction * back_px],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(image, field).astype(np.float64)
            if not np.isfinite(matrix).all() or not _box_keeps_shape(matrix, positive_x, pitch):
                return None
            cost = _template_cost(matrix, template, weights, arc_count, distance)
            return None if cost is None else (cost, matrix, front_px, back_px)

        best = None
        for front_px in fronts:
            ratios = [back / front_px for back in backs] if backs else np.geomspace(0.4, 1.6, 16)
            for ratio in ratios:
                found = solve(front_px, front_px * float(ratio))
                if found is not None and (best is None or found[0] < best[0]):
                    best = found
        if best is None:
            continue
        for spread in (0.06, 0.02, 0.007):
            front_steps = [1.0] if len(fronts) == 1 else [1.0 - spread, 1.0, 1.0 + spread]
            back_steps = [1.0] if backs else [1.0 - spread, 1.0, 1.0 + spread]
            _cost, _matrix, front_px, back_px = best
            for front_step in front_steps:
                for back_step in back_steps:
                    found = solve(front_px * front_step, back_px * back_step)
                    if found is not None and found[0] < best[0]:
                        best = found
        cost, matrix, front_px, back_px = best
        if cost > _BOX_COST_PX:
            continue
        # The middle of the 16.5 m line has to fall between its two ends.
        middle = _project_field(matrix, [(front, width / 2.0)])
        if middle is None:
            continue
        share = float((middle[0] - corner) @ along) / front_px
        if not (0.2 <= share <= 0.8) or not (0.4 <= back_px / front_px <= 2.0):
            continue
        if not _side_matches_image(matrix, positive_x, ink.shape[1]):
            continue
        if not _box_keeps_shape(matrix, positive_x, pitch):
            continue
        # The marks were already checked point by point against the ink. The
        # grass probe only measures how deep the picture goes.
        patch = paint_patch(matrix, positive_x, grass, None, pitch)
        if patch is None:
            continue
        scale = local_scale(matrix, positive_x, pitch)
        if scale is None:
            continue
        return {
            "matrix": matrix,
            "positive_x": bool(positive_x),
            "cm_per_px": scale,
            "region": "penalty",
            "projective": True,
            "toward_goal_cm": patch["toward_goal_cm"],
            "toward_arc_cm": patch["toward_arc_cm"],
            "area_px": 0.5 * (front_px + back_px) * depth_px,
            "cost": cost,
        }
    return None


def _painted_reach(ink, origin, direction, gap_px=40.0):
    """How far the paint of a line goes on from a point, in pixels."""
    wide = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    length, _share, _left = _ray_run(wide, origin, direction, reach=1500.0, gap_px=gap_px, skip=0.0)
    return length


_D_HALF_CM = float(np.sqrt(915.0**2 - 550.0**2))
_D_DEPTH_CM = 915.0 - 550.0


def _painted_d(ink, foot, along, perp):
    """The penalty arc as it leaves the 16.5 m line: its two ends and its curve.

    The arc is the thin curve on the pitch side of the line that starts and
    ends on it. Returns (end_low, end_high, curve points), or None.
    """
    ys, xs = np.where(ink > 0)
    delta = np.stack([xs, ys], axis=1).astype(np.float64) - foot
    out = -(delta @ perp)
    # Leave the 16.5 m line itself out. The arc keeps going right up to it.
    keep = (out > 6.0) & (out < 260.0)
    if int(np.count_nonzero(keep)) < 60:
        return None
    band = np.zeros_like(ink)
    band[ys[keep], xs[keep]] = 255
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(band, connectivity=8)
    member_of = labels[ys[keep], xs[keep]]
    kept_out = out[keep]
    kept_along = (delta @ along)[keep]
    kept_points = np.stack([xs[keep], ys[keep]], axis=1).astype(np.float64)
    best = None
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 60:
            continue
        member = member_of == index
        piece_out = kept_out[member]
        piece_along = kept_along[member]
        touching = piece_out < 16.0
        if int(np.count_nonzero(touching)) < 4 or float(piece_out.max()) < 25.0:
            continue
        low = float(piece_along[touching].min())
        high = float(piece_along[touching].max())
        if high - low < 60.0:
            continue
        # A curve: the far part of it lies between its two ends on the line.
        far = piece_out > 0.5 * float(piece_out.max())
        slack = 0.15 * (high - low)
        if not np.all((piece_along[far] > low - slack) & (piece_along[far] < high + slack)):
            continue
        size = float(stats[index, cv2.CC_STAT_AREA])
        if best is None or size > best[0]:
            points = kept_points[member]
            ends = []
            for end_along in (low, high):
                # The arc meets the line almost along it. Carry its last stretch on to the line.
                near = (piece_out < 40.0) & (np.abs(piece_along - end_along) < 0.35 * (high - low))
                end = foot + along * end_along
                if int(np.count_nonzero(near)) >= 12:
                    vx, vy, x0, y0 = cv2.fitLine(points[near].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
                    reach = _meet(foot, along, np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64))
                    if reach is not None and abs(reach - end_along) < 40.0:
                        end = foot + along * reach
                ends.append(end)
            best = (size, ends[0], ends[1], points[piece_out > 12.0])
    if best is None or len(best[3]) < 20:
        return None
    return best[1], best[2], best[3]


def box_from_arc(measured, grass, ink, pitch=None):
    """Image-to-field matrix from one corner and the painted arc.

    With the goal line off the picture the box cannot be closed. The arc
    still is: its ends sit 7.31 m each side of the middle of the 16.5 m line
    and its tip 3.65 m out. With a corner that is four known points.
    """
    pitch = pitch or SoccerPitchConfiguration()
    if measured is None or not measured["sides"]:
        return None
    foot = measured["foot"]
    along = measured["along"]
    perp = measured["perp"]
    arc = _painted_d(ink, foot, along, perp)
    if arc is None:
        return None
    end_low, end_high, curve = arc
    arc_middle = 0.5 * (float((end_low - foot) @ along) + float((end_high - foot) @ along))
    width = float(pitch.width)
    middle = width / 2.0
    distance = cv2.distanceTransform(cv2.bitwise_not(ink), cv2.DIST_L2, 3)
    picks = curve[np.linspace(0, len(curve) - 1, min(len(curve), 48)).astype(int)]
    best = None
    for name, side in measured["sides"].items():
        station = float((side["corner"] - foot) @ along)
        # A real corner is beyond the arc on its own hand.
        if name == "low" and station > float((end_low - foot) @ along) - 20.0:
            continue
        if name == "high" and station < float((end_high - foot) @ along) + 20.0:
            continue
        for positive_x in (True, False):
            front = _front_x(positive_x, pitch)
            outward = -_goal_sign(positive_x)
            template, weights, arc_count = _box_template(pitch, positive_x)
            for low_y in (middle - _D_HALF_CM, middle + _D_HALF_CM):
                high_y = 2.0 * middle - low_y
                corner_y = (middle - _BOX_HALF_CM) if low_y < middle else (middle + _BOX_HALF_CM)
                if name == "high":
                    corner_y = 2.0 * middle - corner_y
                ends = np.array([side["corner"], end_low, end_high], dtype=np.float64)
                field_ends = [(front, corner_y), (front, low_y), (front, high_y), (front + outward * _D_DEPTH_CM, middle)]
                inward = -outward

                def solve(tip, reach_px, reach_cm):
                    # Three marks lie on the 16.5 m line. The side adds one point off it.
                    on_side = side["corner"] + side["direction"] * reach_px
                    image = np.vstack([ends, [tip], [on_side]]).astype(np.float32)
                    field = np.array(field_ends + [(front + inward * reach_cm, corner_y)], dtype=np.float32)
                    matrix, _mask = cv2.findHomography(image, field, 0)
                    if matrix is None or not np.isfinite(matrix).all():
                        return None
                    if not _box_keeps_shape(matrix, positive_x, pitch):
                        return None
                    cost = _template_cost(matrix, template, weights, arc_count, distance)
                    return None if cost is None else (cost, matrix, tip, reach_cm)

                reach_px = min(float(side["length"]), 200.0)
                found = None
                for tip in picks[:: max(1, len(picks) // 16)]:
                    for reach_cm in np.geomspace(250.0, 1650.0, 10):
                        tried = solve(tip, reach_px, float(reach_cm))
                        if tried is not None and (found is None or tried[0] < found[0]):
                            found = tried
                if found is None:
                    continue
                for step in (4.0, 2.0, 1.0):
                    for dx in (-step, 0.0, step):
                        for dy in (-step, 0.0, step):
                            for grow in (1.0 - step / 40.0, 1.0, 1.0 + step / 40.0):
                                tried = solve(found[2] + np.array([dx, dy]), reach_px, found[3] * grow)
                                if tried is not None and tried[0] < found[0]:
                                    found = tried
                cost, matrix, _tip, _reach = found
                if cost > _BOX_COST_PX:
                    continue
                if not _side_matches_image(matrix, positive_x, ink.shape[1]):
                    continue
                if not _box_keeps_shape(matrix, positive_x, pitch):
                    continue
                goal_side = _project_field(matrix, [(front - outward * 500.0, middle)])
                if goal_side is None or float((goal_side[0] - (foot + along * arc_middle)) @ perp) <= 0.0:
                    continue
                if best is None or cost < best[0]:
                    best = (cost, matrix, positive_x)
    if best is None:
        return None
    cost, matrix, positive_x = best
    patch = paint_patch(matrix, positive_x, grass, None, pitch)
    scale = local_scale(matrix, positive_x, pitch)
    if patch is None or scale is None:
        return None
    return {
        "matrix": matrix,
        "positive_x": bool(positive_x),
        "cm_per_px": scale,
        "region": "penalty",
        "projective": True,
        "toward_goal_cm": patch["toward_goal_cm"],
        "toward_arc_cm": patch["toward_arc_cm"],
        "area_px": 0.0,
        "cost": cost,
    }


def fit_painted_box(frame, grass, white_mask=None, pitch=None, seed=None, ink=None):
    """The whole penalty area on the paint, without the arc fit.

    An open camera makes the lines too thin for the white mask and the arc
    too small for its fit, and the box is all in the picture. The thin marks
    still show the 16.5 m line, both sides and the goal line. seed is the
    16.5 m line of the matrix on screen. Without it every long painted line
    is tried from both hands, and the largest box that holds the arc wins.
    Returns the report and the measured sides, either of which may be None.
    """
    pitch = pitch or SoccerPitchConfiguration()
    if ink is None:
        ink = line_ink(frame, grass, white_mask)
    if ink is None:
        return None, None
    if seed is not None:
        foot, along, perp, start_px = seed
        measured = penalty_sides(frame, grass, foot, along, perp, start_px, ink=ink)
        report = box_from_sides(measured, grass, ink, pitch)
        if report is None:
            report = box_from_arc(measured, grass, ink, pitch)
        return report, measured
    paint = _clean_paint(ink)
    if paint is None:
        return None, None
    best = None
    for origin, direction in _penalty_lines(paint)[:3]:
        middle = _painted_middle(ink, origin, direction)
        if middle is None:
            continue
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        for hand in (normal, -normal):
            measured = penalty_sides(frame, grass, middle, direction, hand, 0.0, ink=ink)
            report = box_from_sides(measured, grass, ink, pitch)
            if report is None:
                report = box_from_arc(measured, grass, ink, pitch)
            if report is not None and (best is None or (report["area_px"], -report["cost"]) > (best[0]["area_px"], -best[0]["cost"])):
                best = (report, measured)
    return best if best is not None else (None, None)


def _picture_exit(origin, direction, shape):
    """Where a ray leaves the grass part of the picture."""
    height, width = shape[:2]
    reach = []
    for value, step, low, high in (
        (float(origin[0]), float(direction[0]), 0.0, width - 1.0),
        (float(origin[1]), float(direction[1]), CROWD_BAND * height, height - 1.0),
    ):
        if abs(step) < 1e-9:
            continue
        reach.append(((high if step > 0 else low) - value) / step)
    reach = [value for value in reach if value > 0.0]
    if not reach:
        return np.asarray(origin, dtype=np.float64)
    return np.asarray(origin, dtype=np.float64) + np.asarray(direction, dtype=np.float64) * min(reach)


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
    front_line=None,
):
    """Draw the 16.5 m line, the arc, and the sides that stay on the grass.

    A measured side replaces the projected one: side_anchor names it, low,
    high or both. front_line is the 16.5 m line cut at a measured corner.
    """
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
    front_points = [(front, y0 + (y1 - y0) * step) for step in np.linspace(0, 1, 24)]
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
        if front_line is not None and len(front_line) == 2:
            _image_segment(frame, front_line[0], front_line[1], color)
        else:
            _polyline(frame, inverse, np.array(front_points, dtype=np.float32), color)
        # A measured side replaces its own projection. The other side stays as it was.
        if side_anchor not in ("high", "both"):
            _polyline(frame, inverse, np.array(near_side, dtype=np.float32), color)
        if side_anchor not in ("low", "both"):
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
    steps = max(16, int(float(np.linalg.norm(end - start)) / 8.0) + 1)
    for step in np.linspace(0.0, 1.0, steps):
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
        self.front_line = None
        self._width_foot = None
        self._width_along = None
        self._drawn_corners = None
        self._side_memory = None
        self._corner_t = None
        self._corner_matrix_id = None
        self.projective = False
        self._side_cache = None
        self._ink_cache = None
        self.accepted_frames = 0
        self.held_frames = 0
        self.rejected_frames = 0
        self.reasons = Counter()

    def update(self, white_line_mask, frame=None, grass_mask=None):
        self._side_cache = None
        ink = self._ink(white_line_mask, frame, grass_mask)
        # A matrix read off the thin marks is checked against the thin marks.
        marks = ink if (self.projective and ink is not None) else white_line_mask
        report = fit_broadcast_penalty(white_line_mask, self.pitch, frame)
        if report is not None and not (SANE_CM_MIN <= float(report["cm_per_px"]) <= SANE_CM_MAX):
            self.reasons["absurd"] += 1
            report = None
        if report is not None and grass_mask is not None:
            patch = paint_patch(report["matrix"], report["positive_x"], grass_mask, white_line_mask, self.pitch)
            if patch is None:
                self.reasons["off_pitch"] += 1
                report = None
            else:
                report["toward_goal_cm"] = patch["toward_goal_cm"]
                report["toward_arc_cm"] = patch["toward_arc_cm"]
        boxed = False
        if report is None and ink is not None and not self._geometry_holds(marks, grass_mask):
            # The arc fit missed and the matrix on screen has left the paint.
            # An open camera still shows the whole box in thin lines.
            report = self._box_report(white_line_mask, frame, grass_mask)
            if report is not None and not (SANE_CM_MIN <= float(report["cm_per_px"]) <= SANE_CM_MAX):
                self.reasons["absurd"] += 1
                report = None
            boxed = report is not None
        if report is None:
            centre = fit_broadcast_centre(white_line_mask, self.pitch, frame)
            if centre is not None and not (SANE_CM_MIN <= float(centre["cm_per_px"]) <= SANE_CM_MAX):
                self.reasons["absurd"] += 1
                centre = None
            if centre is not None and centre_on_paint(centre["matrix"], grass_mask, white_line_mask, self.pitch):
                report = centre
            elif centre is not None:
                self.reasons["off_pitch"] += 1
        coherent = self._geometry_holds(marks, grass_mask)
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
            elif self.projective and not boxed and bool(report["positive_x"]) == bool(self.positive_x):
                # The box only stands in while the arc fit misses.
                fresh_view = True
                self.reasons["arc_back"] += 1
            elif boxed and bool(report["positive_x"]) == bool(self.positive_x):
                # The matrix on screen has left the paint and the box was checked
                # point by point on this frame. A zoom is not a scale jump here.
                fresh_view = True
                self.reasons["box_replaces"] += 1
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
            if boxed and report.get("projective"):
                self.reasons["boxed"] += 1
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
            "boxed_frames": int(self.reasons.get("boxed", 0)),
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
        self.projective = bool(report.get("projective"))
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

    def _ink(self, white_line_mask, frame, grass_mask):
        """Thin marks of this frame, read once."""
        if frame is None or grass_mask is None:
            return None
        if self._ink_cache is None or self._ink_cache[0] is not frame:
            self._ink_cache = (frame, line_ink(frame, grass_mask, white_line_mask))
        return self._ink_cache[1]

    def _box_report(self, white_line_mask, frame, grass_mask):
        """The four painted corners of the penalty area, when they can be read."""
        ink = self._ink(white_line_mask, frame, grass_mask)
        if ink is None:
            return None
        seed = None
        if self.matrix is not None and self.region != "centre" and self.scale:
            geometry = _line_frame(self.matrix, self.positive_x, self.pitch)
            if geometry is not None:
                seed = (geometry[0], geometry[1], geometry[2], _ARC_MEET_CM / float(self.scale) + 24.0)
        report, measured = fit_painted_box(frame, grass_mask, white_line_mask, self.pitch, seed, ink)
        if report is None and seed is not None and not self._arc_on_ink(ink, grass_mask):
            # The matrix on screen has left the paint. Look at every long line again.
            report, measured = fit_painted_box(frame, grass_mask, white_line_mask, self.pitch, None, ink)
        self._side_cache = (frame, measured) if measured is not None else None
        return report

    def _arc_on_ink(self, ink, grass_mask):
        return paint_patch(self.matrix, self.positive_x, grass_mask, ink, self.pitch) is not None

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
        self.front_line = None
        self._width_foot = None
        self._width_along = None
        self._drawn_corners = None
        self._side_memory = None
        self._corner_t = None
        self._corner_matrix_id = None
        self.projective = False
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

    def _turn_from(self, along, direction):
        """Signed angle, in radians, from the 16.5 m line to a side."""
        return float(
            np.arctan2(
                along[0] * direction[1] - along[1] * direction[0],
                float(np.dot(along, direction)),
            )
        )

    def _remembered_direction(self, along):
        """The last measured side, turned with the 16.5 m line, for a few frames."""
        if self._side_memory is None:
            return None
        turn, generation, left = self._side_memory
        if generation != self.generation or left <= 0:
            self._side_memory = None
            return None
        self._side_memory = (turn, generation, left - 1)
        cos, sin = float(np.cos(turn)), float(np.sin(turn))
        return np.array(
            [along[0] * cos - along[1] * sin, along[0] * sin + along[1] * cos],
            dtype=np.float64,
        )

    def _measure_sides(self, found, frame, white_line_mask=None):
        """Put each side on its paint, from its corner to the goal line.

        The side ends where it meets the goal line. With the goal line off the
        picture it runs to the picture edge. When only one side is painted in
        the picture, the other corner takes the same angle and runs to the
        edge. One corner alone draws one side, and the rest stays projected.
        """
        self.side_direction = None
        self.side_lines = None
        self.side_anchor = None
        self.front_line = None
        self._width_foot = None
        self._width_along = None
        self._drawn_corners = None
        if frame is None or self.grass is None or not self.scale:
            return
        geometry = _line_frame(self.matrix, self.positive_x, self.pitch)
        if geometry is None:
            return
        if self._side_cache is not None and self._side_cache[0] is frame:
            measured = self._side_cache[1]
        else:
            start_px = _ARC_MEET_CM / float(self.scale) + 24.0
            measured = penalty_sides(
                frame, self.grass, geometry[0], geometry[1], geometry[2], start_px, _clean_paint(white_line_mask)
            )
        if measured is None:
            return
        sides = measured["sides"]
        along = measured["along"]
        segments = {}
        for name, side in sides.items():
            if side["junction"] is not None:
                end = side["junction"]
            elif side["border"]:
                end = _picture_exit(side["corner"], side["direction"], frame.shape)
            else:
                end = side["corner"] + side["direction"] * side["length"]
            segments[name] = self._point_pair(side["corner"], end)
        best = None
        for side in sides.values():
            rank = (side["junction"] is not None, side["length"])
            if best is None or rank > best[0]:
                best = (rank, side["direction"])
        if best is not None:
            direction = best[1]
            self._side_memory = (self._turn_from(along, direction), self.generation, self.hold_frames)
        else:
            direction = self._remembered_direction(along)
        both = found is not None and found.get("terms") is not None
        if both:
            if direction is None:
                return
            self.side_direction = direction
            self._width_foot = np.asarray(found["foot"], dtype=np.float64)
            self._width_along = np.asarray(found["along"], dtype=np.float64)
            drawn = {}
            lines = []
            for name in ("low", "high"):
                if name in segments:
                    drawn[name] = sides[name]["corner"]
                    lines.append(segments[name])
            for name in ("low", "high"):
                if name in segments:
                    continue
                origin = np.asarray(found["corner_" + name], dtype=np.float64)
                drawn[name] = origin
                copied = self._side_segment(
                    frame, origin, _picture_exit(origin, direction, frame.shape)
                )
                if copied is not None:
                    lines.append(copied)
            self._drawn_corners = (drawn["low"], drawn["high"])
            self.side_lines = lines or None
            return
        if not segments:
            return
        self.side_anchor = "both" if len(segments) == 2 else next(iter(segments))
        self.side_lines = [segments[name] for name in ("low", "high") if name in segments]
        # The 16.5 m line stops at a measured corner and keeps its projected end on the other hand.
        reach = _BOX_HALF_CM / float(self.scale)
        ends = []
        for name, sign in (("low", -1.0), ("high", 1.0)):
            if name in sides:
                ends.append(sides[name]["corner"])
            else:
                ends.append(measured["foot"] + along * sign * reach)
        self.front_line = self._point_pair(ends[0], ends[1])

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
        self.front_line = None
        self._width_foot = None
        self._width_along = None
        self._drawn_corners = None
        if self.matrix is None or self.positive_x is None or self.region == "centre" or not self.scale:
            self._corner_t = None
            self._corner_matrix_id = None
            return
        if self.projective:
            # A true perspective needs no width correction. The sides are drawn where they are painted.
            self._corner_t = None
            self._corner_matrix_id = None
            self._measure_sides(None, frame, white_line_mask)
            return
        paint = _clean_paint(white_line_mask)
        found = _line_ends(paint, self.matrix, self.positive_x, self.scale, self.pitch)
        matrix_id = id(self.matrix)
        if found is not None and found["terms"] is not None:
            self._corner_t = (found["t_low"], found["t_high"])
            self._corner_matrix_id = matrix_id
            self._apply_corners(found)
            self._measure_sides(found, frame, white_line_mask)
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
                    self._measure_sides(restored, frame, white_line_mask)
                    return
        self._measure_sides(None, frame, white_line_mask)

    def _publish(self, decision, white_line_mask, frame=None):
        self._refresh_corners(white_line_mask, frame)
        decision["side_lines"] = self.side_lines
        decision["side_anchor"] = self.side_anchor
        decision["front_line"] = self.front_line
        if self.corner_low is None or self.corner_high is None or self.corner_perp is None:
            return decision
        # The drawing starts each side on its measured corner. The width keeps the line ends.
        low, high = self._drawn_corners or (self.corner_low, self.corner_high)
        decision["corners"] = (
            (float(low[0]), float(low[1])),
            (float(high[0]), float(high[1])),
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
            "front_line": None,
        }
