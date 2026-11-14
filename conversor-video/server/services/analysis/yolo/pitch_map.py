"""Project the visible penalty third, and nothing past it.

The fit uses the painted 16.5 m line and the 9.15 m arc, which fixes the scale
at the penalty spot 5.5 m behind that line. Coordinates are published inside
the 35 m third of the goal the calibration is looking at. The grass probe
accepts or rejects that calibration. It does not erase a player or the ball
who already land inside the third. A short gap keeps the last good matrix,
then drops it. A new fit that flips the same line, or jumps the scale while
the line stays put, is refused. Possession and passes never read this matrix.
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
            return self._accept(report, grass_mask, fresh_view=fresh_view)
        if coherent:
            self.held_frames += 1
            self.hold_left = self.hold_frames
            self.reasons["held"] += 1
            return self._decision(True, True, "held")
        # A few bad frames keep the last good matrix. The counter does not refill.
        if self.matrix is not None and self.hold_left > 0:
            self.hold_left -= 1
            self.held_frames += 1
            self.reasons["bridge"] += 1
            return self._decision(True, True, "held")
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
        for (image_x, image_y), (x, y) in zip(array.reshape(-1, 2), transformed):
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
        }
