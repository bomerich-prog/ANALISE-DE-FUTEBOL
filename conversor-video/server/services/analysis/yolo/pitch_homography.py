"""Accept a pitch homography only when its lines sit on the white paint.

The keypoint model labels a landmark by index and ViewTransformer maps that
index straight onto the pitch template. On a broadcast frame the model often
returns a handful of right-penalty indexes sitting on the wrong lines, or on
the advertising boards. Those points still agree with each other, so a plain
least-squares fit (and RANSAC alone) produces a matrix with a small residual
and a penalty box drawn in open grass.

A mirror of the whole template is itself a homography, so it draws the same
curves in the image. Line overlap is what tells a registered pitch from a
plausible but misplaced one. A frame that fails stays without field
coordinates. A frame that passed keeps being used only while its lines still
cover the paint, which is how a still camera survives a few missed detections.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from roboflow_soccer_config import SoccerPitchConfiguration
from roboflow_view import ViewTransformer

MIN_INLIERS = 6
MIN_LINE_SUPPORT = 0.40
RANSAC_REPROJ_CM = 250.0
HOLD_FRAMES = 15
MIN_SPAN_PX = 100.0
MIN_SAMPLES_INSIDE = 40
_LINE_DILATION = 9
_KEYPOINT_LINE_RADIUS_PX = 18
# The painted penalty area is 16.5m deep and the arc is 9.15m about the
# 11m spot. The keypoint template's 20.15m box never meets that arc, so a
# broadcast frame has to be registered from the paint itself.
_PENALTY_DEPTH_CM = 1650.0
_SPOT_FROM_GOAL_CM = 1100.0
_ARC_RADIUS_CM = 915.0
_SPOT_FROM_FRONT_CM = _PENALTY_DEPTH_CM - _SPOT_FROM_GOAL_CM


def _clean_paint(white_line_mask):
    """Drop player-sized blobs. A pitch line is only a few pixels thick."""
    if white_line_mask is None or white_line_mask.size == 0:
        return None
    if int(np.count_nonzero(white_line_mask)) < 80:
        return None
    distance = cv2.distanceTransform(white_line_mask, cv2.DIST_L2, 3)
    core = np.zeros_like(white_line_mask)
    core[distance >= 5.5] = 255
    players = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    paint = cv2.bitwise_and(white_line_mask, cv2.bitwise_not(players))
    # Advertising boards sit above the grass and look like a long white line.
    paint[: int(paint.shape[0] * 0.14), :] = 0
    if int(np.count_nonzero(paint)) < 80:
        return None
    return paint


def _line_segments(paint):
    edges = cv2.Canny(paint, 40, 120)
    found = cv2.HoughLinesP(edges, 1, np.pi / 180, 36, minLineLength=70, maxLineGap=22)
    if found is None:
        return []
    rows = []
    for x1, y1, x2, y2 in found.reshape(-1, 4):
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
        rows.append((length, angle, float(x1), float(y1), float(x2), float(y2)))
    return rows


def _penalty_line(paint):
    """The long painted edge of the penalty area, as a point and a unit direction."""
    families = []
    for segment in _line_segments(paint):
        length, angle, x1, y1, x2, y2 = segment
        if length < 80 or not (16 <= abs(angle) <= 55):
            continue
        if min(y1, y2) < paint.shape[0] * 0.28:
            continue
        midpoint = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], np.float64)
        for family in families:
            if abs(angle - family[0][1]) >= 8:
                continue
            # Parallel edges of the box share an angle. Only a collinear run is one line.
            _, _, ax, ay, bx, by = family[0]
            family_direction = np.array([bx - ax, by - ay], np.float64)
            family_norm = float(np.linalg.norm(family_direction))
            if family_norm < 1e-6:
                continue
            family_direction /= family_norm
            delta = midpoint - np.array([ax, ay], np.float64)
            offset = abs(float(delta[0] * family_direction[1] - delta[1] * family_direction[0]))
            if offset < 14:
                family.append(segment)
                break
        else:
            families.append([segment])
    if not families:
        return None
    family = max(families, key=lambda items: sum(item[0] for item in items))
    if sum(item[0] for item in family) < 220:
        return None
    points = []
    for _, _, x1, y1, x2, y2 in family:
        points.extend(((x1, y1), (x2, y2)))
    vx, vy, x0, y0 = cv2.fitLine(np.array(points, np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([float(vx), float(vy)], np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction /= norm
    if direction[1] < 0:
        direction = -direction
    return np.array([float(x0), float(y0)], np.float64), direction


def _circle_through(points):
    """Circle through three points. Six-point fits never land on this thin arc."""
    design = np.stack([points[:, 0], points[:, 1], np.ones(3)], axis=1)
    try:
        linear, eccentricity, constant = np.linalg.solve(design, -(points[:, 0] ** 2 + points[:, 1] ** 2))
    except np.linalg.LinAlgError:
        return None
    center_x = -linear / 2
    center_y = -eccentricity / 2
    radius2 = center_x * center_x + center_y * center_y - constant
    if radius2 <= 1:
        return None
    return np.array([float(center_x), float(center_y)], np.float64), float(np.sqrt(radius2))


def _painted_arc_distance(matrix, paint, positive_x):
    """Median distance, in pixels, from the painted penalty arc to the mask.

    The front line is on the paint by construction, so only the arc outside
    the box is scored. None when that arc leaves the frame.
    """
    try:
        inverse = np.linalg.inv(matrix).astype(np.float32)
    except np.linalg.LinAlgError:
        return None
    length = 12000.0
    front = length - _PENALTY_DEPTH_CM if positive_x else _PENALTY_DEPTH_CM
    spot = length - _SPOT_FROM_GOAL_CM if positive_x else _SPOT_FROM_GOAL_CM
    samples = []
    for angle in np.linspace(0, 2 * np.pi, 48):
        x = spot + _ARC_RADIUS_CM * np.cos(angle)
        y = 3500.0 + _ARC_RADIUS_CM * np.sin(angle)
        outside = x < front if positive_x else x > front
        if outside:
            samples.append((x, y))
    projected = cv2.perspectiveTransform(np.array(samples, np.float32).reshape(-1, 1, 2), inverse).reshape(-1, 2)
    height, width = paint.shape[:2]
    distance = cv2.distanceTransform(cv2.bitwise_not(paint), cv2.DIST_L2, 3)
    inside = []
    for x, y in projected:
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height:
            inside.append((float(x), float(y), distance[int(y), int(x)]))
    if len(inside) < 10:
        return None
    xs = [point[0] for point in inside]
    ys = [point[1] for point in inside]
    # A handful of pixels on a sock is not the arc.
    if max(xs) - min(xs) < 70 and max(ys) - min(ys) < 70:
        return None
    return float(np.median([point[2] for point in inside]))


def _affine_matrix(origin, along, perp, px_per_cm, front_x, width_cm, positive_x):
    sign = 1.0 if positive_x else -1.0
    scale = float(px_per_cm)
    ox, oy = float(origin[0]), float(origin[1])
    px, py = float(perp[0]), float(perp[1])
    ax, ay = float(along[0]), float(along[1])
    return np.array(
        [
            [sign * px / scale, sign * py / scale, front_x - sign * (px * ox + py * oy) / scale],
            [ax / scale, ay / scale, width_cm / 2 - (ax * ox + ay * oy) / scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _arc_candidates(paint, origin, direction):
    """Circles centred on the penalty spot: 5.5m behind the line, radius 9.15m."""
    ys, xs = np.where(paint > 0)
    if len(xs) < 40:
        return []
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    offset = (points - origin)[:, 0] * direction[1] - (points - origin)[:, 1] * direction[0]
    beside = points[(np.abs(offset) > 14) & (np.abs(offset) < 240) & (points[:, 1] > paint.shape[0] * 0.42)]
    if len(beside) < 40:
        return []
    rng = np.random.default_rng(0)
    found = []
    for _ in range(1600):
        sample = beside[rng.choice(len(beside), 3, replace=False)]
        if min(
            np.hypot(*(sample[0] - sample[1])),
            np.hypot(*(sample[0] - sample[2])),
            np.hypot(*(sample[1] - sample[2])),
        ) < 25:
            continue
        circle = _circle_through(sample)
        if circle is None:
            continue
        center, radius = circle
        if not (90 <= radius <= 260):
            continue
        signed = float((center[0] - origin[0]) * direction[1] - (center[1] - origin[1]) * direction[0])
        gap = abs(signed)
        if gap < 40 or not (0.48 <= gap / radius <= 0.72):
            continue
        residual = np.abs(np.hypot(beside[:, 0] - center[0], beside[:, 1] - center[1]) - radius)
        keep = beside[residual < 6]
        inliers = int(len(keep))
        if inliers < 100:
            continue
        # A straight line beside the penalty line fits a large circle. The arc bows off that chord.
        chord_vx, chord_vy, chord_x, chord_y = cv2.fitLine(
            keep.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        ).reshape(-1)
        chord = np.array([float(chord_vx), float(chord_vy)], np.float64)
        chord_norm = float(np.linalg.norm(chord))
        if chord_norm < 1e-6:
            continue
        chord /= chord_norm
        chord_origin = np.array([float(chord_x), float(chord_y)], np.float64)
        bow = np.abs((keep - chord_origin)[:, 0] * chord[1] - (keep - chord_origin)[:, 1] * chord[0])
        if float(np.percentile(bow, 90)) < 0.08 * radius:
            continue
        # The painted penalty arc is the short cap outside the box, about 106 degrees.
        # A centre-circle fit covers far more of the circle and must not win.
        angles = np.sort(np.arctan2(keep[:, 1] - center[1], keep[:, 0] - center[0]))
        gaps = np.diff(angles)
        wrap = angles[0] + 2 * np.pi - angles[-1]
        span = 2 * np.pi - max(float(gaps.max()) if len(gaps) else 0.0, float(wrap))
        if not (np.radians(80) <= span <= np.radians(130)):
            continue
        found.append((inliers, center, radius, signed))
    return found


_GOAL_WIDTH_CM = 732.0


def _goal_post_groups(frame):
    """Thin bright verticals on the far half of the frame. One pass per frame."""
    if frame is None:
        return ()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] >= 185) & (hsv[:, :, 1] <= 50)).astype(np.uint8) * 255
    posts = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 22)))
    height, width = posts.shape[:2]
    posts[: int(height * 0.08), :] = 0
    posts[int(height * 0.62) :, :] = 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(posts, 8)
    groups = []
    for index in range(1, count):
        x, y, box_w, box_h, _area = stats[index]
        if box_h < 36 or not (2 <= box_w <= 40):
            continue
        groups.append((float(x), float(x + box_w / 2.0), float(y), float(y + box_h), float(width)))
    return tuple(groups)


def _goal_mouth_contradicts(post_groups, center, gap, perp, positive_x):
    """True when the goal posts are visible and the arc does not face them.

    The arc fixes the scale: the spot is 5.5m behind the line, so the 7.32m
    goal has a known width in pixels and its centre is 11m past the spot.
    A centre-circle cap can satisfy the same arc test, but the real posts
    then sit well away from that predicted centre.
    """
    if not post_groups or gap < 40:
        return False
    groups = []
    for x, midpoint_x, y0, y1, width in post_groups:
        # The scoreboard sits on the left. Posts are searched on the goal's side.
        if positive_x and x < width * 0.45:
            continue
        if not positive_x and x > width * 0.55:
            continue
        groups.append((midpoint_x, y0, y1))
    expect = _GOAL_WIDTH_CM * gap / _SPOT_FROM_FRONT_CM
    goal_x = float(center[0] + perp[0] * (2.0 * gap))
    nearest = None
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            separation = abs(groups[i][0] - groups[j][0])
            if not (0.55 * expect <= separation <= 1.6 * expect):
                continue
            overlap = min(groups[i][2], groups[j][2]) - max(groups[i][1], groups[j][1])
            if overlap < 10:
                continue
            midpoint = 0.5 * (groups[i][0] + groups[j][0])
            error = abs(midpoint - goal_x)
            if nearest is None or error < nearest:
                nearest = error
    return nearest is not None and nearest > 2.2 * expect


def fit_broadcast_penalty(white_line_mask, pitch, frame=None):
    """Image-to-field matrix from the penalty line and the arc outside it.

    The keypoint model on a broadcast frame labels the right-hand penalty
    marks and then draws them in open grass. The white paint still has the
    penalty line and the arc, and the spot sits 5.5m behind that line on a
    9.15m arc. Returns None when that pair is not on the paint, or when the
    goal posts are visible and this arc does not face them.
    """
    paint = _clean_paint(white_line_mask)
    if paint is None:
        return None
    line = _penalty_line(paint)
    if line is None:
        return None
    origin, direction = line
    post_groups = _goal_post_groups(frame)
    best = None
    for inliers, center, _radius, _signed in _arc_candidates(paint, origin, direction):
        foot = origin + direction * float(np.dot(center - origin, direction))
        gap = float(np.linalg.norm(center - foot))
        if gap < 40:
            continue
        perp = (center - foot) / gap
        positive_x = float(center[0]) >= float(foot[0])
        if _goal_mouth_contradicts(post_groups, center, gap, perp, positive_x):
            continue
        front_x = float(pitch.length) - _PENALTY_DEPTH_CM if positive_x else _PENALTY_DEPTH_CM
        matrix = _affine_matrix(foot, direction, perp, gap / _SPOT_FROM_FRONT_CM, front_x, float(pitch.width), positive_x)
        distance = _painted_arc_distance(matrix, paint, positive_x)
        if distance is None or distance > 8.0:
            continue
        # The arc has to sit on the paint. Inlier count breaks the remaining ties.
        rank = (distance, -inliers)
        if best is None or rank < best[0]:
            best = (rank, matrix, bool(positive_x), float(gap), float(front_x))
    if best is None:
        return None
    _rank, matrix, positive_x, gap, front_x = best
    return {
        "matrix": matrix,
        "positive_x": positive_x,
        "cm_per_px": _SPOT_FROM_FRONT_CM / gap,
        "front_x_cm": front_x,
    }


class PitchHomographyGate:
    def __init__(
        self,
        pitch=None,
        min_inliers=MIN_INLIERS,
        min_line_support=MIN_LINE_SUPPORT,
        ransac_reproj_cm=RANSAC_REPROJ_CM,
        hold_frames=HOLD_FRAMES,
        min_span_px=MIN_SPAN_PX,
    ):
        self.pitch = pitch or SoccerPitchConfiguration()
        self.vertices = list(self.pitch.vertices)
        self.min_inliers = int(min_inliers)
        self.min_line_support = float(min_line_support)
        self.ransac_reproj_cm = float(ransac_reproj_cm)
        self.hold_frames = int(hold_frames)
        self.min_span_px = float(min_span_px)
        self.held = None
        self.hold_left = 0
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.held_frames = 0
        self.reasons = Counter()

    def update(self, keypoints, white_line_mask, frame=None):
        decision = self._fit(keypoints, white_line_mask)
        if not decision["accepted"]:
            painted = fit_broadcast_penalty(white_line_mask, self.pitch, frame)
            if painted is not None:
                painted_matrix = painted["matrix"]
                decision = {
                    "accepted": True,
                    "held": False,
                    "reason": "accepted",
                    "transformer": ViewTransformer.from_matrix(painted_matrix),
                    "matrix": painted_matrix,
                    "candidate_matrix": painted_matrix,
                    "inliers": decision["inliers"],
                    "support": line_support(painted_matrix, white_line_mask, self.pitch),
                    "candidate_support": line_support(painted_matrix, white_line_mask, self.pitch),
                }
        if decision["accepted"]:
            self.held = decision["transformer"]
            self.hold_left = self.hold_frames
            self.accepted_frames += 1
            self.reasons["accepted"] += 1
            return decision
        if self._held_still_matches(white_line_mask):
            self.hold_left -= 1
            self.held_frames += 1
            self.reasons["held"] += 1
            return {
                "accepted": True,
                "held": True,
                "reason": "held",
                "transformer": self.held,
                "matrix": self.held.m,
                "inliers": decision["inliers"],
                "support": line_support(self.held.m, white_line_mask, self.pitch),
                "candidate_matrix": decision["candidate_matrix"],
                "candidate_support": decision["support"],
            }
        self.rejected_frames += 1
        self.reasons[decision["reason"]] += 1
        decision["transformer"] = None
        return decision

    def metrics(self):
        return {
            "accepted_frames": self.accepted_frames,
            "rejected_frames": self.rejected_frames,
            "held_frames": self.held_frames,
            "min_inliers": self.min_inliers,
            "min_line_support": self.min_line_support,
            "ransac_reproj_cm": self.ransac_reproj_cm,
            "hold_frames": self.hold_frames,
            "reasons": dict(self.reasons),
        }

    def _fit(self, keypoints, white_line_mask):
        by_index = {}
        for keypoint in keypoints or []:
            index = int(keypoint["index"])
            if index < 0 or index >= len(self.vertices):
                continue
            score = float(keypoint["confidence"]) if keypoint.get("confidence") is not None else -1.0
            previous = by_index.get(index)
            if previous is None or score > previous[0]:
                by_index[index] = (score, keypoint)

        empty = {
            "accepted": False,
            "held": False,
            "reason": "too_few_points",
            "transformer": None,
            "matrix": None,
            "candidate_matrix": None,
            "inliers": len(by_index),
            "support": None,
            "candidate_support": None,
        }
        if len(by_index) < self.min_inliers:
            return empty
        by_index = self._on_white_lines(by_index, white_line_mask)
        empty["inliers"] = len(by_index)
        if len(by_index) < self.min_inliers:
            empty["reason"] = "off_lines"
            return empty

        indexes = sorted(by_index)
        source = np.array(
            [[by_index[index][1]["x"], by_index[index][1]["y"]] for index in indexes],
            dtype=np.float32,
        )
        target = np.array([self.vertices[index] for index in indexes], dtype=np.float32)
        try:
            matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, self.ransac_reproj_cm)
        except cv2.error:
            matrix, mask = None, None
        if matrix is None or mask is None:
            empty["reason"] = "no_homography"
            empty["inliers"] = 0
            return empty

        inlier = mask.ravel().astype(bool)
        inlier_count = int(inlier.sum())
        candidate = np.asarray(matrix, dtype=np.float64)
        rejected = {
            "accepted": False,
            "held": False,
            "reason": "few_inliers",
            "transformer": None,
            "matrix": None,
            "candidate_matrix": candidate,
            "inliers": inlier_count,
            "support": None,
            "candidate_support": None,
        }
        if inlier_count < self.min_inliers:
            return rejected

        inlier_source = source[inlier]
        span_x = float(inlier_source[:, 0].max() - inlier_source[:, 0].min())
        span_y = float(inlier_source[:, 1].max() - inlier_source[:, 1].min())
        if span_x < self.min_span_px or span_y < self.min_span_px:
            rejected["reason"] = "clustered"
            return rejected

        support = line_support(candidate, white_line_mask, self.pitch)
        rejected["support"] = support
        rejected["candidate_support"] = support
        if support is None or support < self.min_line_support:
            rejected["reason"] = "line_support"
            return rejected

        return {
            "accepted": True,
            "held": False,
            "reason": "accepted",
            "transformer": ViewTransformer.from_matrix(candidate),
            "matrix": candidate,
            "candidate_matrix": candidate,
            "inliers": inlier_count,
            "support": support,
            "candidate_support": support,
        }

    def _held_still_matches(self, white_line_mask):
        if self.held is None or self.hold_left <= 0:
            return False
        support = line_support(self.held.m, white_line_mask, self.pitch)
        if support is not None and support >= self.min_line_support:
            return True
        # A broadcast fit only explains the penalty arc, not every template line.
        paint = _clean_paint(white_line_mask)
        if paint is not None:
            for positive_x in (True, False):
                distance = _painted_arc_distance(self.held.m, paint, positive_x)
                if distance is not None and distance <= 8.0:
                    return True
        self.held = None
        self.hold_left = 0
        return False

    def _on_white_lines(self, by_index, white_line_mask):
        if white_line_mask is None:
            return by_index
        diameter = _KEYPOINT_LINE_RADIUS_PX * 2 + 1
        near_lines = cv2.dilate(
            white_line_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)),
        )
        height, width = near_lines.shape[:2]
        kept = {}
        for index, item in by_index.items():
            keypoint = item[1]
            x = int(round(float(keypoint["x"])))
            y = int(round(float(keypoint["y"])))
            if 0 <= x < width and 0 <= y < height and near_lines[y, x] > 0:
                kept[index] = item
        return kept


def line_support(matrix, white_line_mask, pitch):
    """Fraction of projected pitch markings that land on the white-line mask."""
    if white_line_mask is None or matrix is None:
        return None
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        return None
    try:
        inverse = np.linalg.inv(array)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(inverse).all():
        return None

    samples = _pitch_samples(pitch)
    projected = cv2.perspectiveTransform(
        samples.reshape(-1, 1, 2),
        inverse.astype(np.float32),
    ).reshape(-1, 2)
    height, width = white_line_mask.shape[:2]
    support = white_line_mask
    if _LINE_DILATION > 1:
        support = cv2.dilate(
            white_line_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_LINE_DILATION, _LINE_DILATION)),
        )
    inside = 0
    hits = 0
    for x, y in projected:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if abs(x) > width * 4 or abs(y) > height * 4:
            continue
        px = int(round(x))
        py = int(round(y))
        if 0 <= px < width and 0 <= py < height:
            inside += 1
            if support[py, px] > 0:
                hits += 1
    if inside < MIN_SAMPLES_INSIDE:
        return None
    return hits / inside


def draw_projected_pitch(frame, matrix, color, pitch=None):
    """Draw the pitch template through a homography. Debug only."""
    if frame is None or matrix is None:
        return
    pitch = pitch or SoccerPitchConfiguration()
    array = np.asarray(matrix, dtype=np.float64)
    try:
        inverse = np.linalg.inv(array).astype(np.float32)
    except np.linalg.LinAlgError:
        return
    height, width = frame.shape[:2]
    limit = max(width, height) * 4
    for start_id, end_id in pitch.edges:
        start = np.array(pitch.vertices[start_id - 1], dtype=np.float32)
        end = np.array(pitch.vertices[end_id - 1], dtype=np.float32)
        segment = np.stack([start + (end - start) * (step / 15) for step in range(16)])
        projected = cv2.perspectiveTransform(segment.reshape(-1, 1, 2), inverse).reshape(-1, 2)
        if not np.isfinite(projected).all() or np.max(np.abs(projected)) > limit:
            continue
        cv2.polylines(frame, [np.round(projected).astype(np.int32)], False, color, 2, cv2.LINE_AA)
    radius = float(pitch.centre_circle_radius)
    center = np.array([pitch.length / 2, pitch.width / 2], dtype=np.float32)
    circle = []
    for angle in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        circle.append([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)])
    projected = cv2.perspectiveTransform(np.array(circle, dtype=np.float32).reshape(-1, 1, 2), inverse).reshape(-1, 2)
    if np.isfinite(projected).all() and np.max(np.abs(projected)) <= limit:
        cv2.polylines(frame, [np.round(projected).astype(np.int32)], True, color, 2, cv2.LINE_AA)


def _pitch_samples(pitch):
    samples = []
    for start_id, end_id in pitch.edges:
        start = np.array(pitch.vertices[start_id - 1], dtype=np.float32)
        end = np.array(pitch.vertices[end_id - 1], dtype=np.float32)
        for step in range(12):
            samples.append(start + (end - start) * (step / 11))
    radius = float(pitch.centre_circle_radius)
    center = np.array([pitch.length / 2, pitch.width / 2], dtype=np.float32)
    for angle in np.linspace(0, 2 * np.pi, 36, endpoint=False):
        samples.append([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)])
    return np.array(samples, dtype=np.float32)
