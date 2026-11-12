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


def _arc_distance_stats(matrix, paint, positive_x):
    """Median and 90th-percentile distance, in pixels, from the penalty arc to the paint."""
    try:
        inverse = np.linalg.inv(matrix).astype(np.float32)
    except np.linalg.LinAlgError:
        return None, None
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
        return None, None
    xs = [point[0] for point in inside]
    ys = [point[1] for point in inside]
    # A handful of pixels on a sock is not the arc.
    if max(xs) - min(xs) < 70 and max(ys) - min(ys) < 70:
        return None, None
    distances = [point[2] for point in inside]
    return float(np.median(distances)), float(np.percentile(distances, 90))


def _painted_arc_distance(matrix, paint, positive_x):
    """Median distance, in pixels, from the painted penalty arc to the mask.

    The front line is on the paint by construction, so only the arc outside
    the box is scored. None when that arc leaves the frame.
    """
    median, _spread = _arc_distance_stats(matrix, paint, positive_x)
    return median


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
        "anchor": "penalty",
        "cm_per_px": _SPOT_FROM_FRONT_CM / gap,
        "front_x_cm": front_x,
    }


def _curvature_split(paint):
    """Straight paint versus curved paint. A cropped centre circle stays in the curved set."""
    contours, _hierarchy = cv2.findContours(paint, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    curved = []
    straight = []
    step = 8
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float64)
        if len(points) < step * 2 + 2:
            continue
        for index in range(step, len(points) - step):
            before = points[index] - points[index - step]
            after = points[index + step] - points[index]
            before_norm = float(np.linalg.norm(before))
            after_norm = float(np.linalg.norm(after))
            if before_norm < 1.0 or after_norm < 1.0:
                straight.append(points[index])
                continue
            turn = abs(before[0] * after[1] - before[1] * after[0]) / (before_norm * after_norm)
            if turn >= 0.10:
                curved.append(points[index])
            else:
                straight.append(points[index])
    curved = np.array(curved, dtype=np.float64) if curved else np.zeros((0, 2))
    straight = np.array(straight, dtype=np.float64) if straight else np.zeros((0, 2))
    return curved, straight


def _arc_span(points, center):
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    angles = np.sort(angles)
    gaps = np.diff(angles)
    wrap = float(angles[0] + 2 * np.pi - angles[-1])
    largest_gap = max(float(gaps.max()) if len(gaps) else 0.0, wrap)
    return float(np.degrees(2 * np.pi - largest_gap))


def _fit_arcs(points, rounds=3):
    """Circles supported by curved paint. Each round removes the circle it kept."""
    if len(points) < 40:
        return []
    remaining = points
    found = []
    rng = np.random.default_rng(1)
    for _round in range(rounds):
        if len(remaining) < 40:
            break
        best = None
        for _attempt in range(700):
            sample = remaining[rng.choice(len(remaining), 3, replace=False)]
            if min(
                np.hypot(*(sample[0] - sample[1])),
                np.hypot(*(sample[0] - sample[2])),
                np.hypot(*(sample[1] - sample[2])),
            ) < 16:
                continue
            circle = _circle_through(sample)
            if circle is None:
                continue
            center, radius = circle
            if not (40.0 <= radius <= 280.0):
                continue
            residual = np.abs(np.hypot(remaining[:, 0] - center[0], remaining[:, 1] - center[1]) - radius)
            kept = remaining[residual < 4.0]
            if len(kept) < 35:
                continue
            span = _arc_span(kept, center)
            if span < 40.0 or span > 320.0:
                continue
            median = float(np.median(residual[residual < 4.0]))
            rank = (median, -len(kept))
            if best is None or rank < best[0]:
                best = (rank, center, float(radius), span, kept)
        if best is None:
            break
        _rank, center, radius, span, kept = best
        found.append({"center": center, "radius": radius, "span": span, "count": len(kept)})
        residual = np.abs(np.hypot(remaining[:, 0] - center[0], remaining[:, 1] - center[1]) - radius)
        remaining = remaining[residual >= 6.0]
    return found


def _line_offset(point, origin, direction):
    delta = point - origin
    return float(delta[0] * direction[1] - delta[1] * direction[0])


def _collect_lines(paint):
    edges = cv2.Canny(paint, 40, 120)
    found = cv2.HoughLinesP(edges, 1, np.pi / 180, 18, minLineLength=40, maxLineGap=18)
    if found is None:
        return []
    segments = []
    for x1, y1, x2, y2 in found.reshape(-1, 4):
        direction = np.array([float(x2 - x1), float(y2 - y1)], np.float64)
        length = float(np.linalg.norm(direction))
        if length < 36:
            continue
        direction /= length
        if direction[0] < 0:
            direction = -direction
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        origin = np.array([float(x1), float(y1)], np.float64)
        segments.append(
            {
                "origin": origin,
                "direction": direction,
                "length": length,
                "angle": angle,
                "mid_y": float(y1 + y2) / 2.0,
            }
        )
    return segments


def _far_touchline(lines, height):
    """The long shallow line high in the frame, under the advertising boards."""
    shallow = [
        line
        for line in lines
        if abs(line["angle"]) <= 20.0 and line["mid_y"] < height * 0.46 and line["length"] >= 70.0
    ]
    if not shallow:
        return None
    shallow.sort(key=lambda line: (line["mid_y"], -line["length"]))
    return shallow[0]


def _line_continues_through(paint, line, support):
    """True when the straight paint runs past the arc on both sides.

    The halfway line crosses the centre circle and keeps going. The 16.5 m line
    is only the chord of the penalty arc, so it does not come out the other side.
    """
    if len(support) < 8:
        return False
    direction = line["direction"]
    origin = line["origin"]
    along = np.array([float(np.dot(point - origin, direction)) for point in support])
    low, high = np.percentile(along, [8, 92])
    height, width = paint.shape[:2]

    def paint_beyond(distance):
        point = origin + direction * distance
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (2 <= x < width - 2 and 2 <= y < height - 2):
            return False
        return bool(np.count_nonzero(paint[y - 2 : y + 3, x - 2 : x + 3]))

    before = any(paint_beyond(low - step) for step in (24.0, 48.0, 80.0))
    after = any(paint_beyond(high + step) for step in (24.0, 48.0, 80.0))
    return before and after


def _bow_ratio(line, support, radius):
    signed = np.array([_line_offset(point, line["origin"], line["direction"]) for point in support])
    if len(signed) == 0:
        return None, 0.0
    positive = int(np.count_nonzero(signed > 0))
    fraction = max(positive, len(signed) - positive) / len(signed)
    bow = float(np.max(np.abs(signed))) / max(float(radius), 1.0)
    return bow, fraction


def _mark_kind_from_line(paint, line, support, radius):
    """Shallow one-sided cap: penalty arc. Line through both sides: centre circle."""
    if line is None or len(support) < 12:
        return "unknown"
    bow, fraction = _bow_ratio(line, support, radius)
    if bow is None:
        return "unknown"
    through = _line_continues_through(paint, line, support)
    if through and bow >= 0.65:
        return "centre"
    if (not through) and fraction >= 0.85:
        return "penalty"
    if fraction >= 0.9 and bow <= 0.55:
        return "penalty"
    return "unknown"


def _label_arc(arc, lines):
    """Centre if any straight line passes through the centre. Penalty only when none does."""
    if not lines:
        return "unknown", None
    through = []
    offset_lines = []
    for line in lines:
        offset = abs(_line_offset(arc["center"], line["origin"], line["direction"]))
        ratio = offset / arc["radius"]
        if ratio <= 0.40:
            through.append(line)
        elif 0.45 <= ratio <= 0.78:
            offset_lines.append(line)
    if through:
        return "centre", max(through, key=lambda line: line["length"])
    if offset_lines and 80.0 <= arc["span"] <= 140.0:
        return "penalty", max(offset_lines, key=lambda line: line["length"])
    return "unknown", None


def _pair_centre_and_penalty(arcs, lines, height):
    """Two half-circles: the one on the halfway line is central. Depth, not the sideline, decides."""
    labelled = []
    for arc in arcs:
        kind, line = _label_arc(arc, lines)
        labelled.append((kind, line, arc))
    centres = [item for item in labelled if item[0] == "centre"]
    penalties = [item for item in labelled if item[0] == "penalty"]
    if centres:
        tight = [
            item
            for item in centres
            if item[2].get("error", 9.0) <= 2.5
            and item[2]["count"] >= 40
            and item[2]["span"] >= 55.0
            and item[2]["center"][1] > height * 0.18
        ]
        pool = tight or centres
        centres = [min(pool, key=lambda item: (item[2].get("error", 9.0), -item[2]["count"]))]
    if not centres and len(labelled) >= 2:
        first, second = labelled[0], labelled[1]
        delta = second[2]["center"] - first[2]["center"]
        length = float(np.linalg.norm(delta))
        if length < 1.0:
            return None, None, None
        direction = delta / length

        def halfway_alignment(item):
            _kind, line, arc = item
            if line is None:
                return 1.0
            return abs(float(np.dot(line["direction"], direction)))

        if halfway_alignment(first) <= halfway_alignment(second):
            centres = [first]
            penalties = [second]
        else:
            centres = [second]
            penalties = [first]
    if not centres:
        return None, None, None
    centre_kind, centre_line, centre_arc = centres[0]
    penalty_arc = None
    if centre_line is not None:
        for _kind, _line, arc in labelled:
            if arc is centre_arc:
                continue
            offset = abs(_line_offset(arc["center"], centre_line["origin"], centre_line["direction"]))
            ratio = offset / max(centre_arc["radius"], 1.0)
            separation = float(np.linalg.norm(arc["center"] - centre_arc["center"]))
            depth = separation / max(centre_arc["radius"], 1.0)
            # 49 m from the centre spot to the penalty spot is about 5.4 radii.
            if 0.45 <= ratio <= 0.9 and 3.5 <= depth <= 7.5 and arc["span"] <= 150.0:
                penalty_arc = arc
                break
    if penalty_arc is None and penalties:
        penalty_arc = penalties[0][2]
    return centre_arc, centre_line, penalty_arc


def _centre_matrix(center, along, perp, radius):
    centimetres = _ARC_RADIUS_CM / float(radius)
    return np.array(
        [
            [
                centimetres * perp[0],
                centimetres * perp[1],
                6000.0 - centimetres * float(np.dot(perp, center)),
            ],
            [
                centimetres * along[0],
                centimetres * along[1],
                3500.0 - centimetres * float(np.dot(along, center)),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _paint_distances(matrix, paint, field_points):
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        return []
    projected = cv2.perspectiveTransform(
        np.asarray(field_points, dtype=np.float32).reshape(-1, 1, 2), inverse
    ).reshape(-1, 2)
    distance = cv2.distanceTransform(cv2.bitwise_not(paint), cv2.DIST_L2, 3)
    height, width = paint.shape[:2]
    values = []
    for x, y in projected:
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height:
            values.append(float(distance[int(y), int(x)]))
    return values


def _visible_mark_sits_on_paint(values):
    """The painted part of a mark is enough. The hidden half of a circle is not."""
    if len(values) < 8:
        return False
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    count = max(6, int(round(len(ordered) * 0.33)))
    close = int(np.count_nonzero(ordered <= 8.0))
    return close >= 6 and float(np.median(ordered[:count])) <= 6.0


def _mark_score(values):
    if len(values) < 8:
        return None
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    count = max(8, int(round(len(ordered) * 0.4)))
    return float(np.median(ordered[:count]))


def centre_marks_still_on_paint(matrix, paint):
    """True when the visible centre circle and halfway line still sit on the paint."""
    if matrix is None or paint is None:
        return False
    circle = [
        (6000.0 + _ARC_RADIUS_CM * np.cos(angle), 3500.0 + _ARC_RADIUS_CM * np.sin(angle))
        for angle in np.linspace(0, 2 * np.pi, 48)
    ]
    line = [(6000.0, y) for y in np.linspace(200.0, 6800.0, 24)]
    circle_score = _mark_score(_paint_distances(matrix, paint, circle))
    line_score = _mark_score(_paint_distances(matrix, paint, line))
    return circle_score is not None and line_score is not None and circle_score <= 5.0 and line_score <= 8.0


def _orient_centre(center, along, radius, penalty_center):
    """Point field +x toward the penalty arc when it is visible, otherwise to image right."""
    perp = np.array([-along[1], along[0]], np.float64)
    if penalty_center is not None:
        toward = penalty_center - center
        if float(np.dot(toward, perp)) < 0.0:
            perp = -perp
    else:
        probe = center + perp * radius
        other = center - perp * radius
        if float(probe[0]) < float(other[0]):
            perp = -perp
    if float(np.dot(along, np.array([-perp[1], perp[0]]))) < 0.0:
        along = -along
    return along, perp


def _component_arcs(paint, curved):
    """One circle per curved patch, so a cropped centre arc is not mixed with a straight line."""
    if len(curved) < 25:
        return []
    mask = np.zeros(paint.shape, np.uint8)
    xs = np.clip(curved[:, 0].astype(int), 0, paint.shape[1] - 1)
    ys = np.clip(curved[:, 1].astype(int), 0, paint.shape[0] - 1)
    mask[ys, xs] = 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    components, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arcs = []
    rng = np.random.default_rng(1)
    for component in components:
        x, y, width, height = cv2.boundingRect(component)
        if width * height < 500 or max(width, height) < 28:
            continue
        selected = curved[
            (curved[:, 0] >= x)
            & (curved[:, 0] < x + width)
            & (curved[:, 1] >= y)
            & (curved[:, 1] < y + height)
        ]
        if len(selected) < 25:
            continue
        best = None
        for _attempt in range(220):
            sample = selected[rng.choice(len(selected), 3, replace=False)]
            if min(
                np.hypot(*(sample[0] - sample[1])),
                np.hypot(*(sample[0] - sample[2])),
                np.hypot(*(sample[1] - sample[2])),
            ) < 10:
                continue
            circle = _circle_through(sample)
            if circle is None:
                continue
            center, radius = circle
            if not (45.0 <= radius <= 170.0):
                continue
            residual = np.abs(np.hypot(selected[:, 0] - center[0], selected[:, 1] - center[1]) - radius)
            kept = int(np.count_nonzero(residual < 4.0))
            if kept < 22:
                continue
            median = float(np.median(residual[residual < 4.0]))
            span = _arc_span(selected[residual < 4.0], center)
            if span < 35.0:
                continue
            rank = (median, -kept)
            if best is None or rank < best[0]:
                best = (rank, center, float(radius), span, kept)
        if best is None:
            continue
        rank, center, radius, span, kept = best
        arcs.append(
            {
                "center": center,
                "radius": radius,
                "span": span,
                "count": kept,
                "error": float(rank[0]),
            }
        )
    return arcs


def _snap_center_to_line(center, radius, line, support):
    """Move the centre onto the straight line and keep the arc's radius."""
    direction = line["direction"]
    normal = np.array([-direction[1], direction[0]], np.float64)
    signed = _line_offset(center, line["origin"], direction)
    snapped = center - normal * signed
    if abs(_line_offset(snapped, line["origin"], direction)) > abs(signed):
        snapped = center + normal * signed
    distances = np.hypot(support[:, 0] - snapped[0], support[:, 1] - snapped[1])
    fitted = float(np.median(distances))
    if not (45.0 <= fitted <= 170.0):
        return None
    error = np.abs(distances - fitted)
    if int(np.count_nonzero(error < 6.0)) < 20 or float(np.median(error)) > 5.0:
        return None
    return snapped, fitted


def _length_fraction(point, paint, lines):
    """0 at one end of the visible paint, 1 at the other, along the far touchline."""
    touch = _far_touchline(lines, paint.shape[0])
    axis = touch["direction"] if touch is not None else np.array([1.0, 0.0], np.float64)
    ys, xs = np.where(paint > 0)
    if len(xs) < 30:
        return 0.5, axis
    coords = xs.astype(np.float64) * axis[0] + ys.astype(np.float64) * axis[1]
    low, high = np.percentile(coords, [4, 96])
    if high - low < 1.0:
        return 0.5, axis
    value = float(point[0] * axis[0] + point[1] * axis[1])
    return float((value - low) / (high - low)), axis


def _spot_matrix(center, along, perp, radius, spot_x, width_cm):
    """Image point ``center`` is the penalty spot. ``perp`` points toward increasing field x."""
    centimetres = _ARC_RADIUS_CM / float(radius)
    return np.array(
        [
            [
                centimetres * perp[0],
                centimetres * perp[1],
                float(spot_x) - centimetres * float(np.dot(perp, center)),
            ],
            [
                centimetres * along[0],
                centimetres * along[1],
                width_cm / 2.0 - centimetres * float(np.dot(along, center)),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _penalty_from_cap(arc, line, support, paint, pitch, goal_direction):
    """A cap toward one goal is the penalty arc. Its fitted centre is the penalty spot."""
    del support
    if line is None or pitch is None:
        return None
    direction = line["direction"]
    perp = np.array([-direction[1], direction[0]], np.float64)
    if float(np.dot(perp, goal_direction)) < 0.0:
        perp = -perp
    if float(np.dot(direction, np.array([-perp[1], perp[0]]))) < 0.0:
        direction = -direction
    positive_x = float(goal_direction[0]) >= 0.0
    spot_x = float(pitch.length) - _SPOT_FROM_GOAL_CM if positive_x else _SPOT_FROM_GOAL_CM
    best = None
    for factor in np.linspace(0.75, 1.35, 9):
        matrix = _spot_matrix(
            arc["center"], direction, perp, arc["radius"] * float(factor), spot_x, float(pitch.width)
        )
        distance = _painted_arc_distance(matrix, paint, positive_x)
        if distance is None:
            continue
        if best is None or distance < best[0]:
            best = (distance, matrix, arc["radius"] * float(factor))
    if best is None or best[0] > 8.0:
        return None
    return best[1], positive_x, best[2]


def _caps_on_line(paint, origin, direction, ratio_min, ratio_max):
    """Bows that leave a straight line and come back to it.

    The chord is measured first. ``height / (chord / 2)`` is about 0.5 for the
    penalty arc (3.65 m of bow on a 14.6 m chord) and about 1 for a semicircle
    whose centre sits on the line.
    """
    ys, xs = np.where(paint > 0)
    if len(xs) < 40:
        return []
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    delta = points - origin
    along = delta @ direction
    offset = delta[:, 0] * direction[1] - delta[:, 1] * direction[0]
    on_line = np.abs(offset) < 8.0
    if int(np.count_nonzero(on_line)) < 30:
        return []
    low = float(np.percentile(along[on_line], 2))
    high = float(np.percentile(along[on_line], 98))
    step = 14.0
    edges = np.arange(low - step, high + step, step)
    found = []
    for sign in (1.0, -1.0):
        values = []
        for edge in edges:
            chosen = (along >= edge) & (along < edge + step) & (offset * sign > 8.0) & (offset * sign < 220.0)
            if int(np.count_nonzero(chosen)) < 5:
                values.append(0.0)
            else:
                values.append(float(np.percentile(offset[chosen] * sign, 80)))
        values = np.asarray(values, dtype=np.float64)
        index = 0
        while index < len(values):
            if values[index] < 16.0:
                index += 1
                continue
            end = index
            while end < len(values) and values[end] >= 16.0:
                end += 1
            segment = values[index:end]
            peak = int(np.argmax(segment))
            if end - index >= 4 and 0 < peak < len(segment) - 1:
                height = float(segment.max())
                chord = float((end - index) * step)
                left_shoulder = float(values[index - 1]) if index > 0 else height
                right_shoulder = float(values[end]) if end < len(values) else height
                half = chord / 2.0
                ratio = height / half if half > 1.0 else 99.0
                if (
                    left_shoulder < 0.45 * height
                    and right_shoulder < 0.45 * height
                    and height >= 28.0
                    and chord >= 80.0
                    and ratio_min <= ratio <= ratio_max
                ):
                    radius = (height * height + half * half) / (2.0 * height)
                    found.append(
                        {
                            "sign": float(sign),
                            "height": height,
                            "chord": chord,
                            "radius": float(radius),
                            "ratio": float(ratio),
                            "left": float(edges[index]),
                            "right": float(edges[end - 1] + step),
                        }
                    )
            index = end
    return found


def _goal_from_cap(direction, sign):
    """Unit image vector from the 16.5 m line toward the penalty spot.

    The painted bow is on the other side of that line, toward midfield.
    """
    goal = np.array([-float(direction[1]), float(direction[0])], np.float64) * float(sign)
    norm = float(np.linalg.norm(goal))
    if norm < 1e-6:
        return None
    return goal / norm


def _same_painted_line(first, second):
    if first is None or second is None:
        return False
    origin, direction = first
    other_origin, other_direction = second
    if abs(float(np.dot(direction, other_direction))) < np.cos(np.deg2rad(12.0)):
        return False
    return abs(_line_offset(origin, other_origin, other_direction)) < 18.0


def _line_extends_past(paint, origin, direction, left, right):
    """True when the straight paint keeps going past both ends of the bow."""
    height, width = paint.shape[:2]

    def hits(distance):
        point = origin + direction * float(distance)
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if not (2 <= x < width - 2 and 2 <= y < height - 2):
            return False
        return bool(np.count_nonzero(paint[y - 2 : y + 3, x - 2 : x + 3]))

    before = any(hits(left - step) for step in (20.0, 45.0, 80.0))
    after = any(hits(right + step) for step in (20.0, 45.0, 80.0))
    return before and after


def _fit_shallow_penalty(paint, pitch):
    """Penalty arc: a shallow bow on the 16.5 m line, spot 5.5 m behind it."""
    line = _penalty_line(paint)
    if line is None or pitch is None:
        return None
    origin, direction = line
    best = None
    for cap in _caps_on_line(paint, origin, direction, 0.32, 0.85):
        toward_spot = _goal_from_cap(direction, cap["sign"])
        if toward_spot is None:
            continue
        # The right-hand goal is high field x. A spot on the left of its line is the other goal.
        positive_x = bool(toward_spot[0] >= 0.0)
        perp = toward_spot if positive_x else -toward_spot
        along = np.array(direction, dtype=np.float64, copy=True)
        if float(np.dot(along, np.array([-perp[1], perp[0]]))) < 0.0:
            along = -along
        spot_x = float(pitch.length) - _SPOT_FROM_GOAL_CM if positive_x else _SPOT_FROM_GOAL_CM
        midpoint = 0.5 * (cap["left"] + cap["right"])
        for foot_along in (midpoint - 24.0, midpoint, midpoint + 24.0):
            foot = origin + direction * float(foot_along)
            for scale in (0.9, 1.0, 1.1):
                radius = float(cap["radius"]) * float(scale)
                if not (70.0 <= radius <= 340.0):
                    continue
                spot = foot + toward_spot * (radius * _SPOT_FROM_FRONT_CM / _ARC_RADIUS_CM)
                matrix = _spot_matrix(spot, along, perp, radius, spot_x, float(pitch.width))
                median, spread = _arc_distance_stats(matrix, paint, positive_x)
                if median is None or median > 8.0 or spread > 16.0:
                    continue
                if best is None or (median, spread) < best[0]:
                    best = ((median, spread), matrix, positive_x, radius)
    if best is None:
        return None
    _rank, matrix, positive_x, radius = best
    return {
        "matrix": matrix,
        "positive_x": positive_x,
        "anchor": "penalty",
        "cm_per_px": _ARC_RADIUS_CM / float(radius),
        "front_x_cm": float(pitch.length) - _PENALTY_DEPTH_CM if positive_x else _PENALTY_DEPTH_CM,
        "has_penalty_arc": True,
    }


def _fit_deep_centre(paint, pitch):
    """Centre circle: the line crosses the arc and the bow is about one radius deep."""
    del pitch
    penalty_line = _penalty_line(paint)
    best = None
    for line in _collect_lines(paint):
        if line["length"] < 120.0:
            continue
        candidate = (line["origin"], line["direction"])
        if _same_painted_line(candidate, penalty_line):
            continue
        for cap in _caps_on_line(paint, line["origin"], line["direction"], 0.85, 1.35):
            if not _line_extends_past(paint, line["origin"], line["direction"], cap["left"], cap["right"]):
                continue
            radius = float(cap["height"])
            if not (50.0 <= radius <= 260.0):
                continue
            center = line["origin"] + line["direction"] * (0.5 * (cap["left"] + cap["right"]))
            along, perp = _orient_centre(center, line["direction"], radius, None)
            matrix = _centre_matrix(center, along, perp, radius)
            if not centre_marks_still_on_paint(matrix, paint):
                continue
            rank = (abs(cap["ratio"] - 1.0), -line["length"])
            if best is None or rank < best[0]:
                best = (rank, matrix, radius)
    if best is None:
        return None
    _rank, matrix, radius = best
    return {
        "matrix": matrix,
        "positive_x": None,
        "anchor": "centre",
        "cm_per_px": _ARC_RADIUS_CM / float(radius),
        "front_x_cm": 6000.0,
        "has_penalty_arc": False,
    }


def fit_broadcast_centre(white_line_mask, pitch, frame=None):
    """Image-to-field matrix from the bow of the visible arc.

    The straight line is measured first. A shallow bow on one side of the
    16.5 m line is the penalty arc, and its centre is 5.5 m behind that line.
    A deep bow, with the line continuing out of both sides, is the centre
    circle even when the camera crops it. A short arc is not the centre circle
    just because a fitted centre falls near the chord. The matrix is kept only
    when the projected mark covers the white paint.
    """
    del frame
    if pitch is None:
        pitch = SoccerPitchConfiguration()
    paint = _clean_paint(white_line_mask)
    if paint is None:
        return None
    penalty = _fit_shallow_penalty(paint, pitch)
    if penalty is not None:
        return penalty
    return _fit_deep_centre(paint, pitch)


def _paint_points(paint):
    ys, xs = np.where(paint > 0)
    if len(xs) == 0:
        return np.zeros((0, 2))
    return np.stack([xs, ys], axis=1).astype(np.float64)


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
