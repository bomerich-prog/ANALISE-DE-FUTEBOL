"""Image-to-field matrix around the centre circle, read off the thin painted marks.

The centre circle, the halfway line and the touchlines are fitted as a true
perspective, not an affine. A circle cut by the picture edge still fixes the
fit with the halfway line, and when the halfway line is off the picture the
touchline and a visible penalty area take its place.

Every candidate is scored both ways. The projected marks have to land on the
ink, and the long straight paint in the picture has to land on a pitch mark.
That second check stops a fit that matches the circle and throws the rest of
the pitch out of the picture.

Pitch length and width vary between stadiums, but the marks around the
centre do not, so the template is the FIFA standard pitch, 105 by 68 m. The
result is given in the program's field frame: the centre is at (6000, 3500)
cm and distances from it are real.
"""

import cv2
import numpy as np

# Same band as pitch_map: the top of the picture is stands, never paint.
CROWD_BAND = 0.18

FIELD_CENTRE_X = 6000.0
FIELD_CENTRE_Y = 3500.0
CIRCLE_CM = 915.0
HALF_LENGTH_CM = 5250.0
HALF_WIDTH_CM = 3400.0
BOX_HALF_CM = 2016.0
BOX_DEPTH_CM = 1650.0

MAX_COST_PX = 2.5
MAX_UNEXPLAINED_CM = 65.0
MIN_CIRCLE_ON_INK = 0.6
MIN_CIRCLE_ON_GRASS = 0.9
# Area over width plus height of a thin painted curve. A close camera paints
# the circle 8 px thick and a player standing on it adds more; a blob in the
# stands is well over 17.
_THIN_PIECE = 16.0
# Text on the grass (watermark, painted advert): short straight lines (at most
# this share of the width), parallel within a few degrees, this share of the
# height apart, overlapping along most of their length, and at least this share
# of paint (letter strokes) on the line half-way between them.
_TEXT_MAX_LENGTH = 0.35
_TEXT_MAX_ANGLE_DEG = 4.0
_TEXT_MIN_GAP = 0.008
_TEXT_MAX_GAP = 0.06
_TEXT_MIN_OVERLAP = 0.6
_TEXT_INK = 0.2
TWO_WAY_WEIGHT = 0.01
_COST_CAP_PX = 15.0
_UNEXPLAINED_CAP_CM = 500.0
# The far touchline is painted on the grass, below the topmost grass row
# (the grass mask is dilated a few pixels). Paint above that row is the
# bottom edge of the boards or the stands.
_GRASS_EDGE_PX = 0.0

_X = FIELD_CENTRE_X
_Y = FIELD_CENTRE_Y
_CARDINAL = np.array(
    [(_X, _Y - CIRCLE_CM), (_X, _Y + CIRCLE_CM), (_X - CIRCLE_CM, _Y), (_X + CIRCLE_CM, _Y)],
    dtype=np.float32,
)


def _template():
    """Painted marks of the pitch, with weights. The circle comes first.

    The circle and the halfway line weigh most. Far marks weigh less: the
    broadcast lens bends them a few pixels.
    """
    points = []
    weights = []

    def add(marks, weight):
        for mark in marks:
            points.append(mark)
            weights.append(weight)

    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    add([(_X + CIRCLE_CM * np.cos(a), _Y + CIRCLE_CM * np.sin(a)) for a in angles], 2.0)
    circle_count = len(points)
    add([(_X, y) for y in np.linspace(_Y - HALF_WIDTH_CM, _Y + HALF_WIDTH_CM, 60)], 1.5)
    for side in (-1.0, 1.0):
        add([(x, _Y + side * HALF_WIDTH_CM) for x in np.linspace(_X - HALF_LENGTH_CM, _X + HALF_LENGTH_CM, 140)], 0.5)
    for side in (-1.0, 1.0):
        goal = _X + side * HALF_LENGTH_CM
        front = goal - side * BOX_DEPTH_CM
        add([(front, y) for y in np.linspace(_Y - BOX_HALF_CM, _Y + BOX_HALF_CM, 30)], 0.8)
        for edge in (-1.0, 1.0):
            add([(goal - side * BOX_DEPTH_CM * t, _Y + edge * BOX_HALF_CM) for t in np.linspace(0.0, 1.0, 14)], 0.8)
        add([(goal, y) for y in np.linspace(_Y - HALF_WIDTH_CM, _Y + HALF_WIDTH_CM, 40)], 0.4)
        add([(goal - side * 550.0, y) for y in np.linspace(_Y - 916.0, _Y + 916.0, 12)], 0.6)
        spot = goal - side * 1100.0
        arc = [(spot - side * 915.0 * np.cos(a), _Y + 915.0 * np.sin(a)) for a in np.linspace(-np.pi, np.pi, 72)]
        add([mark for mark in arc if (mark[0] - front) * (-side) > 20.0], 1.0)
    return np.array(points, dtype=np.float64), np.array(weights, dtype=np.float64), circle_count


_MARKS, _WEIGHTS, _CIRCLE_COUNT = _template()


def _field_distance(points):
    """Distance, in cm, from field points to the nearest painted mark."""
    x = points[:, 0]
    y = points[:, 1]

    def along_y(x0, y0, y1):
        return np.hypot(x - x0, y - np.clip(y, y0, y1))

    def along_x(y0, x0, x1):
        return np.hypot(x - np.clip(x, x0, x1), y - y0)

    distances = [
        np.abs(np.hypot(x - _X, y - _Y) - CIRCLE_CM),
        along_y(_X, _Y - HALF_WIDTH_CM, _Y + HALF_WIDTH_CM),
        along_x(_Y - HALF_WIDTH_CM, _X - HALF_LENGTH_CM, _X + HALF_LENGTH_CM),
        along_x(_Y + HALF_WIDTH_CM, _X - HALF_LENGTH_CM, _X + HALF_LENGTH_CM),
    ]
    for side in (-1.0, 1.0):
        goal = _X + side * HALF_LENGTH_CM
        front = goal - side * BOX_DEPTH_CM
        low, high = sorted((goal, front))
        distances.append(along_y(goal, _Y - HALF_WIDTH_CM, _Y + HALF_WIDTH_CM))
        distances.append(along_y(front, _Y - BOX_HALF_CM, _Y + BOX_HALF_CM))
        distances.append(along_x(_Y - BOX_HALF_CM, low, high))
        distances.append(along_x(_Y + BOX_HALF_CM, low, high))
        small = goal - side * 550.0
        low, high = sorted((goal, small))
        distances.append(along_y(small, _Y - 916.0, _Y + 916.0))
        distances.append(along_x(_Y - 916.0, low, high))
        distances.append(along_x(_Y + 916.0, low, high))
        spot = goal - side * 1100.0
        arc = np.abs(np.hypot(x - spot, y - _Y) - 915.0)
        distances.append(np.where((x - front) * (-side) > 0.0, arc, 1e9))
    return np.min(np.vstack(distances), axis=0)


def _project(field_to_image, points):
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ field_to_image.T
    depth = homogeneous[:, 2]
    image = np.full((len(points), 2), np.nan)
    ahead = depth > 1e-9
    image[ahead] = homogeneous[ahead, :2] / depth[ahead, None]
    return image


def _on_ink(points, wide):
    height, width = wide.shape[:2]
    finite = np.isfinite(points).all(axis=1)
    xs = np.where(finite, np.rint(points[:, 0]), -1).astype(np.int64)
    ys = np.where(finite, np.rint(points[:, 1]), -1).astype(np.int64)
    inside = (xs >= 0) & (xs < width) & (ys >= height * CROWD_BAND) & (ys < height)
    hit = np.zeros(len(points), dtype=bool)
    hit[inside] = wide[ys[inside], xs[inside]] > 0
    return hit, inside


def _ellipse_points(ellipse, count=240):
    (cx, cy), (a, b), angle = ellipse
    t = np.radians(angle)
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    x = a / 2.0 * np.cos(theta)
    y = b / 2.0 * np.sin(theta)
    return np.stack([cx + x * np.cos(t) - y * np.sin(t), cy + x * np.sin(t) + y * np.cos(t)], axis=1)


def _conic(ellipse):
    (cx, cy), (a, b), angle = ellipse
    t = np.radians(angle)
    c, s = np.cos(t), np.sin(t)
    ra, rb = (a / 2.0) ** 2, (b / 2.0) ** 2
    m = np.array([[c * c / ra + s * s / rb, c * s * (1 / ra - 1 / rb)], [c * s * (1 / ra - 1 / rb), s * s / ra + c * c / rb]])
    centre = np.array([cx, cy])
    conic = np.zeros((3, 3))
    conic[:2, :2] = m
    conic[:2, 2] = -m @ centre
    conic[2, :2] = -m @ centre
    conic[2, 2] = centre @ m @ centre - 1.0
    return conic


def _ellipse_residual(ellipse, points):
    (cx, cy), (a, b), angle = ellipse
    t = np.radians(angle)
    u = (points[:, 0] - cx) * np.cos(t) + (points[:, 1] - cy) * np.sin(t)
    v = -(points[:, 0] - cx) * np.sin(t) + (points[:, 1] - cy) * np.cos(t)
    radius = np.sqrt((u / (a / 2.0)) ** 2 + (v / (b / 2.0)) ** 2)
    return np.abs(radius - 1.0) * min(a, b) / 2.0


def _band_ink(ink, point, direction, length, offset):
    """Share of the line, shifted `offset` px sideways, that lands on paint."""
    height, width = ink.shape[:2]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    steps = np.arange(0.0, float(length), 2.0)
    q = np.rint(np.asarray(point, np.float64) + np.outer(steps, direction) + offset * normal).astype(np.int64)
    inside = (q[:, 0] >= 0) & (q[:, 0] < width) & (q[:, 1] >= 0) & (q[:, 1] < height)
    return float(np.mean(ink[q[inside, 1], q[inside, 0]] > 0)) if inside.any() else 0.0


def text_lines(ink, lines):
    """Indices of the straight lines that are written text, not pitch marks.

    Letters of a watermark or of an advert painted on the grass line up into
    short parallel straight lines a few pixels apart (top, middle and base of
    the letters), with the strokes of the letters packed between them. A pitch
    line is long, or alone, or beside the edge of a board with no strokes in
    between.
    """
    height, width = ink.shape[:2]
    short = width * _TEXT_MAX_LENGTH
    gap_min = height * _TEXT_MIN_GAP
    gap_max = height * _TEXT_MAX_GAP
    found = set()
    for i, (length_i, point_i, dir_i) in enumerate(lines):
        if length_i > short:
            continue
        normal = np.array([-dir_i[1], dir_i[0]], dtype=np.float64)
        for j, (length_j, point_j, dir_j) in enumerate(lines):
            if j <= i or length_j > short:
                continue
            if abs(float(np.dot(dir_i, dir_j))) < np.cos(np.radians(_TEXT_MAX_ANGLE_DEG)):
                continue
            gap = float(np.dot(np.asarray(point_j) - np.asarray(point_i), normal))
            if not gap_min <= abs(gap) <= gap_max:
                continue
            # Overlap along the line.
            start_j = float(np.dot(np.asarray(point_j) - np.asarray(point_i), dir_i))
            ends_j = sorted((start_j, start_j + float(np.dot(dir_j, dir_i)) * length_j))
            overlap = min(length_i, ends_j[1]) - max(0.0, ends_j[0])
            if overlap < _TEXT_MIN_OVERLAP * min(length_i, length_j):
                continue
            if _band_ink(ink, point_i, dir_i, length_i, gap / 2.0) >= _TEXT_INK:
                found.update((i, j))
    return found


def straight_lines(ink, min_length):
    """Long straight paint, longest first. The picture border is left out."""
    segments = cv2.HoughLinesP(ink, 1, np.pi / 360, 60, minLineLength=int(min_length), maxLineGap=25)
    if segments is None:
        return []
    width = ink.shape[1]
    found = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        if max(x1, x2) < 6 or min(x1, x2) > width - 7:
            continue
        direction = np.array([x2 - x1, y2 - y1], dtype=np.float64)
        length = float(np.linalg.norm(direction))
        found.append((length, np.array([x1, y1], dtype=np.float64), direction / length))
    found.sort(key=lambda item: -item[0])
    lines = []
    for length, point, direction in found:
        same = any(
            abs(float((point - other)[0] * d[1] - (point - other)[1] * d[0])) < 10.0
            and abs(float(direction @ d)) > 0.995
            for _length, other, d in lines
        )
        if same:
            continue
        lines.append((length, point, direction))
        if len(lines) >= 10:
            break
    return lines


def _without_lines(ink, lines):
    """The ink with the steep lines cut out, so the circle stands apart from the halfway line."""
    cut = ink.copy()
    for _length, point, direction in lines:
        a = point - direction * 4000.0
        b = point + direction * 4000.0
        cv2.line(cut, tuple(int(v) for v in a), tuple(int(v) for v in b), 0, 15)
    width = cut.shape[1]
    cut[:, :4] = 0
    cut[:, width - 4 :] = 0
    return cut


def _curve_pieces(ink):
    """Thin curved pieces of paint. A straight piece is another line."""
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    ys, xs = np.where(ink > 0)
    member = labels[ys, xs]
    order = np.argsort(member)
    member = member[order]
    points = np.stack([xs, ys], axis=1)[order].astype(np.float64)
    starts = np.searchsorted(member, np.arange(count))
    ends = np.searchsorted(member, np.arange(count), side="right")
    pieces = []
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        span = stats[index, cv2.CC_STAT_WIDTH] + stats[index, cv2.CC_STAT_HEIGHT]
        if area < 40 or max(stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]) < 40:
            continue
        if area > _THIN_PIECE * span:
            continue
        piece = points[starts[index] : ends[index]]
        vx, vy, x0, y0 = cv2.fitLine(piece.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        if float(np.percentile(np.abs((piece - [x0, y0]) @ np.array([-vy, vx])), 90)) < 3.5:
            continue
        pieces.append(piece)
    pieces.sort(key=lambda piece: -len(piece))
    return pieces[:8]


def circle_candidates(ink, wide, steep_lines=(), on_pitch=None):
    """Ellipses that sit on the painted curves. Players and the picture edge are outliers.

    on_pitch, when given, drops curves off the grass before the best few are
    kept, so a logo on the boards cannot take the place of the circle.
    """
    height, width = ink.shape[:2]
    pieces = _curve_pieces(_without_lines(ink, steep_lines))
    if not pieces:
        return []
    every = np.vstack(pieces)
    rng = np.random.default_rng(0)
    if len(every) > 6000:
        every = every[rng.choice(len(every), 6000, replace=False)]
    sizes = np.array([len(piece) for piece in pieces], dtype=np.float64)
    sizes /= sizes.sum()
    found = []
    for _ in range(260):
        piece = pieces[rng.choice(len(pieces), p=sizes)]
        if len(piece) < 20:
            continue
        try:
            ellipse = cv2.fitEllipse(piece[rng.choice(len(piece), 6, replace=False)].astype(np.float32))
        except cv2.error:
            continue
        (cx, cy), (a, b), _angle = ellipse
        if not np.isfinite([cx, cy, a, b]).all() or min(a, b) < 25 or max(a, b) > 2.5 * width:
            continue
        if max(a, b) / max(min(a, b), 1.0) > 8.0:
            continue
        inliers = int(np.count_nonzero(_ellipse_residual(ellipse, every) < 3.0))
        if inliers >= 60:
            found.append((inliers, ellipse))
    found.sort(key=lambda item: -item[0])
    candidates = []
    seen = []
    for _inliers, ellipse in found[:40]:
        support = every[_ellipse_residual(ellipse, every) < 3.0]
        try:
            ellipse = cv2.fitEllipse(support.astype(np.float32))
        except cv2.error:
            continue
        (cx, cy), (a, b), _angle = ellipse
        if min(a, b) < 25 or max(a, b) > 2.5 * width:
            continue
        if any(abs(cx - s[0]) < 8 and abs(cy - s[1]) < 8 and abs(max(a, b) - s[2]) < 10 for s in seen):
            continue
        hit, inside = _on_ink(_ellipse_points(ellipse), wide)
        if int(np.count_nonzero(inside)) < 40:
            continue
        cover = float(hit[inside].mean())
        if cover < MIN_CIRCLE_ON_INK:
            continue
        if on_pitch is not None and not on_pitch(ellipse):
            continue
        seen.append((cx, cy, max(a, b)))
        candidates.append((float(hit.sum()) * cover, ellipse))
        if len(candidates) >= 4:
            break
    candidates.sort(key=lambda item: -item[0])
    return [ellipse for _score, ellipse in candidates]


def _grass_top(grass):
    """Topmost grass row of each column, smoothed. Above it are the boards and the stands."""
    height, width = grass.shape[:2]
    # Green in the stands comes in narrow pockets: pillars, shadows, seats.
    # The pitch is wide, so only grass that runs across the picture counts,
    # and only the piece joined to the bottom of the picture.
    run = max(31, int(width * 0.08)) | 1
    wide = cv2.morphologyEx(
        (grass > 0).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1))
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(wide, connectivity=8)
    if count > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[biggest, cv2.CC_STAT_AREA] >= 0.2 * height * width:
            wide = (labels == biggest).astype(np.uint8)
    has = wide > 0
    rows = np.where(has.any(axis=0), has.argmax(axis=0), height).astype(np.float64)
    padded = np.pad(rows, 15, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 31)
    return np.median(windows, axis=1)


class _Frame:
    """Everything one frame's search needs, computed once."""

    def __init__(self, ink, grass):
        self.ink = ink
        self.height, self.width = ink.shape[:2]
        self.wide = cv2.dilate(ink, np.ones((5, 5), np.uint8))
        self.distance = cv2.distanceTransform(cv2.bitwise_not(ink), cv2.DIST_L2, 3)
        self.grass_top = _grass_top(grass) if grass is not None else None
        every = straight_lines(ink, self.height * 0.18)
        self.steep = [line for line in every if abs(line[2][1]) > 0.55]
        self.flat = [line for line in every if abs(line[2][0]) > 0.9 and line[0] > 0.35 * self.width]
        self.circles = circle_candidates(ink, self.wide, self.steep, on_pitch=self._on_grass)
        self.samples = self._samples(straight_lines(ink, self.height * 0.10))

    def _on_grass(self, ellipse):
        """True when the ellipse lies on the pitch. Round logos on the boards are curves too."""
        curve = _ellipse_points(ellipse, 120)
        inside = (curve[:, 0] >= 0) & (curve[:, 0] < self.width) & (curve[:, 1] >= 0) & (curve[:, 1] < self.height)
        if not np.any(inside):
            return False
        curve = curve[inside]
        xs = curve[:, 0].astype(np.int64)
        if self.grass_top is not None:
            low = curve[:, 1] >= self.grass_top[xs] - _GRASS_EDGE_PX
        else:
            low = curve[:, 1] >= self.height * CROWD_BAND
        return float(low.mean()) >= MIN_CIRCLE_ON_GRASS

    def _samples(self, lines):
        points = []
        text = text_lines(self.ink, lines)
        for index, (length, point, direction) in enumerate(lines):
            if index in text:
                continue
            points.extend(point + direction * t for t in np.linspace(0.0, length, max(2, int(length / 18.0))))
        for ellipse in self.circles[:1]:
            curve = _ellipse_points(ellipse, 60)
            hit, _inside = _on_ink(curve, self.wide)
            points.extend(curve[hit])
        if not points:
            return np.zeros((0, 2))
        points = np.array(points, dtype=np.float64)
        # Only paint on the grass has to be a pitch mark. The edges of the
        # boards, the LED panels and the railings in the stands are long
        # straight lines too, and a line running off the touchline into the
        # boards keeps its straight piece up there.
        xs = np.clip(points[:, 0].astype(np.int64), 0, self.width - 1)
        if self.grass_top is not None:
            on_grass = points[:, 1] >= self.grass_top[xs] - _GRASS_EDGE_PX
        else:
            on_grass = points[:, 1] >= self.height * CROWD_BAND
        return points[on_grass]

    def chamfer(self, field_to_image):
        """Weighted distance, in pixels, from the projected marks to the ink, or None."""
        image = _project(field_to_image, _MARKS)
        inside = (
            np.isfinite(image).all(axis=1)
            & (image[:, 0] >= 0) & (image[:, 0] < self.width)
            & (image[:, 1] >= 0) & (image[:, 1] < self.height)
        )
        circle = inside[:_CIRCLE_COUNT]
        if int(np.count_nonzero(circle)) < _CIRCLE_COUNT * 0.25:
            return None
        spread = np.ptp(image[:_CIRCLE_COUNT][circle], axis=0)
        if float(spread.max()) < 60.0 or float(spread.min()) < 15.0:
            return None
        xs = image[inside, 0].astype(np.int64)
        ys = image[inside, 1].astype(np.int64)
        reach = np.minimum(self.distance[ys, xs], _COST_CAP_PX)
        # A mark in the stands is as wrong as a mark far from the paint.
        if self.grass_top is not None:
            reach = np.where(ys >= self.grass_top[xs] - 25.0, reach, _COST_CAP_PX)
        else:
            reach = np.where(ys >= self.height * CROWD_BAND, reach, _COST_CAP_PX)
        weights = _WEIGHTS[inside]
        return float(np.sum(reach * weights) / np.sum(weights))

    def unexplained(self, field_to_image):
        """Mean distance, in cm, from the long paint in the picture to the nearest pitch mark."""
        if len(self.samples) == 0:
            return 0.0
        try:
            image_to_field = np.linalg.inv(field_to_image)
        except np.linalg.LinAlgError:
            return _UNEXPLAINED_CAP_CM
        homogeneous = np.column_stack([self.samples, np.ones(len(self.samples))]) @ image_to_field.T
        depth = homogeneous[:, 2]
        behind = depth <= 1e-9
        field = homogeneous[:, :2] / np.where(behind, 1.0, depth)[:, None]
        distance = np.minimum(_field_distance(field), _UNEXPLAINED_CAP_CM)
        return float(np.mean(np.where(behind, _UNEXPLAINED_CAP_CM, distance)))

    def circle_on_ink(self, field_to_image):
        hit, inside = _on_ink(_project(field_to_image, _MARKS[:_CIRCLE_COUNT]), self.wide)
        return float(hit[inside].mean()) if np.any(inside) else 0.0

    def total(self, field_to_image):
        chamfer = self.chamfer(field_to_image)
        if chamfer is None:
            return None
        return chamfer + TWO_WAY_WEIGHT * self.unexplained(field_to_image)


def _keeps_hand(field_to_image):
    """The same handedness as the rest of the map: image right is the high-x half."""
    points = _project(field_to_image, np.array([(_X, _Y), (_X + 100.0, _Y), (_X, _Y + 100.0)]))
    if not np.isfinite(points).all():
        return False
    a = points[1] - points[0]
    b = points[2] - points[0]
    return float(a[0] * b[1] - a[1] * b[0]) > 0.0


def _from_cardinals(points):
    matrix = cv2.getPerspectiveTransform(_CARDINAL, np.asarray(points, dtype=np.float32)).astype(np.float64)
    if not np.isfinite(matrix).all() or not _keeps_hand(matrix):
        return None
    return matrix


def _refine(view, cardinals, steps=(8.0, 4.0, 2.0, 1.0, 0.5)):
    """Coordinate descent on the image of the circle's four cardinal points."""

    def evaluate(points):
        matrix = _from_cardinals(points)
        if matrix is None:
            return None
        cost = view.total(matrix)
        return None if cost is None else (cost, matrix, points)

    best = evaluate(np.asarray(cardinals, dtype=np.float64))
    if best is None:
        return None
    for step in steps:
        moved = True
        rounds = 0
        while moved and rounds < 40:
            moved = False
            rounds += 1
            for index in range(4):
                for shift in _MOVES:
                    points = best[2].copy()
                    points[index] = points[index] + shift * step
                    tried = evaluate(points)
                    if tried is not None and tried[0] < best[0] - 1e-4:
                        best = tried
                        moved = True
            # Two ends of the circle moving together follow a valley that single moves cannot.
            for first, second in ((0, 1), (2, 3), (0, 2), (1, 3)):
                for shift in _MOVES:
                    points = best[2].copy()
                    points[first] = points[first] + shift * step
                    points[second] = points[second] + shift * step
                    tried = evaluate(points)
                    if tried is not None and tried[0] < best[0] - 1e-4:
                        best = tried
                        moved = True
    return best


_MOVES = [np.array(move, dtype=np.float64) for move in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))]


def _seeds(ellipse, point, direction):
    """Cardinal points of the circle from its ellipse and the halfway line through it.

    The line meets the ellipse at the circle's two y ends. The pole of the line
    is where the x direction vanishes, so the line from the circle's centre
    toward it crosses the ellipse at the two x ends. The centre's place on the
    halfway line is the one free parameter, and a few are tried.
    """
    curve = _ellipse_points(ellipse, 720)
    normal = np.array([-direction[1], direction[0]])
    side = (curve - point) @ normal
    flips = np.where(np.sign(side[:-1]) != np.sign(side[1:]))[0]
    if len(flips) < 2:
        return []
    crossings = []
    for index in flips[:2]:
        t = side[index] / (side[index] - side[index + 1])
        crossings.append(curve[index] + t * (curve[index + 1] - curve[index]))
    top, bottom = sorted(crossings, key=lambda p: p[1])
    left_reach = -float(side.min())
    right_reach = float(side.max())
    # The halfway line cuts the circle near its middle. The 16.5 m line cuts a penalty arc near its end.
    if not 0.28 <= left_reach / (left_reach + right_reach) <= 0.72:
        return []
    line = np.cross(np.append(point, 1.0), np.append(point + direction, 1.0))
    pole = np.linalg.solve(_conic(ellipse), line)
    seeds = []
    for t in np.linspace(0.3, 0.7, 9):
        centre = top + (bottom - top) * t
        toward = pole[:2] / pole[2] - centre if abs(pole[2]) > 1e-12 else pole[:2]
        toward = toward / np.linalg.norm(toward)
        across = (curve - centre) @ np.array([-toward[1], toward[0]])
        cuts = np.where(np.sign(across[:-1]) != np.sign(across[1:]))[0]
        if len(cuts) < 2:
            continue
        ends = []
        for index in cuts[:2]:
            t2 = across[index] / (across[index] - across[index + 1])
            ends.append(curve[index] + t2 * (curve[index + 1] - curve[index]))
        left, right = sorted(ends, key=lambda p: p[0])
        seeds.append(np.array([top, bottom, left, right]))
    return seeds


def _start_points(view):
    """Starting fits: the circle with each steep line, else the circle with the far touchline."""
    starts = []
    for ellipse in view.circles:
        for _length, point, direction in view.steep:
            for cardinals in _seeds(ellipse, point, direction):
                matrix = _from_cardinals(cardinals)
                cost = None if matrix is None else view.total(matrix)
                if cost is not None:
                    starts.append((cost, cardinals))
    if starts or not view.flat:
        return starts
    # No halfway line in the picture. The x direction vanishes somewhere on the far
    # touchline, and its polar is the halfway line.
    view.flat.sort(key=lambda line: float(line[1][1]))
    _length, point, direction = view.flat[0]
    if direction[0] < 0.0:
        direction = -direction
    for ellipse in view.circles[:3]:
        conic = _conic(ellipse)
        for angle in np.linspace(-1.5, 1.5, 90):
            vanishing = np.append(point + direction * np.tan(angle) * view.width, 1.0)
            polar = conic @ vanishing
            norm = float(np.hypot(polar[0], polar[1]))
            if norm < 1e-12:
                continue
            on_line = -polar[2] * polar[:2] / norm**2
            along = np.array([polar[1], -polar[0]]) / norm
            for cardinals in _seeds(ellipse, on_line, along):
                matrix = _from_cardinals(cardinals)
                cost = None if matrix is None else view.total(matrix)
                if cost is not None:
                    starts.append((cost, cardinals))
    return starts


def _accept(view, found):
    if found is None:
        return None
    chamfer = view.chamfer(found[1])
    if chamfer is None or chamfer > MAX_COST_PX:
        return None
    unexplained = view.unexplained(found[1])
    if unexplained > MAX_UNEXPLAINED_CM or view.circle_on_ink(found[1]) < MIN_CIRCLE_ON_INK:
        return None
    return found[1], chamfer, unexplained


def _report(field_to_image, chamfer, unexplained):
    image_to_field = np.linalg.inv(field_to_image)
    image_to_field = image_to_field / image_to_field[2, 2]
    ends = _project(field_to_image, np.array([(_X - CIRCLE_CM, _Y), (_X + CIRCLE_CM, _Y)]))
    span = float(np.linalg.norm(ends[1] - ends[0]))
    return {
        "matrix": image_to_field,
        "positive_x": True,
        "cm_per_px": 2.0 * CIRCLE_CM / span if span > 1e-6 else None,
        "front_x_cm": FIELD_CENTRE_X,
        "region": "centre",
        "projective": True,
        "cost": chamfer,
        "unexplained_cm": unexplained,
    }


def fit_centre_on_ink(frame, grass, white_mask=None, prior=None, ink=None):
    """Image-to-field report around the centre circle, or None.

    prior is the image-to-field matrix on screen. When it is given it is
    tried first and only refined, which is what a moving camera needs.
    """
    if ink is None:
        from pitch_map import line_ink

        ink = line_ink(frame, grass, white_mask)
    if ink is None:
        return None
    view = _Frame(ink, grass)
    if prior is not None:
        try:
            field_to_image = np.linalg.inv(np.asarray(prior, dtype=np.float64))
        except np.linalg.LinAlgError:
            field_to_image = None
        if field_to_image is not None:
            cardinals = _project(field_to_image, _CARDINAL.astype(np.float64))
            if np.isfinite(cardinals).all():
                accepted = _accept(view, _refine(view, cardinals, steps=(4.0, 2.0, 1.0, 0.5)))
                if accepted is not None:
                    return _report(*accepted)
    starts = _start_points(view)
    starts.sort(key=lambda item: item[0])
    best = None
    for _cost, cardinals in starts[:8]:
        found = _refine(view, cardinals)
        if found is not None and (best is None or found[0] < best[0]):
            best = found
    accepted = _accept(view, best)
    return None if accepted is None else _report(*accepted)


def centre_holds(matrix, grass, ink):
    """True when the centre matrix on screen still lands on this frame's ink."""
    if matrix is None or ink is None:
        return False
    try:
        field_to_image = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    except np.linalg.LinAlgError:
        return False
    view = _Frame.__new__(_Frame)
    view.ink = ink
    view.height, view.width = ink.shape[:2]
    view.wide = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    view.distance = cv2.distanceTransform(cv2.bitwise_not(ink), cv2.DIST_L2, 3)
    view.grass_top = _grass_top(grass) if grass is not None else None
    chamfer = view.chamfer(field_to_image)
    return chamfer is not None and chamfer <= MAX_COST_PX and view.circle_on_ink(field_to_image) >= MIN_CIRCLE_ON_INK
