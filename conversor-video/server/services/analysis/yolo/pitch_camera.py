"""The broadcast camera: fixed on the halfway line, it only pans, tilts and zooms.

Every clip the program takes comes from the main camera, high in the stand
on the halfway line. A free homography per frame has 8 numbers and nothing in
them says where the camera stands, so a fit read off one penalty area can put
the camera near the corner flag and turn the whole map. This module fixes the
camera centre and leaves each frame only its rotation and its zoom.

The camera centre is measured once per clip, on its first frames with a map:
its x is the halfway line, its distance behind the near touchline and its
height are the ones whose camera lays the whole pitch template on the thin
marks of those frames best. Each later frame starts from the previous one
(or from the map fit of that frame) and is pulled onto the thin marks.

All the geometry here is in radar centimetres: a 105 x 68 m pitch, origin at
a corner, as pitch_radar draws it. The map matrices come in their own field
frames, which are the radar frame shifted, so they are moved in and out by a
translation.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

from pitch_centre import _MARKS, _WEIGHTS, FIELD_CENTRE_X, FIELD_CENTRE_Y
from pitch_radar import RADAR_LENGTH_CM, RADAR_WIDTH_CM, to_radar

HALFWAY_X_CM = RADAR_LENGTH_CM / 2.0
_MIDDLE_Y_CM = RADAR_WIDTH_CM / 2.0
# Stands tried for the camera: distance behind the near touchline, height.
BEHIND_CANDIDATES_CM = (2800.0, 3500.0, 4200.0, 5000.0, 6000.0, 7200.0)
HEIGHT_CANDIDATES_CM = (1000.0, 1300.0, 1600.0, 2000.0, 2500.0, 3100.0)
CALIBRATION_FRAMES = 10
# Mean distance in pixels from the projected template to the nearest thin mark.
ACCEPT_COST_PX = 4.5
# Frames a lost camera keeps its last pose.
HOLD_FRAMES = 15
# A pose this close to the marks is not published, but the next frame starts from it.
FOLLOW_COST_PX = 9.0
_COST_CAP_PX = 15.0
_SMOOTH_NEW = 0.6
_SMOOTH_MAX_TURN = np.radians(1.5)
_SMOOTH_MAX_ZOOM = 0.04
_FOCAL_SEEDS = tuple(np.geomspace(800.0, 9000.0, 7))
_PARAM_SCALE = np.array([0.01, 0.01, 0.01, 0.05])

# The whole-pitch marks of pitch_centre, moved into radar centimetres.
_TEMPLATE = _MARKS - np.array([FIELD_CENTRE_X - HALFWAY_X_CM, FIELD_CENTRE_Y - _MIDDLE_Y_CM])


def _shift(region, positive_x):
    """Radar = field + shift, for one map region."""
    return np.array(to_radar(0.0, 0.0, region, positive_x), dtype=np.float64)


def _field_to_radar(shift):
    return np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]])


def _radar_to_field(shift):
    return np.array([[1.0, 0.0, -shift[0]], [0.0, 1.0, -shift[1]], [0.0, 0.0, 1.0]])


def camera_centre(behind_cm, height_cm, x_cm=HALFWAY_X_CM):
    """Radar point of a camera behind the near touchline. z points down, so the height is negative."""
    return np.array([float(x_cm), RADAR_WIDTH_CM + float(behind_cm), -float(height_cm)])


def _homography(params, centre, width, height):
    """Radar-to-image homography of a camera at centre. params: rotation vector, log focal."""
    rotation, _jacobian = cv2.Rodrigues(np.asarray(params[:3], dtype=np.float64))
    focal = float(np.exp(params[3]))
    k = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    t = -rotation @ centre
    return k @ np.column_stack([rotation[:, 0], rotation[:, 1], t])


def _project(homography, points):
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ homography.T
    depth = homogeneous[:, 2]
    image = np.full((len(points), 2), np.nan)
    ahead = depth > 1e-9
    image[ahead] = homogeneous[ahead, :2] / depth[ahead, None]
    return image


def _camera_project(params, points, centre, width, height):
    """Image points of radar points, NaN behind the camera."""
    rotation, _jacobian = cv2.Rodrigues(np.asarray(params[:3], dtype=np.float64))
    focal = float(np.exp(params[3]))
    camera = (np.column_stack([points, np.zeros(len(points))]) - centre) @ rotation.T
    depth = camera[:, 2]
    ahead = depth > 1e-6
    depth = np.where(ahead, depth, 1.0)
    image = np.column_stack(
        [focal * camera[:, 0] / depth + width / 2.0, focal * camera[:, 1] / depth + height / 2.0]
    )
    image[~ahead] = np.nan
    return image


def _trusted_marks(region, positive_x, projective):
    """Radar points where this kind of fit is known to sit on the paint."""
    points = []
    if region == "centre":
        for angle in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
            points.append((HALFWAY_X_CM + 915.0 * np.cos(angle), _MIDDLE_Y_CM + 915.0 * np.sin(angle)))
        for y in np.linspace(_MIDDLE_Y_CM - 1800.0, _MIDDLE_Y_CM + 1800.0, 16):
            points.append((HALFWAY_X_CM, y))
        return np.array(points)
    goal = RADAR_LENGTH_CM if positive_x else 0.0
    inward = -1.0 if positive_x else 1.0
    front = goal + inward * 1650.0
    for y in np.linspace(_MIDDLE_Y_CM - 2016.0, _MIDDLE_Y_CM + 2016.0, 24):
        points.append((front, y))
    spot = goal + inward * 1100.0
    reach = np.arccos(550.0 / 915.0)
    for angle in np.linspace(-reach, reach, 16):
        points.append((spot + inward * 915.0 * np.cos(angle), _MIDDLE_Y_CM + 915.0 * np.sin(angle)))
    # An arc fit is exact on the 16.5 m line and the arc only. A box fit holds the whole box.
    depth = 1650.0 if projective else 500.0
    for edge in (-1.0, 1.0):
        for step in np.linspace(0.0, 1.0, 8):
            points.append((front - inward * depth * step, _MIDDLE_Y_CM + edge * 2016.0))
    if projective:
        for y in np.linspace(_MIDDLE_Y_CM - 2016.0, _MIDDLE_Y_CM + 2016.0, 12):
            points.append((goal, y))
    return np.array(points)


def _look_at(centre, target):
    """Rotation vector of a level camera at centre looking at a pitch point."""
    forward = target - centre
    forward = forward / np.linalg.norm(forward)
    down = np.array([0.0, 0.0, 1.0])
    y_axis = down - forward * float(forward @ down)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, forward)
    rotation = np.vstack([x_axis, y_axis, forward])
    rvec, _jacobian = cv2.Rodrigues(rotation)
    return rvec.reshape(-1)


class InkField:
    """The thin marks of one frame: distance to the nearest mark and where that mark is."""

    def __init__(self, ink, grass):
        self.height, self.width = ink.shape[:2]
        marks = (ink > 0).astype(np.uint8) * 255
        self.empty = not marks.any()
        if self.empty:
            return
        self.distance, labels = cv2.distanceTransformWithLabels(
            cv2.bitwise_not(marks), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        ys, xs = np.nonzero(marks)
        self.nearest = np.zeros((int(labels.max()) + 1, 2), dtype=np.float64)
        self.nearest[labels[ys, xs]] = np.column_stack([xs, ys])
        self.labels = labels
        has = grass > 0
        self.top = np.where(has.any(axis=0), has.argmax(axis=0), self.height).astype(np.float64)

    def _inside(self, image):
        inside = (
            np.isfinite(image).all(axis=1)
            & (image[:, 0] >= 0)
            & (image[:, 0] < self.width - 1)
            & (image[:, 1] >= 0)
            & (image[:, 1] < self.height - 1)
        )
        index = np.nonzero(inside)[0]
        xs = image[index, 0].astype(int)
        ys = image[index, 1].astype(int)
        # A mark drawn above the grass is in the crowd: it never matches.
        on_grass = ys >= self.top[xs] - 25
        return index, xs, ys, on_grass

    def cost(self, params, centre):
        """Weighted mean pixel distance from the template to the thin marks, capped per point."""
        if self.empty:
            return _COST_CAP_PX
        image = _camera_project(params, _TEMPLATE, centre, self.width, self.height)
        index, xs, ys, on_grass = self._inside(image)
        if len(index) < 40:
            return _COST_CAP_PX
        reach = np.minimum(self.distance[ys, xs], _COST_CAP_PX)
        reach = np.where(on_grass, reach, _COST_CAP_PX)
        weights = _WEIGHTS[index]
        return float((reach * weights).sum() / weights.sum())

    def pairs(self, params, centre, reach_px):
        """Template points near a thin mark, and the nearest mark pixel of each."""
        if self.empty:
            return np.zeros(0, dtype=int), np.zeros((0, 2))
        image = _camera_project(params, _TEMPLATE, centre, self.width, self.height)
        index, xs, ys, on_grass = self._inside(image)
        near = on_grass & (self.distance[ys, xs] < reach_px)
        targets = self.nearest[self.labels[ys[near], xs[near]]]
        return index[near], targets


def _pull(field, params, centre, rounds=6):
    """Rotation and zoom pulled onto the thin marks, point to nearest mark, a few rounds."""
    params = np.asarray(params, dtype=np.float64)
    for round_index in range(rounds):
        index, targets = field.pairs(params, centre, 40.0 if round_index == 0 else 25.0 if round_index == 1 else 12.0)
        if len(index) < 20:
            break
        points = _TEMPLATE[index]
        weights = np.sqrt(_WEIGHTS[index])[:, None]

        def residual(values, points=points, targets=targets, weights=weights):
            image = _camera_project(values, points, centre, field.width, field.height)
            return np.nan_to_num((image - targets) * weights, nan=1e3).ravel()

        try:
            result = least_squares(
                residual, params, loss="soft_l1", f_scale=2.0, max_nfev=60, x_scale=_PARAM_SCALE
            )
        except ValueError:
            break
        params = result.x
    return params, field.cost(params, centre)


def _seed(fit_to_image, region, positive_x, projective, centre, width, height):
    """Rotation and zoom of a camera at centre that lay the trusted marks where the map fit puts them."""
    marks = _trusted_marks(region, positive_x, projective)
    image = _project(fit_to_image, marks)
    keep = (
        np.isfinite(image).all(axis=1)
        & (image[:, 0] > 0)
        & (image[:, 0] < width)
        & (image[:, 1] > 0)
        & (image[:, 1] < height)
    )
    if keep.sum() < 8:
        return None
    marks = marks[keep]
    image = image[keep]
    rotation = _look_at(centre, np.array([*marks.mean(axis=0), 0.0]))
    best = None
    for focal in _FOCAL_SEEDS:

        def residual(values):
            projected = _camera_project(values, marks, centre, width, height)
            return np.nan_to_num(projected - image, nan=1e4).ravel()

        try:
            result = least_squares(residual, np.array([*rotation, np.log(focal)]), max_nfev=300)
        except ValueError:
            continue
        if best is None or result.cost < best.cost:
            best = result
    return None if best is None else best.x


def _radar_to_image_of_fit(image_to_field, region, positive_x):
    """The map fit (image to field) as a radar-to-image homography."""
    shift = _shift(region, positive_x)
    homography = np.linalg.inv(np.asarray(image_to_field, dtype=np.float64)) @ _radar_to_field(shift)
    return homography / homography[2, 2]


def _motion(previous, new):
    """Rotation and zoom from one pose to the next."""
    old_rotation, _jacobian = cv2.Rodrigues(np.asarray(previous[:3], dtype=np.float64))
    new_rotation, _jacobian = cv2.Rodrigues(np.asarray(new[:3], dtype=np.float64))
    turn, _jacobian = cv2.Rodrigues(new_rotation @ old_rotation.T)
    return turn.reshape(-1), float(new[3] - previous[3])


def _moved(pose, motion):
    """The pose one frame on, if the camera keeps panning and zooming as it did."""
    turn, zoom = motion
    step, _jacobian = cv2.Rodrigues(np.asarray(turn, dtype=np.float64))
    rotation, _jacobian = cv2.Rodrigues(np.asarray(pose[:3], dtype=np.float64))
    rvec, _jacobian = cv2.Rodrigues(step @ rotation)
    return np.array([*rvec.reshape(-1), pose[3] + zoom])


class BroadcastCamera:
    """The clip's main camera. Re-solves each map frame with the camera centre fixed."""

    def __init__(
        self,
        behind_candidates=BEHIND_CANDIDATES_CM,
        height_candidates=HEIGHT_CANDIDATES_CM,
        calibration_frames=CALIBRATION_FRAMES,
        accept_cost_px=ACCEPT_COST_PX,
        hold_frames=HOLD_FRAMES,
    ):
        self.candidates = [camera_centre(b, h) for b in behind_candidates for h in height_candidates]
        self.calibration_frames = int(calibration_frames)
        self.accept_cost_px = float(accept_cost_px)
        self.hold_frames = int(hold_frames)
        self.centre = None
        self._pending = []
        self._params = None
        self._follow = None
        self._motion = None
        self._shape = None
        self._missed = 0
        self.solved_frames = 0
        self.held_frames = 0
        self.missed_frames = 0
        self.last_cost = None

    def solve(self, image_to_field, region, positive_x, projective, ink, grass, shape):
        """Image-to-field matrix of this frame from the fixed camera, in the fit's field frame, or None.

        image_to_field is this frame's map matrix, the start for the camera.
        None means the camera is not known yet, or lost: the caller keeps its own matrix.
        """
        region = region or "penalty"
        if ink is None or grass is None or image_to_field is None:
            return self._held(region, positive_x)
        height, width = shape[:2]
        if self._shape is not None and self._shape != (height, width):
            self.reset()
        self._shape = (height, width)
        fit = _radar_to_image_of_fit(image_to_field, region, positive_x)
        field = InkField(ink, grass)
        if self.centre is None:
            self._pending.append((field, fit, region, positive_x, bool(projective)))
            if len(self._pending) < self.calibration_frames:
                return None
            self._calibrate()
        previous = self._follow if self._follow is not None else self._params
        motion = self._motion
        params, cost = self._track(field, fit, region, positive_x, projective, previous, motion)
        self.last_cost = cost
        if params is not None and previous is not None and cost <= FOLLOW_COST_PX:
            self._motion = _motion(previous, params)
        if params is None or cost > FOLLOW_COST_PX:
            self._follow = None
            self._motion = None
        if params is None or cost > self.accept_cost_px:
            if params is not None and cost <= FOLLOW_COST_PX:
                self._follow = params
            return self._held(region, positive_x, missed=True)
        self._follow = None
        # The smoothing starts from where the pan going on puts the camera, so a steady pan does not lag.
        expected = self._params
        if expected is not None and motion is not None and previous is self._params:
            expected = _moved(expected, motion)
        self._params = self._smooth(expected, params)
        self._missed = 0
        self.solved_frames += 1
        return self._image_to_field(region, positive_x)

    def reset(self):
        self.centre = None
        self._pending = []
        self._params = None
        self._follow = None
        self._motion = None
        self._missed = 0

    def radar_to_image(self):
        if self.centre is None or self._params is None or self._shape is None:
            return None
        height, width = self._shape
        return _homography(self._params, self.centre, width, height)

    def metrics(self):
        centre = self.centre
        return {
            "method": "camera fixed near the halfway line, pan, tilt and zoom read off the thin marks",
            "camera_x_m": None if centre is None else round(float(centre[0]) / 100.0, 2),
            "camera_behind_touchline_m": None
            if centre is None
            else round(float(centre[1] - RADAR_WIDTH_CM) / 100.0, 2),
            "camera_height_m": None if centre is None else round(float(-centre[2]) / 100.0, 2),
            "solved_frames": self.solved_frames,
            "held_frames": self.held_frames,
            "missed_frames": self.missed_frames,
        }

    def _track(self, field, fit, region, positive_x, projective, previous, motion=None):
        """Best pose of this frame from the previous pose, the pan going on, and the map fit."""
        best = (None, _COST_CAP_PX)
        starts = []
        if previous is not None:
            starts.append(previous)
            if motion is not None:
                starts.append(_moved(previous, motion))
        for start in starts:
            params, cost = _pull(field, start, self.centre)
            if cost < best[1]:
                best = (params, cost)
            # The pose already lands: the frame moved little.
            if best[1] <= self.accept_cost_px * 0.6:
                return best
        seed = _seed(fit, region, positive_x, projective, self.centre, field.width, field.height)
        if seed is not None:
            params, cost = _pull(field, seed, self.centre)
            if cost < best[1]:
                best = (params, cost)
        return best

    def _calibrate(self):
        """The stand whose camera follows the first map frames best.

        Every candidate follows the first few frames, the best ones follow all
        of them, then the distance and the height are fine-tuned around the winner.
        """
        quick = self._pending[: max(3, self.calibration_frames // 3)]
        ranked = sorted(self.candidates, key=lambda centre: self._follow_cost(centre, quick))
        scored = [(self._follow_cost(centre, self._pending), tuple(centre)) for centre in ranked[:4]]
        best_cost, best = min(scored)
        best = np.array(best)
        step_behind, step_height = 400.0, 200.0
        for _round in range(6):
            moved = False
            for d_behind, d_height in (
                (step_behind, 0.0),
                (-step_behind, 0.0),
                (0.0, step_height),
                (0.0, -step_height),
                (step_behind, step_height),
                (-step_behind, -step_height),
            ):
                centre = best + np.array([0.0, d_behind, -d_height])
                if centre[1] < RADAR_WIDTH_CM + 1000.0 or -centre[2] < 500.0:
                    continue
                cost = self._follow_cost(centre, self._pending)
                if cost < best_cost:
                    best_cost, best, moved = cost, centre, True
            if not moved:
                step_behind /= 2.0
                step_height /= 2.0
        self.centre = best
        # The calibration frames are past: tracking goes on from their last pose.
        self._params = None
        for field, fit, region, positive_x, projective in self._pending:
            params, cost = self._track(field, fit, region, positive_x, projective, self._params)
            if params is not None and cost <= FOLLOW_COST_PX:
                self._params = params
        self._pending = []

    def _follow_cost(self, centre, frames):
        """Summed pixel cost of a camera at centre following frames, one from the last."""
        self.centre = np.asarray(centre, dtype=np.float64)
        previous = None
        total = 0.0
        for field, fit, region, positive_x, projective in frames:
            params, cost = self._track(field, fit, region, positive_x, projective, previous)
            total += min(cost, 8.0)
            if params is not None and cost <= FOLLOW_COST_PX:
                previous = params
        return total

    def _smooth(self, previous, new):
        """A small move is blended with the last pose. A real pan or zoom is taken as it is."""
        if previous is None:
            return new
        old_rotation, _jacobian = cv2.Rodrigues(np.asarray(previous[:3], dtype=np.float64))
        new_rotation, _jacobian = cv2.Rodrigues(np.asarray(new[:3], dtype=np.float64))
        turn, _jacobian = cv2.Rodrigues(new_rotation @ old_rotation.T)
        if np.linalg.norm(turn) > _SMOOTH_MAX_TURN or abs(new[3] - previous[3]) > _SMOOTH_MAX_ZOOM:
            return new
        blended_turn, _jacobian = cv2.Rodrigues(turn.reshape(-1) * _SMOOTH_NEW)
        rotation, _jacobian = cv2.Rodrigues(blended_turn @ old_rotation)
        log_focal = previous[3] + _SMOOTH_NEW * (new[3] - previous[3])
        return np.array([*rotation.reshape(-1), log_focal])

    def _held(self, region, positive_x, missed=False):
        if missed:
            self.missed_frames += 1
        if self._params is None or self.centre is None or self._missed >= self.hold_frames:
            return None
        self._missed += 1
        self.held_frames += 1
        return self._image_to_field(region, positive_x)

    def _image_to_field(self, region, positive_x):
        homography = self.radar_to_image()
        if homography is None:
            return None
        shift = _shift(region, positive_x)
        matrix = _radar_to_field(shift) @ np.linalg.inv(homography)
        return matrix / matrix[2, 2]
