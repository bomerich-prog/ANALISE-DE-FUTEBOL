"""How far the broadcast camera turned since the last frame the map was solved on.

The 2D map is solved on one frame out of every few. In between, the camera
keeps panning, so the frames in the middle reuse that solved map after one
step: points on the grass are followed from the solved frame to this one, and
the move that most of them agree on is the camera's. A camera that only turns
and zooms moves every point of the picture by one homography, whatever its
depth. Players run on their own and are left out of the search. When too few
points agree, the move is not read and the caller solves the map again.
"""

from __future__ import annotations

import cv2
import numpy as np

# The search runs on a picture half the size: the pan is still a few pixels there.
MOTION_SCALE = 0.5
MOTION_MAX_POINTS = 500
MOTION_QUALITY = 0.005
MOTION_MIN_DISTANCE_PX = 10
# A point is kept only when following it back lands where it started.
MOTION_ROUND_TRIP_PX = 1.0
MOTION_RANSAC_PX = 1.5
MOTION_MIN_POINTS = 30
MOTION_MIN_SHARE = 0.5
# Around a player's box: the shadow and the legs move with him.
MOTION_PLAYER_MARGIN_PX = 12
# Past these, the picture changed rather than turned: a cut, a replay, a zoom jump.
MOTION_MAX_ZOOM_STEP = 0.2
MOTION_MAX_SHIFT_SHARE = 0.25


def _small_gray(frame):
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, None, fx=MOTION_SCALE, fy=MOTION_SCALE, interpolation=cv2.INTER_AREA)


def _search_mask(shape, grass_mask, people_boxes):
    """Where points are looked for on the small picture: the grass without the players."""
    height, width = shape[:2]
    if grass_mask is None:
        mask = np.full((height, width), 255, dtype=np.uint8)
    else:
        mask = cv2.resize(grass_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        # The lines and the edge of the grass carry the corners: keep a rim around it.
        mask = cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8))
    margin = MOTION_PLAYER_MARGIN_PX * MOTION_SCALE
    for box in people_boxes or ():
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = (float(value) * MOTION_SCALE for value in box[:4])
        left = int(max(0, np.floor(x1 - margin)))
        top = int(max(0, np.floor(y1 - margin)))
        right = int(min(width, np.ceil(x2 + margin)))
        bottom = int(min(height, np.ceil(y2 + margin)))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 0
    return mask


def to_full_size(small_homography):
    """The homography found on the small picture, for the full-size picture."""
    scale = np.diag([MOTION_SCALE, MOTION_SCALE, 1.0])
    return np.linalg.inv(scale) @ np.asarray(small_homography, dtype=np.float64) @ scale


def plausible_turn(homography, shape):
    """True when the move is a camera turning and zooming, not a different picture."""
    if homography is None or not np.isfinite(homography).all():
        return False
    height, width = shape[:2]
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
    if not np.isfinite(moved).all():
        return False
    area = abs(cv2.contourArea(moved.astype(np.float32)))
    zoom = np.sqrt(area / float(width * height))
    if not (1.0 - MOTION_MAX_ZOOM_STEP <= zoom <= 1.0 + MOTION_MAX_ZOOM_STEP):
        return False
    shift = np.linalg.norm(moved.mean(axis=0) - corners.mean(axis=0))
    return shift <= MOTION_MAX_SHIFT_SHARE * float(np.hypot(width, height))


class FrameMotion:
    """The camera move from an anchor frame to a later one, as an image homography."""

    def __init__(self):
        self._gray = None
        self._points = None
        self._shape = None
        self.read = 0
        self.failed = 0

    def clear(self):
        self._gray = None
        self._points = None
        self._shape = None

    def anchor(self, frame, grass_mask=None, people_boxes=None):
        """Remember this frame and the grass points the next frames are followed from."""
        self._shape = frame.shape[:2]
        gray = _small_gray(frame)
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MOTION_MAX_POINTS,
            qualityLevel=MOTION_QUALITY,
            minDistance=MOTION_MIN_DISTANCE_PX,
            mask=_search_mask(gray.shape, grass_mask, people_boxes),
            blockSize=5,
        )
        if points is None or len(points) < MOTION_MIN_POINTS:
            self._gray = None
            self._points = None
            return False
        self._gray = gray
        self._points = points.astype(np.float32)
        return True

    def since_anchor(self, frame):
        """Full-size homography from the anchor picture to this one, or None when it cannot be read."""
        homography = self._read(frame)
        if homography is None:
            self.failed += 1
        else:
            self.read += 1
        return homography

    def _read(self, frame):
        if self._gray is None or self._points is None or frame.shape[:2] != self._shape:
            return None
        gray = _small_gray(frame)
        options = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        ahead, status, _ = cv2.calcOpticalFlowPyrLK(self._gray, gray, self._points, None, **options)
        if ahead is None:
            return None
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, self._gray, ahead, None, **options)
        if back is None:
            return None
        round_trip = np.linalg.norm((back - self._points).reshape(-1, 2), axis=1)
        good = (status.reshape(-1) == 1) & (status_back.reshape(-1) == 1) & (round_trip <= MOTION_ROUND_TRIP_PX)
        if int(good.sum()) < MOTION_MIN_POINTS:
            return None
        start = self._points.reshape(-1, 2)[good]
        end = ahead.reshape(-1, 2)[good]
        homography, inliers = cv2.findHomography(start, end, cv2.RANSAC, MOTION_RANSAC_PX)
        if homography is None or inliers is None:
            return None
        agreeing = int(inliers.sum())
        if agreeing < MOTION_MIN_POINTS or agreeing < MOTION_MIN_SHARE * len(start):
            return None
        full = to_full_size(homography)
        if not plausible_turn(full, self._shape):
            return None
        return full
