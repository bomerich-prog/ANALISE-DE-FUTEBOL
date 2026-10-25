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

    def update(self, keypoints, white_line_mask):
        decision = self._fit(keypoints, white_line_mask)
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
        if support is None or support < self.min_line_support:
            self.held = None
            self.hold_left = 0
            return False
        return True

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
