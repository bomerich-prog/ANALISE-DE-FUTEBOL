#!/usr/bin/env python3
"""Run YOLOv8 detections over a soccer video and write an annotated MP4."""

import argparse
import base64
from collections import Counter, deque
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from ball_heatmap import BallHeatmapAccumulator
from pass_detector import PassDetector
from person_class import PersonClassLock, reject_non_person_detections
from possession import PossessionEngine, build_ball_state, build_possession_players
from possession_stats import PossessionTeamAccumulator
from roboflow_ball import BallTracker
from roboflow_soccer_config import SoccerPitchConfiguration
from roboflow_team import TeamClassifier, TeamTrackLock, reject_training_crop
from roboflow_view import ViewTransformer


BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
PERSON_DUPLICATE_IOU_THRESHOLD = 0.85
MIN_WHITE_LINE_PIXELS = 80
FIELD_LINE_OVERLAP_DILATION = 31
FIELD_BOUNDARY_LINE_MIN_GRASS_DELTA = 0.35
FIELD_BOUNDARY_LINE_SAMPLE_OFFSET = 24
LIVE_DEBUG_FRAME_MAX_WIDTH = 960
LIVE_DEBUG_FRAME_JPEG_QUALITY = 72
LIVE_DEBUG_FRAME_NAME = "live-frame.jpg"
MIN_FIELD_AREA_RATIO = 0.12
LIVE_DEBUG_FRAME_REPLACE_RETRIES = 6
LIVE_DEBUG_FRAME_RETRY_BASE_DELAY_SECONDS = 0.03
PLAYER_MODEL_RELATIVE_PATH = Path("models/football-player-detection.pt")
PITCH_MODEL_RELATIVE_PATH = Path("models/football-pitch-detection.pt")
BALL_MODEL_RELATIVE_PATH = Path("models/football-ball-detection.pt")
PITCH_KEYPOINT_MIN_CONFIDENCE = 0.25
BALL_TRACKER_BUFFER_SIZE = 20
BALL_MEMORY_MAX_MISSING_FRAMES = 5
BALL_TRAJECTORY_HISTORY_SIZE = 6
BALL_ROI_MIN_RADIUS_PX = 45.0
BALL_ROI_SCALE_FROM_BOX = 2.25
BALL_PREDICTION_NEAR_DISTANCE_PX = 80.0
BALL_CONFIDENCE_NEAR_MIN = 0.12
BALL_CONFIDENCE_FAR_MIN = 0.35
PREDICTIVE_BALL_CONFIDENCE_NEAR_MIN = 0.08
PREDICTIVE_BALL_CONFIDENCE_FAR_MIN = 0.20
HYBRID_BALL_CONFIDENCE_NEAR_MIN = 0.05
HYBRID_BALL_CONFIDENCE_FAR_MIN = 0.1
FIELD_PREDICTION_MAX_DELTA_PX = 180.0
HYBRID_CONFIDENCE_BONUS_DISTANCE_PX = 40.0
HYBRID_CONFIDENCE_BONUS_DELTA = 0.03
PHYSICAL_BALL_VALIDATION_DEFAULT = os.getenv("PHYSICAL_BALL_VALIDATION", "on").strip().lower()
if PHYSICAL_BALL_VALIDATION_DEFAULT not in {"on", "off"}:
    PHYSICAL_BALL_VALIDATION_DEFAULT = "on"
PHYSICAL_ACCEPT_DISTANCE = 120.0
PHYSICAL_VERIFY_DISTANCE = 300.0
PHYSICAL_MAX_DISTANCE = 300.0
PHYSICAL_CONSECUTIVE_CONFIRMATIONS = 2
PHYSICAL_RESET_AFTER_MISSING = 10
PHYSICAL_HIGH_SPEED_THRESHOLD = 120.0
BALL_DETECTOR_DEFAULT_CONFIDENCE = 0.25
BALL_YOLO_RECALL_CONFIDENCE = 0.02
BALL_STRONG_CONFIDENCE = 0.25
BALL_NEARBY_STRONG_DISTANCE_PX = PHYSICAL_ACCEPT_DISTANCE
BALL_DIAGNOSTIC_MIN_CONFIDENCE = 0.05
PREDICTIVE_BALL_TRACKING_DEFAULT = os.getenv("PREDICTIVE_BALL_TRACKING", "on").strip().lower()
if PREDICTIVE_BALL_TRACKING_DEFAULT not in {"on", "off"}:
    PREDICTIVE_BALL_TRACKING_DEFAULT = "on"
HYBRID_BALL_SEARCH_DEFAULT = os.getenv("HYBRID_BALL_SEARCH", "on").strip().lower()
if HYBRID_BALL_SEARCH_DEFAULT not in {"on", "off"}:
    HYBRID_BALL_SEARCH_DEFAULT = "on"
PREDICTIVE_ROI_SIZE_SEQUENCE = [200.0, 300.0, 450.0, 650.0]
PREDICTIVE_GLOBAL_FALLBACK_AFTER_MISSES = 4
PREDICTIVE_HISTORY_SIZE = 10
TILE_SIZE = 640
TILE_OVERLAP = 0.25
INITIAL_DETECTIONS_REQUIRED = 3
CROP_INITIAL_SIZE = 250
CROP_MAX_SIZE = 1400
MAX_FRAMES_BEFORE_GLOBAL_SEARCH = 3
SEMI_GLOBAL_WINDOW_SIZE = 1200
CROP_EXPANSION_LEVELS_DEFAULT = [250, 500, 900, 1400]
SEMI_GLOBAL_RECOVERY_COOLDOWN_FRAMES = 10
BALL_DETECTOR_BACKEND_DEFAULT = os.getenv("BALL_DETECTOR_BACKEND", "yolo").strip().lower()
BALL_DETECTOR_BACKENDS = {"yolo", "rfdetr"}
if BALL_DETECTOR_BACKEND_DEFAULT not in BALL_DETECTOR_BACKENDS:
    BALL_DETECTOR_BACKEND_DEFAULT = "yolo"
RFDETR_SOCCERNET_REPO_ID_DEFAULT = "julianzu9612/RFDETR-Soccernet"
RFDETR_SOCCERNET_CHECKPOINT_FILENAME = "weights/checkpoint_best_regular.pth"
RFDETR_SOCCERNET_CONFIG_FILENAME = "config.json"
RFDETR_SOCCERNET_MODEL_DIR = Path("models/rfdetr-soccernet")
BALL_DIAGNOSTIC_FRAME_CONFIDENCE_BINS = [
    {"label": "0.05-0.10", "lower": 0.05, "upper": 0.10},
    {"label": "0.10-0.20", "lower": 0.10, "upper": 0.20},
    {"label": "0.20-0.30", "lower": 0.20, "upper": 0.30},
    {"label": "0.30-0.40", "lower": 0.30, "upper": 0.40},
    {"label": ">0.40", "lower": 0.40, "upper": None},
]
TEAM_CLASSIFIER_STRIDE = 10
TEAM_CLASSIFIER_MAX_TRAINING_FRAMES = 400
TEAM_CLASSIFIER_TARGET_SAMPLES = 60
TEAM_CLASSIFIER_MIN_CROP_HEIGHT = 50
TEAM_CLASSIFIER_MIN_CROP_WIDTH = 30
TEAM_CLASSIFIER_BORDER_MARGIN = 2
TEAM_CLASSIFIER_MIN_LAPLACIAN_VARIANCE = 45.0
TEAM_CLASSIFIER_REFRESH_INTERVAL = 30
BYTE_TRACK_MIN_CONSECUTIVE_FRAMES = 3
BYTE_TRACK_LOST_TRACK_BUFFER = 60
BYTE_TRACK_MINIMUM_MATCHING_THRESHOLD = 0.72
PLAYER_REFEREE_HISTORY_WINDOW = 9
HOMOGRAPHY_MIN_KEYPOINTS = 4
PITCH_KEYPOINT_COLOR = (255, 0, 255)
PITCH_KEYPOINT_OUTLINE_COLOR = (255, 255, 255)
PITCH_KEYPOINT_RADIUS = 4
DETECTION_LABEL_BY_CLASS_ID = {
    PLAYER_CLASS_ID: "Player",
    GOALKEEPER_CLASS_ID: "Goalkeeper",
    REFEREE_CLASS_ID: "Referee",
    BALL_CLASS_ID: "Ball",
}
CLASS_COLOR_OVERRIDES = {
    "Player": (0, 200, 83),
    "Goalkeeper": (255, 193, 7),
    "Referee": (138, 43, 226),
    "Ball": (0, 179, 255),
}
SOCCER_PITCH_CONFIGURATION = SoccerPitchConfiguration()
PROFILE_STAGE_NAMES = [
    "player_detection",
    "ball_detection",
    "pitch_detection",
    "bytetrack",
    "ball_tracker",
    "team_classifier_fit",
    "team_classifier_predict",
    "goalkeeper_assignment",
    "homography",
    "pass_detection",
    "render_bounding_boxes",
    "video_writer",
    "live_frame_update",
    "frame_total",
]


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def log_live_frame_debug(message):
    print(f"[live-frame-debug] {message}", file=sys.stderr, flush=True)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def suppress_duplicate_person_detections(detections):
    kept = []
    suppressed = 0
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if any(
            box_iou(detection["box"], kept_detection["box"]) >= PERSON_DUPLICATE_IOU_THRESHOLD
            for kept_detection in kept
        ):
            suppressed += 1
            continue
        kept.append(detection)
    return kept, suppressed


def build_grass_support_mask(frame):
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = (25, 25, 35)
    upper_green = (95, 255, 255)
    hsv_mask = cv2.inRange(hsv, lower_green, upper_green)

    blue, green, red = cv2.split(frame)
    dominant_green = (
        (green.astype("float32") > red.astype("float32") * 0.95)
        & (green.astype("float32") > blue.astype("float32") * 0.95)
        & (green > 35)
    ).astype("uint8") * 255

    grass_mask = cv2.bitwise_and(hsv_mask, dominant_green)
    grass_mask = cv2.medianBlur(grass_mask, 5)
    grass_mask = cv2.morphologyEx(
        grass_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    grass_mask = cv2.morphologyEx(
        grass_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    grass_mask = cv2.dilate(
        grass_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    return grass_mask


def build_white_line_mask(frame, grass_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 135), (180, 80, 255))
    nearby_grass = cv2.dilate(
        grass_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    )
    white_mask = cv2.bitwise_and(white_mask, nearby_grass)
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
    )
    return white_mask


def detect_field_line_segments(white_line_mask):
    height, width = white_line_mask.shape[:2]
    edges = cv2.Canny(white_line_mask, 50, 150)
    segments = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=45,
        minLineLength=max(60, int(width * 0.05)),
        maxLineGap=max(12, int(width * 0.018)),
    )
    if segments is None:
        return []

    min_length = max(50, min(width, height) * 0.055)
    long_segments = []
    for segment in np.asarray(segments).reshape(-1, 4):
        x1, y1, x2, y2 = [int(value) for value in segment]
        length = math.hypot(x2 - x1, y2 - y1)
        if length >= min_length:
            long_segments.append((x1, y1, x2, y2))
    return long_segments


def build_segment_mask(shape, segments):
    mask = np.zeros(shape, dtype=np.uint8)
    for x1, y1, x2, y2 in segments:
        cv2.line(mask, (x1, y1), (x2, y2), 255, 8)
    return mask


def sample_grass_ratio_along_line_side(grass_mask, segment, normal, side):
    x1, y1, x2, y2 = segment
    length = max(1, int(math.hypot(x2 - x1, y2 - y1)))
    sample_count = max(12, min(80, length // 12))
    grass_hits = 0
    valid_samples = 0
    height, width = grass_mask.shape[:2]

    for index in range(sample_count):
        t = index / max(1, sample_count - 1)
        x = x1 + (x2 - x1) * t + normal[0] * FIELD_BOUNDARY_LINE_SAMPLE_OFFSET * side
        y = y1 + (y2 - y1) * t + normal[1] * FIELD_BOUNDARY_LINE_SAMPLE_OFFSET * side
        px = int(round(x))
        py = int(round(y))
        if 0 <= px < width and 0 <= py < height:
            valid_samples += 1
            if grass_mask[py, px] > 0:
                grass_hits += 1

    return grass_hits / valid_samples if valid_samples else 0.0


def distance_to_contour_boundary(contour, point):
    return abs(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))


def find_field_boundary_constraints(grass_mask, selected_contour, line_segments):
    height, width = grass_mask.shape[:2]
    min_length = max(100, width * 0.12)
    min_centroid_guided_length = max(140, width * 0.18)
    max_boundary_distance = max(70, height * 0.10)
    constraints = []
    moments = cv2.moments(selected_contour)
    if moments["m00"]:
        contour_center = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float32,
        )
    else:
        contour_center = np.array([width / 2, height / 2], dtype=np.float32)

    for segment in line_segments:
        x1, y1, x2, y2 = segment
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_length:
            continue

        normal = np.array([-dy / length, dx / length], dtype=np.float32)
        positive_grass = sample_grass_ratio_along_line_side(grass_mask, segment, normal, 1)
        negative_grass = sample_grass_ratio_along_line_side(grass_mask, segment, normal, -1)
        grass_delta = abs(positive_grass - negative_grass)

        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
        boundary_distance = distance_to_contour_boundary(selected_contour, midpoint)

        if grass_delta >= FIELD_BOUNDARY_LINE_MIN_GRASS_DELTA:
            if boundary_distance > max_boundary_distance:
                continue
            keep_sign = 1 if positive_grass > negative_grass else -1
            score = length * grass_delta
        elif length >= min_centroid_guided_length and boundary_distance <= max_boundary_distance:
            signed_center = float(np.dot(contour_center - np.array([x1, y1], dtype=np.float32), normal))
            keep_sign = 1 if signed_center >= 0 else -1
            distance_weight = max(0.1, 1 - boundary_distance / (max_boundary_distance + 1))
            score = length * 0.5 * distance_weight
        else:
            continue

        inward_normal = normal * keep_sign
        center_distance = float(
            np.dot(contour_center - np.array([x1, y1], dtype=np.float32), inward_normal)
        )
        if center_distance <= 0:
            continue

        constraints.append(
            {
                "point": np.array([x1, y1], dtype=np.float32),
                "normal": normal,
                "keep_sign": keep_sign,
                "inward_normal": inward_normal,
                "center_distance": center_distance,
                "midpoint": np.array(midpoint, dtype=np.float32),
                "score": score,
            }
        )

    constraints = filter_external_boundary_constraints(constraints, width, height)
    constraints.sort(key=lambda item: item["center_distance"], reverse=True)
    selected_constraints = []
    for constraint in constraints:
        duplicate = False
        for selected in selected_constraints:
            same_boundary_side = (
                float(np.dot(constraint["inward_normal"], selected["inward_normal"])) > 0.90
            )
            if same_boundary_side:
                duplicate = True
                break
        if not duplicate:
            selected_constraints.append(constraint)
        if len(selected_constraints) >= 4:
            break

    return selected_constraints


def filter_external_boundary_constraints(constraints, width, height):
    external_constraints = []
    for constraint in constraints:
        midpoint = constraint["midpoint"]
        inward_normal = constraint["inward_normal"]
        horizontal_boundary = abs(inward_normal[1]) >= abs(inward_normal[0])

        if horizontal_boundary:
            if not (midpoint[1] <= height * 0.45 or midpoint[1] >= height * 0.55):
                continue
        else:
            # Goal posts, nets and most inner vertical markings live near the center of the image.
            if width * 0.20 < midpoint[0] < width * 0.80:
                continue

        external_constraints.append(constraint)

    return external_constraints


def clip_polygon_with_constraint(polygon, constraint):
    if len(polygon) < 3:
        return polygon

    point = constraint["point"]
    normal = constraint["normal"]
    keep_sign = constraint["keep_sign"]

    def signed_distance(vertex):
        return float(np.dot(vertex - point, normal) * keep_sign)

    clipped = []
    previous = polygon[-1]
    previous_distance = signed_distance(previous)
    previous_inside = previous_distance >= -1.0

    for current in polygon:
        current_distance = signed_distance(current)
        current_inside = current_distance >= -1.0

        if current_inside != previous_inside:
            denominator = previous_distance - current_distance
            if abs(denominator) > 1e-6:
                t = previous_distance / denominator
                clipped.append(previous + (current - previous) * t)

        if current_inside:
            clipped.append(current)

        previous = current
        previous_distance = current_distance
        previous_inside = current_inside

    return np.array(clipped, dtype=np.float32)


def apply_field_line_constraints(hull, grass_mask, selected_contour, line_segments):
    polygon = hull.reshape(-1, 2).astype(np.float32)
    constraints = find_field_boundary_constraints(grass_mask, selected_contour, line_segments)
    for constraint in constraints:
        clipped = clip_polygon_with_constraint(polygon, constraint)
        if len(clipped) >= 3:
            polygon = clipped
    return polygon.astype(np.int32), bool(constraints)


def select_grass_contour_by_field_lines(grass_mask, line_reference_mask):
    height, width = grass_mask.shape[:2]
    contours, _ = cv2.findContours(grass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = width * height * MIN_FIELD_AREA_RATIO
    large_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    if not large_contours:
        return None, False

    if cv2.countNonZero(line_reference_mask) < MIN_WHITE_LINE_PIXELS:
        return max(large_contours, key=cv2.contourArea), False

    line_reference_mask = cv2.dilate(
        line_reference_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (FIELD_LINE_OVERLAP_DILATION, FIELD_LINE_OVERLAP_DILATION),
        ),
        iterations=1,
    )

    best_contour = None
    best_score = 0
    for contour in large_contours:
        contour_mask = np.zeros_like(grass_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        overlap = cv2.countNonZero(cv2.bitwise_and(contour_mask, line_reference_mask))
        score = overlap * (1 + cv2.contourArea(contour) / (width * height))
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None or best_score < MIN_WHITE_LINE_PIXELS:
        return max(large_contours, key=cv2.contourArea), False
    return best_contour, True


def detect_field_region(frame, previous_mask=None):
    height, width = frame.shape[:2]
    grass_mask = build_grass_support_mask(frame)
    white_line_mask = build_white_line_mask(frame, grass_mask)
    line_segments = detect_field_line_segments(white_line_mask)
    line_reference_mask = build_segment_mask(white_line_mask.shape, line_segments)
    if cv2.countNonZero(line_reference_mask) < MIN_WHITE_LINE_PIXELS:
        line_reference_mask = white_line_mask

    selected_contour, line_guided = select_grass_contour_by_field_lines(
        grass_mask, line_reference_mask
    )

    if selected_contour is None:
        if previous_mask is not None:
            previous_contours, _ = cv2.findContours(
                previous_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            return previous_mask, previous_contours, False
        full_mask = np.full((height, width), 255, dtype=np.uint8)
        return full_mask, [], False

    hull = cv2.convexHull(selected_contour)
    playable_polygon, constraints_applied = apply_field_line_constraints(
        hull, grass_mask, selected_contour, line_segments
    )
    field_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(field_mask, [playable_polygon], 255)

    field_mask = cv2.erode(
        field_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    field_contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return field_mask, field_contours, line_guided or constraints_applied


def detection_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def detection_foot_point(box):
    x1, _, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


def build_view_transformer_from_pitch_keypoints(keypoints):
    keypoints_by_index = {}
    for keypoint in keypoints:
        keypoint_index = int(keypoint["index"])
        if keypoint_index < 0 or keypoint_index >= len(SOCCER_PITCH_CONFIGURATION.vertices):
            continue

        score = float(keypoint["confidence"]) if keypoint.get("confidence") is not None else -1.0
        previous = keypoints_by_index.get(keypoint_index)
        if previous is None or score > previous["score"]:
            keypoints_by_index[keypoint_index] = {"keypoint": keypoint, "score": score}

    if len(keypoints_by_index) < HOMOGRAPHY_MIN_KEYPOINTS:
        return None

    source_points = []
    target_points = []
    for keypoint_index in sorted(keypoints_by_index.keys()):
        keypoint = keypoints_by_index[keypoint_index]["keypoint"]
        source_points.append([keypoint["x"], keypoint["y"]])
        target_points.append(SOCCER_PITCH_CONFIGURATION.vertices[keypoint_index])

    try:
        return ViewTransformer(
            source=np.array(source_points, dtype=np.float32),
            target=np.array(target_points, dtype=np.float32),
        )
    except ValueError:
        return None


def assign_field_coordinates_to_detections(detections, view_transformer, anchor_getter):
    if not detections:
        return
    if view_transformer is None:
        for detection in detections:
            detection["field_x"] = None
            detection["field_y"] = None
        return

    anchor_points = np.array([anchor_getter(detection["box"]) for detection in detections], dtype=np.float32)
    transformed_points = view_transformer.transform_points(anchor_points)
    for index, detection in enumerate(detections):
        detection["field_x"] = float(transformed_points[index][0])
        detection["field_y"] = float(transformed_points[index][1])


def is_detection_inside_field(detection, field_mask):
    foot_x, foot_y = detection_foot_point(detection["box"])
    foot_x = clamp(foot_x, 0, field_mask.shape[1] - 1)
    foot_y = clamp(foot_y, 0, field_mask.shape[0] - 1)
    return field_mask[foot_y, foot_x] > 0


def draw_field_debug_overlay(frame, field_contours):
    if field_contours:
        cv2.drawContours(frame, field_contours, -1, (0, 255, 255), 3)


def get_crops(frame, detections):
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def get_player_crops_from_payload(frame, detections):
    crops = []
    for detection in detections:
        crops.append(sv.crop_image(frame, np.array(detection["box"], dtype=np.int32)))
    return crops


def classify_training_crop_quality(box, frame_width, frame_height):
    x1, y1, x2, y2 = [float(value) for value in box]
    if not all(np.isfinite([x1, y1, x2, y2])):
        return False, "empty"

    crop_width = x2 - x1
    crop_height = y2 - y1
    if crop_width <= 0 or crop_height <= 0:
        return False, "empty"
    if crop_height < TEAM_CLASSIFIER_MIN_CROP_HEIGHT or crop_width < TEAM_CLASSIFIER_MIN_CROP_WIDTH:
        return False, "small"

    if (
        x1 <= TEAM_CLASSIFIER_BORDER_MARGIN
        or y1 <= TEAM_CLASSIFIER_BORDER_MARGIN
        or x2 >= frame_width - TEAM_CLASSIFIER_BORDER_MARGIN
        or y2 >= frame_height - TEAM_CLASSIFIER_BORDER_MARGIN
    ):
        return False, "border"

    return True, "ok"


def is_crop_blurry(crop, min_laplacian_variance):
    if crop is None or crop.size == 0:
        return True
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray_crop, cv2.CV_64F).var() < min_laplacian_variance


def stabilize_player_referee_class(class_history, fallback_class_id):
    if not class_history:
        return fallback_class_id
    counts = Counter(class_history)
    most_common_count = max(counts.values())
    candidates = {class_id for class_id, count in counts.items() if count == most_common_count}
    if len(candidates) == 1:
        return next(iter(candidates))
    for class_id in reversed(class_history):
        if class_id in candidates:
            return class_id
    return fallback_class_id


def resolve_goalkeepers_team_id(players, players_team_id, goalkeepers):
    if len(goalkeepers) == 0:
        return np.array([])
    if len(players) == 0 or len(players_team_id) == 0:
        return np.full(len(goalkeepers), -1, dtype=np.int32)

    players_xy = np.array(
        [detection_foot_point(detection["box"]) for detection in players],
        dtype=np.float32,
    )
    goalkeepers_xy = np.array(
        [detection_foot_point(detection["box"]) for detection in goalkeepers],
        dtype=np.float32,
    )
    team_0_mask = players_team_id == 0
    team_1_mask = players_team_id == 1
    if not np.any(team_0_mask) or not np.any(team_1_mask):
        return np.full(len(goalkeepers), -1, dtype=np.int32)

    team_0_centroid = players_xy[team_0_mask].mean(axis=0)
    team_1_centroid = players_xy[team_1_mask].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def resolve_model_path(model_arg):
    candidate = Path(model_arg)
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parent / candidate
    return candidate.resolve()


def resolve_pitch_model_path(pitch_model_arg):
    return resolve_model_path(pitch_model_arg)


def resolve_ball_model_path(ball_model_arg):
    return resolve_model_path(ball_model_arg)


def resolve_player_model_path(player_model_arg):
    return resolve_model_path(player_model_arg)


def empty_detections():
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int32),
    )


def load_rfdetr_ball_class_id(config_path):
    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return BALL_CLASS_ID

    class_mapping = (
        config_data.get("classes", {}).get("class_mapping", {})
        if isinstance(config_data, dict)
        else {}
    )
    for class_id, class_name in class_mapping.items():
        if str(class_name).strip().lower() == "ball":
            try:
                return int(class_id)
            except (TypeError, ValueError):
                continue
    return BALL_CLASS_ID


def download_huggingface_model_file(repo_id, filename, target_directory):
    target_directory.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub não está disponível para baixar o checkpoint RF-DETR."
        ) from exc

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_directory),
            local_dir_use_symlinks=False,
        )
    except TypeError:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_directory),
        )
    return Path(downloaded_path).resolve()


def ensure_rfdetr_soccernet_files(repo_id, model_directory):
    checkpoint_path = model_directory / RFDETR_SOCCERNET_CHECKPOINT_FILENAME
    config_path = model_directory / RFDETR_SOCCERNET_CONFIG_FILENAME
    if not checkpoint_path.exists():
        checkpoint_path = download_huggingface_model_file(
            repo_id=repo_id,
            filename=RFDETR_SOCCERNET_CHECKPOINT_FILENAME,
            target_directory=model_directory,
        )
    if not config_path.exists():
        config_path = download_huggingface_model_file(
            repo_id=repo_id,
            filename=RFDETR_SOCCERNET_CONFIG_FILENAME,
            target_directory=model_directory,
        )
    return checkpoint_path.resolve(), config_path.resolve()


class BallDetectorYOLO:
    def __init__(self, model, confidence_threshold):
        self.model = model
        self.confidence_threshold = float(confidence_threshold)
        self.source_name = "yolo-football-ball-detection"

    def detect(self, frame, roi_xyxy=None):
        roi_frame = frame
        offset_x = 0
        offset_y = 0
        if roi_xyxy is not None:
            x1, y1, x2, y2 = [int(value) for value in roi_xyxy]
            roi_frame = frame[y1:y2, x1:x2]
            offset_x = x1
            offset_y = y1

        result = self.model(
            roi_frame,
            imgsz=640,
            conf=self.confidence_threshold,
            verbose=False,
        )[0]
        detections = extract_ball_detections(result)
        if roi_xyxy is None or len(detections) == 0:
            return detections
        return translate_detections_to_full_frame(detections, offset_x, offset_y)


class BallDetectorRFDETR:
    def __init__(self, checkpoint_path, confidence_threshold, ball_class_id, device="auto"):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.confidence_threshold = float(confidence_threshold)
        self.ball_class_id = int(ball_class_id)
        self.source_name = "rfdetr-soccernet"
        self.device = device
        self.model = None

    def _ensure_model(self):
        if self.model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch não está disponível para carregar o RF-DETR.") from exc
        try:
            from rfdetr import RFDETR
        except ImportError as exc:
            raise RuntimeError("Pacote oficial rfdetr não está instalado.") from exc

        runtime_device = self.device
        if runtime_device == "auto":
            runtime_device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(
            str(self.checkpoint_path),
            map_location=runtime_device,
            weights_only=False,
        )
        checkpoint_args = checkpoint.get("args")
        args_dict = {}
        if isinstance(checkpoint_args, dict):
            args_dict = checkpoint_args
        elif checkpoint_args is not None and hasattr(checkpoint_args, "__dict__"):
            args_dict = vars(checkpoint_args)

        checkpoint_num_classes = int(args_dict.get("num_classes", 4))
        model_kwargs = {
            "encoder": args_dict.get("encoder", "dinov2_windowed_base"),
            "out_feature_indexes": args_dict.get("out_feature_indexes", [2, 5, 8, 11]),
            "dec_layers": args_dict.get("dec_layers", 3),
            "projector_scale": args_dict.get("projector_scale", ["P3", "P5"]),
            "hidden_dim": args_dict.get("hidden_dim", 384),
            "patch_size": args_dict.get("patch_size", 14),
            "num_windows": args_dict.get("num_windows", 4),
            "sa_nheads": args_dict.get("sa_nheads", 12),
            "ca_nheads": args_dict.get("ca_nheads", 24),
            "dec_n_points": args_dict.get("dec_n_points", 4),
            "resolution": args_dict.get("resolution", 560),
            "positional_encoding_size": args_dict.get("positional_encoding_size", 37),
            "num_classes": max(1, checkpoint_num_classes - 1),
            "pretrain_weights": args_dict.get("pretrain_weights", "rf-detr-large.pth"),
        }

        self.model = RFDETR(**model_kwargs)
        model_state = checkpoint.get("model") or checkpoint.get("model_state_dict") or checkpoint
        self.model.model.model.load_state_dict(model_state, strict=False)
        self.model.model.model.to(runtime_device)
        self.model.model.model.eval()
        self.device = runtime_device

    def detect(self, frame, roi_xyxy=None):
        self._ensure_model()
        roi_frame = frame
        offset_x = 0
        offset_y = 0
        if roi_xyxy is not None:
            x1, y1, x2, y2 = [int(value) for value in roi_xyxy]
            roi_frame = frame[y1:y2, x1:x2]
            offset_x = x1
            offset_y = y1

        frame_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(
            frame_rgb,
            threshold=self.confidence_threshold,
            include_source_image=False,
        )
        if isinstance(detections, list):
            detections = detections[0] if detections else empty_detections()
        if detections is None or len(detections) == 0:
            return empty_detections()

        class_ids = detections.class_id
        if class_ids is None:
            return detections
        class_ids_array = np.asarray(class_ids, dtype=np.int32)
        if class_ids_array.size != len(detections):
            return detections
        ball_mask = class_ids_array == self.ball_class_id
        if not np.any(ball_mask):
            return empty_detections()
        filtered = detections[ball_mask]
        if roi_xyxy is None or len(filtered) == 0:
            return filtered
        return translate_detections_to_full_frame(filtered, offset_x, offset_y)


def build_ball_detector(args, confidence_threshold):
    backend = str(args.ball_detector).strip().lower()
    if backend not in BALL_DETECTOR_BACKENDS:
        raise ValueError(
            f"Detector de bola inválido '{backend}'. Opções: {sorted(BALL_DETECTOR_BACKENDS)}"
        )

    if backend == "yolo":
        ball_model_path = resolve_ball_model_path(args.ball_model)
        if not ball_model_path.exists():
            raise FileNotFoundError(f"Modelo de detecção da bola não encontrado: {ball_model_path}")
        return (
            BallDetectorYOLO(YOLO(str(ball_model_path)), confidence_threshold),
            backend,
            str(ball_model_path),
        )

    rfdetr_model_directory = resolve_model_path(args.rfdetr_model_dir)
    checkpoint_path, config_path = ensure_rfdetr_soccernet_files(
        repo_id=args.rfdetr_repo_id,
        model_directory=rfdetr_model_directory,
    )
    ball_class_id = load_rfdetr_ball_class_id(config_path)
    return (
        BallDetectorRFDETR(
            checkpoint_path=checkpoint_path,
            confidence_threshold=confidence_threshold,
            ball_class_id=ball_class_id,
            device=args.rfdetr_device,
        ),
        backend,
        str(checkpoint_path),
    )


def extract_pitch_keypoints(pitch_result, confidence_threshold):
    keypoints = getattr(pitch_result, "keypoints", None)
    if keypoints is None:
        return []

    xy = getattr(keypoints, "xy", None)
    if xy is None:
        return []

    xy_array = xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy)
    if xy_array.size == 0:
        return []

    confidence_array = None
    confidence = getattr(keypoints, "conf", None)
    if confidence is not None:
        confidence_array = (
            confidence.cpu().numpy() if hasattr(confidence, "cpu") else np.asarray(confidence)
        )

    visible_keypoints = []
    for detection_index, detection_keypoints in enumerate(xy_array):
        for keypoint_index, point in enumerate(detection_keypoints):
            x, y = float(point[0]), float(point[1])
            if not np.isfinite(x) or not np.isfinite(y):
                continue

            keypoint_confidence = None
            if confidence_array is not None:
                keypoint_confidence = float(confidence_array[detection_index][keypoint_index])
                if keypoint_confidence < confidence_threshold:
                    continue

            visible_keypoints.append(
                {
                    "x": x,
                    "y": y,
                    "confidence": keypoint_confidence,
                    "index": keypoint_index,
                }
            )

    return visible_keypoints


def draw_pitch_keypoints(frame, keypoints):
    if frame is None:
        return

    for keypoint in keypoints:
        x = clamp(int(round(keypoint["x"])), 0, frame.shape[1] - 1)
        y = clamp(int(round(keypoint["y"])), 0, frame.shape[0] - 1)
        cv2.circle(frame, (x, y), PITCH_KEYPOINT_RADIUS + 2, PITCH_KEYPOINT_OUTLINE_COLOR, 1)
        cv2.circle(frame, (x, y), PITCH_KEYPOINT_RADIUS, PITCH_KEYPOINT_COLOR, -1)
        cv2.putText(
            frame,
            str(keypoint["index"] + 1),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            PITCH_KEYPOINT_OUTLINE_COLOR,
            1,
            cv2.LINE_AA,
        )


def extract_ball_detections(ball_result):
    return sv.Detections.from_ultralytics(ball_result)


def ball_detections_to_dicts(detections, source_name):
    if len(detections) == 0:
        return []

    confidence_values = detections.confidence
    if confidence_values is None:
        confidence_values = np.zeros(len(detections), dtype=np.float32)
    class_ids = detections.class_id
    if class_ids is None:
        class_ids = np.full(len(detections), BALL_CLASS_ID, dtype=np.int32)
    else:
        class_ids = np.asarray(class_ids, dtype=np.int32)

    detection_dicts = []
    for index in range(len(detections)):
        box = detections.xyxy[index].tolist()
        center = detection_center_np(box)
        detection_dicts.append(
            {
                "box": box,
                "bounding_box": box,
                "confidence": float(confidence_values[index]),
                "source": source_name,
                "class_id": int(class_ids[index]) if index < len(class_ids) else BALL_CLASS_ID,
                "class": "Ball",
                "center": [float(center[0]), float(center[1])],
                "field_x": None,
                "field_y": None,
            }
        )
    return detection_dicts


def detection_center_np(box):
    x1, y1, x2, y2 = [float(value) for value in box]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def predict_ball_center(position_history, current_frame_number):
    if not position_history:
        return None

    latest_frame, latest_center = position_history[-1]
    if len(position_history) < 2:
        return latest_center.copy()

    velocity_samples = []
    sampled_positions = list(position_history)[-4:]
    for index in range(1, len(sampled_positions)):
        previous_frame, previous_center = sampled_positions[index - 1]
        next_frame, next_center = sampled_positions[index]
        frame_delta = max(1, int(next_frame - previous_frame))
        velocity_samples.append((next_center - previous_center) / float(frame_delta))

    if not velocity_samples:
        return latest_center.copy()

    average_velocity = np.mean(np.stack(velocity_samples), axis=0)
    prediction_frame_delta = max(1, int(current_frame_number - latest_frame))
    return latest_center + average_velocity * float(prediction_frame_delta)


def predict_motion_state_from_history(position_history, current_frame_number):
    if not position_history:
        return None

    sampled_positions = list(position_history)[-3:]
    latest_frame, latest_position = sampled_positions[-1]
    frame_delta_to_target = max(1, int(current_frame_number - latest_frame))

    if len(sampled_positions) == 1:
        velocity = np.zeros(2, dtype=np.float32)
        acceleration = np.zeros(2, dtype=np.float32)
        predicted = latest_position.copy()
    elif len(sampled_positions) == 2:
        previous_frame, previous_position = sampled_positions[0]
        dt = max(1, int(latest_frame - previous_frame))
        velocity = (latest_position - previous_position) / float(dt)
        acceleration = np.zeros(2, dtype=np.float32)
        predicted = latest_position + velocity * float(frame_delta_to_target)
    else:
        first_frame, first_position = sampled_positions[0]
        middle_frame, middle_position = sampled_positions[1]
        dt1 = max(1, int(middle_frame - first_frame))
        dt2 = max(1, int(latest_frame - middle_frame))
        velocity_previous = (middle_position - first_position) / float(dt1)
        velocity_latest = (latest_position - middle_position) / float(dt2)
        average_dt = max(1.0, (float(dt1) + float(dt2)) * 0.5)
        acceleration = (velocity_latest - velocity_previous) / average_dt
        velocity = velocity_latest
        dt_to_target = float(frame_delta_to_target)
        predicted = latest_position + velocity * dt_to_target + 0.5 * acceleration * (dt_to_target**2)

    direction_norm = float(np.linalg.norm(velocity))
    if direction_norm > 0:
        direction = (velocity / direction_norm).astype(np.float32)
    else:
        direction = np.zeros(2, dtype=np.float32)

    return {
        "predicted": predicted.astype(np.float32),
        "velocity": velocity.astype(np.float32),
        "acceleration": acceleration.astype(np.float32),
        "direction": direction.astype(np.float32),
        "latest_frame": int(latest_frame),
    }


def transform_field_point_to_image(view_transformer, field_point):
    if view_transformer is None or field_point is None:
        return None
    try:
        inverse_matrix = np.linalg.inv(view_transformer.m)
    except np.linalg.LinAlgError:
        return None
    reshaped_point = np.asarray(field_point, dtype=np.float32).reshape(1, 1, 2)
    transformed = cv2.perspectiveTransform(reshaped_point, inverse_matrix).reshape(-1, 2)
    if transformed.size == 0:
        return None
    image_point = transformed[0]
    if not np.all(np.isfinite(image_point)):
        return None
    return image_point.astype(np.float32)


def transform_image_point_to_field(view_transformer, image_point):
    if view_transformer is None or image_point is None:
        return None
    transformed = view_transformer.transform_points(
        np.asarray(image_point, dtype=np.float32).reshape(1, 2)
    )
    if transformed.size == 0:
        return None
    field_point = transformed[0]
    if not np.all(np.isfinite(field_point)):
        return None
    return field_point.astype(np.float32)


def clamp_center_to_frame(center, width, height):
    if center is None:
        return None
    return np.array(
        [
            float(clamp(center[0], 0.0, max(0.0, float(width) - 1.0))),
            float(clamp(center[1], 0.0, max(0.0, float(height) - 1.0))),
        ],
        dtype=np.float32,
    )


def build_square_roi_from_center(center, roi_size_px, width, height):
    if center is None:
        return None
    if roi_size_px <= 1:
        return None

    half = float(roi_size_px) * 0.5
    x1 = int(math.floor(center[0] - half))
    y1 = int(math.floor(center[1] - half))
    x2 = int(math.ceil(center[0] + half))
    y2 = int(math.ceil(center[1] + half))

    x1 = max(0, min(x1, max(0, width - 1)))
    y1 = max(0, min(y1, max(0, height - 1)))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return (x1, y1, x2, y2)


def translate_detections_to_full_frame(detections, offset_x, offset_y):
    if detections is None or len(detections) == 0:
        return detections if detections is not None else empty_detections()
    shifted = detections.xyxy.copy()
    shifted[:, [0, 2]] += float(offset_x)
    shifted[:, [1, 3]] += float(offset_y)
    return sv.Detections(
        xyxy=shifted,
        mask=detections.mask,
        confidence=detections.confidence,
        class_id=detections.class_id,
        tracker_id=detections.tracker_id,
        data=detections.data,
    )


def merge_detection_batches(detection_batches):
    valid_batches = [batch for batch in detection_batches if batch is not None and len(batch) > 0]
    if not valid_batches:
        return empty_detections()

    xyxy = np.concatenate([batch.xyxy for batch in valid_batches], axis=0).astype(np.float32)

    confidence_batches = []
    class_id_batches = []
    for batch in valid_batches:
        if batch.confidence is None:
            confidence_batches.append(np.zeros((len(batch),), dtype=np.float32))
        else:
            confidence_batches.append(np.asarray(batch.confidence, dtype=np.float32))
        if batch.class_id is None:
            class_id_batches.append(np.full((len(batch),), BALL_CLASS_ID, dtype=np.int32))
        else:
            class_id_batches.append(np.asarray(batch.class_id, dtype=np.int32))

    return sv.Detections(
        xyxy=xyxy,
        confidence=np.concatenate(confidence_batches, axis=0),
        class_id=np.concatenate(class_id_batches, axis=0),
    )


def build_tile_ranges(frame_width, frame_height, tile_size, tile_overlap):
    normalized_tile_size = max(64, int(tile_size))
    normalized_overlap = float(max(0.0, min(0.9, tile_overlap)))
    stride = max(1, int(round(normalized_tile_size * (1.0 - normalized_overlap))))

    x_starts = list(range(0, max(1, frame_width - normalized_tile_size + 1), stride))
    y_starts = list(range(0, max(1, frame_height - normalized_tile_size + 1), stride))
    if not x_starts:
        x_starts = [0]
    if not y_starts:
        y_starts = [0]

    last_x = max(0, frame_width - normalized_tile_size)
    last_y = max(0, frame_height - normalized_tile_size)
    if x_starts[-1] != last_x:
        x_starts.append(last_x)
    if y_starts[-1] != last_y:
        y_starts.append(last_y)

    tiles = []
    for y_start in y_starts:
        for x_start in x_starts:
            x1 = int(x_start)
            y1 = int(y_start)
            x2 = min(frame_width, x1 + normalized_tile_size)
            y2 = min(frame_height, y1 + normalized_tile_size)
            tiles.append((x1, y1, x2, y2))
    return tiles


def detect_ball_with_global_tiling(frame, ball_detector, tile_size, tile_overlap, region_xyxy=None):
    height, width = frame.shape[:2]
    if region_xyxy is None:
        region_x1, region_y1, region_x2, region_y2 = 0, 0, width, height
    else:
        region_x1, region_y1, region_x2, region_y2 = [int(value) for value in region_xyxy]
        region_x1 = max(0, min(region_x1, max(0, width - 1)))
        region_y1 = max(0, min(region_y1, max(0, height - 1)))
        region_x2 = max(region_x1 + 1, min(region_x2, width))
        region_y2 = max(region_y1 + 1, min(region_y2, height))

    region_width = region_x2 - region_x1
    region_height = region_y2 - region_y1
    tiles = build_tile_ranges(region_width, region_height, tile_size, tile_overlap)
    tile_detections = []
    for local_tile_xyxy in tiles:
        tile_x1, tile_y1, tile_x2, tile_y2 = local_tile_xyxy
        global_tile_xyxy = (
            int(region_x1 + tile_x1),
            int(region_y1 + tile_y1),
            int(region_x1 + tile_x2),
            int(region_y1 + tile_y2),
        )
        tile_detections.append(ball_detector.detect(frame, roi_xyxy=global_tile_xyxy))
    return merge_detection_batches(tile_detections)


def build_adaptive_crop_schedule(initial_size, max_size):
    normalized_initial = float(max(64, int(initial_size)))
    normalized_max = float(max(normalized_initial, int(max_size)))
    schedule = []
    for level in CROP_EXPANSION_LEVELS_DEFAULT:
        level_size = float(level)
        if level_size < normalized_initial:
            continue
        bounded_size = min(normalized_max, level_size)
        if not schedule or bounded_size > schedule[-1]:
            schedule.append(bounded_size)

    if not schedule:
        schedule = [normalized_initial]
    if schedule[0] > normalized_initial:
        schedule.insert(0, normalized_initial)
    if schedule[-1] < normalized_max:
        schedule.append(normalized_max)
    return schedule


def pick_adaptive_crop_size(schedule, missing_frames):
    if not schedule:
        return None
    index = max(0, min(int(missing_frames), len(schedule) - 1))
    return float(schedule[index])


def ball_roi_radius(last_detection):
    if not last_detection:
        return BALL_ROI_MIN_RADIUS_PX
    x1, y1, x2, y2 = [float(value) for value in last_detection["box"]]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return max(BALL_ROI_MIN_RADIUS_PX, max(width, height) * BALL_ROI_SCALE_FROM_BOX)


def pick_detection_with_confidence_fusion(
    detections,
    predicted_center,
    near_min_confidence=BALL_CONFIDENCE_NEAR_MIN,
    far_min_confidence=BALL_CONFIDENCE_FAR_MIN,
    proximity_bonus_distance_px=None,
    proximity_bonus_delta=0.0,
):
    if not detections:
        return None
    if predicted_center is None:
        return max(detections, key=lambda detection: detection["confidence"])

    accepted = []
    for detection in detections:
        center = detection_center_np(detection["box"])
        distance = float(np.linalg.norm(center - predicted_center))
        near_threshold = float(near_min_confidence)
        if (
            proximity_bonus_distance_px is not None
            and distance <= float(proximity_bonus_distance_px)
            and proximity_bonus_delta > 0.0
        ):
            near_threshold = max(0.0, near_threshold - float(proximity_bonus_delta))
        confidence_threshold = (
            near_threshold
            if distance <= BALL_PREDICTION_NEAR_DISTANCE_PX
            else far_min_confidence
        )
        if detection["confidence"] < confidence_threshold:
            continue
        accepted.append((distance, detection))

    if not accepted:
        return None

    accepted.sort(key=lambda item: (item[0], -item[1]["confidence"]))
    return accepted[0][1]


def keep_recoverable_ball_candidates(
    detections,
    predicted_center,
    strong_min_confidence=BALL_STRONG_CONFIDENCE,
    weak_min_confidence=BALL_YOLO_RECALL_CONFIDENCE,
    weak_max_distance_px=BALL_PREDICTION_NEAR_DISTANCE_PX,
    nearby_strong_distance_px=BALL_NEARBY_STRONG_DISTANCE_PX,
):
    """Keep a low-confidence ball when it continues the current track.

    On the orange-ball recording the detector often emits the true box between
    0.02 and 0.25 while a distant box scores higher. That nearby box has to
    stay in the candidate list. A box at or above 0.25 within 120 px still
    wins, and a weak box cannot open a track when there is no prediction.
    """
    if not detections:
        return []
    strong_min = float(strong_min_confidence)
    if predicted_center is None:
        return [
            detection
            for detection in detections
            if float(detection["confidence"]) >= strong_min
        ]

    predicted = np.asarray(predicted_center, dtype=np.float32)
    nearby_strong = []
    continuation = []
    distant_strong = []
    nearby_limit = float(nearby_strong_distance_px)
    weak_limit = float(weak_max_distance_px)
    for detection in detections:
        confidence = float(detection["confidence"])
        center = detection_center_np(detection["box"])
        distance = float(np.linalg.norm(center - predicted))
        if confidence >= strong_min and distance <= nearby_limit:
            nearby_strong.append(detection)
        elif confidence >= float(weak_min_confidence) and distance <= weak_limit:
            continuation.append(detection)
        elif confidence >= strong_min:
            distant_strong.append(detection)
    if nearby_strong:
        return nearby_strong
    if continuation:
        return continuation
    return distant_strong


def select_player_ball_fallback(
    ball_candidates,
    predicted_center,
    last_valid_detection,
    max_distance_px=BALL_PREDICTION_NEAR_DISTANCE_PX,
    min_confidence=BALL_STRONG_CONFIDENCE,
):
    """Continue an open ball track with the player detector's ball class.

    The dedicated ball detector keeps its own result whenever it produced a
    ball. This source is consulted only to fill that absence, and only when a
    track and a pixel prediction already exist. A box farther than 80 px, or
    below 0.25, is ignored, including the sideline false balls from the player
    model. It never opens a track.
    """
    decision = {
        "used": False,
        "rejected_distance": False,
        "rejected_confidence": False,
        "confidence": None,
        "distance": None,
    }
    if not ball_candidates:
        return None, decision

    min_confidence = float(min_confidence)
    max_distance = float(max_distance_px)
    if predicted_center is None or last_valid_detection is None:
        has_strong_candidate = any(
            float(candidate["confidence"]) >= min_confidence for candidate in ball_candidates
        )
        if has_strong_candidate:
            decision["rejected_distance"] = True
        else:
            decision["rejected_confidence"] = True
        return None, decision

    predicted = np.asarray(predicted_center, dtype=np.float32)
    eligible = []
    saw_near_weak = False
    for candidate in ball_candidates:
        confidence = float(candidate["confidence"])
        center = detection_center_np(candidate["box"])
        distance = float(np.linalg.norm(center - predicted))
        if distance <= max_distance and confidence >= min_confidence:
            eligible.append((distance, -confidence, candidate))
        elif distance <= max_distance:
            saw_near_weak = True

    if eligible:
        eligible.sort(key=lambda item: (item[0], item[1]))
        chosen = eligible[0][2]
        decision["used"] = True
        decision["confidence"] = float(chosen["confidence"])
        decision["distance"] = float(eligible[0][0])
        return chosen, decision

    if saw_near_weak:
        decision["rejected_confidence"] = True
    else:
        decision["rejected_distance"] = True
    return None, decision


def select_ball_detection_with_roi(
    detections,
    predicted_center,
    roi_radius_pixels,
    near_min_confidence=BALL_CONFIDENCE_NEAR_MIN,
    far_min_confidence=BALL_CONFIDENCE_FAR_MIN,
):
    if not detections:
        return None, False, False
    if predicted_center is None:
        return pick_detection_with_confidence_fusion(detections, None), False, False

    roi_candidates = []
    for detection in detections:
        center = detection_center_np(detection["box"])
        distance = float(np.linalg.norm(center - predicted_center))
        if distance <= roi_radius_pixels:
            roi_candidates.append(detection)

    selected_from_roi = pick_detection_with_confidence_fusion(
        roi_candidates,
        predicted_center,
        near_min_confidence=near_min_confidence,
        far_min_confidence=far_min_confidence,
    )
    if selected_from_roi is not None:
        return selected_from_roi, True, True

    selected_global = pick_detection_with_confidence_fusion(
        detections,
        predicted_center,
        near_min_confidence=near_min_confidence,
        far_min_confidence=far_min_confidence,
    )
    return selected_global, True, False


def ball_confidence_threshold_for_distance(distance_to_prediction):
    if distance_to_prediction is None:
        return None
    return (
        BALL_CONFIDENCE_NEAR_MIN
        if distance_to_prediction <= BALL_PREDICTION_NEAR_DISTANCE_PX
        else BALL_CONFIDENCE_FAR_MIN
    )


def ball_dicts_to_detections(detections):
    if not detections:
        return empty_detections()

    xyxy = np.asarray([detection["box"] for detection in detections], dtype=np.float32)
    confidence = np.asarray(
        [float(detection.get("confidence", 0.0)) for detection in detections],
        dtype=np.float32,
    )
    class_id = np.asarray(
        [int(detection.get("class_id", BALL_CLASS_ID)) for detection in detections],
        dtype=np.int32,
    )
    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
    )


def apply_physical_ball_validation(
    selected_detection,
    predicted_center,
    pending_candidate_state,
    current_frame_number,
    missing_frames,
    last_valid_detection,
    predicted_speed,
    accept_distance,
    verify_distance,
    max_distance,
    consecutive_confirmations,
    reset_after_missing,
    high_speed_threshold,
):
    stats = {
        "accepts": 0,
        "rejects": 0,
        "pending": 0,
        "confirmed": 0,
        "false_candidates": 0,
        "teleports_avoided": 0,
        "distance": None,
    }

    pending_state = pending_candidate_state
    if pending_state is not None:
        frames_since_last_pending = int(current_frame_number - int(pending_state["last_frame"]))
        if frames_since_last_pending > 1 or int(missing_frames) > int(reset_after_missing):
            stats["false_candidates"] += 1
            pending_state = None

    if selected_detection is None:
        return None, pending_state, stats

    candidate_center = detection_center_np(selected_detection["box"])
    distance_to_prediction = None
    if predicted_center is not None:
        distance_to_prediction = float(np.linalg.norm(candidate_center - predicted_center))
        stats["distance"] = distance_to_prediction

    has_reset_condition = (
        predicted_center is None
        or last_valid_detection is None
        or int(missing_frames) > int(reset_after_missing)
        or float(predicted_speed) >= float(high_speed_threshold)
    )
    if distance_to_prediction is None or has_reset_condition:
        if pending_state is not None:
            stats["false_candidates"] += 1
            pending_state = None
        stats["accepts"] += 1
        return selected_detection, pending_state, stats

    if distance_to_prediction <= float(accept_distance):
        if pending_state is not None:
            stats["false_candidates"] += 1
            pending_state = None
        stats["accepts"] += 1
        return selected_detection, pending_state, stats

    if distance_to_prediction <= float(verify_distance):
        if (
            pending_state is not None
            and np.linalg.norm(candidate_center - pending_state["center"]) <= float(accept_distance)
        ):
            pending_state["consecutive"] = int(pending_state["consecutive"]) + 1
            pending_state["center"] = candidate_center.astype(np.float32)
            pending_state["detection"] = selected_detection
            pending_state["last_frame"] = int(current_frame_number)
        else:
            if pending_state is not None:
                stats["false_candidates"] += 1
            pending_state = {
                "center": candidate_center.astype(np.float32),
                "detection": selected_detection,
                "consecutive": 1,
                "first_frame": int(current_frame_number),
                "last_frame": int(current_frame_number),
            }

        stats["pending"] += 1
        if int(pending_state["consecutive"]) >= int(consecutive_confirmations):
            confirmed_detection = pending_state["detection"]
            pending_state = None
            stats["accepts"] += 1
            stats["confirmed"] += 1
            return confirmed_detection, pending_state, stats

        stats["rejects"] += 1
        return None, pending_state, stats

    if pending_state is not None:
        stats["false_candidates"] += 1
        pending_state = None
    stats["rejects"] += 1
    if distance_to_prediction > float(max_distance):
        stats["teleports_avoided"] += 1
    else:
        stats["teleports_avoided"] += 1
    return None, pending_state, stats


def detection_box_to_xywh(box):
    x1, y1, x2, y2 = [float(value) for value in box]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return [x1, y1, width, height]


def detection_box_area(box):
    _, _, width, height = detection_box_to_xywh(box)
    return width * height


def compute_nms_keep_mask(detections, iou_threshold):
    if not detections:
        return []

    sorted_indexes = sorted(
        range(len(detections)),
        key=lambda index: detections[index]["confidence"],
        reverse=True,
    )
    keep_mask = [False] * len(detections)
    kept_indexes = []
    for index in sorted_indexes:
        candidate_box = detections[index]["box"]
        if any(box_iou(candidate_box, detections[kept_index]["box"]) > iou_threshold for kept_index in kept_indexes):
            continue
        keep_mask[index] = True
        kept_indexes.append(index)
    return keep_mask


def pick_best_ball_diagnostic_candidate(detections, predicted_center):
    if not detections:
        return None

    if predicted_center is None:
        return max(range(len(detections)), key=lambda index: detections[index]["confidence"])

    return min(
        range(len(detections)),
        key=lambda index: (
            np.linalg.norm(detection_center_np(detections[index]["box"]) - predicted_center),
            -detections[index]["confidence"],
        ),
    )


def frame_has_candidate_in_confidence_range(detections, lower, upper):
    if upper is None:
        return any(detection["confidence"] > lower for detection in detections)
    return any(lower <= detection["confidence"] < upper for detection in detections)


def histogram_of_confidences(confidence_values):
    histogram_bins = {
        "<0.05": 0,
        "0.05-0.10": 0,
        "0.10-0.20": 0,
        "0.20-0.30": 0,
        "0.30-0.40": 0,
        ">0.40": 0,
    }
    for confidence in confidence_values:
        if confidence < 0.05:
            histogram_bins["<0.05"] += 1
        elif confidence < 0.10:
            histogram_bins["0.05-0.10"] += 1
        elif confidence < 0.20:
            histogram_bins["0.10-0.20"] += 1
        elif confidence < 0.30:
            histogram_bins["0.20-0.30"] += 1
        elif confidence < 0.40:
            histogram_bins["0.30-0.40"] += 1
        else:
            histogram_bins[">0.40"] += 1
    return histogram_bins


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def draw_label(frame, x1, y1, label, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_y1 = max(0, y1 - text_height - baseline - 8)
    label_y2 = label_y1 + text_height + baseline + 8
    label_x2 = min(frame.shape[1] - 1, x1 + text_width + 10)

    cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 5, label_y2 - baseline - 4),
        font,
        font_scale,
        (10, 10, 10),
        thickness,
        cv2.LINE_AA,
    )


def draw_detection(frame, box, class_name, confidence, color=None):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = clamp(x1, 0, frame.shape[1] - 1)
    x2 = clamp(x2, 0, frame.shape[1] - 1)
    y1 = clamp(y1, 0, frame.shape[0] - 1)
    y2 = clamp(y2, 0, frame.shape[0] - 1)

    track_id = None
    team_id = None
    display_name = class_name
    if isinstance(class_name, tuple):
        display_name = class_name[0]
        if len(class_name) > 1:
            track_id = class_name[1]
        if len(class_name) > 2:
            team_id = class_name[2]
    color = color or CLASS_COLOR_OVERRIDES.get(display_name, (0, 179, 255))
    if display_name in {"Player", "Goalkeeper"} and track_id is not None and team_id is not None:
        label = f"{display_name} #{track_id} [Team {team_id}]"
    elif track_id is not None:
        label = f"{display_name} #{track_id} - {round(confidence * 100)}%"
    else:
        label = f"{display_name} - {round(confidence * 100)}%"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_label(frame, x1, y1, label, color)


def open_writer(output_path, fps, width, height):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Não foi possível criar o vídeo processado.")
    return writer


def encode_live_debug_frame(frame):
    if frame is None:
        return None

    height, width = frame.shape[:2]
    if width > LIVE_DEBUG_FRAME_MAX_WIDTH:
        scale = LIVE_DEBUG_FRAME_MAX_WIDTH / width
        frame = cv2.resize(frame, (LIVE_DEBUG_FRAME_MAX_WIDTH, int(height * scale)))

    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_DEBUG_FRAME_JPEG_QUALITY],
    )
    if not ok:
        return None
    return base64.b64encode(buffer).decode("ascii")


def write_live_debug_frame(debug_frame_dir, frame):
    if frame is None or not debug_frame_dir:
        return False

    frame_dir = Path(debug_frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    final_path = frame_dir / LIVE_DEBUG_FRAME_NAME
    temp_path = frame_dir / f".{LIVE_DEBUG_FRAME_NAME}.tmp.jpg"
    log_live_frame_debug(f"write_live_debug_frame temp_path={temp_path.resolve()} final_path={final_path.resolve()}")
    write_ok = cv2.imwrite(str(temp_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    log_live_frame_debug(f"cv2.imwrite returned {write_ok}")
    if not write_ok:
        return False

    for attempt in range(LIVE_DEBUG_FRAME_REPLACE_RETRIES):
        try:
            os.replace(temp_path, final_path)
            log_live_frame_debug(f"os.replace succeeded attempt={attempt + 1}")
            return True
        except PermissionError as exc:
            log_live_frame_debug(
                f"PermissionError on os.replace attempt={attempt + 1} errno={getattr(exc, 'errno', None)} message={exc}"
            )
            if attempt == LIVE_DEBUG_FRAME_REPLACE_RETRIES - 1:
                temp_path.unlink(missing_ok=True)
                return False
            time.sleep(LIVE_DEBUG_FRAME_RETRY_BASE_DELAY_SECONDS * (attempt + 1))
        except Exception:
            log_live_frame_debug(
                "Unexpected exception during live-frame replace:\n" + traceback.format_exc().rstrip()
            )
            raise

    return False


def should_write_live_debug_frame(debug_frame_dir, frames_processed, interval):
    if not debug_frame_dir:
        return False
    return frames_processed == 1 or frames_processed % max(1, interval) == 0


def finalize_browser_video(raw_output_path, final_output_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raw_output_path.replace(final_output_path)
        return

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_output_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(final_output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raw_output_path.replace(final_output_path)
        return

    raw_output_path.unlink(missing_ok=True)


def create_profile_state():
    return {
        "stages": {
            stage_name: {
                "total_seconds": 0.0,
                "calls": 0,
                "min_seconds": None,
                "max_seconds": 0.0,
            }
            for stage_name in PROFILE_STAGE_NAMES
        },
        "frame_total_seconds": [],
    }


def record_profile_stage(profile_state, stage_name, elapsed_seconds):
    if stage_name not in profile_state["stages"]:
        profile_state["stages"][stage_name] = {
            "total_seconds": 0.0,
            "calls": 0,
            "min_seconds": None,
            "max_seconds": 0.0,
        }
    stage = profile_state["stages"][stage_name]
    elapsed = max(0.0, float(elapsed_seconds))
    stage["total_seconds"] += elapsed
    stage["calls"] += 1
    if stage["min_seconds"] is None or elapsed < stage["min_seconds"]:
        stage["min_seconds"] = elapsed
    if elapsed > stage["max_seconds"]:
        stage["max_seconds"] = elapsed


def collect_hardware_info():
    cpu_model = None
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as cpuinfo_file:
            for line in cpuinfo_file:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpu_model = None

    if not cpu_model:
        cpu_model = platform.processor() or platform.machine() or "unknown"

    gpu_info = {
        "available": False,
        "backend": None,
        "count": 0,
        "names": [],
    }
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["backend"] = "torch.cuda"
            gpu_info["count"] = int(torch.cuda.device_count())
            gpu_info["names"] = [
                torch.cuda.get_device_name(index)
                for index in range(gpu_info["count"])
            ]
    except Exception as exc:
        gpu_info["torch_error"] = str(exc)

    if not gpu_info["available"]:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            completed = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
                if names:
                    gpu_info["available"] = True
                    gpu_info["backend"] = "nvidia-smi"
                    gpu_info["count"] = len(names)
                    gpu_info["names"] = names
            else:
                gpu_info["nvidia_smi_error"] = completed.stderr.strip()

    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu": {
            "model": cpu_model,
            "logical_cores": os.cpu_count(),
            "architecture": platform.machine(),
        },
        "gpu": gpu_info,
    }


def build_performance_profile_report(
    profile_state,
    analysis_total_seconds,
    frames_processed,
    width,
    height,
    input_path,
    output_path,
    hardware_info,
    team_classifier_cache_metrics,
    ball_tracking_metrics,
    physical_validation_enabled,
    physical_accept_distance,
    physical_verify_distance,
    physical_max_distance,
    physical_consecutive_confirmations,
    physical_reset_after_missing,
    physical_high_speed_threshold,
):
    frame_total_seconds = profile_state["frame_total_seconds"]
    avg_frame_seconds = (
        sum(frame_total_seconds) / len(frame_total_seconds)
        if frame_total_seconds
        else 0.0
    )
    min_frame_seconds = min(frame_total_seconds) if frame_total_seconds else 0.0
    max_frame_seconds = max(frame_total_seconds) if frame_total_seconds else 0.0
    avg_processing_fps = (
        frames_processed / analysis_total_seconds
        if analysis_total_seconds > 0
        else 0.0
    )

    stages = {}
    for stage_name, stage_stats in profile_state["stages"].items():
        total_seconds = float(stage_stats["total_seconds"])
        calls = int(stage_stats["calls"])
        min_seconds = float(stage_stats["min_seconds"] or 0.0)
        max_seconds = float(stage_stats["max_seconds"])
        avg_seconds = total_seconds / calls if calls > 0 else 0.0
        percent_of_total = (
            (total_seconds / analysis_total_seconds) * 100.0
            if analysis_total_seconds > 0
            else 0.0
        )
        stages[stage_name] = {
            "total_seconds": total_seconds,
            "total_ms": total_seconds * 1000.0,
            "calls": calls,
            "avg_seconds_per_call": avg_seconds,
            "avg_ms_per_call": avg_seconds * 1000.0,
            "min_seconds_per_call": min_seconds,
            "max_seconds_per_call": max_seconds,
            "percent_of_total_analysis_time": percent_of_total,
        }

    ranking_candidates = [
        {"stage": stage_name, **stage_report}
        for stage_name, stage_report in stages.items()
        if stage_name != "frame_total"
    ]
    top_5_slowest_stages = sorted(
        ranking_candidates,
        key=lambda item: item["total_seconds"],
        reverse=True,
    )[:5]

    predict_calls_before_cache = int(team_classifier_cache_metrics["predict_calls_before_cache"])
    predict_calls_after_cache = int(team_classifier_cache_metrics["predict_calls_after_cache"])
    cache_hits = int(team_classifier_cache_metrics["cache_hits"])
    cache_misses = int(team_classifier_cache_metrics["cache_misses"])
    total_cache_decisions = cache_hits + cache_misses
    hit_rate_percent = (
        (cache_hits / total_cache_decisions) * 100.0
        if total_cache_decisions > 0
        else 0.0
    )
    reduction_calls = max(0, predict_calls_before_cache - predict_calls_after_cache)
    reduction_percent = (
        (reduction_calls / predict_calls_before_cache) * 100.0
        if predict_calls_before_cache > 0
        else 0.0
    )
    team_classifier_predict_total_seconds = float(
        profile_state["stages"]["team_classifier_predict"]["total_seconds"]
    )
    avg_predict_call_seconds_after = (
        team_classifier_predict_total_seconds / predict_calls_after_cache
        if predict_calls_after_cache > 0
        else 0.0
    )
    estimated_time_saved_seconds = reduction_calls * avg_predict_call_seconds_after
    prediction_error_sum = float(ball_tracking_metrics["ball_prediction_error_sum"])
    prediction_error_count = int(ball_tracking_metrics["ball_prediction_error_count"])
    average_prediction_error = (
        prediction_error_sum / prediction_error_count if prediction_error_count > 0 else 0.0
    )
    prediction_error_pixels_sum = float(ball_tracking_metrics["prediction_error_pixels_sum"])
    prediction_error_pixels_count = int(ball_tracking_metrics["prediction_error_pixels_count"])
    average_prediction_error_pixels = (
        prediction_error_pixels_sum / prediction_error_pixels_count
        if prediction_error_pixels_count > 0
        else 0.0
    )
    prediction_error_field_sum = float(ball_tracking_metrics["prediction_error_field_sum"])
    prediction_error_field_count = int(ball_tracking_metrics["prediction_error_field_count"])
    average_prediction_error_field = (
        prediction_error_field_sum / prediction_error_field_count
        if prediction_error_field_count > 0
        else 0.0
    )
    roi_size_sum = float(ball_tracking_metrics["roi_size_sum"])
    roi_size_count = int(ball_tracking_metrics["roi_size_count"])
    roi_average_size = roi_size_sum / roi_size_count if roi_size_count > 0 else 0.0
    interruptions_before = int(ball_tracking_metrics["ball_track_interruptions_before"])
    interruptions_after = int(ball_tracking_metrics["ball_track_interruptions_after"])
    continuity_gain_percent = (
        ((interruptions_before - interruptions_after) / interruptions_before) * 100.0
        if interruptions_before > 0
        else 0.0
    )
    frames_using_global_tiling = int(ball_tracking_metrics["frames_using_global_tiling"])
    frames_using_full_global_search = int(ball_tracking_metrics["frames_using_full_global_search"])
    semi_global_searches = int(ball_tracking_metrics["semi_global_searches"])
    semi_global_success = int(ball_tracking_metrics["semi_global_success"])
    global_searches = int(ball_tracking_metrics["global_searches"])
    global_search_success = int(ball_tracking_metrics["global_search_success"])
    frames_using_crop = int(ball_tracking_metrics["frames_using_crop"])
    number_of_global_searches = int(ball_tracking_metrics["number_of_global_searches"])
    number_of_crop_searches = int(ball_tracking_metrics["number_of_crop_searches"])
    crop_expansion_levels_used = dict(ball_tracking_metrics["crop_expansion_levels_used"])
    average_crop_size = (
        float(ball_tracking_metrics["average_crop_size_sum"])
        / float(ball_tracking_metrics["average_crop_size_count"])
        if int(ball_tracking_metrics["average_crop_size_count"]) > 0
        else 0.0
    )
    largest_crop_size = float(ball_tracking_metrics["largest_crop_size"])
    time_spent_global_search = float(ball_tracking_metrics["time_spent_global_search"])
    time_spent_crop_search = float(ball_tracking_metrics["time_spent_crop_search"])
    recoveries_after_global_search = int(ball_tracking_metrics["recoveries_after_global_search"])
    recoveries_after_crop = int(ball_tracking_metrics["recoveries_after_crop"])
    semi_global_success_rate = (
        (semi_global_success / semi_global_searches) * 100.0
        if semi_global_searches > 0
        else 0.0
    )
    global_search_success_rate = (
        (global_search_success / global_searches) * 100.0
        if global_searches > 0
        else 0.0
    )
    global_recovery_success_rate = (
        (recoveries_after_global_search / number_of_global_searches) * 100.0
        if number_of_global_searches > 0
        else 0.0
    )
    crop_search_success_rate = (
        (recoveries_after_crop / number_of_crop_searches) * 100.0
        if number_of_crop_searches > 0
        else 0.0
    )
    percentage_of_frames_using_full_global_search = (
        (frames_using_full_global_search / frames_processed) * 100.0
        if frames_processed > 0
        else 0.0
    )
    physical_detection_distance_count = int(ball_tracking_metrics["physical_detection_distance_count"])
    average_detection_distance = (
        float(ball_tracking_metrics["physical_detection_distance_sum"]) / physical_detection_distance_count
        if physical_detection_distance_count > 0
        else 0.0
    )
    max_detection_distance = float(ball_tracking_metrics["physical_detection_distance_max"])

    return {
        "analysis": {
            "input_video_path": str(input_path),
            "output_video_path": str(output_path),
            "frames_processed": int(frames_processed),
            "resolution": {"width": int(width), "height": int(height)},
            "total_analysis_seconds": float(analysis_total_seconds),
            "total_analysis_ms": float(analysis_total_seconds * 1000.0),
            "avg_processing_fps": float(avg_processing_fps),
            "avg_frame_seconds": float(avg_frame_seconds),
            "avg_frame_ms": float(avg_frame_seconds * 1000.0),
            "max_frame_seconds": float(max_frame_seconds),
            "max_frame_ms": float(max_frame_seconds * 1000.0),
            "min_frame_seconds": float(min_frame_seconds),
            "min_frame_ms": float(min_frame_seconds * 1000.0),
        },
        "hardware": hardware_info,
        "team_classifier_cache": {
            "refresh_interval_frames": TEAM_CLASSIFIER_REFRESH_INTERVAL,
            "predict_calls_before_cache": predict_calls_before_cache,
            "predict_calls_after_cache": predict_calls_after_cache,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate_percent": hit_rate_percent,
            "reduction_calls": reduction_calls,
            "reduction_percent": reduction_percent,
            "estimated_time_saved_seconds": estimated_time_saved_seconds,
        },
        "ball_tracking_enhancements": {
            "memory_max_missing_frames": BALL_MEMORY_MAX_MISSING_FRAMES,
            "ball_memory_hits": int(ball_tracking_metrics["ball_memory_hits"]),
            "ball_memory_misses": int(ball_tracking_metrics["ball_memory_misses"]),
            "ball_roi_searches": int(ball_tracking_metrics["ball_roi_searches"]),
            "ball_roi_success": int(ball_tracking_metrics["ball_roi_success"]),
            "ball_prediction_error": average_prediction_error,
            "ball_prediction_error_samples": prediction_error_count,
            "ball_prediction_hits": int(ball_tracking_metrics["ball_prediction_hits"]),
            "ball_prediction_misses": int(ball_tracking_metrics["ball_prediction_misses"]),
            "roi_search_success": int(ball_tracking_metrics["roi_search_success"]),
            "roi_search_fail": int(ball_tracking_metrics["roi_search_fail"]),
            "average_prediction_error_pixels": average_prediction_error_pixels,
            "average_prediction_error_field": average_prediction_error_field,
            "roi_average_size": roi_average_size,
            "roi_expansions": int(ball_tracking_metrics["roi_expansions"]),
            "global_fallbacks": int(ball_tracking_metrics["global_fallbacks"]),
            "largest_gap_before_prediction": int(
                ball_tracking_metrics["largest_gap_before_prediction"]
            ),
            "largest_gap_after_prediction": int(ball_tracking_metrics["largest_gap_after_prediction"]),
            "prediction_mode_selected": str(ball_tracking_metrics["prediction_mode_selected"]),
            "frames_using_global_tiling": frames_using_global_tiling,
            "frames_using_full_global_search": frames_using_full_global_search,
            "semi_global_searches": semi_global_searches,
            "semi_global_success": semi_global_success,
            "semi_global_success_rate": semi_global_success_rate,
            "crop_expansion_levels_used": crop_expansion_levels_used,
            "frames_using_crop": frames_using_crop,
            "number_of_global_searches": number_of_global_searches,
            "global_searches": global_searches,
            "global_search_success": global_search_success,
            "percentage_of_frames_using_full_global_search": (
                percentage_of_frames_using_full_global_search
            ),
            "number_of_crop_searches": number_of_crop_searches,
            "average_crop_size": average_crop_size,
            "largest_crop_size": largest_crop_size,
            "time_spent_global_search": time_spent_global_search,
            "time_spent_crop_search": time_spent_crop_search,
            "recoveries_after_global_search": recoveries_after_global_search,
            "recoveries_after_crop": recoveries_after_crop,
            "global_search_success_rate": global_search_success_rate,
            "global_recovery_success_rate": global_recovery_success_rate,
            "crop_success_rate": crop_search_success_rate,
            "ball_track_interruptions_before": interruptions_before,
            "ball_track_interruptions_after": interruptions_after,
            "ball_track_continuity_gain_percent": continuity_gain_percent,
            "physical_validation_accepts": int(ball_tracking_metrics["physical_validation_accepts"]),
            "physical_validation_rejects": int(ball_tracking_metrics["physical_validation_rejects"]),
            "physical_validation_pending": int(ball_tracking_metrics["physical_validation_pending"]),
            "physical_validation_confirmed": int(ball_tracking_metrics["physical_validation_confirmed"]),
            "physical_validation_false_candidates": int(
                ball_tracking_metrics["physical_validation_false_candidates"]
            ),
            "physical_validation_teleports_avoided": int(
                ball_tracking_metrics["physical_validation_teleports_avoided"]
            ),
            "average_detection_distance": average_detection_distance,
            "max_detection_distance": max_detection_distance,
            "physical_validation_enabled": bool(physical_validation_enabled),
            "physical_validation_config": {
                "accept_distance": float(physical_accept_distance),
                "verify_distance": float(physical_verify_distance),
                "max_distance": float(physical_max_distance),
                "consecutive_confirmations": int(physical_consecutive_confirmations),
                "reset_after_missing": int(physical_reset_after_missing),
                "high_speed_threshold": float(physical_high_speed_threshold),
            },
        },
        "ball_source": {
            "primary_detector_frames": int(ball_tracking_metrics["primary_detector_frames"]),
            "fallback_frames": int(ball_tracking_metrics["fallback_frames_used"]),
            "neither_frames": int(ball_tracking_metrics["neither_detector_frames"]),
        },
        "ball_fallback": {
            "frames_considered": int(ball_tracking_metrics["fallback_frames_considered"]),
            "frames_used": int(ball_tracking_metrics["fallback_frames_used"]),
            "frames_rejected_distance": int(
                ball_tracking_metrics["fallback_frames_rejected_distance"]
            ),
            "frames_rejected_confidence": int(
                ball_tracking_metrics["fallback_frames_rejected_confidence"]
            ),
            "average_confidence": (
                float(ball_tracking_metrics["fallback_confidence_sum"])
                / float(ball_tracking_metrics["fallback_frames_used"])
                if int(ball_tracking_metrics["fallback_frames_used"]) > 0
                else 0.0
            ),
            "average_distance": (
                float(ball_tracking_metrics["fallback_distance_sum"])
                / float(ball_tracking_metrics["fallback_frames_used"])
                if int(ball_tracking_metrics["fallback_frames_used"]) > 0
                else 0.0
            ),
        },
        "stages": stages,
        "top_5_slowest_stages": top_5_slowest_stages,
        "frame_total_seconds": frame_total_seconds,
        "generated_at_unix_seconds": time.time(),
    }


def write_performance_profile_report(report_path, report):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_suffix(f"{report_path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as file_pointer:
        json.dump(report, file_pointer, ensure_ascii=False, indent=2)
        file_pointer.write("\n")
    os.replace(temp_path, report_path)


def write_ball_diagnostics_report(report_path, report):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_suffix(f"{report_path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as file_pointer:
        json.dump(report, file_pointer, ensure_ascii=False, indent=2)
        file_pointer.write("\n")
    os.replace(temp_path, report_path)


def build_ball_diagnostics_summary(ball_diagnostics_state):
    consecutive_missing_sequences = list(ball_diagnostics_state["consecutive_missing_sequences"])
    current_missing_streak = int(ball_diagnostics_state["current_missing_streak"])
    if current_missing_streak > 0:
        consecutive_missing_sequences.append(current_missing_streak)

    max_missing_streak = max(consecutive_missing_sequences) if consecutive_missing_sequences else 0
    average_missing_streak = (
        float(sum(consecutive_missing_sequences) / len(consecutive_missing_sequences))
        if consecutive_missing_sequences
        else 0.0
    )
    missing_frames_count = int(len(ball_diagnostics_state["frames_without_ball"]))

    reason_counts = {
        "no_candidate": int(ball_diagnostics_state["frames_without_any_candidate"]),
        "confidence": int(ball_diagnostics_state["frames_discarded_by_confidence"]),
        "roi": int(ball_diagnostics_state["frames_discarded_by_roi"]),
        "nms": int(ball_diagnostics_state["frames_discarded_by_nms"]),
    }
    for other_filter_name, count in ball_diagnostics_state["other_filter_frame_counts"].items():
        reason_counts[f"other:{other_filter_name}"] = int(count)

    sorted_reasons = sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)
    top_reasons = [item for item in sorted_reasons if item[1] > 0][:3]
    top_reason_key = top_reasons[0][0] if top_reasons else None

    recommended_adjustment = "no_change_recommended"
    if top_reason_key == "confidence":
        recommended_adjustment = "evaluate lowering confidence gate for predicted-near candidates"
    elif top_reason_key == "roi":
        recommended_adjustment = "evaluate widening ROI radius around predicted position"
    elif top_reason_key == "nms":
        recommended_adjustment = "evaluate relaxing NMS IoU threshold for dense candidate clusters"
    elif top_reason_key == "no_candidate":
        recommended_adjustment = "evaluate improving detector sensitivity for small/blurred ball appearances"
    elif top_reason_key and top_reason_key.startswith("other:"):
        recommended_adjustment = "evaluate downstream ball candidate gating after detector output"

    top_reason_count = top_reasons[0][1] if top_reasons else 0
    estimated_gain_percent = (
        (float(top_reason_count) / float(missing_frames_count)) * 100.0
        if missing_frames_count > 0
        else 0.0
    )

    return {
        "max_consecutive_frames_without_ball": max_missing_streak,
        "avg_consecutive_frames_without_ball": average_missing_streak,
        "consecutive_missing_sequences": consecutive_missing_sequences,
        "missing_frames_count": missing_frames_count,
        "discarded_candidate_confidence_distribution": histogram_of_confidences(
            ball_diagnostics_state["discarded_candidate_confidences"]
        ),
        "frames_with_candidate_confidence_range": dict(
            ball_diagnostics_state["frames_with_candidate_confidence_range"]
        ),
        "frame_counts_by_reason": reason_counts,
        "top_reasons": top_reasons,
        "recommended_adjustment": recommended_adjustment,
        "estimated_gain_percent_if_applied": estimated_gain_percent,
    }


def analyze(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {input_path}")
    analysis_start_time = time.perf_counter()
    profile_state = create_profile_state()
    hardware_info = collect_hardware_info()
    profile_report_path = output_path.with_suffix(f"{output_path.suffix}.profile.json")
    ball_diagnostics_output_path = (
        Path(args.ball_diagnostics_output).resolve()
        if args.ball_diagnostics_output
        else None
    )
    ball_diagnostics_enabled = ball_diagnostics_output_path is not None
    ball_detector_confidence_for_run = (
        max(0.0, float(args.ball_diagnostics_min_confidence))
        if ball_diagnostics_enabled
        else BALL_DETECTOR_DEFAULT_CONFIDENCE
    )
    if (
        not ball_diagnostics_enabled
        and str(args.ball_detector).strip().lower() == "yolo"
    ):
        ball_detector_confidence_for_run = BALL_YOLO_RECALL_CONFIDENCE

    player_model_path = resolve_player_model_path(args.model)
    if not player_model_path.exists():
        raise FileNotFoundError(
            f"Modelo oficial de detecção de jogadores não encontrado: {player_model_path}"
        )
    model = YOLO(str(player_model_path))
    pitch_model_path = resolve_pitch_model_path(args.pitch_model)
    if not pitch_model_path.exists():
        raise FileNotFoundError(f"Modelo de detecção de campo não encontrado: {pitch_model_path}")
    pitch_model = YOLO(str(pitch_model_path))
    ball_detector, ball_detector_backend, ball_detector_model_path = build_ball_detector(
        args=args,
        confidence_threshold=ball_detector_confidence_for_run,
    )
    predictive_ball_tracking_enabled = (
        str(args.predictive_ball_tracking).strip().lower() == "on"
        and ball_detector_backend == "rfdetr"
    )
    hybrid_ball_search_enabled = (
        predictive_ball_tracking_enabled
        and str(args.hybrid_ball_search).strip().lower() == "on"
        and ball_detector_backend == "rfdetr"
    )
    configured_tile_size = max(64, int(args.tile_size))
    configured_tile_overlap = float(max(0.0, min(0.9, float(args.tile_overlap))))
    configured_initial_detections_required = max(1, int(args.initial_detections_required))
    configured_crop_initial_size = max(64, int(args.crop_initial_size))
    configured_crop_max_size = max(configured_crop_initial_size, int(args.crop_max_size))
    configured_max_frames_before_global_search = max(1, int(args.max_frames_before_global_search))
    physical_ball_validation_enabled = str(args.physical_ball_validation).strip().lower() == "on"
    configured_physical_accept_distance = max(1.0, float(args.physical_accept_distance))
    configured_physical_verify_distance = max(
        configured_physical_accept_distance,
        float(args.physical_verify_distance),
    )
    configured_physical_max_distance = max(
        configured_physical_verify_distance,
        float(args.physical_max_distance),
    )
    configured_physical_consecutive_confirmations = max(
        1,
        int(args.physical_consecutive_confirmations),
    )
    configured_physical_reset_after_missing = max(1, int(args.physical_reset_after_missing))
    configured_physical_high_speed_threshold = max(
        0.0,
        float(args.physical_high_speed_threshold),
    )
    adaptive_crop_schedule = build_adaptive_crop_schedule(
        configured_crop_initial_size,
        configured_crop_max_size,
    )
    ball_tracker = BallTracker(buffer_size=BALL_TRACKER_BUFFER_SIZE)
    team_classifier = TeamClassifier(device="cpu")

    def ball_slicer_callback(image_slice):
        return ball_detector.detect(image_slice)

    # NOTE: InferenceSlicer wiring is intentionally kept in place (callback preserved)
    # for easy reactivation, but temporarily disabled from the runtime flow.
    ball_slicer = None
    team_classifier_crops = []
    team_training_stats = {
        "stride": TEAM_CLASSIFIER_STRIDE,
        "max_training_frames": TEAM_CLASSIFIER_MAX_TRAINING_FRAMES,
        "target_samples": TEAM_CLASSIFIER_TARGET_SAMPLES,
        "fit_called_count": 0,
        "predict_called_count": 0,
        "frames_scanned": 0,
        "training_completed_frame": 0,
        "training_stop_reason": "unknown",
        "player_detections_seen": 0,
        "crops_used_for_fit": 0,
        "crops_total": 0,
        "discarded_small": 0,
        "discarded_empty": 0,
        "discarded_border": 0,
        "discarded_blurry": 0,
        "discarded_non_player": {},
        "discarded_low_chroma_official": 0,
        "kept_low_chroma_official": 0,
    }
    low_chroma_training_crops = []
    training_capture = cv2.VideoCapture(str(input_path))
    if not training_capture.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo para treinar TeamClassifier.")
    training_frame_index = 0
    while (
        training_frame_index < TEAM_CLASSIFIER_MAX_TRAINING_FRAMES
        and team_training_stats["crops_used_for_fit"] < TEAM_CLASSIFIER_TARGET_SAMPLES
    ):
        ok, training_frame = training_capture.read()
        if not ok:
            team_training_stats["training_stop_reason"] = "video_ended"
            break
        training_frame_index += 1
        team_training_stats["frames_scanned"] += 1
        team_training_stats["training_completed_frame"] = training_frame_index
        if training_frame_index % TEAM_CLASSIFIER_STRIDE != 0:
            continue

        frame_height, frame_width = training_frame.shape[:2]
        team_training_result = model.predict(
            training_frame,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            classes=[PLAYER_CLASS_ID],
        )[0]
        team_training_detections = sv.Detections.from_ultralytics(team_training_result)
        player_training_detections = team_training_detections[
            team_training_detections.class_id == PLAYER_CLASS_ID
        ]
        team_training_stats["player_detections_seen"] += len(player_training_detections)
        team_training_stats["crops_total"] += len(player_training_detections)
        for player_box in player_training_detections.xyxy:
            can_use_crop, reason = classify_training_crop_quality(
                player_box,
                frame_width,
                frame_height,
            )
            if not can_use_crop:
                if reason == "small":
                    team_training_stats["discarded_small"] += 1
                elif reason == "border":
                    team_training_stats["discarded_border"] += 1
                else:
                    team_training_stats["discarded_empty"] += 1
                continue

            player_crop = sv.crop_image(training_frame, player_box)
            if is_crop_blurry(player_crop, TEAM_CLASSIFIER_MIN_LAPLACIAN_VARIANCE):
                team_training_stats["discarded_blurry"] += 1
                continue
            non_player_reason = reject_training_crop(player_crop)
            if non_player_reason == "low_chroma_official":
                low_chroma_training_crops.append(player_crop)
                continue
            if non_player_reason:
                reason_counts = team_training_stats["discarded_non_player"]
                reason_counts[non_player_reason] = int(reason_counts.get(non_player_reason, 0)) + 1
                continue

            team_classifier_crops.append(player_crop)
            team_training_stats["crops_used_for_fit"] += 1
            if team_training_stats["crops_used_for_fit"] >= TEAM_CLASSIFIER_TARGET_SAMPLES:
                team_training_stats["training_stop_reason"] = "target_samples_reached"
                break

    if team_training_stats["training_stop_reason"] == "unknown":
        if training_frame_index >= TEAM_CLASSIFIER_MAX_TRAINING_FRAMES:
            team_training_stats["training_stop_reason"] = "max_training_frames_reached"
        else:
            team_training_stats["training_stop_reason"] = "video_ended"
    training_capture.release()
    if len(team_classifier_crops) < 12 and low_chroma_training_crops:
        team_training_stats["kept_low_chroma_official"] = len(low_chroma_training_crops)
        team_classifier_crops.extend(low_chroma_training_crops)
        low_chroma_training_crops = []
    team_training_stats["discarded_low_chroma_official"] = len(low_chroma_training_crops)

    if len(team_classifier_crops) == 0:
        raise RuntimeError("TeamClassifier: nenhum crop de jogador encontrado para treinamento.")
    team_classifier_fit_start = time.perf_counter()
    team_classifier.fit(team_classifier_crops)
    record_profile_stage(
        profile_state,
        "team_classifier_fit",
        time.perf_counter() - team_classifier_fit_start,
    )
    team_training_stats["fit_called_count"] = 1
    total_discarded_training_crops = (
        team_training_stats["discarded_small"]
        + team_training_stats["discarded_empty"]
        + team_training_stats["discarded_border"]
        + team_training_stats["discarded_blurry"]
        + sum(team_training_stats["discarded_non_player"].values())
        + team_training_stats["discarded_low_chroma_official"]
    )
    emit(
        {
            "type": "teamClassifierTraining",
            "samplesUsed": team_training_stats["crops_used_for_fit"],
            "totalCrops": team_training_stats["crops_total"],
            "playerDetectionsSeen": team_training_stats["player_detections_seen"],
            "discarded": total_discarded_training_crops,
            "discardedSmall": team_training_stats["discarded_small"],
            "discardedEmpty": team_training_stats["discarded_empty"],
            "discardedBorder": team_training_stats["discarded_border"],
            "discardedBlurry": team_training_stats["discarded_blurry"],
            "stride": TEAM_CLASSIFIER_STRIDE,
            "trainingCompletedFrame": team_training_stats["training_completed_frame"],
            "trainingStopReason": team_training_stats["training_stop_reason"],
            "fitCalledCount": team_training_stats["fit_called_count"],
        }
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo para análise.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    player_tracker = sv.ByteTrack(
        minimum_consecutive_frames=BYTE_TRACK_MIN_CONSECUTIVE_FRAMES,
        lost_track_buffer=BYTE_TRACK_LOST_TRACK_BUFFER,
        minimum_matching_threshold=BYTE_TRACK_MINIMUM_MATCHING_THRESHOLD,
        frame_rate=float(fps),
    )
    raw_output_path = output_path.with_name(f"{output_path.stem}.opencv{output_path.suffix}")
    writer = open_writer(raw_output_path, fps, width, height)

    emit(
        {
            "type": "progress",
            "progress": 1,
            "message": "YOLOv8 carregado. Iniciando detecção de player/goalkeeper/referee/ball e keypoints do campo...",
            "framesProcessed": 0,
            "playersDetected": 0,
            "ballDetected": False,
        }
    )

    frames_processed = 0
    max_players_detected = 0
    raw_player_detections = 0
    total_player_detections = 0
    total_goalkeeper_detections = 0
    total_ball_detections = 0
    duplicate_player_detections_suppressed = 0
    filtered_player_detections = 0
    discarded_outside_field = 0
    field_mask_detected_frames = 0
    previous_field_mask = None
    ball_detected = False
    confidence_sum = {"Player": 0.0, "Ball": 0.0}
    confidence_count = {"Player": 0, "Ball": 0}
    homography_valid_frames = 0
    total_pitch_keypoints_detected = 0
    homography_examples = []
    ball_position_history = deque(maxlen=BALL_TRAJECTORY_HISTORY_SIZE)
    ball_field_position_history = deque(maxlen=BALL_TRAJECTORY_HISTORY_SIZE)
    ball_motion_history = deque(maxlen=PREDICTIVE_HISTORY_SIZE)
    ball_search_mode = "global_search" if hybrid_ball_search_enabled else "standard"
    ball_consecutive_detections_for_crop = 0
    semi_global_recovery_cooldown_remaining = 0
    physical_pending_candidate_state = None
    ball_last_valid_detection = None
    ball_frames_without_detection = 0
    ball_prediction_mode_selected = "pixel"
    ball_previous_raw_present = False
    ball_previous_final_present = False
    ball_current_raw_gap = 0
    ball_current_final_gap = 0
    ball_tracking_metrics = {
        "ball_memory_hits": 0,
        "ball_memory_misses": 0,
        "ball_roi_searches": 0,
        "ball_roi_success": 0,
        "ball_prediction_error_sum": 0.0,
        "ball_prediction_error_count": 0,
        "prediction_error_pixels_sum": 0.0,
        "prediction_error_pixels_count": 0,
        "prediction_error_field_sum": 0.0,
        "prediction_error_field_count": 0,
        "prediction_error_field_to_pixel_sum": 0.0,
        "prediction_error_field_to_pixel_count": 0,
        "ball_prediction_hits": 0,
        "ball_prediction_misses": 0,
        "roi_search_success": 0,
        "roi_search_fail": 0,
        "roi_size_sum": 0.0,
        "roi_size_count": 0,
        "roi_expansions": 0,
        "global_fallbacks": 0,
        "largest_gap_before_prediction": 0,
        "largest_gap_after_prediction": 0,
        "prediction_mode_selected": ball_prediction_mode_selected,
        "frames_using_global_tiling": 0,
        "frames_using_full_global_search": 0,
        "semi_global_searches": 0,
        "semi_global_success": 0,
        "frames_using_crop": 0,
        "number_of_global_searches": 0,
        "global_searches": 0,
        "global_search_success": 0,
        "number_of_crop_searches": 0,
        "crop_expansion_levels_used": {},
        "average_crop_size_sum": 0.0,
        "average_crop_size_count": 0,
        "largest_crop_size": 0.0,
        "time_spent_global_search": 0.0,
        "time_spent_crop_search": 0.0,
        "recoveries_after_global_search": 0,
        "recoveries_after_crop": 0,
        "ball_track_interruptions_before": 0,
        "ball_track_interruptions_after": 0,
        "physical_validation_accepts": 0,
        "physical_validation_rejects": 0,
        "physical_validation_pending": 0,
        "physical_validation_confirmed": 0,
        "physical_validation_false_candidates": 0,
        "physical_validation_teleports_avoided": 0,
        "physical_detection_distance_sum": 0.0,
        "physical_detection_distance_count": 0,
        "physical_detection_distance_max": 0.0,
        "primary_detector_frames": 0,
        "fallback_frames_used": 0,
        "neither_detector_frames": 0,
        "fallback_frames_considered": 0,
        "fallback_frames_rejected_distance": 0,
        "fallback_frames_rejected_confidence": 0,
        "fallback_confidence_sum": 0.0,
        "fallback_distance_sum": 0.0,
    }
    ball_diagnostics_state = {
        "enabled": ball_diagnostics_enabled,
        "inference_confidence": ball_detector_confidence_for_run,
        "detector_backend": ball_detector_backend,
        "detector_model_path": ball_detector_model_path,
        "frames_without_ball": [],
        "final_ball_timeline": [],
        "consecutive_missing_sequences": [],
        "current_missing_streak": 0,
        "discarded_candidate_confidences": [],
        "frames_with_candidate_confidence_range": {
            confidence_range["label"]: 0
            for confidence_range in BALL_DIAGNOSTIC_FRAME_CONFIDENCE_BINS
        },
        "frames_discarded_by_confidence": 0,
        "frames_discarded_by_roi": 0,
        "frames_discarded_by_nms": 0,
        "frames_without_any_candidate": 0,
        "other_filter_frame_counts": {},
    }
    team_prediction_counts = {0: 0, 1: 0}
    team_track_lock = TeamTrackLock()
    team_classifier_cache_metrics = {
        "predict_calls_before_cache": 0,
        "predict_calls_after_cache": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }
    goalkeeper_team_prediction_counts = {0: 0, 1: 0}
    goalkeepers_with_team_id = 0
    goalkeeper_assignment_frame_examples = []
    player_team_mutation_violations = 0
    human_track_last_seen_frame = {}
    recovered_track_gap_events = 0
    recovered_track_ids = set()
    recovered_track_examples = []
    player_referee_class_history = {}
    last_raw_player_referee_class = {}
    last_stabilized_player_referee_class = {}
    player_referee_raw_swap_events = 0
    player_referee_stabilized_swap_events = 0
    person_class_lock = PersonClassLock()
    rejected_non_person = 0
    rejected_non_person_reasons = {}
    team_classification_frame_examples = []
    last_emitted_progress = -1
    possession_engine = PossessionEngine()
    possession_team_stats = PossessionTeamAccumulator(fps)
    ball_heatmap = BallHeatmapAccumulator(pitch=SOCCER_PITCH_CONFIGURATION)
    last_possession_events = possession_engine.snapshot().as_dict()
    pass_detector = PassDetector()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_start_time = time.perf_counter()
        current_frame_number = frames_processed + 1

        field_mask, field_contours, field_detected = detect_field_region(frame, previous_field_mask)
        previous_field_mask = field_mask
        if field_detected:
            field_mask_detected_frames += 1

        player_detection_stage_start = time.perf_counter()
        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
        )[0]
        player_model_detections = sv.Detections.from_ultralytics(result)
        player_ball_candidates = []
        if (
            len(player_model_detections) == 0
            or player_model_detections.class_id is None
        ):
            tracked_human_detections = player_model_detections
        else:
            player_class_ids = np.asarray(player_model_detections.class_id)
            player_ball_mask = player_class_ids == BALL_CLASS_ID
            human_class_ids = np.asarray(
                [GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, REFEREE_CLASS_ID],
                dtype=player_class_ids.dtype,
            )
            human_mask = np.isin(player_class_ids, human_class_ids)
            player_ball_candidates = ball_detections_to_dicts(
                player_model_detections[player_ball_mask],
                source_name="player-detector-ball",
            )
            tracked_human_detections = player_model_detections[human_mask]
        tracked_human_detections, frame_non_person_reasons = reject_non_person_detections(
            tracked_human_detections,
            frame,
        )
        for reason, count in frame_non_person_reasons.items():
            rejected_non_person += int(count)
            rejected_non_person_reasons[reason] = int(rejected_non_person_reasons.get(reason, 0)) + int(count)
        record_profile_stage(
            profile_state,
            "player_detection",
            time.perf_counter() - player_detection_stage_start,
        )

        bytetrack_stage_start = time.perf_counter()
        tracked_human_detections = player_tracker.update_with_detections(tracked_human_detections)
        record_profile_stage(
            profile_state,
            "bytetrack",
            time.perf_counter() - bytetrack_stage_start,
        )

        pitch_detection_stage_start = time.perf_counter()
        pitch_result = pitch_model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.pitch_keypoint_conf,
            verbose=False,
        )[0]
        pitch_keypoints = extract_pitch_keypoints(pitch_result, args.pitch_keypoint_conf)
        total_pitch_keypoints_detected += len(pitch_keypoints)
        record_profile_stage(
            profile_state,
            "pitch_detection",
            time.perf_counter() - pitch_detection_stage_start,
        )

        homography_stage_start = time.perf_counter()
        view_transformer = build_view_transformer_from_pitch_keypoints(pitch_keypoints)
        if view_transformer is not None:
            homography_valid_frames += 1
        record_profile_stage(
            profile_state,
            "homography",
            time.perf_counter() - homography_stage_start,
        )

        pixel_motion_state = predict_motion_state_from_history(ball_position_history, current_frame_number)
        predicted_pixel_center = (
            clamp_center_to_frame(pixel_motion_state["predicted"], width, height)
            if pixel_motion_state is not None
            else None
        )
        field_motion_state = predict_motion_state_from_history(
            ball_field_position_history,
            current_frame_number,
        )
        predicted_field_center = (
            field_motion_state["predicted"] if field_motion_state is not None else None
        )
        predicted_field_center_in_pixels = clamp_center_to_frame(
            transform_field_point_to_image(view_transformer, predicted_field_center),
            width,
            height,
        )

        pixel_error_average = (
            float(ball_tracking_metrics["prediction_error_pixels_sum"])
            / float(ball_tracking_metrics["prediction_error_pixels_count"])
            if int(ball_tracking_metrics["prediction_error_pixels_count"]) > 0
            else float("inf")
        )
        pixel_error_count = int(ball_tracking_metrics["prediction_error_pixels_count"])
        field_to_pixel_error_average = (
            float(ball_tracking_metrics["prediction_error_field_to_pixel_sum"])
            / float(ball_tracking_metrics["prediction_error_field_to_pixel_count"])
            if int(ball_tracking_metrics["prediction_error_field_to_pixel_count"]) > 0
            else float("inf")
        )
        field_to_pixel_error_count = int(ball_tracking_metrics["prediction_error_field_to_pixel_count"])
        # Keep motion prediction stability in pixel space for tracker decisions.
        # Field-space velocity is still leveraged to place the search crop when available.
        use_field_prediction = False
        predicted_ball_center = predicted_pixel_center
        ball_prediction_mode_selected = "pixel"
        ball_tracking_metrics["prediction_mode_selected"] = ball_prediction_mode_selected
        predicted_center_for_search = predicted_ball_center
        if (
            view_transformer is not None
            and predicted_field_center_in_pixels is not None
            and len(ball_field_position_history) >= 2
        ):
            if predicted_pixel_center is None:
                predicted_center_for_search = predicted_field_center_in_pixels
            else:
                prediction_delta = float(
                    np.linalg.norm(predicted_field_center_in_pixels - predicted_pixel_center)
                )
                if prediction_delta <= FIELD_PREDICTION_MAX_DELTA_PX:
                    predicted_center_for_search = predicted_field_center_in_pixels

        ball_detector_roi_xyxy = None
        predictive_roi_size_used = None
        search_mode_for_frame = "standard"
        detection_source_mode = "standard"
        semi_global_window_xyxy = None
        crop_expansion_threshold = int(configured_max_frames_before_global_search)
        if hybrid_ball_search_enabled:
            if (
                ball_search_mode == "smart_crop"
                and ball_frames_without_detection >= crop_expansion_threshold
            ):
                ball_search_mode = "global_search"
                ball_consecutive_detections_for_crop = 0
                ball_tracking_metrics["global_fallbacks"] += 1

            if ball_search_mode == "smart_crop":
                search_mode_for_frame = "crop"
                predictive_roi_size_used = pick_adaptive_crop_size(
                    adaptive_crop_schedule,
                    ball_frames_without_detection,
                )
                if semi_global_recovery_cooldown_remaining > 0:
                    cooldown_progress = (
                        float(SEMI_GLOBAL_RECOVERY_COOLDOWN_FRAMES - semi_global_recovery_cooldown_remaining)
                        / float(max(1, SEMI_GLOBAL_RECOVERY_COOLDOWN_FRAMES - 1))
                    )
                    cooldown_start = float(max(adaptive_crop_schedule))
                    cooldown_end = float(min(adaptive_crop_schedule))
                    cooldown_size = cooldown_start + (
                        (cooldown_end - cooldown_start) * cooldown_progress
                    )
                    predictive_roi_size_used = max(
                        float(predictive_roi_size_used),
                        float(max(cooldown_end, cooldown_size)),
                    )
                    semi_global_recovery_cooldown_remaining -= 1
                crop_center = predicted_center_for_search
                if crop_center is None and ball_last_valid_detection is not None:
                    crop_center = detection_center_np(ball_last_valid_detection["box"])
                ball_detector_roi_xyxy = (
                    build_square_roi_from_center(crop_center, predictive_roi_size_used, width, height)
                    if crop_center is not None and predictive_roi_size_used is not None
                    else None
                )
                if ball_detector_roi_xyxy is None:
                    search_mode_for_frame = "semi_global"
                    ball_search_mode = "global_search"
                    ball_consecutive_detections_for_crop = 0
                elif predictive_roi_size_used is not None:
                    crop_level_label = str(int(round(float(predictive_roi_size_used))))
                    crop_levels = ball_tracking_metrics["crop_expansion_levels_used"]
                    crop_levels[crop_level_label] = int(crop_levels.get(crop_level_label, 0)) + 1
                    if ball_frames_without_detection > 0:
                        ball_tracking_metrics["roi_expansions"] += 1
                    ball_tracking_metrics["roi_size_sum"] += float(predictive_roi_size_used)
                    ball_tracking_metrics["roi_size_count"] += 1
                    ball_tracking_metrics["average_crop_size_sum"] += float(predictive_roi_size_used)
                    ball_tracking_metrics["average_crop_size_count"] += 1
                    ball_tracking_metrics["largest_crop_size"] = max(
                        float(ball_tracking_metrics["largest_crop_size"]),
                        float(predictive_roi_size_used),
                    )
                    ball_tracking_metrics["ball_roi_searches"] += 1

            if ball_search_mode == "global_search":
                search_mode_for_frame = "semi_global"
                semi_global_center = predicted_center_for_search
                if semi_global_center is None and ball_last_valid_detection is not None:
                    semi_global_center = detection_center_np(ball_last_valid_detection["box"])
                if semi_global_center is None:
                    semi_global_center = np.array([width * 0.5, height * 0.5], dtype=np.float32)
                semi_global_window_xyxy = build_square_roi_from_center(
                    semi_global_center,
                    SEMI_GLOBAL_WINDOW_SIZE,
                    width,
                    height,
                )
        elif predictive_ball_tracking_enabled and predicted_ball_center is not None:
            predictive_roi_stage = min(
                max(0, int(ball_frames_without_detection)),
                PREDICTIVE_GLOBAL_FALLBACK_AFTER_MISSES,
            )
            if predictive_roi_stage >= PREDICTIVE_GLOBAL_FALLBACK_AFTER_MISSES:
                ball_tracking_metrics["global_fallbacks"] += 1
            else:
                predictive_roi_size_used = float(
                    PREDICTIVE_ROI_SIZE_SEQUENCE[
                        min(predictive_roi_stage, len(PREDICTIVE_ROI_SIZE_SEQUENCE) - 1)
                    ]
                )
                ball_detector_roi_xyxy = build_square_roi_from_center(
                    predicted_ball_center,
                    predictive_roi_size_used,
                    width,
                    height,
                )
                if ball_detector_roi_xyxy is not None:
                    ball_tracking_metrics["ball_roi_searches"] += 1
                    ball_tracking_metrics["roi_size_sum"] += predictive_roi_size_used
                    ball_tracking_metrics["roi_size_count"] += 1
                    if predictive_roi_stage > 0:
                        ball_tracking_metrics["roi_expansions"] += 1

        ball_detection_stage_start = time.perf_counter()
        if hybrid_ball_search_enabled and search_mode_for_frame == "semi_global":
            ball_tracking_metrics["semi_global_searches"] += 1
            semi_global_detections = detect_ball_with_global_tiling(
                frame=frame,
                ball_detector=ball_detector,
                tile_size=configured_tile_size,
                tile_overlap=configured_tile_overlap,
                region_xyxy=semi_global_window_xyxy,
            )
            semi_global_detections_nms = (
                semi_global_detections.with_nms(threshold=0.1)
                if len(semi_global_detections) > 0
                else semi_global_detections
            )
            if len(semi_global_detections_nms) > 0:
                ball_detector_raw_detections = semi_global_detections
                detection_source_mode = "semi_global"
            else:
                ball_tracking_metrics["global_searches"] += 1
                ball_tracking_metrics["number_of_global_searches"] += 1
                ball_tracking_metrics["frames_using_global_tiling"] += 1
                ball_tracking_metrics["frames_using_full_global_search"] += 1
                full_global_start_time = time.perf_counter()
                ball_detector_raw_detections = detect_ball_with_global_tiling(
                    frame=frame,
                    ball_detector=ball_detector,
                    tile_size=configured_tile_size,
                    tile_overlap=configured_tile_overlap,
                )
                ball_tracking_metrics["time_spent_global_search"] += (
                    time.perf_counter() - full_global_start_time
                )
                detection_source_mode = "full_global"
        elif hybrid_ball_search_enabled and search_mode_for_frame == "crop":
            ball_tracking_metrics["frames_using_crop"] += 1
            ball_tracking_metrics["number_of_crop_searches"] += 1
            crop_start_time = time.perf_counter()
            ball_detector_raw_detections = ball_detector.detect(frame, roi_xyxy=ball_detector_roi_xyxy)
            ball_tracking_metrics["time_spent_crop_search"] += time.perf_counter() - crop_start_time
            detection_source_mode = "crop"
        else:
            ball_detector_raw_detections = ball_detector.detect(frame, roi_xyxy=ball_detector_roi_xyxy)
        ball_detector_raw_dicts = ball_detections_to_dicts(
            ball_detector_raw_detections,
            source_name=ball_detector.source_name,
        )
        ball_detector_nms_keep_mask = compute_nms_keep_mask(ball_detector_raw_dicts, 0.1)
        ball_detections_after_nms = ball_detector_raw_detections.with_nms(threshold=0.1)
        ball_detections_after_nms_dicts = ball_detections_to_dicts(
            ball_detections_after_nms,
            source_name=ball_detector.source_name,
        )
        dedicated_ball_detections = ball_detections_after_nms
        ball_detection_elapsed = time.perf_counter() - ball_detection_stage_start
        record_profile_stage(
            profile_state,
            "ball_detection",
            ball_detection_elapsed,
        )

        player_detections = []
        goalkeeper_detections = []
        referee_detections = []
        ball_detections = []

        tracker_ids = tracked_human_detections.tracker_id
        if tracker_ids is None:
            tracker_ids = np.full(len(tracked_human_detections), -1, dtype=np.int32)
        current_frame_track_ids = set()
        for tracker_id_value in tracker_ids:
            tracker_id_int = int(tracker_id_value)
            if tracker_id_int >= 0:
                current_frame_track_ids.add(tracker_id_int)
        for track_id in current_frame_track_ids:
            last_seen_frame = human_track_last_seen_frame.get(track_id)
            if last_seen_frame is not None and current_frame_number - last_seen_frame > 1:
                missing_frames = current_frame_number - last_seen_frame - 1
                recovered_track_gap_events += 1
                recovered_track_ids.add(track_id)
                if len(recovered_track_examples) < 5:
                    recovered_track_examples.append(
                        {
                            "track_id": track_id,
                            "missing_frames": missing_frames,
                            "recovered_frame": current_frame_number,
                        }
                    )
            human_track_last_seen_frame[track_id] = current_frame_number

        for detection_index in range(len(tracked_human_detections)):
            raw_class_id = int(tracked_human_detections.class_id[detection_index])
            confidence = float(tracked_human_detections.confidence[detection_index])
            box = tracked_human_detections.xyxy[detection_index].tolist()
            track_id = int(tracker_ids[detection_index])
            class_id = raw_class_id
            if track_id >= 0 and raw_class_id in {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID, REFEREE_CLASS_ID}:
                previous_raw_class = last_raw_player_referee_class.get(track_id)
                if (
                    previous_raw_class is not None
                    and previous_raw_class != raw_class_id
                    and raw_class_id in {PLAYER_CLASS_ID, REFEREE_CLASS_ID}
                    and previous_raw_class in {PLAYER_CLASS_ID, REFEREE_CLASS_ID}
                ):
                    player_referee_raw_swap_events += 1
                last_raw_player_referee_class[track_id] = raw_class_id
                class_id = person_class_lock.assign(
                    track_id,
                    raw_class_id,
                    current_frame_number,
                    verifying=person_class_lock.is_locked(track_id),
                )

            detection_payload = {
                "box": box,
                "bounding_box": box,
                "confidence": confidence,
                "track_id": track_id,
                "field_x": None,
                "field_y": None,
            }

            if class_id == PLAYER_CLASS_ID:
                raw_player_detections += 1
                player_detections.append(detection_payload)
            elif class_id == GOALKEEPER_CLASS_ID:
                goalkeeper_detections.append(detection_payload)
            elif class_id == REFEREE_CLASS_ID:
                referee_detections.append(detection_payload)

        selection_ball_candidates = keep_recoverable_ball_candidates(
            ball_detections_after_nms_dicts,
            predicted_ball_center,
        )
        if (
            hybrid_ball_search_enabled
            and detection_source_mode in {"semi_global", "full_global"}
            and not selection_ball_candidates
            and ball_detector_raw_dicts
        ):
            selection_ball_candidates = keep_recoverable_ball_candidates(
                ball_detector_raw_dicts,
                predicted_ball_center,
            )

        roi_radius_pixels = (
            max(BALL_ROI_MIN_RADIUS_PX, predictive_roi_size_used * 0.5)
            if predictive_roi_size_used is not None
            else ball_roi_radius(ball_last_valid_detection)
        )
        if hybrid_ball_search_enabled:
            if detection_source_mode in {"semi_global", "full_global"}:
                selected_ball_detection = pick_detection_with_confidence_fusion(
                    selection_ball_candidates,
                    None,
                )
                roi_attempted = False
                roi_success = selected_ball_detection is not None
            else:
                selected_ball_detection = pick_detection_with_confidence_fusion(
                    selection_ball_candidates,
                    predicted_center_for_search,
                    near_min_confidence=HYBRID_BALL_CONFIDENCE_NEAR_MIN,
                    far_min_confidence=HYBRID_BALL_CONFIDENCE_FAR_MIN,
                    proximity_bonus_distance_px=HYBRID_CONFIDENCE_BONUS_DISTANCE_PX,
                    proximity_bonus_delta=HYBRID_CONFIDENCE_BONUS_DELTA,
                )
                roi_attempted = search_mode_for_frame == "crop"
                roi_success = roi_attempted and selected_ball_detection is not None
        elif predictive_ball_tracking_enabled:
            selected_ball_detection = pick_detection_with_confidence_fusion(
                selection_ball_candidates,
                predicted_ball_center,
                near_min_confidence=PREDICTIVE_BALL_CONFIDENCE_NEAR_MIN,
                far_min_confidence=PREDICTIVE_BALL_CONFIDENCE_FAR_MIN,
            )
            roi_attempted = ball_detector_roi_xyxy is not None
            roi_success = roi_attempted and selected_ball_detection is not None
        else:
            yolo_ball_detector = ball_detector_backend == "yolo"
            selected_ball_detection, roi_attempted, roi_success = select_ball_detection_with_roi(
                selection_ball_candidates,
                predicted_ball_center,
                roi_radius_pixels,
                near_min_confidence=(
                    BALL_YOLO_RECALL_CONFIDENCE
                    if yolo_ball_detector
                    else BALL_CONFIDENCE_NEAR_MIN
                ),
                far_min_confidence=(
                    BALL_STRONG_CONFIDENCE
                    if yolo_ball_detector
                    else BALL_CONFIDENCE_FAR_MIN
                ),
            )

        predicted_speed_for_validation = (
            float(np.linalg.norm(pixel_motion_state["velocity"]))
            if pixel_motion_state is not None
            else 0.0
        )
        if physical_ball_validation_enabled:
            selected_ball_detection, physical_pending_candidate_state, physical_validation_stats = (
                apply_physical_ball_validation(
                    selected_detection=selected_ball_detection,
                    predicted_center=predicted_center_for_search,
                    pending_candidate_state=physical_pending_candidate_state,
                    current_frame_number=current_frame_number,
                    missing_frames=ball_frames_without_detection,
                    last_valid_detection=ball_last_valid_detection,
                    predicted_speed=predicted_speed_for_validation,
                    accept_distance=configured_physical_accept_distance,
                    verify_distance=configured_physical_verify_distance,
                    max_distance=configured_physical_max_distance,
                    consecutive_confirmations=configured_physical_consecutive_confirmations,
                    reset_after_missing=configured_physical_reset_after_missing,
                    high_speed_threshold=configured_physical_high_speed_threshold,
                )
            )
            ball_tracking_metrics["physical_validation_accepts"] += int(physical_validation_stats["accepts"])
            ball_tracking_metrics["physical_validation_rejects"] += int(physical_validation_stats["rejects"])
            ball_tracking_metrics["physical_validation_pending"] += int(physical_validation_stats["pending"])
            ball_tracking_metrics["physical_validation_confirmed"] += int(physical_validation_stats["confirmed"])
            ball_tracking_metrics["physical_validation_false_candidates"] += int(
                physical_validation_stats["false_candidates"]
            )
            ball_tracking_metrics["physical_validation_teleports_avoided"] += int(
                physical_validation_stats["teleports_avoided"]
            )
            if physical_validation_stats["distance"] is not None:
                physical_distance = float(physical_validation_stats["distance"])
                ball_tracking_metrics["physical_detection_distance_sum"] += physical_distance
                ball_tracking_metrics["physical_detection_distance_count"] += 1
                ball_tracking_metrics["physical_detection_distance_max"] = max(
                    float(ball_tracking_metrics["physical_detection_distance_max"]),
                    physical_distance,
                )
        else:
            physical_pending_candidate_state = None

        used_player_ball_fallback = False
        if selected_ball_detection is not None:
            ball_tracking_metrics["primary_detector_frames"] += 1
        else:
            ball_tracking_metrics["fallback_frames_considered"] += 1
            fallback_detection, fallback_decision = select_player_ball_fallback(
                player_ball_candidates,
                predicted_ball_center,
                ball_last_valid_detection,
            )
            if fallback_decision["used"]:
                selected_ball_detection = fallback_detection
                used_player_ball_fallback = True
                ball_tracking_metrics["fallback_frames_used"] += 1
                ball_tracking_metrics["fallback_confidence_sum"] += float(
                    fallback_decision["confidence"]
                )
                ball_tracking_metrics["fallback_distance_sum"] += float(
                    fallback_decision["distance"]
                )
                # The dedicated detector left no ball. Drop its unconfirmed
                # far candidate so it cannot replace this continuation later.
                physical_pending_candidate_state = None
            else:
                if fallback_decision["rejected_distance"]:
                    ball_tracking_metrics["fallback_frames_rejected_distance"] += 1
                elif fallback_decision["rejected_confidence"]:
                    ball_tracking_metrics["fallback_frames_rejected_confidence"] += 1
                ball_tracking_metrics["neither_detector_frames"] += 1

        dedicated_ball_detections = ball_dicts_to_detections(
            [selected_ball_detection] if selected_ball_detection is not None else []
        )
        ball_tracker_stage_start = time.perf_counter()
        dedicated_ball_detections = ball_tracker.update(dedicated_ball_detections)
        record_profile_stage(
            profile_state,
            "ball_tracker",
            time.perf_counter() - ball_tracker_stage_start,
        )
        tracker_ball_detections = ball_detections_to_dicts(
            dedicated_ball_detections,
            source_name=ball_detector.source_name,
        )
        if not tracker_ball_detections and selected_ball_detection is not None:
            tracker_ball_detections = [selected_ball_detection]
        if tracker_ball_detections:
            selected_ball_detection = tracker_ball_detections[0]
            if used_player_ball_fallback:
                selected_ball_detection["source"] = "player-detector-ball"
        else:
            selected_ball_detection = None

        raw_ball_present = bool(tracker_ball_detections)
        if raw_ball_present:
            ball_current_raw_gap = 0
        else:
            ball_current_raw_gap += 1
            ball_tracking_metrics["largest_gap_before_prediction"] = max(
                int(ball_tracking_metrics["largest_gap_before_prediction"]),
                ball_current_raw_gap,
            )
        if ball_previous_raw_present and not raw_ball_present:
            ball_tracking_metrics["ball_track_interruptions_before"] += 1

        if roi_attempted:
            if not predictive_ball_tracking_enabled:
                ball_tracking_metrics["ball_roi_searches"] += 1
            if roi_success:
                ball_tracking_metrics["ball_roi_success"] += 1
                ball_tracking_metrics["roi_search_success"] += 1
            else:
                ball_tracking_metrics["roi_search_fail"] += 1

        if predicted_center_for_search is not None:
            if selected_ball_detection is not None:
                ball_tracking_metrics["ball_prediction_hits"] += 1
            else:
                ball_tracking_metrics["ball_prediction_misses"] += 1

        previous_missing_frames = int(ball_frames_without_detection)
        # A box below 0.25 may continue the orange ball, but it must not keep a
        # false track alive. After the normal 5-frame memory the prediction is
        # dropped, so a later strong ball can be acquired again.
        weak_yolo_continuation = (
            selected_ball_detection is not None
            and ball_detector_backend == "yolo"
            and float(selected_ball_detection["confidence"]) < BALL_STRONG_CONFIDENCE
        )
        if selected_ball_detection is not None:
            ball_detections = [selected_ball_detection]
            if weak_yolo_continuation:
                ball_frames_without_detection += 1
            else:
                ball_frames_without_detection = 0
            ball_last_valid_detection = selected_ball_detection
            if hybrid_ball_search_enabled:
                if detection_source_mode in {"semi_global", "full_global"}:
                    ball_consecutive_detections_for_crop += 1
                    if detection_source_mode == "semi_global":
                        ball_tracking_metrics["semi_global_success"] += 1
                        semi_global_recovery_cooldown_remaining = (
                            SEMI_GLOBAL_RECOVERY_COOLDOWN_FRAMES
                        )
                    if detection_source_mode == "full_global":
                        ball_tracking_metrics["global_search_success"] += 1
                    if previous_missing_frames > 0:
                        ball_tracking_metrics["recoveries_after_global_search"] += 1
                elif search_mode_for_frame == "crop":
                    ball_consecutive_detections_for_crop = configured_initial_detections_required
                    if previous_missing_frames > 0:
                        ball_tracking_metrics["recoveries_after_crop"] += 1
                if (
                    ball_search_mode == "global_search"
                    and ball_consecutive_detections_for_crop >= configured_initial_detections_required
                ):
                    ball_search_mode = "smart_crop"
            selected_center = detection_center_np(selected_ball_detection["box"])
            ball_position_history.append((current_frame_number, selected_center))
            selected_field_center = transform_image_point_to_field(view_transformer, selected_center)
            if selected_field_center is not None:
                ball_field_position_history.append((current_frame_number, selected_field_center))

            chosen_motion_state = (
                field_motion_state
                if ball_prediction_mode_selected == "field" and field_motion_state is not None
                else pixel_motion_state
            )
            velocity = (
                chosen_motion_state["velocity"]
                if chosen_motion_state is not None
                else np.zeros(2, dtype=np.float32)
            )
            direction = (
                chosen_motion_state["direction"]
                if chosen_motion_state is not None
                else np.zeros(2, dtype=np.float32)
            )
            ball_motion_history.append(
                {
                    "frame": int(current_frame_number),
                    "position_x": float(selected_center[0]),
                    "position_y": float(selected_center[1]),
                    "vx": float(velocity[0]),
                    "vy": float(velocity[1]),
                    "direction_x": float(direction[0]),
                    "direction_y": float(direction[1]),
                    "mode": ball_prediction_mode_selected,
                }
            )

            if predicted_ball_center is not None:
                prediction_error = float(np.linalg.norm(selected_center - predicted_ball_center))
                ball_tracking_metrics["ball_prediction_error_sum"] += prediction_error
                ball_tracking_metrics["ball_prediction_error_count"] += 1
                ball_tracking_metrics["prediction_error_pixels_sum"] += prediction_error
                ball_tracking_metrics["prediction_error_pixels_count"] += 1
            if (
                predicted_field_center is not None
                and selected_field_center is not None
            ):
                field_error = float(np.linalg.norm(selected_field_center - predicted_field_center))
                ball_tracking_metrics["prediction_error_field_sum"] += field_error
                ball_tracking_metrics["prediction_error_field_count"] += 1
            if predicted_field_center_in_pixels is not None:
                field_to_pixel_error = float(
                    np.linalg.norm(selected_center - predicted_field_center_in_pixels)
                )
                ball_tracking_metrics["prediction_error_field_to_pixel_sum"] += field_to_pixel_error
                ball_tracking_metrics["prediction_error_field_to_pixel_count"] += 1
            if (
                weak_yolo_continuation
                and ball_frames_without_detection > BALL_MEMORY_MAX_MISSING_FRAMES
            ):
                ball_last_valid_detection = None
                ball_position_history.clear()
                ball_field_position_history.clear()
                ball_motion_history.clear()
        else:
            ball_detections = []
            ball_frames_without_detection += 1
            if hybrid_ball_search_enabled and ball_search_mode == "global_search":
                ball_consecutive_detections_for_crop = 0
            if (
                ball_last_valid_detection is not None
                and ball_frames_without_detection <= BALL_MEMORY_MAX_MISSING_FRAMES
            ):
                ball_tracking_metrics["ball_memory_hits"] += 1
            else:
                ball_tracking_metrics["ball_memory_misses"] += 1

            if ball_frames_without_detection > BALL_MEMORY_MAX_MISSING_FRAMES:
                ball_last_valid_detection = None
                ball_position_history.clear()
                ball_field_position_history.clear()
                ball_motion_history.clear()
            if (
                hybrid_ball_search_enabled
                and ball_search_mode == "smart_crop"
                and ball_frames_without_detection >= crop_expansion_threshold
            ):
                ball_search_mode = "global_search"
                ball_consecutive_detections_for_crop = 0
        visual_ball_present = bool(ball_detections)
        if visual_ball_present:
            ball_current_final_gap = 0
        else:
            ball_current_final_gap += 1
            ball_tracking_metrics["largest_gap_after_prediction"] = max(
                int(ball_tracking_metrics["largest_gap_after_prediction"]),
                ball_current_final_gap,
            )
        internal_ball_track_alive = visual_ball_present or (
            ball_last_valid_detection is not None
            and ball_frames_without_detection <= BALL_MEMORY_MAX_MISSING_FRAMES
        )
        if ball_previous_final_present and not internal_ball_track_alive:
            ball_tracking_metrics["ball_track_interruptions_after"] += 1
        ball_previous_raw_present = raw_ball_present
        ball_previous_final_present = internal_ball_track_alive
        if ball_diagnostics_state["enabled"]:
            timeline_entry = {
                "frame_id": int(current_frame_number),
                "detected": bool(visual_ball_present),
                "bbox_xyxy": None,
                "bbox_xywh": None,
                "confidence": None,
                "center": None,
                "class": "Ball",
                "class_id": BALL_CLASS_ID,
            }
            if visual_ball_present and ball_detections:
                chosen_detection = ball_detections[0]
                timeline_entry["bbox_xyxy"] = chosen_detection["box"]
                timeline_entry["bbox_xywh"] = detection_box_to_xywh(chosen_detection["box"])
                timeline_entry["confidence"] = float(chosen_detection["confidence"])
                timeline_entry["center"] = chosen_detection.get("center")
                timeline_entry["class_id"] = int(chosen_detection.get("class_id", BALL_CLASS_ID))
            ball_diagnostics_state["final_ball_timeline"].append(timeline_entry)

            if visual_ball_present:
                if ball_diagnostics_state["current_missing_streak"] > 0:
                    ball_diagnostics_state["consecutive_missing_sequences"].append(
                        ball_diagnostics_state["current_missing_streak"]
                    )
                    ball_diagnostics_state["current_missing_streak"] = 0
            else:
                ball_diagnostics_state["current_missing_streak"] += 1
                for confidence_range in BALL_DIAGNOSTIC_FRAME_CONFIDENCE_BINS:
                    if frame_has_candidate_in_confidence_range(
                        ball_detector_raw_dicts,
                        confidence_range["lower"],
                        confidence_range["upper"],
                    ):
                        label = confidence_range["label"]
                        ball_diagnostics_state["frames_with_candidate_confidence_range"][label] += 1

                best_candidate_index = pick_best_ball_diagnostic_candidate(
                    ball_detector_raw_dicts,
                    predicted_ball_center,
                )
                had_prediction = predicted_ball_center is not None
                frame_diagnostic_entry = {
                    "frame_id": int(current_frame_number),
                    "had_predicted_position": bool(had_prediction),
                    "distance_prediction_to_best_candidate": None,
                    "best_candidate_confidence": None,
                    "detector_candidates_count": int(len(ball_detector_raw_dicts)),
                    "best_candidate_bbox_xywh": None,
                    "best_candidate_bbox_area": None,
                    "discarded_by_confidence": False,
                    "discarded_by_nms": False,
                    "discarded_by_roi": False,
                    "discarded_by_other_filter": None,
                }
                if best_candidate_index is None:
                    ball_diagnostics_state["frames_without_any_candidate"] += 1
                    frame_diagnostic_entry["discarded_by_other_filter"] = "no_detector_candidates"
                else:
                    best_candidate = ball_detector_raw_dicts[best_candidate_index]
                    best_candidate_center = detection_center_np(best_candidate["box"])
                    prediction_distance = (
                        float(np.linalg.norm(best_candidate_center - predicted_ball_center))
                        if had_prediction
                        else None
                    )
                    confidence_threshold = ball_confidence_threshold_for_distance(prediction_distance)
                    discarded_by_confidence = (
                        confidence_threshold is not None
                        and best_candidate["confidence"] < confidence_threshold
                    )
                    discarded_by_nms = (
                        best_candidate_index < len(ball_detector_nms_keep_mask)
                        and not ball_detector_nms_keep_mask[best_candidate_index]
                    )
                    discarded_by_roi = (
                        had_prediction
                        and prediction_distance is not None
                        and prediction_distance > roi_radius_pixels
                    )

                    other_filter_name = None
                    if not discarded_by_confidence and not discarded_by_nms and not discarded_by_roi:
                        candidate_present_after_ball_tracker = any(
                            box_iou(best_candidate["box"], tracker_candidate["box"]) >= 0.99
                            for tracker_candidate in tracker_ball_detections
                        )
                        if not candidate_present_after_ball_tracker:
                            other_filter_name = "ball_tracker_selection"
                        elif selected_ball_detection is None:
                            other_filter_name = "selection_rule"

                    if discarded_by_confidence:
                        ball_diagnostics_state["frames_discarded_by_confidence"] += 1
                    if discarded_by_nms:
                        ball_diagnostics_state["frames_discarded_by_nms"] += 1
                    if discarded_by_roi:
                        ball_diagnostics_state["frames_discarded_by_roi"] += 1
                    if other_filter_name is not None:
                        ball_diagnostics_state["other_filter_frame_counts"][other_filter_name] = (
                            int(ball_diagnostics_state["other_filter_frame_counts"].get(other_filter_name, 0)) + 1
                        )
                    if (
                        discarded_by_confidence
                        or discarded_by_nms
                        or discarded_by_roi
                        or other_filter_name is not None
                    ):
                        ball_diagnostics_state["discarded_candidate_confidences"].append(
                            float(best_candidate["confidence"])
                        )

                    frame_diagnostic_entry["distance_prediction_to_best_candidate"] = prediction_distance
                    frame_diagnostic_entry["best_candidate_confidence"] = float(best_candidate["confidence"])
                    frame_diagnostic_entry["best_candidate_bbox_xywh"] = detection_box_to_xywh(
                        best_candidate["box"]
                    )
                    frame_diagnostic_entry["best_candidate_bbox_area"] = float(
                        detection_box_area(best_candidate["box"])
                    )
                    frame_diagnostic_entry["discarded_by_confidence"] = bool(discarded_by_confidence)
                    frame_diagnostic_entry["discarded_by_nms"] = bool(discarded_by_nms)
                    frame_diagnostic_entry["discarded_by_roi"] = bool(discarded_by_roi)
                    frame_diagnostic_entry["discarded_by_other_filter"] = other_filter_name
                ball_diagnostics_state["frames_without_ball"].append(frame_diagnostic_entry)

        player_detections, suppressed = suppress_duplicate_person_detections(player_detections)
        duplicate_player_detections_suppressed += suppressed
        goalkeeper_detections, _ = suppress_duplicate_person_detections(goalkeeper_detections)
        referee_detections, _ = suppress_duplicate_person_detections(referee_detections)

        accepted_player_detections = list(player_detections)
        accepted_goalkeeper_detections = list(goalkeeper_detections)
        accepted_referee_detections = list(referee_detections)
        rejected_player_detections = []

        team_classifier_predict_stage_start = time.perf_counter()
        if accepted_player_detections:
            team_classifier_cache_metrics["predict_calls_before_cache"] += 1

        detections_needing_prediction = []
        for idx, detection in enumerate(accepted_player_detections):
            track_id = int(detection.get("track_id", -1))
            if track_id >= 0 and not team_track_lock.needs_observation(
                track_id,
                current_frame_number,
                TEAM_CLASSIFIER_REFRESH_INTERVAL,
            ):
                detection["team_id"] = int(team_track_lock.current_team(track_id))
                team_classifier_cache_metrics["cache_hits"] += 1
                continue
            verifying = track_id >= 0 and team_track_lock.is_locked(track_id)
            detections_needing_prediction.append((idx, verifying))
            team_classifier_cache_metrics["cache_misses"] += 1

        if detections_needing_prediction:
            team_training_stats["predict_called_count"] += 1
            team_classifier_cache_metrics["predict_calls_after_cache"] += 1
            detections_to_predict = [
                accepted_player_detections[idx] for idx, _verifying in detections_needing_prediction
            ]
            prediction_crops = get_player_crops_from_payload(frame, detections_to_predict)
            predicted_team_ids = team_classifier.predict(prediction_crops) if prediction_crops else np.array([])
            for prediction_idx, (detection_idx, verifying) in enumerate(detections_needing_prediction):
                predicted_team_id = (
                    int(predicted_team_ids[prediction_idx])
                    if prediction_idx < len(predicted_team_ids)
                    else 0
                )
                detection = accepted_player_detections[detection_idx]
                track_id = int(detection.get("track_id", -1))
                if track_id >= 0:
                    detection["team_id"] = team_track_lock.assign(
                        track_id,
                        predicted_team_id,
                        current_frame_number,
                        verifying,
                    )
                else:
                    detection["team_id"] = predicted_team_id

        players_team_id_list = []
        for detection in accepted_player_detections:
            team_id = int(detection["team_id"]) if detection.get("team_id") is not None else 0
            detection["team_id"] = team_id
            players_team_id_list.append(team_id)
            if team_id in team_prediction_counts:
                team_prediction_counts[team_id] += 1
        players_team_id = np.array(players_team_id_list, dtype=np.int32)
        record_profile_stage(
            profile_state,
            "team_classifier_predict",
            time.perf_counter() - team_classifier_predict_stage_start,
        )
        if accepted_player_detections and len(team_classification_frame_examples) < 3:
            team_classification_frame_examples.append(
                {
                    "frame": frames_processed + 1,
                    "players": [
                        {
                            "track_id": int(detection.get("track_id", -1)),
                            "team_id": int(detection.get("team_id", 0)),
                        }
                        for detection in accepted_player_detections[:5]
                    ],
                }
            )
        goalkeeper_assignment_stage_start = time.perf_counter()
        player_team_ids_before_goalkeeper_assignment = [
            int(detection.get("team_id", 0)) for detection in accepted_player_detections
        ]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            accepted_player_detections,
            players_team_id,
            accepted_goalkeeper_detections,
        )
        for idx, detection in enumerate(accepted_goalkeeper_detections):
            goalkeeper_team_id = None
            if idx < len(goalkeepers_team_id):
                predicted_team_id = int(goalkeepers_team_id[idx])
                if predicted_team_id in goalkeeper_team_prediction_counts:
                    goalkeeper_team_id = predicted_team_id
                    goalkeeper_team_prediction_counts[goalkeeper_team_id] += 1
                    goalkeepers_with_team_id += 1
            detection["team_id"] = goalkeeper_team_id
        if accepted_goalkeeper_detections and len(goalkeeper_assignment_frame_examples) < 3:
            goalkeeper_assignment_frame_examples.append(
                {
                    "frame": frames_processed + 1,
                    "goalkeepers": [
                        {
                            "track_id": int(detection.get("track_id", -1)),
                            "team_id": detection.get("team_id"),
                        }
                        for detection in accepted_goalkeeper_detections
                    ],
                }
            )
        player_team_ids_after_goalkeeper_assignment = [
            int(detection.get("team_id", 0)) for detection in accepted_player_detections
        ]
        if player_team_ids_before_goalkeeper_assignment != player_team_ids_after_goalkeeper_assignment:
            player_team_mutation_violations += 1
        record_profile_stage(
            profile_state,
            "goalkeeper_assignment",
            time.perf_counter() - goalkeeper_assignment_stage_start,
        )

        homography_stage_start = time.perf_counter()
        assign_field_coordinates_to_detections(
            accepted_player_detections,
            view_transformer,
            detection_foot_point,
        )
        assign_field_coordinates_to_detections(
            accepted_goalkeeper_detections,
            view_transformer,
            detection_foot_point,
        )
        assign_field_coordinates_to_detections(
            accepted_referee_detections,
            view_transformer,
            detection_foot_point,
        )
        assign_field_coordinates_to_detections(
            ball_detections,
            view_transformer,
            detection_center,
        )

        if len(homography_examples) < 8:
            for class_name, detections in (
                ("Player", accepted_player_detections),
                ("Goalkeeper", accepted_goalkeeper_detections),
                ("Referee", accepted_referee_detections),
                ("Ball", ball_detections),
            ):
                for detection in detections:
                    if detection.get("field_x") is None or detection.get("field_y") is None:
                        continue
                    homography_examples.append(
                        {
                            "frame": frames_processed + 1,
                            "class": class_name,
                            "track_id": detection.get("track_id"),
                            "field_x": float(detection["field_x"]),
                            "field_y": float(detection["field_y"]),
                        }
                    )
                    if len(homography_examples) >= 8:
                        break
                if len(homography_examples) >= 8:
                    break
        ball_heatmap.observe(ball_detections)
        record_profile_stage(
            profile_state,
            "homography",
            time.perf_counter() - homography_stage_start,
        )

        remembered_ball = None
        if ball_last_valid_detection is not None:
            remembered_ball = dict(ball_last_valid_detection)
            if remembered_ball.get("field_x") is None or remembered_ball.get("field_y") is None:
                remembered_center = detection_center_np(remembered_ball["box"])
                remembered_field = transform_image_point_to_field(view_transformer, remembered_center)
                if remembered_field is not None:
                    remembered_ball["field_x"] = float(remembered_field[0])
                    remembered_ball["field_y"] = float(remembered_field[1])

        possession_players = build_possession_players(
            list(accepted_player_detections) + list(accepted_goalkeeper_detections)
        )
        possession_ball = build_ball_state(
            observed_ball=ball_detections[0] if ball_detections else None,
            remembered_ball=remembered_ball,
            predicted_field=predicted_field_center,
            predicted_pixel=predicted_center_for_search,
            internal_alive=internal_ball_track_alive,
        )
        possession_snapshot = possession_engine.update(possession_players, possession_ball)
        possession_events = possession_snapshot.as_dict()
        possession_changed = possession_events != last_possession_events
        last_possession_events = possession_events
        pass_detection_stage_start = time.perf_counter()
        pass_metrics_live = pass_detector.update(
            frame_number=current_frame_number,
            fps=fps,
            snapshot=possession_snapshot,
            players=possession_players,
            ball=possession_ball,
        )
        possession_by_team_live = possession_team_stats.observe(
            possession_snapshot.current_possession_team,
            pass_metrics_live["events"],
        )
        record_profile_stage(
            profile_state,
            "pass_detection",
            time.perf_counter() - pass_detection_stage_start,
        )

        discarded_outside_field += len(rejected_player_detections)
        filtered_player_detections += len(accepted_player_detections)

        output_frame = frame.copy()
        debug_frame = output_frame.copy() if args.debug_frame_dir else None

        draw_pitch_keypoints(output_frame, pitch_keypoints)
        if debug_frame is not None:
            draw_pitch_keypoints(debug_frame, pitch_keypoints)

        render_bounding_boxes_stage_start = time.perf_counter()
        for detection in accepted_player_detections:
            confidence = detection["confidence"]
            confidence_sum["Player"] += confidence
            confidence_count["Player"] += 1
            draw_detection(
                output_frame,
                detection["box"],
                ("Player", detection.get("track_id"), detection.get("team_id")),
                confidence,
            )
            if debug_frame is not None:
                draw_detection(
                    debug_frame,
                    detection["box"],
                    ("Player", detection.get("track_id"), detection.get("team_id")),
                    confidence,
                )

        for detection in accepted_goalkeeper_detections:
            draw_detection(
                output_frame,
                detection["box"],
                ("Goalkeeper", detection.get("track_id"), detection.get("team_id")),
                detection["confidence"],
            )
            if debug_frame is not None:
                draw_detection(
                    debug_frame,
                    detection["box"],
                    ("Goalkeeper", detection.get("track_id"), detection.get("team_id")),
                    detection["confidence"],
                )

        for detection in accepted_referee_detections:
            draw_detection(
                output_frame,
                detection["box"],
                ("Referee", detection.get("track_id")),
                detection["confidence"],
            )
            if debug_frame is not None:
                draw_detection(
                    debug_frame,
                    detection["box"],
                    ("Referee", detection.get("track_id")),
                    detection["confidence"],
                )

        if debug_frame is not None:
            for detection in rejected_player_detections:
                draw_detection(debug_frame, detection["box"], "Rejected", detection["confidence"], (0, 0, 255))

        for detection in ball_detections:
            confidence = detection["confidence"]
            ball_detected = True
            confidence_sum["Ball"] += confidence
            confidence_count["Ball"] += 1
            draw_detection(output_frame, detection["box"], "Ball", confidence)
            if debug_frame is not None:
                draw_detection(debug_frame, detection["box"], "Ball", confidence)
        record_profile_stage(
            profile_state,
            "render_bounding_boxes",
            time.perf_counter() - render_bounding_boxes_stage_start,
        )

        frame_players = len(accepted_player_detections)
        frame_goalkeepers = len(accepted_goalkeeper_detections)
        frame_ball = bool(ball_detections)
        total_player_detections += frame_players
        total_goalkeeper_detections += frame_goalkeepers
        total_ball_detections += len(ball_detections)

        frames_processed += 1
        max_players_detected = max(max_players_detected, frame_players)
        video_writer_stage_start = time.perf_counter()
        writer.write(output_frame)
        record_profile_stage(
            profile_state,
            "video_writer",
            time.perf_counter() - video_writer_stage_start,
        )

        if total_frames:
            progress = int(min(99, max(1, math.floor((frames_processed / total_frames) * 99))))
        else:
            progress = min(99, 1 + frames_processed % 98)

        should_emit = progress != last_emitted_progress and (
            frames_processed == 1 or progress % 5 == 0 or frames_processed == total_frames
        )
        should_emit_debug_frame = should_write_live_debug_frame(
            args.debug_frame_dir, frames_processed, args.debug_frame_interval
        )
        if should_emit or should_emit_debug_frame or possession_changed:
            last_emitted_progress = progress
            payload = {
                "type": "progress",
                "progress": progress,
                "message": f"YOLO processou {frames_processed} frame(s).",
                "framesProcessed": frames_processed,
                "playersDetected": max_players_detected,
                "ballDetected": ball_detected or frame_ball,
                "events": possession_events,
                "passes": {
                    "total_passes": int(pass_metrics_live["total_passes"]),
                    "events": pass_metrics_live["events"],
                    "turnovers": int(pass_metrics_live.get("turnovers", 0)),
                },
                "possession_by_team": possession_by_team_live,
                "ball_heatmap": ball_heatmap.as_dict(),
                "playerDetections": [
                    {
                        "track_id": int(detection.get("track_id", -1)),
                        "team_id": int(detection.get("team_id", 0)),
                        "confidence": float(detection["confidence"]),
                        "bounding_box": detection["bounding_box"],
                        "field_x": detection.get("field_x"),
                        "field_y": detection.get("field_y"),
                    }
                    for detection in accepted_player_detections
                ],
                "goalkeeperDetections": [
                    {
                        "track_id": int(detection.get("track_id", -1)),
                        "team_id": (
                            int(detection["team_id"])
                            if detection.get("team_id") is not None
                            else None
                        ),
                        "confidence": float(detection["confidence"]),
                        "bounding_box": detection["bounding_box"],
                        "field_x": detection.get("field_x"),
                        "field_y": detection.get("field_y"),
                    }
                    for detection in accepted_goalkeeper_detections
                ],
                "refereeDetections": [
                    {
                        "track_id": int(detection.get("track_id", -1)),
                        "confidence": float(detection["confidence"]),
                        "bounding_box": detection["bounding_box"],
                        "field_x": detection.get("field_x"),
                        "field_y": detection.get("field_y"),
                    }
                    for detection in accepted_referee_detections
                ],
                "ballDetections": [
                    {
                        "confidence": float(detection["confidence"]),
                        "bounding_box": detection["bounding_box"],
                        "field_x": detection.get("field_x"),
                        "field_y": detection.get("field_y"),
                    }
                    for detection in ball_detections
                ],
            }
            if should_emit_debug_frame:
                live_frame_update_stage_start = time.perf_counter()
                debug_frame_written = write_live_debug_frame(args.debug_frame_dir, debug_frame)
                debug_frame_jpeg = encode_live_debug_frame(debug_frame)
                if debug_frame_written:
                    log_live_frame_debug(
                        f"debugFrameUpdated sent to Node framesProcessed={frames_processed} debugFrameDir={args.debug_frame_dir}"
                    )
                if debug_frame_written or debug_frame_jpeg:
                    payload["debugFrameUpdated"] = True
                if debug_frame_jpeg:
                    payload["debugFrameJpeg"] = debug_frame_jpeg
                record_profile_stage(
                    profile_state,
                    "live_frame_update",
                    time.perf_counter() - live_frame_update_stage_start,
                )
            emit(payload)

        frame_total_elapsed = time.perf_counter() - frame_start_time
        profile_state["frame_total_seconds"].append(frame_total_elapsed)
        record_profile_stage(profile_state, "frame_total", frame_total_elapsed)

    capture.release()
    writer.release()
    pass_metrics = pass_detector.finalize()
    finalize_browser_video(raw_output_path, output_path)
    analysis_total_seconds = time.perf_counter() - analysis_start_time
    profile_report = build_performance_profile_report(
        profile_state=profile_state,
        analysis_total_seconds=analysis_total_seconds,
        frames_processed=frames_processed,
        width=width,
        height=height,
        input_path=input_path,
        output_path=output_path,
        hardware_info=hardware_info,
        team_classifier_cache_metrics=team_classifier_cache_metrics,
        ball_tracking_metrics=ball_tracking_metrics,
        physical_validation_enabled=physical_ball_validation_enabled,
        physical_accept_distance=configured_physical_accept_distance,
        physical_verify_distance=configured_physical_verify_distance,
        physical_max_distance=configured_physical_max_distance,
        physical_consecutive_confirmations=configured_physical_consecutive_confirmations,
        physical_reset_after_missing=configured_physical_reset_after_missing,
        physical_high_speed_threshold=configured_physical_high_speed_threshold,
    )
    profile_report["pass_detection"] = {
        "total_passes": int(pass_metrics["total_passes"]),
        "raw_candidate_passes": int(pass_metrics["raw_candidate_passes"]),
        "discarded_short_passes": int(pass_metrics["discarded_short_passes"]),
        "discarded_long_passes": int(pass_metrics["discarded_long_passes"]),
        "discarded_unstable_possession": int(pass_metrics["discarded_unstable_possession"]),
        "discarded_timeout": int(pass_metrics["discarded_timeout"]),
        "discarded_direction_check": int(pass_metrics["discarded_direction_check"]),
        "discarded_intercepted": int(pass_metrics.get("discarded_intercepted", 0)),
        "discarded_implausible_speed": int(pass_metrics.get("discarded_implausible_speed", 0)),
        "extended_window_passes": int(pass_metrics.get("extended_window_passes", 0)),
        "possession_gap_passes": int(pass_metrics.get("possession_gap_passes", 0)),
        "turnovers": int(pass_metrics.get("turnovers", 0)),
        "disputes": int(pass_metrics.get("disputes", 0)),
        "conductions": int(pass_metrics.get("conductions", 0)),
        "unconfirmed_candidates": int(pass_metrics.get("unconfirmed_candidates", 0)),
        "pass_score_threshold": int(pass_metrics.get("pass_score_threshold", 0)),
    }
    write_performance_profile_report(profile_report_path, profile_report)
    ball_diagnostics_report = None
    if ball_diagnostics_state["enabled"]:
        ball_diagnostics_summary = build_ball_diagnostics_summary(ball_diagnostics_state)
        ball_diagnostics_report = {
            "input_video_path": str(input_path),
            "output_video_path": str(output_path),
            "frames_processed": int(frames_processed),
            "detector_used": ball_diagnostics_state["detector_backend"],
            "detector_model_path": ball_diagnostics_state["detector_model_path"],
            "detector_inference_confidence": float(ball_diagnostics_state["inference_confidence"]),
            "nms_iou_threshold": 0.1,
            "roi_radius": {
                "min_px": BALL_ROI_MIN_RADIUS_PX,
                "scale_from_box": BALL_ROI_SCALE_FROM_BOX,
            },
            "confidence_gate": {
                "near_distance_px": BALL_PREDICTION_NEAR_DISTANCE_PX,
                "near_min_confidence": BALL_CONFIDENCE_NEAR_MIN,
                "far_min_confidence": BALL_CONFIDENCE_FAR_MIN,
            },
            "summary": ball_diagnostics_summary,
            "frames_without_ball": ball_diagnostics_state["frames_without_ball"],
            "final_ball_timeline": ball_diagnostics_state["final_ball_timeline"],
        }
        write_ball_diagnostics_report(ball_diagnostics_output_path, ball_diagnostics_report)

    avg_player_confidence = (
        confidence_sum["Player"] / confidence_count["Player"] if confidence_count["Player"] else 0
    )
    avg_ball_confidence = (
        confidence_sum["Ball"] / confidence_count["Ball"] if confidence_count["Ball"] else 0
    )
    avg_pitch_keypoints_detected = (
        total_pitch_keypoints_detected / frames_processed if frames_processed else 0
    )

    result = {
        "version": "yolo-1.0",
        "provider": "YoloAnalysisProvider",
        "model": str(player_model_path),
        "architecture": {
            "yolo": "implemented",
            "tracking": "implemented (sv.ByteTrack)",
            "feature_extraction": "implemented (TeamClassifier/SigLIP)",
            "machine_learning": "pending",
        },
        "metrics": {
            "frames_processed": frames_processed,
            "players_detected": max_players_detected,
            "ball_detected": ball_detected,
            "total_player_detections": total_player_detections,
            "total_goalkeeper_detections": total_goalkeeper_detections,
            "raw_player_detections": raw_player_detections,
            "filtered_player_detections": filtered_player_detections,
            "discarded_outside_field": discarded_outside_field,
            "duplicate_player_detections_suppressed": duplicate_player_detections_suppressed,
            "rejected_non_person": rejected_non_person,
            "rejected_non_person_reasons": rejected_non_person_reasons,
            "total_ball_detections": total_ball_detections,
            "field_mask_detected_frames": field_mask_detected_frames,
            "avg_player_confidence": avg_player_confidence,
            "avg_ball_confidence": avg_ball_confidence,
        },
        "detections": {
            "classes": ["Player", "Goalkeeper", "Referee", "Ball"],
            "class_id_map": DETECTION_LABEL_BY_CLASS_ID,
            "player_detection_model": str(player_model_path),
            "ball_detector_backend": ball_detector_backend,
            "ball_detector_model": ball_detector_model_path,
            "confidence_threshold": args.conf,
            "player_duplicate_iou_threshold": PERSON_DUPLICATE_IOU_THRESHOLD,
            "field_filter": "player/goalkeeper/referee bbox bottom-center foot point must be inside white-line-guided playable-field polygon",
            "labels": "class name and confidence are burned into the annotated video",
            "field_coordinates": "field_x and field_y are homography-transformed pitch coordinates for player/goalkeeper/referee/ball",
        },
        "tracking": {
            "status": "implemented",
            "method": "sv.ByteTrack",
            "bytetrack_parameters": {
                "minimum_consecutive_frames": BYTE_TRACK_MIN_CONSECUTIVE_FRAMES,
                "lost_track_buffer": BYTE_TRACK_LOST_TRACK_BUFFER,
                "minimum_matching_threshold": BYTE_TRACK_MINIMUM_MATCHING_THRESHOLD,
                "frame_rate": float(fps),
            },
            "id_storage": "track_id is stored per detection dict for player/goalkeeper/referee and rendered in labels",
            "team_classifier": "TeamClassifier predicts team_id (0/1) for players based on crops",
            "stability": {
                "recovered_track_gap_events": recovered_track_gap_events,
                "recovered_track_ids_count": len(recovered_track_ids),
                "recovered_track_examples": recovered_track_examples,
                "player_referee_class_swaps_before_history": player_referee_raw_swap_events,
                "player_referee_class_swaps_after_history": person_class_lock.metrics()["class_changes_after_lock"],
                "player_referee_history_window": PLAYER_REFEREE_HISTORY_WINDOW,
                "person_class_lock": person_class_lock.metrics(),
                "rejected_non_person": rejected_non_person,
                "rejected_non_person_reasons": rejected_non_person_reasons,
            },
        },
        "team_classifier": {
            "training": {
                "architecture": "SigLIP + UMAP + KMeans",
                "stride": team_training_stats["stride"],
                "frames_scanned": team_training_stats["frames_scanned"],
                "player_detections_seen": team_training_stats["player_detections_seen"],
                "samples_used_for_fit": team_training_stats["crops_used_for_fit"],
                "crops_total": team_training_stats["crops_total"],
                "discarded_small": team_training_stats["discarded_small"],
                "discarded_empty": team_training_stats["discarded_empty"],
                "discarded_border": team_training_stats["discarded_border"],
                "discarded_blurry": team_training_stats["discarded_blurry"],
                "discarded_non_player": team_training_stats["discarded_non_player"],
                "discarded_low_chroma_official": team_training_stats["discarded_low_chroma_official"],
                "kept_low_chroma_official": team_training_stats["kept_low_chroma_official"],
                "discarded_total": total_discarded_training_crops,
                "training_completed_frame": team_training_stats["training_completed_frame"],
                "training_stop_reason": team_training_stats["training_stop_reason"],
            },
            "prediction_counts": {
                "team_0": team_prediction_counts[0],
                "team_1": team_prediction_counts[1],
            },
            "runtime": {
                "fit_called_count": team_training_stats["fit_called_count"],
                "predict_called_count": team_training_stats["predict_called_count"],
                "collecting_after_fit": False,
            },
            "frame_examples": team_classification_frame_examples,
            "color_anchor": team_classifier.color_anchor,
            "stability": team_track_lock.metrics(),
        },
        "goalkeeper_assignment": {
            "method": "official centroid proximity heuristic from roboflow/sports",
            "goalkeepers_detected": total_goalkeeper_detections,
            "goalkeepers_with_team_id": goalkeepers_with_team_id,
            "team_0": goalkeeper_team_prediction_counts[0],
            "team_1": goalkeeper_team_prediction_counts[1],
            "player_team_mutation_violations": player_team_mutation_violations,
            "frame_examples": goalkeeper_assignment_frame_examples,
        },
        "homography": {
            "status": "implemented (official ViewTransformer)",
            "valid_frames": homography_valid_frames,
            "total_frames": frames_processed,
            "avg_keypoints_detected": avg_pitch_keypoints_detected,
            "examples": homography_examples,
        },
        "features": {"status": "implemented (ball possession state machine + pass detector refinements)"},
        "machine_learning": {"status": "pending"},
        "events": possession_engine.snapshot().as_dict(),
        "possession": possession_engine.metrics(),
        "possession_by_team": possession_team_stats.as_dict(),
        "ball_heatmap": ball_heatmap.as_dict(),
        "passes": pass_metrics,
        "artifacts": {
            "processed_video_path": str(output_path),
            "ball_diagnostics_path": (
                str(ball_diagnostics_output_path)
                if ball_diagnostics_output_path is not None
                else None
            ),
        },
    }
    emit(
        {
            "type": "progress",
            "progress": 100,
            "message": "Análise visual YOLO concluída.",
            "framesProcessed": frames_processed,
            "playersDetected": max_players_detected,
            "ballDetected": ball_detected,
            "events": possession_engine.snapshot().as_dict(),
            "passes": {
                "total_passes": int(pass_metrics["total_passes"]),
                "events": pass_metrics["events"],
                "turnovers": int(pass_metrics.get("turnovers", 0)),
            },
            "possession_by_team": possession_team_stats.as_dict(),
            "ball_heatmap": ball_heatmap.as_dict(),
        }
    )
    emit({"type": "result", "result": result})


def parse_args():
    player_model_default = Path(__file__).resolve().parent / PLAYER_MODEL_RELATIVE_PATH
    pitch_model_default = Path(__file__).resolve().parent / PITCH_MODEL_RELATIVE_PATH
    ball_model_default = Path(__file__).resolve().parent / BALL_MODEL_RELATIVE_PATH
    parser = argparse.ArgumentParser(description="Analyze a soccer video with YOLOv8.")
    parser.add_argument("--input", required=True, help="Input MP4 path")
    parser.add_argument("--output", required=True, help="Annotated MP4 output path")
    parser.add_argument(
        "--model",
        default=str(player_model_default),
        help="Path to the official football player detector model (.pt)",
    )
    parser.add_argument(
        "--pitch-model",
        default=str(pitch_model_default),
        help="Path to the football pitch keypoint model (.pt)",
    )
    parser.add_argument(
        "--ball-model",
        default=str(ball_model_default),
        help="Path to the football ball detection model (.pt)",
    )
    parser.add_argument(
        "--ball-detector",
        choices=sorted(BALL_DETECTOR_BACKENDS),
        default=BALL_DETECTOR_BACKEND_DEFAULT,
        help="Ball detector backend: yolo (football-ball-detection.pt) or rfdetr (SoccerNet checkpoint)",
    )
    parser.add_argument(
        "--rfdetr-repo-id",
        default=RFDETR_SOCCERNET_REPO_ID_DEFAULT,
        help="Hugging Face repository id for RF-DETR SoccerNet checkpoint",
    )
    parser.add_argument(
        "--rfdetr-model-dir",
        default=str(RFDETR_SOCCERNET_MODEL_DIR),
        help="Local directory to store RF-DETR SoccerNet model files",
    )
    parser.add_argument(
        "--rfdetr-device",
        default="auto",
        help='RF-DETR runtime device: "auto", "cpu" or "cuda"',
    )
    parser.add_argument(
        "--predictive-ball-tracking",
        choices=["on", "off"],
        default=PREDICTIVE_BALL_TRACKING_DEFAULT,
        help="Enable predictive ball tracking with adaptive ROI search for RF-DETR",
    )
    parser.add_argument(
        "--hybrid-ball-search",
        choices=["on", "off"],
        default=HYBRID_BALL_SEARCH_DEFAULT,
        help="Use hybrid ball inference: global tiling bootstrap + smart crop tracking",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=TILE_SIZE,
        help="Tile size used during global tiling search",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=TILE_OVERLAP,
        help="Tile overlap ratio [0.0-0.9] used during global tiling search",
    )
    parser.add_argument(
        "--initial-detections-required",
        type=int,
        default=INITIAL_DETECTIONS_REQUIRED,
        help="Consecutive ball detections required to switch from global tiling to crop mode",
    )
    parser.add_argument(
        "--crop-initial-size",
        type=int,
        default=CROP_INITIAL_SIZE,
        help="Initial crop size around predicted ball center",
    )
    parser.add_argument(
        "--crop-max-size",
        type=int,
        default=CROP_MAX_SIZE,
        help="Maximum adaptive crop size around predicted ball center",
    )
    parser.add_argument(
        "--max-frames-before-global-search",
        type=int,
        default=MAX_FRAMES_BEFORE_GLOBAL_SEARCH,
        help="Consecutive misses before abandoning crop and returning to global tiling",
    )
    parser.add_argument(
        "--physical-ball-validation",
        choices=["on", "off"],
        default=PHYSICAL_BALL_VALIDATION_DEFAULT,
        help="Enable physical consistency validation between detector output and BallTracker updates",
    )
    parser.add_argument(
        "--physical-accept-distance",
        type=float,
        default=PHYSICAL_ACCEPT_DISTANCE,
        help="Pixel distance threshold for immediate acceptance of a detected ball",
    )
    parser.add_argument(
        "--physical-verify-distance",
        type=float,
        default=PHYSICAL_VERIFY_DISTANCE,
        help="Pixel distance upper bound for pending confirmation candidates",
    )
    parser.add_argument(
        "--physical-max-distance",
        type=float,
        default=PHYSICAL_MAX_DISTANCE,
        help="Pixel distance beyond which detections are rejected as implausible teleports",
    )
    parser.add_argument(
        "--physical-consecutive-confirmations",
        type=int,
        default=PHYSICAL_CONSECUTIVE_CONFIRMATIONS,
        help="Consecutive frames required to confirm medium-distance pending candidates",
    )
    parser.add_argument(
        "--physical-reset-after-missing",
        type=int,
        default=PHYSICAL_RESET_AFTER_MISSING,
        help="Missing-frame threshold to reset physical validation and allow distant reacquisition",
    )
    parser.add_argument(
        "--physical-high-speed-threshold",
        type=float,
        default=PHYSICAL_HIGH_SPEED_THRESHOLD,
        help="Predicted speed threshold (px/frame) that allows distant detections to be accepted",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument(
        "--pitch-keypoint-conf",
        type=float,
        default=PITCH_KEYPOINT_MIN_CONFIDENCE,
        help="Confidence threshold for football pitch keypoints",
    )
    parser.add_argument("--debug-frame-dir", default=None, help="Directory for opt-in live debug frames")
    parser.add_argument(
        "--debug-frame-interval",
        type=int,
        default=1,
        help="Write one live debug frame every N processed frames when debug mode is enabled",
    )
    parser.add_argument(
        "--ball-diagnostics-output",
        default=None,
        help="Optional JSON path to write frame-level ball detector diagnostics",
    )
    parser.add_argument(
        "--ball-diagnostics-min-confidence",
        type=float,
        default=BALL_DIAGNOSTIC_MIN_CONFIDENCE,
        help="Detector confidence used during diagnostic runs (single inference, no extra model)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        analyze(parse_args())
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        sys.exit(1)
