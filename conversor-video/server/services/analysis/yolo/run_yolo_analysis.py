#!/usr/bin/env python3
"""Run YOLOv8 detections over a soccer video and write an annotated MP4."""

import argparse
import base64
import json
import math
import os
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

from roboflow_ball import BallTracker


BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
PERSON_DUPLICATE_IOU_THRESHOLD = 0.85
MIN_FIELD_AREA_RATIO = 0.10
MIN_WHITE_LINE_PIXELS = 80
FIELD_LINE_OVERLAP_DILATION = 31
FIELD_BOUNDARY_LINE_MIN_GRASS_DELTA = 0.35
FIELD_BOUNDARY_LINE_SAMPLE_OFFSET = 24
LIVE_DEBUG_FRAME_MAX_WIDTH = 960
LIVE_DEBUG_FRAME_JPEG_QUALITY = 72
LIVE_DEBUG_FRAME_NAME = "live-frame.jpg"
LIVE_DEBUG_FRAME_REPLACE_RETRIES = 6
LIVE_DEBUG_FRAME_RETRY_BASE_DELAY_SECONDS = 0.03
PLAYER_MODEL_RELATIVE_PATH = Path("models/football-player-detection.pt")
PITCH_MODEL_RELATIVE_PATH = Path("models/football-pitch-detection.pt")
BALL_MODEL_RELATIVE_PATH = Path("models/football-ball-detection.pt")
PITCH_KEYPOINT_MIN_CONFIDENCE = 0.25
BALL_TRACKER_BUFFER_SIZE = 20
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


def is_detection_inside_field(detection, field_mask):
    foot_x, foot_y = detection_foot_point(detection["box"])
    foot_x = clamp(foot_x, 0, field_mask.shape[1] - 1)
    foot_y = clamp(foot_y, 0, field_mask.shape[0] - 1)
    return field_mask[foot_y, foot_x] > 0


def draw_field_debug_overlay(frame, field_contours):
    if field_contours:
        cv2.drawContours(frame, field_contours, -1, (0, 255, 255), 3)


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


def ball_detections_to_dicts(detections):
    if len(detections) == 0:
        return []

    confidence_values = detections.confidence
    if confidence_values is None:
        confidence_values = np.zeros(len(detections), dtype=np.float32)

    return [
        {
            "box": detections.xyxy[index].tolist(),
            "confidence": float(confidence_values[index]),
            "source": "ball_model",
        }
        for index in range(len(detections))
    ]


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

    color = color or CLASS_COLOR_OVERRIDES.get(class_name, (0, 179, 255))
    label = f"{class_name} - {round(confidence * 100)}%"
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


def analyze(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {input_path}")

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
    ball_model_path = resolve_ball_model_path(args.ball_model)
    if not ball_model_path.exists():
        raise FileNotFoundError(f"Modelo de detecção da bola não encontrado: {ball_model_path}")
    ball_model = YOLO(str(ball_model_path))
    ball_tracker = BallTracker(buffer_size=BALL_TRACKER_BUFFER_SIZE)

    def ball_slicer_callback(image_slice):
        ball_slice_result = ball_model(image_slice, imgsz=640, verbose=False)[0]
        return extract_ball_detections(ball_slice_result)

    # NOTE: InferenceSlicer wiring is intentionally kept in place (callback preserved)
    # for easy reactivation, but temporarily disabled from the runtime flow.
    ball_slicer = None
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo para análise.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
    total_ball_detections = 0
    duplicate_player_detections_suppressed = 0
    filtered_player_detections = 0
    discarded_outside_field = 0
    field_mask_detected_frames = 0
    previous_field_mask = None
    ball_detected = False
    confidence_sum = {"Player": 0.0, "Ball": 0.0}
    confidence_count = {"Player": 0, "Ball": 0}
    last_emitted_progress = -1

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        field_mask, field_contours, field_detected = detect_field_region(frame, previous_field_mask)
        previous_field_mask = field_mask
        if field_detected:
            field_mask_detected_frames += 1

        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            classes=[GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, REFEREE_CLASS_ID],
        )[0]
        pitch_result = pitch_model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.pitch_keypoint_conf,
            verbose=False,
        )[0]
        pitch_keypoints = extract_pitch_keypoints(pitch_result, args.pitch_keypoint_conf)
        ball_result = ball_model(frame, imgsz=640, verbose=False)[0]
        dedicated_ball_detections = extract_ball_detections(ball_result).with_nms(threshold=0.1)
        dedicated_ball_detections = ball_tracker.update(dedicated_ball_detections)

        player_detections = []
        goalkeeper_detections = []
        referee_detections = []
        ball_detections = []

        if result.boxes is not None:
            for detected_box in result.boxes:
                class_id = int(detected_box.cls[0])
                confidence = float(detected_box.conf[0])
                box = detected_box.xyxy[0].tolist()

                if class_id == PLAYER_CLASS_ID:
                    raw_player_detections += 1
                    player_detections.append({"box": box, "confidence": confidence})

                elif class_id == GOALKEEPER_CLASS_ID:
                    goalkeeper_detections.append({"box": box, "confidence": confidence})
                elif class_id == REFEREE_CLASS_ID:
                    referee_detections.append({"box": box, "confidence": confidence})

        ball_detections = ball_detections_to_dicts(dedicated_ball_detections)

        player_detections, suppressed = suppress_duplicate_person_detections(player_detections)
        duplicate_player_detections_suppressed += suppressed
        goalkeeper_detections, _ = suppress_duplicate_person_detections(goalkeeper_detections)
        referee_detections, _ = suppress_duplicate_person_detections(referee_detections)

        accepted_player_detections = []
        accepted_goalkeeper_detections = []
        accepted_referee_detections = []
        rejected_player_detections = []
        for detection in player_detections:
            if is_detection_inside_field(detection, field_mask):
                accepted_player_detections.append(detection)
            else:
                rejected_player_detections.append(detection)
        for detection in goalkeeper_detections:
            if is_detection_inside_field(detection, field_mask):
                accepted_goalkeeper_detections.append(detection)
            else:
                rejected_player_detections.append(detection)
        for detection in referee_detections:
            if is_detection_inside_field(detection, field_mask):
                accepted_referee_detections.append(detection)
            else:
                rejected_player_detections.append(detection)

        discarded_outside_field += len(rejected_player_detections)
        filtered_player_detections += len(accepted_player_detections)

        output_frame = frame.copy()
        debug_frame = output_frame.copy() if args.debug_frame_dir else None

        draw_pitch_keypoints(output_frame, pitch_keypoints)
        if debug_frame is not None:
            draw_pitch_keypoints(debug_frame, pitch_keypoints)

        draw_field_debug_overlay(output_frame, field_contours)
        if debug_frame is not None:
            draw_field_debug_overlay(debug_frame, field_contours)

        for detection in accepted_player_detections:
            confidence = detection["confidence"]
            confidence_sum["Player"] += confidence
            confidence_count["Player"] += 1
            draw_detection(output_frame, detection["box"], "Player", confidence)
            if debug_frame is not None:
                draw_detection(debug_frame, detection["box"], "Player", confidence)

        for detection in accepted_goalkeeper_detections:
            draw_detection(output_frame, detection["box"], "Goalkeeper", detection["confidence"])
            if debug_frame is not None:
                draw_detection(debug_frame, detection["box"], "Goalkeeper", detection["confidence"])

        for detection in accepted_referee_detections:
            draw_detection(output_frame, detection["box"], "Referee", detection["confidence"])
            if debug_frame is not None:
                draw_detection(debug_frame, detection["box"], "Referee", detection["confidence"])

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

        frame_players = len(accepted_player_detections)
        frame_ball = bool(ball_detections)
        total_player_detections += frame_players
        total_ball_detections += len(ball_detections)

        frames_processed += 1
        max_players_detected = max(max_players_detected, frame_players)
        writer.write(output_frame)

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
        if should_emit or should_emit_debug_frame:
            last_emitted_progress = progress
            payload = {
                "type": "progress",
                "progress": progress,
                "message": f"YOLO processou {frames_processed} frame(s).",
                "framesProcessed": frames_processed,
                "playersDetected": max_players_detected,
                "ballDetected": ball_detected or frame_ball,
            }
            if should_emit_debug_frame:
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
            emit(payload)

    capture.release()
    writer.release()
    finalize_browser_video(raw_output_path, output_path)

    avg_player_confidence = (
        confidence_sum["Player"] / confidence_count["Player"] if confidence_count["Player"] else 0
    )
    avg_ball_confidence = (
        confidence_sum["Ball"] / confidence_count["Ball"] if confidence_count["Ball"] else 0
    )

    result = {
        "version": "yolo-1.0",
        "provider": "YoloAnalysisProvider",
        "model": str(player_model_path),
        "architecture": {
            "yolo": "implemented",
            "tracking": "pending",
            "feature_extraction": "pending",
            "machine_learning": "pending",
        },
        "metrics": {
            "frames_processed": frames_processed,
            "players_detected": max_players_detected,
            "ball_detected": ball_detected,
            "total_player_detections": total_player_detections,
            "raw_player_detections": raw_player_detections,
            "filtered_player_detections": filtered_player_detections,
            "discarded_outside_field": discarded_outside_field,
            "duplicate_player_detections_suppressed": duplicate_player_detections_suppressed,
            "total_ball_detections": total_ball_detections,
            "field_mask_detected_frames": field_mask_detected_frames,
            "avg_player_confidence": avg_player_confidence,
            "avg_ball_confidence": avg_ball_confidence,
        },
        "detections": {
            "classes": ["Player", "Goalkeeper", "Referee", "Ball"],
            "class_id_map": DETECTION_LABEL_BY_CLASS_ID,
            "player_detection_model": str(player_model_path),
            "confidence_threshold": args.conf,
            "player_duplicate_iou_threshold": PERSON_DUPLICATE_IOU_THRESHOLD,
            "field_filter": "player/goalkeeper/referee bbox bottom-center foot point must be inside white-line-guided playable-field polygon",
            "labels": "class name and confidence are burned into the annotated video",
        },
        "tracking": {"status": "pending"},
        "features": {"status": "pending"},
        "machine_learning": {"status": "pending"},
        "artifacts": {
            "processed_video_path": str(output_path),
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
    return parser.parse_args()


if __name__ == "__main__":
    try:
        analyze(parse_args())
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        sys.exit(1)
