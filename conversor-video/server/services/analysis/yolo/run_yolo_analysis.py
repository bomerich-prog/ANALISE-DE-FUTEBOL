#!/usr/bin/env python3
"""Run YOLOv8 detections over a soccer video and write an annotated MP4."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32
PERSON_DUPLICATE_IOU_THRESHOLD = 0.85
LIVE_DEBUG_FRAME_NAME = "live-frame.jpg"
MIN_FIELD_AREA_RATIO = 0.12
LIVE_DEBUG_FRAME_REPLACE_RETRIES = 6
LIVE_DEBUG_FRAME_RETRY_BASE_DELAY_SECONDS = 0.03


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


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


def detect_field_region(frame, previous_mask=None):
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
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
    )
    grass_mask = cv2.morphologyEx(
        grass_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    grass_mask = cv2.dilate(
        grass_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=1,
    )

    contours, _ = cv2.findContours(grass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = width * height * MIN_FIELD_AREA_RATIO
    large_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    if not large_contours:
        if previous_mask is not None:
            return previous_mask, [], False
        full_mask = np.full((height, width), 255, dtype=np.uint8)
        return full_mask, [], False

    merged_points = np.vstack(large_contours)
    hull = cv2.convexHull(merged_points)
    field_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(field_mask, hull, 255)

    field_mask = cv2.erode(
        field_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    field_contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return field_mask, field_contours, True


def detection_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def is_detection_inside_field(detection, field_mask):
    center_x, center_y = detection_center(detection["box"])
    center_x = clamp(center_x, 0, field_mask.shape[1] - 1)
    center_y = clamp(center_y, 0, field_mask.shape[0] - 1)
    return field_mask[center_y, center_x] > 0


def draw_field_debug_overlay(frame, field_contours):
    if field_contours:
        cv2.drawContours(frame, field_contours, -1, (0, 255, 255), 3)


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

    color = color or ((0, 200, 83) if class_name == "Player" else (0, 179, 255))
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


def write_live_debug_frame(debug_frame_dir, frame):
    if not debug_frame_dir:
        return False

    frame_dir = Path(debug_frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    final_path = frame_dir / LIVE_DEBUG_FRAME_NAME
    temp_path = frame_dir / f".{LIVE_DEBUG_FRAME_NAME}.tmp.jpg"
    if not cv2.imwrite(str(temp_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]):
        return False

    for attempt in range(LIVE_DEBUG_FRAME_REPLACE_RETRIES):
        try:
            os.replace(temp_path, final_path)
            return True
        except PermissionError:
            if attempt == LIVE_DEBUG_FRAME_REPLACE_RETRIES - 1:
                temp_path.unlink(missing_ok=True)
                return False
            time.sleep(LIVE_DEBUG_FRAME_RETRY_BASE_DELAY_SECONDS * (attempt + 1))

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

    model = YOLO(args.model)
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
            "message": "YOLOv8 carregado. Iniciando detecção de jogadores e bola...",
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
            classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
        )[0]

        player_detections = []
        ball_detections = []

        if result.boxes is not None:
            for detected_box in result.boxes:
                class_id = int(detected_box.cls[0])
                confidence = float(detected_box.conf[0])
                box = detected_box.xyxy[0].tolist()

                if class_id == PERSON_CLASS_ID:
                    raw_player_detections += 1
                    player_detections.append({"box": box, "confidence": confidence})
                elif class_id == SPORTS_BALL_CLASS_ID:
                    ball_detections.append({"box": box, "confidence": confidence})

        player_detections, suppressed = suppress_duplicate_person_detections(player_detections)
        duplicate_player_detections_suppressed += suppressed

        accepted_player_detections = []
        rejected_player_detections = []
        for detection in player_detections:
            if is_detection_inside_field(detection, field_mask):
                accepted_player_detections.append(detection)
            else:
                rejected_player_detections.append(detection)

        discarded_outside_field += len(rejected_player_detections)
        filtered_player_detections += len(accepted_player_detections)

        output_frame = frame.copy()
        debug_frame = output_frame.copy() if args.debug_frame_dir else None

        if debug_frame is not None:
            draw_field_debug_overlay(debug_frame, field_contours)

        for detection in accepted_player_detections:
            confidence = detection["confidence"]
            confidence_sum["Player"] += confidence
            confidence_count["Player"] += 1
            draw_detection(output_frame, detection["box"], "Player", confidence)
            if debug_frame is not None:
                draw_detection(debug_frame, detection["box"], "Player", confidence, (0, 200, 83))

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
            if should_emit_debug_frame and write_live_debug_frame(args.debug_frame_dir, debug_frame):
                payload["debugFrameUpdated"] = True
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
        "model": args.model,
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
            "classes": ["Player", "Ball"],
            "player_class": "person",
            "ball_class": "sports ball",
            "confidence_threshold": args.conf,
            "player_duplicate_iou_threshold": PERSON_DUPLICATE_IOU_THRESHOLD,
            "field_filter": "person bbox center must be inside detected grass mask",
            "labels": "class name and confidence are burned into the annotated video",
        },
        "tracking": {"status": "pending"},
        "features": {"status": "pending"},
        "machine_learning": {"status": "pending"},
        "artifacts": {
            "processed_video_path": str(output_path),
        },
    }
    if args.debug_frame_dir:
        result["artifacts"]["debug_live_frame_path"] = str(
            Path(args.debug_frame_dir) / LIVE_DEBUG_FRAME_NAME
        )

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
    parser = argparse.ArgumentParser(description="Analyze a soccer video with YOLOv8.")
    parser.add_argument("--input", required=True, help="Input MP4 path")
    parser.add_argument("--output", required=True, help="Annotated MP4 output path")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
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
