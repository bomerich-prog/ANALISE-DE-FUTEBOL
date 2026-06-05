#!/usr/bin/env python3
"""Run YOLOv8 detections over a soccer video and write an annotated MP4."""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


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


def draw_detection(frame, box, class_name, confidence):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = clamp(x1, 0, frame.shape[1] - 1)
    x2 = clamp(x2, 0, frame.shape[1] - 1)
    y1 = clamp(y1, 0, frame.shape[0] - 1)
    y2 = clamp(y2, 0, frame.shape[0] - 1)

    color = (0, 200, 83) if class_name == "Player" else (0, 179, 255)
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
    total_player_detections = 0
    total_ball_detections = 0
    ball_detected = False
    confidence_sum = {"Player": 0.0, "Ball": 0.0}
    confidence_count = {"Player": 0, "Ball": 0}
    last_emitted_progress = -1

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
        )[0]

        frame_players = 0
        frame_ball = False

        if result.boxes is not None:
            for detected_box in result.boxes:
                class_id = int(detected_box.cls[0])
                confidence = float(detected_box.conf[0])
                box = detected_box.xyxy[0].tolist()

                if class_id == PERSON_CLASS_ID:
                    class_name = "Player"
                    frame_players += 1
                    total_player_detections += 1
                elif class_id == SPORTS_BALL_CLASS_ID:
                    class_name = "Ball"
                    frame_ball = True
                    ball_detected = True
                    total_ball_detections += 1
                else:
                    continue

                confidence_sum[class_name] += confidence
                confidence_count[class_name] += 1
                draw_detection(frame, box, class_name, confidence)

        frames_processed += 1
        max_players_detected = max(max_players_detected, frame_players)
        writer.write(frame)

        if total_frames:
            progress = int(min(99, max(1, math.floor((frames_processed / total_frames) * 99))))
        else:
            progress = min(99, 1 + frames_processed % 98)

        should_emit = progress != last_emitted_progress and (
            frames_processed == 1 or progress % 5 == 0 or frames_processed == total_frames
        )
        if should_emit:
            last_emitted_progress = progress
            emit(
                {
                    "type": "progress",
                    "progress": progress,
                    "message": f"YOLO processou {frames_processed} frame(s).",
                    "framesProcessed": frames_processed,
                    "playersDetected": max_players_detected,
                    "ballDetected": ball_detected or frame_ball,
                }
            )

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
            "total_ball_detections": total_ball_detections,
            "avg_player_confidence": avg_player_confidence,
            "avg_ball_confidence": avg_ball_confidence,
        },
        "detections": {
            "classes": ["Player", "Ball"],
            "player_class": "person",
            "ball_class": "sports ball",
            "confidence_threshold": args.conf,
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
    parser = argparse.ArgumentParser(description="Analyze a soccer video with YOLOv8.")
    parser.add_argument("--input", required=True, help="Input MP4 path")
    parser.add_argument("--output", required=True, help="Annotated MP4 output path")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        analyze(parse_args())
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        sys.exit(1)
