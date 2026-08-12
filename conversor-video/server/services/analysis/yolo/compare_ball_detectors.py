#!/usr/bin/env python3
"""Run YOLO and RF-DETR ball detector comparison on the same video."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    current_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare YOLO vs RF-DETR ball detectors.")
    parser.add_argument("--input", required=True, help="Input MP4 path")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison artifacts")
    parser.add_argument(
        "--comparison-output",
        default=None,
        help="Optional comparison JSON output path",
    )
    parser.add_argument(
        "--analysis-script",
        default=str(current_dir / "run_yolo_analysis.py"),
        help="Path to run_yolo_analysis.py",
    )
    parser.add_argument("--model", default=None, help="Player model path")
    parser.add_argument("--pitch-model", default=None, help="Pitch model path")
    parser.add_argument("--ball-model", default=None, help="YOLO ball model path")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--pitch-keypoint-conf",
        type=float,
        default=0.25,
        help="Pitch keypoint confidence threshold",
    )
    parser.add_argument(
        "--ball-diagnostics-min-confidence",
        type=float,
        default=0.25,
        help="Detector confidence used during comparison runs",
    )
    parser.add_argument(
        "--rfdetr-repo-id",
        default="julianzu9612/RFDETR-Soccernet",
        help="Hugging Face repo id for RF-DETR SoccerNet",
    )
    parser.add_argument(
        "--rfdetr-model-dir",
        default="models/rfdetr-soccernet",
        help="Local directory for RF-DETR model files",
    )
    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=0.1,
        help="IoU threshold used for cross-detector agreement",
    )
    return parser.parse_args()


def run_detector_analysis(args, detector_backend, output_dir):
    analysis_script = Path(args.analysis_script).resolve()
    output_video = output_dir / f"comparison_{detector_backend}.mp4"
    diagnostics_json = output_dir / f"comparison_{detector_backend}.ball_diagnostics.json"
    run_log = output_dir / f"comparison_{detector_backend}.log"

    command = [
        sys.executable,
        str(analysis_script),
        "--input",
        str(Path(args.input).resolve()),
        "--output",
        str(output_video),
        "--ball-detector",
        detector_backend,
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--pitch-keypoint-conf",
        str(args.pitch_keypoint_conf),
        "--ball-diagnostics-output",
        str(diagnostics_json),
        "--ball-diagnostics-min-confidence",
        str(args.ball_diagnostics_min_confidence),
        "--rfdetr-repo-id",
        str(args.rfdetr_repo_id),
        "--rfdetr-model-dir",
        str(args.rfdetr_model_dir),
    ]
    if args.model:
        command.extend(["--model", str(args.model)])
    if args.pitch_model:
        command.extend(["--pitch-model", str(args.pitch_model)])
    if args.ball_model:
        command.extend(["--ball-model", str(args.ball_model)])

    started_at = time.perf_counter()
    with run_log.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    elapsed_seconds = time.perf_counter() - started_at
    if completed.returncode != 0:
        raise RuntimeError(
            f"Detector {detector_backend} falhou. Verifique log: {run_log} (exit={completed.returncode})"
        )

    profile_path = output_video.with_suffix(f"{output_video.suffix}.profile.json")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    timeline = diagnostics.get("final_ball_timeline", [])
    detected_confidences = [
        float(entry["confidence"])
        for entry in timeline
        if entry.get("detected") and entry.get("confidence") is not None
    ]
    avg_confidence = statistics.mean(detected_confidences) if detected_confidences else 0.0

    return {
        "detector_used": diagnostics.get("detector_used", detector_backend),
        "detector_model_path": diagnostics.get("detector_model_path"),
        "output_video_path": str(output_video),
        "profile_path": str(profile_path),
        "diagnostics_path": str(diagnostics_json),
        "log_path": str(run_log),
        "frames_processed": int(profile["analysis"]["frames_processed"]),
        "tempo_total_analise_seconds": float(profile["analysis"]["total_analysis_seconds"]),
        "tempo_medio_por_frame_seconds": float(profile["analysis"]["avg_frame_seconds"]),
        "fps_medio": float(profile["analysis"]["avg_processing_fps"]),
        "frames_sem_bola": int(diagnostics["summary"]["missing_frames_count"]),
        "maior_sequencia_sem_bola": int(diagnostics["summary"]["max_consecutive_frames_without_ball"]),
        "confianca_media": float(avg_confidence),
        "timeline": timeline,
        "wall_clock_seconds": float(elapsed_seconds),
    }


def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def build_timeline_map(timeline):
    timeline_map = {}
    for entry in timeline:
        timeline_map[int(entry["frame_id"])] = entry
    return timeline_map


def compute_precision_recall_against_other(subject_timeline, reference_timeline, iou_threshold):
    subject_map = build_timeline_map(subject_timeline)
    reference_map = build_timeline_map(reference_timeline)
    frame_ids = sorted(set(subject_map.keys()) | set(reference_map.keys()))

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for frame_id in frame_ids:
        subject = subject_map.get(frame_id, {"detected": False})
        reference = reference_map.get(frame_id, {"detected": False})
        subject_detected = bool(subject.get("detected"))
        reference_detected = bool(reference.get("detected"))
        matched = False
        if subject_detected and reference_detected:
            subject_box = subject.get("bbox_xyxy")
            reference_box = reference.get("bbox_xyxy")
            if subject_box and reference_box:
                matched = iou_xyxy(subject_box, reference_box) >= iou_threshold

        if subject_detected and matched:
            true_positive += 1
        elif subject_detected and not matched:
            false_positive += 1

        if reference_detected and not matched:
            false_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(true_positive),
        "fp": int(false_positive),
        "fn": int(false_negative),
    }


def percent_change(new_value, old_value):
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100.0


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_output = (
        Path(args.comparison_output).resolve()
        if args.comparison_output
        else output_dir / "ball_detector_comparison.json"
    )

    yolo_report = run_detector_analysis(args, "yolo", output_dir)
    rfdetr_report = run_detector_analysis(args, "rfdetr", output_dir)

    yolo_match = compute_precision_recall_against_other(
        subject_timeline=yolo_report["timeline"],
        reference_timeline=rfdetr_report["timeline"],
        iou_threshold=args.match_iou_threshold,
    )
    rfdetr_match = compute_precision_recall_against_other(
        subject_timeline=rfdetr_report["timeline"],
        reference_timeline=yolo_report["timeline"],
        iou_threshold=args.match_iou_threshold,
    )

    yolo_report["precision"] = yolo_match["precision"]
    yolo_report["recall"] = yolo_match["recall"]
    yolo_report["agreement_counts"] = yolo_match
    rfdetr_report["precision"] = rfdetr_match["precision"]
    rfdetr_report["recall"] = rfdetr_match["recall"]
    rfdetr_report["agreement_counts"] = rfdetr_match

    comparison_percent = {
        "recall_rf_detr_vs_yolo_percent": percent_change(
            rfdetr_report["recall"],
            yolo_report["recall"],
        ),
        "precision_rf_detr_vs_yolo_percent": percent_change(
            rfdetr_report["precision"],
            yolo_report["precision"],
        ),
        "frames_sem_bola_rf_detr_vs_yolo_percent": percent_change(
            rfdetr_report["frames_sem_bola"],
            yolo_report["frames_sem_bola"],
        ),
        "maior_sequencia_sem_bola_rf_detr_vs_yolo_percent": percent_change(
            rfdetr_report["maior_sequencia_sem_bola"],
            yolo_report["maior_sequencia_sem_bola"],
        ),
        "tempo_medio_por_frame_rf_detr_vs_yolo_percent": percent_change(
            rfdetr_report["tempo_medio_por_frame_seconds"],
            yolo_report["tempo_medio_por_frame_seconds"],
        ),
        "fps_rf_detr_vs_yolo_percent": percent_change(
            rfdetr_report["fps_medio"],
            yolo_report["fps_medio"],
        ),
        "tempo_total_analise_rf_detr_vs_yolo_percent": percent_change(
            rfdetr_report["tempo_total_analise_seconds"],
            yolo_report["tempo_total_analise_seconds"],
        ),
        "confianca_media_rf_detr_vs_yolo_percent": percent_change(
            rfdetr_report["confianca_media"],
            yolo_report["confianca_media"],
        ),
    }

    final_report = {
        "input_video_path": str(Path(args.input).resolve()),
        "metric_definition": {
            "recall_precision": (
                "Cross-detector agreement metric on same frame using IoU >= "
                f"{args.match_iou_threshold}; each detector is evaluated against the other."
            ),
        },
        "detectors": {
            "detector_atual_yolo": yolo_report,
            "detector_rf_detr_soccernet": rfdetr_report,
        },
        "comparison_percent": comparison_percent,
        "generated_at_unix_seconds": time.time(),
    }

    comparison_output.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"comparison_report_path": str(comparison_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
