#!/usr/bin/env python3
"""The pitch homography is kept only when its lines land on the white paint."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from pitch_homography import (  # noqa: E402
    PitchHomographyGate,
    _affine_matrix,
    _side_matches_image,
    fit_broadcast_penalty,
    line_support,
)
from roboflow_soccer_config import SoccerPitchConfiguration  # noqa: E402


def assert_close(actual, expected, label, tolerance=1e-6):
    if actual is None or abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(condition, label):
    if not condition:
        raise AssertionError(label)


def broadcast_scene():
    """A known homography, the white lines it explains, and the keypoints on them."""
    pitch = SoccerPitchConfiguration()
    image_corners = np.array([[180, 90], [1160, 70], [1280, 680], [40, 700]], dtype=np.float32)
    field_corners = np.array(
        [[0, 0], [pitch.length, 0], [pitch.length, pitch.width], [0, pitch.width]],
        dtype=np.float32,
    )
    image_to_field, _ = cv2.findHomography(image_corners, field_corners)
    field_to_image = np.linalg.inv(image_to_field)
    mask = np.zeros((750, 1338), dtype=np.uint8)
    for start_id, end_id in pitch.edges:
        start = np.array(pitch.vertices[start_id - 1], dtype=np.float32)
        end = np.array(pitch.vertices[end_id - 1], dtype=np.float32)
        segment = np.stack([start + (end - start) * (step / 20) for step in range(21)])
        projected = cv2.perspectiveTransform(segment.reshape(-1, 1, 2), field_to_image.astype(np.float32))
        cv2.polylines(mask, [np.round(projected.reshape(-1, 2)).astype(np.int32)], False, 255, 3, cv2.LINE_AA)
    center = np.array([pitch.length / 2, pitch.width / 2], dtype=np.float32)
    circle = []
    for angle in np.linspace(0, 2 * np.pi, 60, endpoint=False):
        circle.append(
            [
                center[0] + pitch.centre_circle_radius * np.cos(angle),
                center[1] + pitch.centre_circle_radius * np.sin(angle),
            ]
        )
    projected = cv2.perspectiveTransform(
        np.array(circle, dtype=np.float32).reshape(-1, 1, 2),
        field_to_image.astype(np.float32),
    )
    cv2.polylines(mask, [np.round(projected.reshape(-1, 2)).astype(np.int32)], True, 255, 3, cv2.LINE_AA)

    visible = [0, 1, 9, 13, 14, 16, 17, 24, 25, 31]
    keypoints = []
    for index in visible:
        point = cv2.perspectiveTransform(
            np.array([[pitch.vertices[index]]], dtype=np.float32),
            field_to_image.astype(np.float32),
        )[0, 0]
        keypoints.append(
            {"index": index, "x": float(point[0]), "y": float(point[1]), "confidence": 0.9}
        )
    return keypoints, mask, image_to_field


def shift(keypoints, dx, dy):
    return [{**keypoint, "x": keypoint["x"] + dx, "y": keypoint["y"] + dy} for keypoint in keypoints]


def test_lines_on_the_paint_are_accepted():
    keypoints, mask, _matrix = broadcast_scene()
    gate = PitchHomographyGate()
    decision = gate.update(keypoints, mask)
    assert_true(decision["accepted"], "a template that covers the paint is accepted")
    assert_true(decision["support"] >= 0.55, f"support stays high, got {decision['support']}")
    assert_true(decision["held"] is False, "the first good frame is not a hold")


def test_shifted_keypoints_are_rejected():
    keypoints, mask, _matrix = broadcast_scene()
    gate = PitchHomographyGate()
    decision = gate.update(shift(keypoints, 80, 40), mask)
    assert_true(not decision["accepted"], "a box drawn off the paint is rejected")
    assert_true(
        decision["reason"] in {"line_support", "off_lines"},
        f"reason is the paint check, got {decision['reason']}",
    )
    assert_true(decision["transformer"] is None, "a rejected frame has no field transform")
    assert_true(decision["support"] is None or decision["support"] < 0.25, "shifted support stays low")


def test_one_outlier_does_not_move_the_fit():
    keypoints, mask, _matrix = broadcast_scene()
    noisy = shift(keypoints[:1], 220, -160) + keypoints[1:]
    gate = PitchHomographyGate()
    decision = gate.update(noisy, mask)
    assert_true(decision["accepted"], "RANSAC drops the outlier and keeps the paint fit")
    assert_true(decision["inliers"] <= len(noisy) - 1, "the outlier is not an inlier")


def test_last_good_matrix_is_kept_only_while_the_lines_still_match():
    keypoints, mask, _matrix = broadcast_scene()
    gate = PitchHomographyGate(hold_frames=2)
    assert_true(gate.update(keypoints, mask)["accepted"], "seed frame")
    held = gate.update([], mask)
    assert_true(held["held"], "a missed detection keeps the matrix while the paint still agrees")
    assert_true(held["transformer"] is not None, "held matrix can still project")
    blank = np.zeros_like(mask)
    dropped = gate.update([], blank)
    assert_true(dropped["transformer"] is None, "the held matrix is dropped when the lines leave")
    assert_true(dropped["reason"] == "too_few_points", "there was no new fit to accept")


def test_hold_expires():
    keypoints, mask, _matrix = broadcast_scene()
    gate = PitchHomographyGate(hold_frames=1)
    gate.update(keypoints, mask)
    assert_true(gate.update([], mask)["held"], "one held frame is allowed")
    expired = gate.update([], mask)
    assert_true(expired["transformer"] is None, "the hold does not last past its window")


def test_four_points_do_not_register_the_pitch():
    keypoints, mask, _matrix = broadcast_scene()
    gate = PitchHomographyGate()
    decision = gate.update(keypoints[:4], mask)
    assert_true(not decision["accepted"], "four points are enough to invent a wrong box")
    assert_true(
        decision["reason"] in {"too_few_points", "few_inliers", "off_lines"},
        f"four-point fit must be refused, got {decision['reason']}",
    )
    clustered = []
    for offset, keypoint in enumerate(keypoints[:4]):
        clustered.append({**keypoint, "x": 400 + offset, "y": 300 + offset * 0.5})
    decision = gate.update(clustered, mask)
    assert_true(not decision["accepted"], "four points on top of each other do not register a pitch")


def test_support_of_the_true_matrix_is_high():
    _keypoints, mask, matrix = broadcast_scene()
    support = line_support(matrix, mask, SoccerPitchConfiguration())
    assert_close(support, 1.0, "exact lines cover the mask", tolerance=0.05)


def test_a_line_on_the_right_cannot_be_the_left_goal():
    along = np.array([0.0, 1.0])
    perp = np.array([1.0, 0.0])
    scale = 100.0 / 550.0
    right = _affine_matrix(np.array([1800.0, 500.0]), along, perp, scale, 12000.0 - 1650.0, 7000.0, True)
    mirrored = _affine_matrix(np.array([1800.0, 500.0]), along, perp, scale, 1650.0, 7000.0, False)
    assert_true(_side_matches_image(right, True, 1920), "a line on the right may be the right goal")
    assert_true(not _side_matches_image(mirrored, False, 1920), "the same line is not the left goal")
    left = _affine_matrix(np.array([120.0, 500.0]), along, perp, scale, 1650.0, 7000.0, False)
    flipped = _affine_matrix(np.array([120.0, 500.0]), along, perp, scale, 12000.0 - 1650.0, 7000.0, True)
    assert_true(_side_matches_image(left, False, 1920), "a line on the left may be the left goal")
    assert_true(not _side_matches_image(flipped, True, 1920), "the same line is not the right goal")


def _penalty_mask(shift, width=1920, height=1080):
    mask = np.zeros((height, width), dtype=np.uint8)
    start = np.array([220.0, 420.0]) + np.array(shift, dtype=np.float64)
    end = np.array([980.0, 680.0]) + np.array(shift, dtype=np.float64)
    direction = end - start
    direction = direction / np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]])
    center = (start + end) / 2.0 + perp * 100.0
    radius = 100.0 * 9.15 / 5.5
    cv2.line(mask, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), 255, 5)
    base = np.arctan2(-perp[1], -perp[0])
    points = []
    for degrees in np.linspace(-50, 50, 70):
        angle = base + np.deg2rad(degrees)
        points.append(center + radius * np.array([np.cos(angle), np.sin(angle)]))
    cv2.polylines(mask, [np.round(np.array(points)).astype(np.int32)], False, 255, 5)
    return mask


def test_a_right_hand_penalty_is_published_on_the_right():
    pitch = SoccerPitchConfiguration()
    left = fit_broadcast_penalty(_penalty_mask((0.0, 0.0)), pitch, frame=None)
    right = fit_broadcast_penalty(_penalty_mask((800.0, 0.0)), pitch, frame=None)
    assert_true(left is not None and left["positive_x"] is False, "the left box stays the near goal")
    assert_true(right is not None and right["positive_x"] is True, "the right box is the far goal")
    assert_true(right["front_x_cm"] > 8000.0, "the right box uses the far 16.5 m line")


def main():
    cv2.setRNGSeed(0)
    test_support_of_the_true_matrix_is_high()
    test_lines_on_the_paint_are_accepted()
    test_shifted_keypoints_are_rejected()
    test_one_outlier_does_not_move_the_fit()
    test_last_good_matrix_is_kept_only_while_the_lines_still_match()
    test_hold_expires()
    test_four_points_do_not_register_the_pitch()
    test_a_line_on_the_right_cannot_be_the_left_goal()
    test_a_right_hand_penalty_is_published_on_the_right()
    print("pitch homography gate tests passed")


if __name__ == "__main__":
    main()
