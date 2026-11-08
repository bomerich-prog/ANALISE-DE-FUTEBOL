#!/usr/bin/env python3
"""The visible-third map stays off the event coordinates."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from ball_heatmap import BallHeatmapAccumulator  # noqa: E402
from pitch_homography import fit_broadcast_penalty  # noqa: E402
from pitch_map import (  # noqa: E402
    VisiblePitchMap,
    box_matches_visible_paint,
    inside_visible,
    scale_is_stable,
    visible_bounds,
)
from possession import _player_distance  # noqa: E402
from roboflow_soccer_config import SoccerPitchConfiguration  # noqa: E402
from spatial_metrics import SpatialMetrics, hull_area_m2  # noqa: E402


def assert_true(condition, label):
    if not condition:
        raise AssertionError(label)


def assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_close(actual, expected, label, tolerance=1.0):
    if actual is None or abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def paint_penalty(gap=100.0):
    """Penalty line plus the arc the fitter expects 5.5 m behind it."""
    mask = np.zeros((750, 1338), dtype=np.uint8)
    start = np.array([220.0, 300.0])
    end = np.array([980.0, 560.0])
    direction = end - start
    direction = direction / np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]])
    center = (start + end) / 2.0 + perp * gap
    radius = gap * 9.15 / 5.5
    cv2.line(mask, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), 255, 5)
    base = np.arctan2(-perp[1], -perp[0])
    points = []
    for degrees in np.linspace(-50, 50, 70):
        angle = base + np.deg2rad(degrees)
        points.append(center + radius * np.array([np.cos(angle), np.sin(angle)]))
    cv2.polylines(mask, [np.round(np.array(points)).astype(np.int32)], False, 255, 5)
    return mask


def test_scale_jump_is_rejected_and_the_third_is_35_m():
    assert_true(scale_is_stable(None, 4.0), "the first scale has nothing to jump from")
    assert_true(scale_is_stable(4.0, 5.0), "a quarter more is still the same shot")
    assert_true(not scale_is_stable(4.0, 8.0), "doubling the scale is a bad matrix")
    low_x0, low_x1, y0, y1 = visible_bounds(False)
    high_x0, high_x1, _y0, _y1 = visible_bounds(True)
    assert_equal((low_x0, low_x1), (0.0, 3500.0), "the near goal owns the first 35 m")
    assert_equal((high_x0, high_x1), (8500.0, 12000.0), "the far goal owns the last 35 m")
    assert_equal((y0, y1), (0.0, 7000.0), "the width stays the pitch width")
    assert_true(inside_visible(3400, 3500, False), "34 m from the goal is inside")
    assert_true(not inside_visible(4200, 3500, False), "42 m is already the next third")
    assert_true(inside_visible(11000, 3500, True), "the far penalty spot is inside its third")
    assert_true(not inside_visible(4000, 3500, True), "midfield is not published from a penalty camera")


def test_map_coordinates_do_not_change_possession_distance():
    player = {"x": 10.0, "y": 20.0, "field_x": None, "field_y": None, "map_x": 1000.0, "map_y": 1000.0}
    ball = {"x": 40.0, "y": 20.0, "field_x": None, "field_y": None, "map_x": 8000.0, "map_y": 4000.0}
    _distance, space = _player_distance(player, ball)
    assert_equal(space, "pixel", "possession keeps the pixel distance when only the map is filled")


def test_heatmap_reads_the_map_channel():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe([{"map_x": 500.0, "map_y": 400.0, "field_x": 6100.0, "field_y": 3600.0}])
    report = heatmap.as_dict()
    assert_equal(report["cells"][0][0], 1, "map_x wins over a field coordinate")
    assert_equal(report["observed_points"], 1, "the map point is counted")


def test_projection_drops_the_far_half():
    pitch_map = VisiblePitchMap()
    pitch_map.transformer = pitch_map.transformer
    from roboflow_view import ViewTransformer

    pitch_map.transformer = ViewTransformer.from_matrix(np.eye(3))
    pitch_map.positive_x = False
    pitch_map.matrix = np.eye(3)
    mapped = pitch_map.project(np.array([[1000, 3500], [5000, 3500], [2000, 9000]], dtype=np.float32))
    assert_close(mapped[0][0], 1000.0, "a point in the visible third is kept", tolerance=0.1)
    assert_true(mapped[1] is None, "a point 50 m from the goal is dropped")
    assert_true(mapped[2] is None, "a point off the pitch width is dropped")


def test_spatial_metrics_use_only_mapped_players():
    metrics = SpatialMetrics(fps=10.0, goal_at_high_x=False)
    players = [
        {"track_id": 1, "team_id": 0, "map_x": 500.0, "map_y": 2000.0},
        {"track_id": 2, "team_id": 0, "map_x": 1500.0, "map_y": 4000.0},
        {"track_id": 3, "team_id": 0, "map_x": 1000.0, "map_y": 2500.0},
        {"track_id": 4, "team_id": 1, "map_x": 800.0, "map_y": 2500.0, "field_x": 9000.0, "field_y": 1000.0},
        {"track_id": 5, "team_id": 1, "map_x": None, "map_y": None, "field_x": 10000.0, "field_y": 6000.0},
    ]
    ball = {"map_x": 1200.0, "map_y": 3000.0}
    metrics.observe(1, players, ball, generation=1, goal_at_high_x=False)
    metrics.observe(2, players, {"map_x": 1300.0, "map_y": 3000.0}, generation=1, goal_at_high_x=False)
    report = metrics.as_dict()
    assert_close(report["teams"]["Team 0"]["width_m"], 20.0, "team 0 spans 20 m of width")
    assert_close(report["teams"]["Team 0"]["length_m"], 10.0, "team 0 spans 10 m of length")
    assert_true(report["teams"]["Team 0"]["compactness_m2"] > 0, "three players have a hull")
    assert_close(report["teams"]["Team 0"]["line_from_visible_goal_m"], 5.0, "the deepest player is 5 m out")
    assert_equal(report["teams"]["Team 1"]["shape_samples"], 0, "one mapped player is not a shape")
    assert_true(report["teams"]["Team 0"]["speed_m_s"] == 0.0, "nobody moved")
    assert_close(report["ball_speed_m_s"], 10.0, "the ball moved 1 m in a tenth of a second", tolerance=0.1)
    assert_equal(report["feeds_events"], False, "the report says it does not feed events")
    assert_true(hull_area_m2([(0, 0), (100, 0)]) is None, "two points have no area")
    jumped = metrics.observe(3, players, {"map_x": 9000.0, "map_y": 3000.0}, generation=2, goal_at_high_x=False)
    assert_true(jumped["ball_speed_samples"] == 1, "a new matrix does not invent ball speed")


def test_a_box_in_the_stands_is_not_published():
    grass = np.zeros((750, 1338), dtype=np.uint8)
    grass[220:, :] = 255
    white = np.zeros_like(grass)
    into_the_stands = np.array([[10.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    assert_true(
        not box_matches_visible_paint(into_the_stands, True, grass, white),
        "a box drawn across the crowd is refused",
    )
    on_the_grass_but_off_the_paint = np.array([[15.0, 0.0, 0.0], [0.0, 15.0, -2500.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    assert_true(
        not box_matches_visible_paint(on_the_grass_but_off_the_paint, True, grass, white),
        "a box on open grass, off the white lines, is refused",
    )
    assert_true(
        box_matches_visible_paint(into_the_stands, True, None, white),
        "without a grass picture the paint-only tests still accept",
    )


def test_blank_paint_publishes_nothing():
    pitch_map = VisiblePitchMap()
    decision = pitch_map.update(np.zeros((750, 1338), dtype=np.uint8))
    assert_true(not decision["accepted"], "empty paint is not a homography")
    assert_true(pitch_map.project(np.array([[400, 400]], dtype=np.float32)) == [None] or pitch_map.transformer is None, "nothing is projected")
    assert_true(decision["matrix"] is None, "a rejected frame keeps no matrix")


def test_real_penalty_paint_locks_and_a_blank_frame_drops_it():
    mask = paint_penalty()
    report = fit_broadcast_penalty(mask, SoccerPitchConfiguration(), frame=None)
    if report is None:
        raise AssertionError("the painted penalty line and arc should lock")
    assert_true(report["positive_x"] is False, "this line faces the near goal")
    assert_close(report["front_x_cm"], 1650.0, "the front line is the 16.5 m line", tolerance=0.1)
    assert_close(report["cm_per_px"], 5.5, "100 px is 5.5 m from the line to the spot", tolerance=0.4)
    pitch_map = VisiblePitchMap(hold_frames=2)
    first = pitch_map.update(mask)
    assert_true(first["accepted"] and not first["held"], "paint acceptance is a fresh fit")
    inside = pitch_map.project(np.array([[500.0, 400.0]], dtype=np.float32))
    assert_true(inside[0] is not None, "a point on this penalty projects inside the third")
    assert_true(inside[0][0] <= 3600.0, "the published point stays inside 35 m")
    jumped = paint_penalty(gap=150.0)
    refused = pitch_map.update(jumped)
    assert_true(pitch_map.reasons["scale_jump"] >= 1, "a doubled image scale is refused")
    assert_true(refused["matrix"] is None or abs(pitch_map.scale - report["cm_per_px"]) < 1.0, "the jump does not replace the map")
    blank = np.zeros_like(mask)
    dropped = pitch_map.update(blank)
    if dropped["accepted"]:
        dropped = pitch_map.update(blank)
    assert_true(not dropped["accepted"], "a blank frame does not keep a stale matrix")
    assert_true(dropped["matrix"] is None, "the dropped frame has no coordinates")


def main():
    cv2.setRNGSeed(0)
    test_scale_jump_is_rejected_and_the_third_is_35_m()
    test_map_coordinates_do_not_change_possession_distance()
    test_heatmap_reads_the_map_channel()
    test_projection_drops_the_far_half()
    test_spatial_metrics_use_only_mapped_players()
    test_a_box_in_the_stands_is_not_published()
    test_blank_paint_publishes_nothing()
    test_real_penalty_paint_locks_and_a_blank_frame_drops_it()
    print("pitch map tests passed")


if __name__ == "__main__":
    main()
