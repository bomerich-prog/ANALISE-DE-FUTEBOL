#!/usr/bin/env python3
"""The visible-third map stays off the event coordinates."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from ball_heatmap import BallHeatmapAccumulator  # noqa: E402
from pitch_homography import fit_broadcast_penalty  # noqa: E402
from pitch_homography import _affine_matrix  # noqa: E402
from pitch_map import (  # noqa: E402
    LINE_MOVE_PX,
    VisiblePitchMap,
    box_matches_visible_paint,
    field_y_from_line,
    front_image_point,
    inside_measured,
    inside_visible,
    marker_bgr,
    paint_patch,
    penalty_corners,
    replacement_verdict,
    scale_is_stable,
    visible_bounds,
    _width_terms,
)
from possession import _player_distance  # noqa: E402
from roboflow_soccer_config import SoccerPitchConfiguration  # noqa: E402
from roboflow_view import ViewTransformer  # noqa: E402
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


def paint_penalty(gap=100.0, shift=(0.0, 0.0)):
    """Penalty line plus the arc the fitter expects 5.5 m behind it."""
    mask = np.zeros((750, 1338), dtype=np.uint8)
    start = np.array([220.0, 300.0]) + np.array(shift, dtype=np.float64)
    end = np.array([980.0, 560.0]) + np.array(shift, dtype=np.float64)
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


def test_marker_colours_follow_the_team():
    assert_equal(marker_bgr("player", 0), (255, 64, 0), "team 0 is blue")
    assert_equal(marker_bgr("player", 1), (0, 0, 255), "team 1 is red")
    assert_equal(marker_bgr("goalkeeper", 0)[0], 255, "a goalkeeper keeps the team colour")
    assert_equal(marker_bgr("referee"), (0, 255, 255), "the referee is yellow")
    assert_equal(marker_bgr("ball"), (0, 140, 255), "the ball is orange")
    assert_equal(marker_bgr("player", None), (180, 180, 180), "a player without a team stays neutral")


def test_the_centre_circle_publishes_the_middle():
    pitch_map = VisiblePitchMap()
    decision = pitch_map.update(_centre_circle_mask())
    assert_true(decision["accepted"] and decision["region"] == "centre", "the open camera locks the middle")
    near = pitch_map.project(np.array([[520.0, 430.0]], dtype=np.float32))
    assert_true(near[0] is not None, "a player beside the circle stays on the map")
    assert_true(3000.0 <= near[0][0] <= 9000.0, "that player is in the middle 60 m")
    blank = np.zeros((750, 1338), dtype=np.uint8)
    for _ in range(15):
        assert_true(pitch_map.update(blank)["accepted"], "a short gap keeps the centre")
    assert_true(not pitch_map.update(blank)["accepted"], "the centre bridge ends")


def _centre_circle_mask():
    mask = np.zeros((750, 1338), dtype=np.uint8)
    cv2.line(mask, (400, 180), (400, 680), 255, 5)
    cv2.circle(mask, (400, 430), 120, 255, 5)
    return mask


def _uneven_penalty(continue_past=False):
    """A 16.5 m line whose far half is shorter in pixels than its near half."""
    foot = np.array([640.0, 340.0])
    along = np.array([0.88, 0.47])
    along = along / np.linalg.norm(along)
    perp = np.array([-along[1], along[0]])
    matrix = _affine_matrix(foot, along, perp, 1.0 / 7.0, 10350.0, 7000.0, True)
    mask = np.zeros((750, 1338), dtype=np.uint8)
    for station in list(range(-216, -20, 2)) + list(range(20, 150, 2)) + list(range(214, 392, 2)):
        point = foot + along * float(station)
        cv2.circle(mask, (int(round(point[0])), int(round(point[1]))), 2, 255, -1)
    if continue_past:
        for station in range(-270, -218):
            point = foot + along * float(station) + perp * 16.0
            cv2.circle(mask, (int(round(point[0])), int(round(point[1]))), 1, 255, -1)
    return mask, matrix, foot, along


def test_uneven_corners_move_the_width_and_keep_the_arc_scale():
    terms = _width_terms(-216.0, 392.0, 3500.0)
    assert_close(field_y_from_line(0.0, terms, 3500.0), 3500.0, "the arc stays the middle of the box", tolerance=1.0)
    assert_close(field_y_from_line(-216.0, terms, 3500.0), 1484.0, "the far corner is 20.16 m off centre", tolerance=1.0)
    assert_close(field_y_from_line(392.0, terms, 3500.0), 5516.0, "the near corner is 20.16 m off centre", tolerance=1.0)
    mask, matrix, foot, along = _uneven_penalty()
    found = penalty_corners(mask, matrix, True, 7.0)
    assert_true(found is not None, "the two ends of the line are the corners")
    assert_close(found["t_low"], -216.0, "the short side is the far corner", tolerance=12.0)
    assert_close(found["t_high"], 392.0, "the long side is the near corner", tolerance=12.0)
    pitch_map = VisiblePitchMap()
    pitch_map.matrix = matrix
    pitch_map.transformer = ViewTransformer.from_matrix(matrix)
    pitch_map.positive_x = True
    pitch_map.region = "penalty"
    pitch_map.scale = 7.0
    pitch_map.toward_goal_cm = 400.0
    decision = pitch_map._publish(pitch_map._decision(True, False, "accepted"), mask)
    assert_true(decision["corners"] is not None, "the drawing uses the two corners")
    far = foot + along * found["t_low"]
    mapped = pitch_map.project(np.array([far], dtype=np.float32))
    assert_true(mapped[0] is not None, "a player on the far corner stays on the map")
    assert_close(mapped[0][1], 1484.0, "that player is on the corner, not 5 m inside the box", tolerance=80.0)
    drifting, drift_matrix, _foot, _along = _uneven_penalty(continue_past=True)
    assert_true(
        penalty_corners(drifting, drift_matrix, True, 7.0) is None,
        "a line that keeps going is not a corner",
    )


def test_a_line_cut_by_the_frame_is_not_a_corner():
    foot = np.array([900.0, 500.0])
    along = np.array([0.95, 0.31])
    along = along / np.linalg.norm(along)
    perp = np.array([-along[1], along[0]])
    matrix = _affine_matrix(foot, along, perp, 1.0 / 4.0, 10350.0, 7000.0, True)
    mask = np.zeros((750, 1338), dtype=np.uint8)
    for station in range(-400, 900, 2):
        point = foot + along * float(station)
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            cv2.circle(mask, (x, y), 2, 255, -1)
    assert_true(
        penalty_corners(mask, matrix, True, 4.0) is None,
        "the picture edge is not a painted corner",
    )


def test_the_measured_strip_does_not_erase_the_visible_third():
    pitch_map = VisiblePitchMap()
    pitch_map.transformer = ViewTransformer.from_matrix(np.eye(3))
    pitch_map.matrix = np.eye(3)
    pitch_map.positive_x = False
    pitch_map.toward_goal_cm = 200.0
    pitch_map.toward_arc_cm = 200.0
    deep = pitch_map.project(np.array([[2500.0, 3500.0]], dtype=np.float32))
    assert_true(deep[0] is not None, "25 m stays on the map when the probe stopped at 2 m")
    assert_close(deep[0][0], 2500.0, "the published point keeps its field position", tolerance=0.1)
    assert_true(
        not inside_measured(2500.0, 3500.0, False, 200.0, 200.0),
        "the probe itself still marks that point as past the measured depth",
    )
    far = pitch_map.project(np.array([[5000.0, 3500.0]], dtype=np.float32))
    assert_true(far[0] is None, "50 m is outside the visible third")


def test_projection_drops_the_far_half():
    pitch_map = VisiblePitchMap()
    pitch_map.transformer = pitch_map.transformer
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


def _stands_matrix():
    """16.5 m line on the grass, goal line in the crowd. 4 px per centimetre of depth."""
    return np.array(
        [[0.0, -4.0, 12350.0], [20.0 / 3.0, 0.0, 1484.0 - 200.0 / 0.15], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _paint_projected(mask, matrix, field_points, radius=2):
    inverse = np.linalg.inv(matrix).astype(np.float32)
    projected = cv2.perspectiveTransform(
        np.asarray(field_points, dtype=np.float32).reshape(-1, 1, 2), inverse
    ).reshape(-1, 2)
    for x, y in projected:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(mask, (int(round(x)), int(round(y))), radius, 255, -1)


def test_goal_line_in_the_stands_keeps_the_grass_patch():
    grass = np.zeros((750, 1338), dtype=np.uint8)
    grass[220:, :] = 255
    white = np.zeros_like(grass)
    matrix = _stands_matrix()
    front = 10350.0
    y0 = (7000.0 - 4032.0) / 2.0
    y1 = (7000.0 + 4032.0) / 2.0
    front_line = [(front, y0 + (y1 - y0) * step) for step in np.linspace(0, 1, 40)]
    spot = 12000.0 - 1100.0
    arc = []
    for angle in np.linspace(0, 2 * np.pi, 48):
        x = spot + 915.0 * np.cos(angle)
        y = 3500.0 + 915.0 * np.sin(angle)
        if x < front:
            arc.append((x, y))
    _paint_projected(white, matrix, front_line, radius=2)
    _paint_projected(white, matrix, arc, radius=2)
    patch = paint_patch(matrix, True, grass, white)
    assert_true(patch is not None, "the line and the arc keep the frame")
    assert_true(patch["toward_goal_cm"] >= 400.0, "several metres toward the goal stay on the grass")
    assert_true(patch["toward_goal_cm"] < 1650.0, "the goal line in the stands is outside the patch")
    assert_true(
        inside_measured(front + 400.0, 3500.0, True, patch["toward_goal_cm"], patch["toward_arc_cm"]),
        "4 m toward the goal is published",
    )
    assert_true(
        not inside_measured(12000.0, 3500.0, True, patch["toward_goal_cm"], patch["toward_arc_cm"]),
        "the goal line is not published",
    )
    pitch_map = VisiblePitchMap()
    pitch_map.matrix = matrix
    pitch_map.transformer = ViewTransformer.from_matrix(matrix)
    pitch_map.positive_x = True
    pitch_map.toward_goal_cm = patch["toward_goal_cm"]
    pitch_map.toward_arc_cm = patch["toward_arc_cm"]
    pitch_map.grass = grass
    near = pitch_map.project(np.array([[502.0, 400.0]], dtype=np.float32))
    far = pitch_map.project(np.array([[502.0, 87.0]], dtype=np.float32))
    assert_true(near[0] is not None, "a grass point inside the patch is published")
    assert_true(far[0] is None, "the same player in the crowd publishes nothing")


def test_blank_paint_publishes_nothing():
    pitch_map = VisiblePitchMap()
    decision = pitch_map.update(np.zeros((750, 1338), dtype=np.uint8))
    assert_true(not decision["accepted"], "empty paint is not a homography")
    assert_true(pitch_map.project(np.array([[400, 400]], dtype=np.float32)) == [None] or pitch_map.transformer is None, "nothing is projected")
    assert_true(decision["matrix"] is None, "a rejected frame keeps no matrix")


def test_a_camera_move_replaces_a_matrix_that_left_the_paint():
    pitch_map = VisiblePitchMap(hold_frames=15)
    first = pitch_map.update(paint_penalty())
    assert_true(first["accepted"] and not first["held"], "the opening penalty is accepted")
    old_front = front_image_point(pitch_map.matrix, pitch_map.positive_x)
    moved = pitch_map.update(paint_penalty(shift=(300.0, 20.0)))
    assert_true(moved["accepted"] and not moved["held"], "the other line replaces the old matrix")
    assert_true(pitch_map.reasons["camera_move"] >= 1, "a real change of line is not blocked at 12 m")
    new_front = front_image_point(pitch_map.matrix, pitch_map.positive_x)
    travel = float(np.hypot(new_front[0] - old_front[0], new_front[1] - old_front[1]))
    assert_true(travel >= LINE_MOVE_PX, "the published line followed the paint")
    assert_true(pitch_map.generation >= 2, "a new region does not inherit the old shot")
    inside = pitch_map.project(np.array([[700.0, 430.0]], dtype=np.float32))
    assert_true(inside[0] is not None, "the new penalty still publishes inside its third")
    assert_true(inside[0][0] <= 3600.0, "the new side does not draw the other half")


def test_a_short_gap_keeps_the_last_matrix_then_drops_it():
    pitch_map = VisiblePitchMap(hold_frames=15)
    assert_true(pitch_map.update(paint_penalty())["accepted"], "the penalty locks")
    blank = np.zeros((750, 1338), dtype=np.uint8)
    point = np.array([[500.0, 400.0]], dtype=np.float32)
    for _ in range(15):
        decision = pitch_map.update(blank)
        assert_true(decision["accepted"] and decision["held"], "a short gap keeps the last matrix")
        assert_true(pitch_map.project(point)[0] is not None, "players stay on the map during the gap")
    dropped = pitch_map.update(blank)
    assert_true(not dropped["accepted"], "the bridge ends")
    assert_true(dropped["matrix"] is None, "the matrix is gone after the gap")
    assert_true(pitch_map.reasons["bridge"] == 15, "each gap frame is counted")
    assert_true(pitch_map.reasons["no_visible_penalty"] >= 1, "the drop is recorded")


def test_replacement_refuses_a_mirror_and_allows_a_far_camera():
    assert_equal(
        replacement_verdict(True, True, 400.0, 20.0, True),
        "nudge",
        "a few metres on the same line is the same penalty",
    )
    assert_equal(
        replacement_verdict(True, True, 5000.0, 20.0, True),
        "field_jump",
        "12 m still blocks a teleport while the arc is on the paint",
    )
    assert_equal(
        replacement_verdict(False, True, 5000.0, 30.0, True),
        "mirrored",
        "flipping the same line is a mirror",
    )
    assert_equal(
        replacement_verdict(False, True, 400.0, 40.0, False),
        "mirrored",
        "a flip in place is still a mirror after the old arc slips",
    )
    assert_equal(
        replacement_verdict(False, False, 8000.0, 400.0, False),
        "camera_move",
        "the other penalty may cross 12 m once its line has moved",
    )
    assert_equal(
        replacement_verdict(True, False, 800.0, 30.0, False),
        "scale_jump",
        "a doubled scale on the same line is not a new camera",
    )


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
    dropped = None
    for _ in range(pitch_map.hold_frames + 1):
        dropped = pitch_map.update(blank)
    assert_true(not dropped["accepted"], "the bridge does not keep a stale matrix")
    assert_true(dropped["matrix"] is None, "the dropped frame has no coordinates")


def main():
    cv2.setRNGSeed(0)
    test_scale_jump_is_rejected_and_the_third_is_35_m()
    test_map_coordinates_do_not_change_possession_distance()
    test_heatmap_reads_the_map_channel()
    test_projection_drops_the_far_half()
    test_marker_colours_follow_the_team()
    test_the_centre_circle_publishes_the_middle()
    test_uneven_corners_move_the_width_and_keep_the_arc_scale()
    test_a_line_cut_by_the_frame_is_not_a_corner()
    test_the_measured_strip_does_not_erase_the_visible_third()
    test_spatial_metrics_use_only_mapped_players()
    test_a_box_in_the_stands_is_not_published()
    test_goal_line_in_the_stands_keeps_the_grass_patch()
    test_blank_paint_publishes_nothing()
    test_real_penalty_paint_locks_and_a_blank_frame_drops_it()
    test_a_camera_move_replaces_a_matrix_that_left_the_paint()
    test_a_short_gap_keeps_the_last_matrix_then_drops_it()
    test_replacement_refuses_a_mirror_and_allows_a_far_camera()
    print("pitch map tests passed")


if __name__ == "__main__":
    main()
