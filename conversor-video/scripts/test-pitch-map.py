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
    fit_painted_box,
    front_image_point,
    inside_measured,
    inside_visible,
    marker_bgr,
    paint_patch,
    penalty_corners,
    penalty_sides,
    replacement_verdict,
    _line_ends,
    _line_frame,
    scale_is_stable,
    visible_bounds,
    _width_terms,
)
from pitch_centre import fit_centre_on_ink  # noqa: E402
from pitch_radar import RADAR_LENGTH_CM, RADAR_WIDTH_CM, RadarMemory, draw_radar, to_radar  # noqa: E402
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


def _paint_ridge(frame, start, end, value=155):
    cv2.line(
        frame,
        tuple(np.round(start).astype(int)),
        tuple(np.round(end).astype(int)),
        (value, value, value),
        3,
    )


def test_the_side_ridge_sets_the_width_and_keeps_the_depth():
    mask, matrix, _foot, _along = _uneven_penalty()
    found = penalty_corners(mask, matrix, True, 7.0)
    _foot, along, perp = _line_frame(matrix, True, SoccerPitchConfiguration())
    theta = np.radians(40.0)
    direction = np.cos(theta) * along + np.sin(theta) * perp
    direction = direction / np.linalg.norm(direction)
    corner = np.asarray(found["corner_low"], dtype=np.float64)
    frame = np.full((750, 1338, 3), 100, np.uint8)
    grass = np.full((750, 1338), 255, np.uint8)
    _paint_ridge(frame, corner, corner + direction * 210.0)
    board = np.round(corner + direction * 260.0).astype(int)
    cv2.circle(frame, tuple(board), 28, (230, 230, 230), -1)
    cv2.circle(grass, tuple(board), 28, 0, -1)
    pitch_map = VisiblePitchMap()
    pitch_map.matrix = matrix
    pitch_map.transformer = ViewTransformer.from_matrix(matrix)
    pitch_map.positive_x = True
    pitch_map.region = "penalty"
    pitch_map.scale = 7.0
    pitch_map.toward_goal_cm = 400.0
    pitch_map.grass = grass
    decision = pitch_map._publish(pitch_map._decision(True, False, "accepted"), mask, frame)
    assert_true(decision["side_lines"] is not None and len(decision["side_lines"]) == 2, "one painted side is copied to the other corner")
    traced = decision["side_lines"][0]
    copied = decision["side_lines"][1]
    traced_dir = np.array(traced[1]) - np.array(traced[0])
    copied_dir = np.array(copied[1]) - np.array(copied[0])
    traced_dir = traced_dir / np.linalg.norm(traced_dir)
    copied_dir = copied_dir / np.linalg.norm(copied_dir)
    assert_true(float(np.dot(traced_dir, direction)) > 0.99, "the ink follows the painted side")
    assert_true(float(np.dot(copied_dir, traced_dir)) > 0.99, "the other side keeps that same angle")
    assert_true(
        float(np.linalg.norm(np.array(traced[1]) - board)) > 20.0,
        "the ink stops before the boards",
    )
    on_side = corner + direction * 100.0
    through_middle = _foot + direction * 120.0
    mapped_side = pitch_map.project(np.array([on_side], dtype=np.float32))
    mapped_middle = pitch_map.project(np.array([through_middle], dtype=np.float32))
    raw_middle = pitch_map.transformer.transform_points(np.array([through_middle], dtype=np.float32))[0]
    assert_true(mapped_side[0] is not None, "a player on the side stays on the map")
    assert_close(mapped_side[0][1], 1484.0, "he takes the corner's width, not the perpendicular", tolerance=40.0)
    assert_close(mapped_middle[0][1], 3500.0, "the middle of the 16.5 m line stays the middle", tolerance=40.0)
    assert_close(mapped_middle[0][0], float(raw_middle[0]), "the depth stays the arc scale", tolerance=0.5)


def test_a_board_is_not_a_side_and_one_corner_draws_one_side():
    mask, matrix, _foot, _along = _uneven_penalty()
    boards = np.full((750, 1338, 3), 230, np.uint8)
    grass = np.zeros((750, 1338), np.uint8)
    on_boards = penalty_sides(
        boards, grass, np.array([400.0, 300.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), 60.0
    )
    assert_true(on_boards is not None and not on_boards["sides"], "the advertising is not a side line")
    pitch_map = VisiblePitchMap()
    pitch_map.matrix = matrix
    pitch_map.transformer = ViewTransformer.from_matrix(matrix)
    pitch_map.positive_x = True
    pitch_map.region = "penalty"
    pitch_map.scale = 7.0
    pitch_map.grass = grass
    decision = pitch_map._publish(pitch_map._decision(True, False, "accepted"), mask, boards)
    assert_true(decision["corners"] is not None, "the two corners still set the width")
    assert_true(decision["side_lines"] is None, "no side is invented on the boards")

    foot = np.array([700.0, 400.0])
    along = np.array([0.95, 0.31])
    along = along / np.linalg.norm(along)
    perp = np.array([-along[1], along[0]])
    single = _affine_matrix(foot, along, perp, 1.0 / 4.0, 10350.0, 7000.0, True)
    single_mask = np.zeros((750, 1338), dtype=np.uint8)
    for station in range(-360, -20, 2):
        point = foot + along * float(station)
        cv2.circle(single_mask, (int(round(point[0])), int(round(point[1]))), 2, 255, -1)
    ends = _line_ends(single_mask, single, True, 4.0)
    assert_true(ends is not None and ends["corner_low"] is not None and ends["corner_high"] is None, "one end off the picture leaves the other")
    theta = np.radians(40.0)
    direction = np.cos(theta) * ends["along"] + np.sin(theta) * ends["perp"]
    direction = direction / np.linalg.norm(direction)
    picture = np.full((750, 1338, 3), 100, np.uint8)
    live_grass = np.full((750, 1338), 255, np.uint8)
    _paint_ridge(picture, ends["corner_low"], ends["corner_low"] + direction * 180.0)
    alone = VisiblePitchMap()
    alone.matrix = single
    alone.transformer = ViewTransformer.from_matrix(single)
    alone.positive_x = True
    alone.region = "penalty"
    alone.scale = 4.0
    alone.grass = live_grass
    one = alone._publish(alone._decision(True, False, "accepted"), single_mask, picture)
    assert_true(one["corners"] is None, "one corner does not move the width")
    assert_equal(one["side_anchor"], "low", "the visible side is the far one")
    assert_true(one["side_lines"] is not None and len(one["side_lines"]) == 1, "only that side is drawn")
    assert_true(one["front_line"] is not None, "the 16.5 m line is cut at the measured corner")
    assert_true(
        float(np.linalg.norm(np.array(one["front_line"][0]) - ends["corner_low"])) < 8.0,
        "the 16.5 m line starts on the corner, not past it",
    )
    sample = ends["corner_low"] + direction * 40.0
    mapped = alone.project(np.array([sample], dtype=np.float32))
    raw = alone.transformer.transform_points(np.array([sample], dtype=np.float32))[0]
    assert_true(mapped[0] is not None, "the player stays on the affine map")
    assert_close(mapped[0][1], float(raw[1]), "his width stays affine when the other corner is hidden", tolerance=0.5)


def _painted_box(goal_line=True, past_corner=True):
    """A broadcast penalty: 16.5 m line, far side, goal line, touchline and a board."""
    foot = np.array([640.0, 340.0])
    along = np.array([0.88, 0.47])
    along = along / np.linalg.norm(along)
    perp = np.array([along[1], -along[0]])
    corner = foot - along * 216.0
    near = foot + along * 392.0
    side = np.array([np.cos(np.radians(-5.0)), np.sin(np.radians(-5.0))])
    goal = np.array([np.cos(np.radians(20.0)), np.sin(np.radians(20.0))])
    junction = corner + side * 450.0
    frame = np.full((750, 1338, 3), (60, 120, 70), np.uint8)
    grass = np.full((750, 1338), 255, np.uint8)
    _paint_ridge(frame, corner, near, 200)
    _paint_ridge(frame, corner, junction, 170)
    if goal_line:
        _paint_ridge(frame, junction - goal * 320.0, junction + goal * 420.0, 170)
        # The side's direction runs on into a board. The ink has to stop on the junction.
        _paint_ridge(frame, junction + side * 12.0, junction + side * 70.0, 170)
    # The touchline is longer than the side and never touches the 16.5 m line.
    _paint_ridge(frame, corner - along * 190.0 - side * 300.0, corner - along * 190.0 + side * 700.0, 170)
    if past_corner:
        # A player stands on the line past the corner.
        cv2.circle(frame, tuple(np.round(corner - along * 70.0).astype(int)), 9, (235, 235, 235), -1)
    return frame, grass, foot, along, perp, corner, near, side, junction


def test_the_side_ends_where_it_meets_the_goal_line():
    frame, grass, foot, along, perp, corner, near, side, junction = _painted_box()
    # The calibrated line lags the paint by a few pixels, as a held matrix does.
    measured = penalty_sides(frame, grass, foot + perp * 9.0, along, perp, 120.0)
    assert_true(measured is not None and "low" in measured["sides"], "the far side is found from the paint")
    far = measured["sides"]["low"]
    assert_true(float(np.linalg.norm(far["corner"] - corner)) < 6.0, "the corner is where the side leaves, not the player past it")
    assert_true(float(np.dot(far["direction"], side)) > 0.999, "the side follows its paint, not the touchline")
    assert_true(far["junction"] is not None, "the side ends on the goal line")
    assert_true(float(np.linalg.norm(far["junction"] - junction)) < 6.0, "the end is the point where the three lines meet")
    assert_true("high" not in measured["sides"], "the near side has no paint in this picture")

    open_frame, _grass, _foot, _along, _perp, corner, _near, side, _junction = _painted_box(goal_line=False)
    _paint_ridge(open_frame, corner, corner + side * 1200.0, 170)
    to_edge = penalty_sides(open_frame, grass, foot, along, perp, 120.0)["sides"]["low"]
    assert_true(to_edge["junction"] is None and to_edge["border"], "without the goal line the side runs to the picture edge")


def test_the_other_corner_copies_the_angle_to_the_picture_edge():
    frame, grass, foot, along, perp, corner, near, side, junction = _painted_box()
    mask, _matrix, _foot, _along = _uneven_penalty()
    # Same 16.5 m line as the mask, with the goal on the painted side's hand.
    matrix = _affine_matrix(foot, along, perp, 1.0 / 7.0, 10350.0, 7000.0, True)
    pitch_map = VisiblePitchMap()
    pitch_map.matrix = matrix
    pitch_map.transformer = ViewTransformer.from_matrix(matrix)
    pitch_map.positive_x = True
    pitch_map.region = "penalty"
    pitch_map.scale = 7.0
    pitch_map.toward_goal_cm = 400.0
    pitch_map.grass = grass
    decision = pitch_map._publish(pitch_map._decision(True, False, "accepted"), mask, frame)
    assert_true(decision["side_lines"] is not None and len(decision["side_lines"]) == 2, "both sides are drawn")
    measured, copied = decision["side_lines"]
    assert_true(float(np.linalg.norm(np.array(measured[1]) - junction)) < 6.0, "the far side stops on the junction")
    assert_true(float(np.linalg.norm(np.array(copied[0]) - near)) < 14.0, "the copy starts on the near corner")
    copied_dir = np.array(copied[1]) - np.array(copied[0])
    copied_dir = copied_dir / np.linalg.norm(copied_dir)
    assert_true(float(np.dot(copied_dir, side)) > 0.999, "the near side takes the far side's angle")
    assert_true(copied[1][0] > 1330.0, "and runs to the end of the picture")
    assert_true(
        float(np.linalg.norm(np.array(decision["corners"][0]) - corner)) < 6.0,
        "the 16.5 m line is drawn from the measured corner",
    )
    # A frame with no readable side keeps the last angle for a moment.
    blank = np.full((750, 1338, 3), (60, 120, 70), np.uint8)
    held = pitch_map._publish(pitch_map._decision(True, True, "held"), mask, blank)
    assert_true(held["side_lines"] is not None and len(held["side_lines"]) == 2, "a missed frame keeps both sides")
    held_dir = np.array(held["side_lines"][0][1]) - np.array(held["side_lines"][0][0])
    held_dir = held_dir / np.linalg.norm(held_dir)
    assert_true(float(np.dot(held_dir, side)) > 0.995, "and they keep the measured angle, not the perpendicular")



# Corners of the box as the clip's camera saw them: all inside, and frame 165 zoomed out.
_WHOLE_BOX_VIEW = [(560.0, 230.0), (1230.0, 640.0), (1120.0, 160.0), (1600.0, 360.0)]
_FAR_CORNER_VIEW = [(917.0, 279.0), (1606.0, 649.0), (1321.0, 254.0), (2110.0, 524.0)]


def _broadcast_box(goal_line=True, view=None):
    """A penalty area in perspective, painted as thin lines on the grass.

    Returns the frame, a grass mask and the true field-to-image matrix.
    """
    pitch = SoccerPitchConfiguration()
    field = np.array([(10350.0, 1484.0), (10350.0, 5516.0), (12000.0, 1484.0), (12000.0, 5516.0)], np.float32)
    image = np.array(view or _WHOLE_BOX_VIEW, np.float32)
    to_image = cv2.getPerspectiveTransform(field, image)
    frame = np.full((750, 1338, 3), (60, 120, 70), np.uint8)
    grass = np.full((750, 1338), 255, np.uint8)
    grass[: int(750 * 0.10)] = 0

    def paint(points):
        pts = cv2.perspectiveTransform(np.array(points, np.float32).reshape(-1, 1, 2), to_image).reshape(-1, 2)
        for a, b in zip(pts[:-1], pts[1:]):
            _paint_ridge(frame, a, b, 180)

    steps = np.linspace(0.0, 1.0, 40)
    paint([(10350.0, 1484.0 + 4032.0 * t) for t in steps])
    paint([(10350.0 + 1650.0 * t, 1484.0) for t in steps])
    paint([(10350.0 + 1650.0 * t, 5516.0) for t in steps])
    if goal_line:
        paint([(12000.0, 300.0 + 6400.0 * t) for t in steps])
    paint([(11450.0, 2584.0 + 1832.0 * t) for t in steps])
    paint([(11450.0 + 550.0 * t, 2584.0) for t in steps])
    arc = [
        (10900.0 - 915.0 * np.cos(a), 3500.0 + 915.0 * np.sin(a))
        for a in np.linspace(-np.arccos(550.0 / 915.0), np.arccos(550.0 / 915.0), 40)
    ]
    paint(arc)
    return frame, grass, to_image, pitch


def test_an_open_camera_reads_the_box_from_its_painted_corners():
    frame, grass, to_image, pitch = _broadcast_box()
    report, _measured = fit_painted_box(frame, grass, None, pitch)
    assert_true(report is not None, "the whole painted box gives a matrix without the arc fit")
    assert_true(report["positive_x"], "the goal is on the right")
    checks = np.array([(10350.0, 1484.0), (10350.0, 3500.0), (11450.0, 2584.0), (10900.0 - 915.0, 3500.0)], np.float32)
    truth = cv2.perspectiveTransform(checks.reshape(-1, 1, 2), to_image).reshape(-1, 2)
    inverse = np.linalg.inv(report["matrix"])
    found = cv2.perspectiveTransform(checks.reshape(-1, 1, 2), inverse.astype(np.float32)).reshape(-1, 2)
    error = float(np.max(np.linalg.norm(found - truth, axis=1)))
    assert_true(error < 5.0, f"corner, middle, small box and arc land on the paint ({error:.1f} px)")


def test_without_the_goal_line_the_arc_closes_the_box():
    frame, grass, to_image, pitch = _broadcast_box(goal_line=False, view=_FAR_CORNER_VIEW)
    report, _measured = fit_painted_box(frame, grass, None, pitch)
    assert_true(report is not None, "one corner and the arc are enough")
    checks = np.array([(10350.0, 1484.0), (10350.0, 3500.0), (10900.0 - 915.0, 3500.0)], np.float32)
    truth = cv2.perspectiveTransform(checks.reshape(-1, 1, 2), to_image).reshape(-1, 2)
    inverse = np.linalg.inv(report["matrix"])
    found = cv2.perspectiveTransform(checks.reshape(-1, 1, 2), inverse.astype(np.float32)).reshape(-1, 2)
    error = float(np.max(np.linalg.norm(found - truth, axis=1)))
    assert_true(error < 6.0, f"the corner, the middle and the arc tip land on the paint ({error:.1f} px)")


def _broadcast_centre(view):
    """The centre circle and the halfway line in perspective, painted as thin lines.

    view gives the image of the circle's four ends: top, bottom, left, right.
    Returns the frame, a grass mask and the true field-to-image matrix.
    """
    cardinals = np.array([(6000.0, 2585.0), (6000.0, 4415.0), (5085.0, 3500.0), (6915.0, 3500.0)], np.float32)
    to_image = cv2.getPerspectiveTransform(cardinals, np.array(view, np.float32))
    frame = np.full((1080, 1920, 3), (60, 120, 70), np.uint8)
    grass = np.full((1080, 1920), 255, np.uint8)
    grass[:110] = 0

    def paint(points):
        pts = cv2.perspectiveTransform(np.array(points, np.float32).reshape(-1, 1, 2), to_image).reshape(-1, 2)
        for a, b in zip(pts[:-1], pts[1:]):
            _paint_ridge(frame, a, b, 185)

    steps = np.linspace(0, 1, 120)
    # A 105 x 68 m pitch around the centre: every mark the camera could see is painted.
    paint([(6000.0 + 915.0 * np.cos(a), 3500.0 + 915.0 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 120)])
    paint([(6000.0, 100.0 + 6800.0 * t) for t in steps])
    for y in (100.0, 6900.0):
        paint([(750.0 + 10500.0 * t, y) for t in steps])
    for goal, inward in ((750.0, 1.0), (11250.0, -1.0)):
        front = goal + inward * 1650.0
        paint([(goal, 100.0 + 6800.0 * t) for t in steps])
        paint([(front, 1484.0 + 4032.0 * t) for t in steps])
        for y in (1484.0, 5516.0):
            paint([(goal + inward * 1650.0 * t, y) for t in steps])
        small = goal + inward * 550.0
        paint([(small, 2584.0 + 1832.0 * t) for t in steps])
        for y in (2584.0, 4416.0):
            paint([(goal + inward * 550.0 * t, y) for t in steps])
        spot = goal + inward * 1100.0
        reach = np.arccos(550.0 / 915.0)
        paint([(spot + inward * 915.0 * np.cos(a), 3500.0 + 915.0 * np.sin(a)) for a in np.linspace(-reach, reach, 40)])
    return frame, grass, to_image


def _centre_error(report, to_image):
    checks = np.array([(6000.0, 3500.0), (5085.0, 3500.0), (6915.0, 3500.0), (6000.0, 2585.0), (6000.0, 4415.0)], np.float32)
    truth = cv2.perspectiveTransform(checks.reshape(-1, 1, 2), to_image).reshape(-1, 2)
    found = cv2.perspectiveTransform(
        checks.reshape(-1, 1, 2), np.linalg.inv(report["matrix"]).astype(np.float32)
    ).reshape(-1, 2)
    return float(np.max(np.linalg.norm(found - truth, axis=1)))


def test_the_centre_circle_is_read_off_the_thin_marks():
    # The whole circle in the picture, the halfway line through it.
    frame, grass, to_image = _broadcast_centre([(980.0, 470.0), (960.0, 690.0), (650.0, 575.0), (1300.0, 575.0)])
    report = fit_centre_on_ink(frame, grass)
    assert_true(report is not None, "the whole circle and the halfway line give a matrix")
    assert_true(report["region"] == "centre" and report["projective"], "it is a true perspective of the centre")
    error = _centre_error(report, to_image)
    assert_true(error < 6.0, f"the centre and the four ends of the circle land on the paint ({error:.1f} px)")
    middle = cv2.perspectiveTransform(np.array([[[960.0, 575.0]]], np.float32), report["matrix"].astype(np.float32))[0, 0]
    assert_true(abs(float(middle[0]) - 6000.0) < 60.0, "the image middle maps to the halfway line")


def test_a_circle_cut_by_the_picture_edge_still_gives_the_centre():
    # The camera has panned: the right half of the circle is off the picture.
    frame, grass, to_image = _broadcast_centre([(1700.0, 300.0), (1650.0, 520.0), (1290.0, 400.0), (2200.0, 420.0)])
    report = fit_centre_on_ink(frame, grass)
    assert_true(report is not None, "the cut circle and the halfway line still give a matrix")
    error = _centre_error(report, to_image)
    # The circle is 900 px wide here; 12 px is under 1.5 % of it.
    assert_true(error < 12.0, f"the cut circle lands on the paint ({error:.1f} px)")


def test_grass_without_the_centre_marks_gives_no_centre():
    frame = np.full((1080, 1920, 3), (60, 120, 70), np.uint8)
    grass = np.full((1080, 1920), 255, np.uint8)
    _paint_ridge(frame, (100.0, 300.0), (1800.0, 260.0), 185)
    assert_true(fit_centre_on_ink(frame, grass) is None, "a lone straight line is not a centre")


def test_the_radar_keeps_real_distances_from_the_mark_it_was_read_from():
    # Centre calibration: the centre spot is the middle of the 105 x 68 m radar.
    assert_true(to_radar(6000.0, 3500.0, "centre", True) == (RADAR_LENGTH_CM / 2.0, RADAR_WIDTH_CM / 2.0), "centre spot")
    # Penalty calibration: the goal line is the end of the radar, at either end.
    x, y = to_radar(12000.0, 3500.0, "penalty", True)
    assert_true(abs(x - RADAR_LENGTH_CM) < 1e-6 and abs(y - RADAR_WIDTH_CM / 2.0) < 1e-6, "high-x goal")
    x, _y = to_radar(10350.0, 3500.0, "penalty", True)
    assert_true(abs(x - (RADAR_LENGTH_CM - 1650.0)) < 1e-6, "16.5 m line stays 16.5 m from the goal")
    x, _y = to_radar(1650.0, 3500.0, "penalty", False)
    assert_true(abs(x - 1650.0) < 1e-6, "low-x 16.5 m line")


def test_the_radar_puts_a_player_where_the_centre_matrix_sees_him():
    frame, grass, to_image = _broadcast_centre([(980.0, 470.0), (960.0, 690.0), (650.0, 575.0), (1300.0, 575.0)])
    report = fit_centre_on_ink(frame, grass)
    pitch_map = VisiblePitchMap()
    pitch_map.matrix = report["matrix"]
    pitch_map.transformer = ViewTransformer.from_matrix(report["matrix"])
    pitch_map.positive_x = True
    pitch_map.region = "centre"
    pitch_map.projective = True
    pitch_map.grass = grass
    # A player 10 m into the high-x half and 5 m toward the far touchline.
    foot = cv2.perspectiveTransform(np.array([[[7000.0, 3000.0]]], np.float32), to_image)[0, 0]
    radar = pitch_map.radar_points([foot])[0]
    assert_true(radar is not None, "a player on the grass is on the radar")
    expect = (RADAR_LENGTH_CM / 2.0 + 1000.0, RADAR_WIDTH_CM / 2.0 - 500.0)
    miss = float(np.hypot(radar[0] - expect[0], radar[1] - expect[1]))
    assert_true(miss < 150.0, f"he is 10 m past halfway and 5 m off the middle ({miss:.0f} cm off)")
    blank = np.zeros((1080, 1920, 3), np.uint8)
    draw_radar(blank, [{"x": radar[0], "y": radar[1], "role": "Player", "team_id": 0, "track_id": 7}], radar, pitch_map.radar_view(blank.shape))
    corner = blank[700:, 1300:]
    assert_true(int(np.count_nonzero(corner)) > 0 and int(np.count_nonzero(blank[:300, :600])) == 0, "drawn in the bottom-right corner only")

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


def test_the_camera_on_the_halfway_line_follows_a_pan():
    import pitch_camera as camera_module

    width, height = 1280, 720
    truth = camera_module.camera_centre(4200.0, 1800.0)
    grass = np.full((height, width), 255, dtype=np.uint8)
    camera = camera_module.BroadcastCamera(calibration_frames=6)
    published = []
    for step in range(14):
        # The camera pans from the middle toward the high-x penalty area.
        target = np.array([5250.0 + 260.0 * step, 3000.0, 0.0])
        pose = np.array([*camera_module._look_at(truth, target), np.log(1500.0)])
        radar_to_image = camera_module._homography(pose, truth, width, height)
        ink = np.zeros((height, width), dtype=np.uint8)
        for x, y in camera_module._project(radar_to_image, camera_module._TEMPLATE):
            if np.isfinite(x) and 0 <= x < width and 0 <= y < height:
                cv2.circle(ink, (int(x), int(y)), 2, 255, -1)
        # The map fit of the frame: right on the box, a few per cent off on the far side.
        skew = np.array([[1.0, 0.03, 0.0], [0.0, 1.02, 0.0], [0.0, 0.0, 1.0]])
        shift = camera_module._shift("penalty", True)
        fit = camera_module._radar_to_field(shift) @ np.linalg.inv(skew @ radar_to_image)
        matrix = camera.solve(fit, "penalty", True, True, ink, grass, (height, width))
        published.append((matrix, radar_to_image, shift, fit))
    centre = camera.centre
    assert_true(centre is not None, "the camera centre is measured on the first frames")
    assert_true(abs(centre[0] - camera_module.HALFWAY_X_CM) < 1.0, "the camera stays on the halfway line")
    assert_true(abs(centre[1] - truth[1]) < 900.0 and abs(centre[2] - truth[2]) < 500.0, "the stand is found")
    matrix, radar_to_image, shift, fit = published[-1]
    assert_true(matrix is not None, "a frame after the calibration is published from the camera")
    far = np.array([[7000.0, 0.0], [9000.0, 0.0], [10500.0, 3400.0]])
    image = camera_module._project(radar_to_image, far).reshape(-1, 1, 2)

    def error(image_to_field):
        field = cv2.perspectiveTransform(image, image_to_field).reshape(-1, 2)
        return float(np.linalg.norm(field + shift - far, axis=1).max())

    assert_true(error(matrix) < 250.0, "the far touchline lands within 2.5 m")
    assert_true(error(matrix) < 0.5 * error(fit), "the camera beats the skewed fit on the far side")


def _dot(track, x, y, team=0, role="Player"):
    return {"x": x, "y": y, "role": role, "team_id": team, "track_id": track}


def test_a_missed_player_keeps_his_dot_on_his_last_spot():
    memory = RadarMemory(hold_frames=60)
    memory.update(0, [_dot(7, 3000.0, 2000.0), _dot(8, 6000.0, 4000.0, team=1)])
    held = memory.update(1, [_dot(8, 6050.0, 4000.0, team=1)])
    assert_true(len(held) == 2, "the missed player is still on the radar")
    seven = [dot for dot in held if dot["track_id"] == 7][0]
    assert_true((seven["x"], seven["y"]) == (3000.0, 2000.0) and seven["held"], "his dot stays where he was last seen")
    back = memory.update(5, [_dot(7, 3400.0, 2100.0), _dot(8, 6100.0, 4000.0, team=1)])
    seven = [dot for dot in back if dot["track_id"] == 7][0]
    assert_true((seven["x"], seven["y"]) == (3400.0, 2100.0) and not seven["held"], "the dot moves when he is seen again")
    for frame in range(6, 66):
        dots = memory.update(frame, [_dot(8, 6100.0, 4000.0, team=1)])
    assert_true(len(dots) == 2, "the dot is held for the 60 frames")
    dots = memory.update(66, [_dot(8, 6100.0, 4000.0, team=1)])
    assert_true(len(dots) == 1, "after 60 frames without him the dot goes")


def test_a_new_id_near_a_lost_dot_continues_it():
    memory = RadarMemory(hold_frames=60)
    memory.update(0, [_dot(7, 3000.0, 2000.0), _dot(9, 8000.0, 2000.0), _dot(8, 3300.0, 2000.0, team=1)])
    memory.update(1, [_dot(9, 8000.0, 2000.0), _dot(8, 3300.0, 2000.0, team=1)])
    dots = memory.update(4, [_dot(21, 3150.0, 2050.0), _dot(9, 8000.0, 2000.0), _dot(8, 3300.0, 2000.0, team=1)])
    assert_true(len(dots) == 3, "the new ID takes over the lost dot: no double")
    assert_true(memory.relinked_ids == 1, "one ID relinked")
    assert_true(7 in [dot["track_id"] for dot in dots], "the dot keeps the player's first number")
    # Another lost dot of the other team nearby is never taken.
    memory.update(5, [_dot(21, 3150.0, 2050.0), _dot(9, 8000.0, 2000.0)])
    dots = memory.update(6, [_dot(21, 3150.0, 2050.0), _dot(9, 8000.0, 2000.0), _dot(30, 3320.0, 2000.0, team=0)])
    assert_true(len(dots) == 4, "a new ID of the other team keeps its own dot")


def test_two_lost_dots_about_as_close_are_not_guessed():
    memory = RadarMemory(hold_frames=60)
    memory.update(0, [_dot(1, 3000.0, 2000.0), _dot(2, 3400.0, 2000.0)])
    memory.update(1, [])
    dots = memory.update(2, [_dot(5, 3200.0, 2000.0)])
    assert_true(len(dots) == 3 and memory.relinked_ids == 0, "an ambiguous new ID gets its own dot")


def main():
    cv2.setRNGSeed(0)
    test_scale_jump_is_rejected_and_the_third_is_35_m()
    test_map_coordinates_do_not_change_possession_distance()
    test_heatmap_reads_the_map_channel()
    test_projection_drops_the_far_half()
    test_marker_colours_follow_the_team()
    test_the_centre_circle_publishes_the_middle()
    test_uneven_corners_move_the_width_and_keep_the_arc_scale()
    test_the_side_ridge_sets_the_width_and_keeps_the_depth()
    test_a_board_is_not_a_side_and_one_corner_draws_one_side()
    test_the_side_ends_where_it_meets_the_goal_line()
    test_the_other_corner_copies_the_angle_to_the_picture_edge()
    test_an_open_camera_reads_the_box_from_its_painted_corners()
    test_without_the_goal_line_the_arc_closes_the_box()
    test_the_centre_circle_is_read_off_the_thin_marks()
    test_a_circle_cut_by_the_picture_edge_still_gives_the_centre()
    test_grass_without_the_centre_marks_gives_no_centre()
    test_the_radar_keeps_real_distances_from_the_mark_it_was_read_from()
    test_the_radar_puts_a_player_where_the_centre_matrix_sees_him()
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
    test_the_camera_on_the_halfway_line_follows_a_pan()
    test_a_missed_player_keeps_his_dot_on_his_last_spot()
    test_a_new_id_near_a_lost_dot_continues_it()
    test_two_lost_dots_about_as_close_are_not_guessed()
    print("pitch map tests passed")


if __name__ == "__main__":
    main()
