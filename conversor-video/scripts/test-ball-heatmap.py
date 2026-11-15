#!/usr/bin/env python3
"""The ball heatmap counts the observed ball on the 2D radar pitch, one metre per cell,
and fills a short gap with estimated balls on the straight line between two sightings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from ball_heatmap import BallHeatmapAccumulator  # noqa: E402


def assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def ball(source=None):
    detection = {"box": [0, 0, 4, 4], "confidence": 0.8}
    if source is not None:
        detection["source"] = source
    return [detection]


def test_observed_ball_lands_in_its_metre():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(1, ball(), (550.0, 420.0))
    heatmap.observe(2, ball(), (5260.0, 3410.0))
    report = heatmap.as_dict()
    assert_equal(report["coordinates"], "radar", "radar pitch")
    assert_equal(report["columns"], 105, "one column per metre of 105 m")
    assert_equal(report["rows"], 68, "one row per metre of 68 m")
    assert_equal(report["cells"][4][5], 1, "5,5 m by 4,2 m")
    assert_equal(report["cells"][34][52], 1, "the centre spot")
    assert_equal(report["seen_points"], 2, "two sightings")
    assert_equal(report["estimated_points"], 0, "consecutive frames need no estimate")


def test_a_short_gap_is_filled_on_the_straight_line():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(10, ball(), (3000.0, 1000.0))
    for frame in range(11, 15):
        heatmap.observe(frame, [], None)
    heatmap.observe(15, ball(), (3500.0, 1300.0))
    report = heatmap.as_dict()
    assert_equal(report["seen_points"], 2, "two real sightings")
    assert_equal(report["estimated_points"], 4, "one estimate for each lost frame")
    assert_equal(report["observed_points"], 6, "all of them are painted")
    # Evenly spaced: 3100,1060 / 3200,1120 / 3300,1180 / 3400,1240.
    for x, y in ((31, 10), (32, 11), (33, 11), (34, 12)):
        assert_equal(report["cells"][y][x], 1, f"estimate at {x} m, {y} m")


def test_a_long_gap_or_an_impossible_jump_is_not_filled():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(1, ball(), (1000.0, 1000.0))
    heatmap.observe(80, ball(), (2000.0, 1000.0))
    assert_equal(heatmap.as_dict()["estimated_points"], 0, "a gap over 45 frames stays empty")
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(1, ball(), (500.0, 500.0))
    heatmap.observe(4, ball(), (9500.0, 6000.0))
    assert_equal(heatmap.as_dict()["estimated_points"], 0, "90 m in 3 frames is not a ball")


def test_missing_and_predicted_ball_are_not_counted():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(1, [], None)
    heatmap.observe(2, ball("predicted"), (5000.0, 3000.0))
    heatmap.observe(3, ball("remembered"), (5000.0, 3000.0))
    report = heatmap.as_dict()
    assert_equal(report["frames_total"], 3, "every frame is seen")
    assert_equal(report["frames_observed_ball"], 0, "no observed ball")
    assert_equal(report["observed_points"], 0, "nothing painted")


def test_ball_without_map_or_off_the_pitch_is_kept_off_the_grid():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(1, ball(), None)
    heatmap.observe(2, ball(), (float("nan"), 3000.0))
    heatmap.observe(3, ball(), (-50.0, -20.0))
    heatmap.observe(4, ball(), (5000.0, 9000.0))
    report = heatmap.as_dict()
    assert_equal(report["frames_without_field_point"], 2, "no map point")
    assert_equal(report["frames_outside_field"], 1, "9 m past the touchline")
    assert_equal(report["cells"][0][0], 1, "touchline jitter is clamped to the corner")


def test_report_is_a_copy():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(1, ball(), (500.0, 400.0))
    report = heatmap.as_dict()
    report["cells"][4][5] = 99
    assert_equal(heatmap.as_dict()["cells"][4][5], 1, "the live grid is not shared")


if __name__ == "__main__":
    test_observed_ball_lands_in_its_metre()
    test_a_short_gap_is_filled_on_the_straight_line()
    test_a_long_gap_or_an_impossible_jump_is_not_filled()
    test_missing_and_predicted_ball_are_not_counted()
    test_ball_without_map_or_off_the_pitch_is_kept_off_the_grid()
    test_report_is_a_copy()
    print("ball heatmap tests passed")
