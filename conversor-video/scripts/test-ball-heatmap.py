#!/usr/bin/env python3
"""The ball heatmap counts only the observed ball, in field coordinates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from ball_heatmap import BallHeatmapAccumulator  # noqa: E402


def assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def ball(field_x, field_y, source=None):
    detection = {"box": [0, 0, 4, 4], "confidence": 0.8, "field_x": field_x, "field_y": field_y}
    if source is not None:
        detection["source"] = source
    return [detection]


def test_observed_ball_lands_in_its_cell():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(ball(500.0, 400.0))
    heatmap.observe(ball(6100.0, 3600.0))
    heatmap.observe(ball(6100.0, 3600.0))
    report = heatmap.as_dict()
    assert_equal(report["columns"], 12, "columns")
    assert_equal(report["rows"], 8, "rows")
    assert_equal(len(report["cells"]), 8, "one list per row")
    assert_equal(len(report["cells"][0]), 12, "one count per column")
    assert_equal(report["cells"][0][0], 1, "5 m by 4 m is the first cell")
    assert_equal(report["cells"][4][6], 2, "61 m by 36 m is just past the centre")
    assert_equal(report["observed_points"], 3, "every point is counted")
    assert_equal(report["max_cell_count"], 2, "busiest cell")
    assert_equal(report["frames_observed_ball"], 3, "three observed frames")


def test_missing_and_predicted_ball_are_not_counted():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe([])
    heatmap.observe(None)
    heatmap.observe(ball(3000.0, 3000.0, source="predicted"))
    heatmap.observe(ball(3000.0, 3000.0, source="remembered"))
    report = heatmap.as_dict()
    assert_equal(report["frames_total"], 4, "every frame is seen")
    assert_equal(report["frames_observed_ball"], 0, "no observed ball")
    assert_equal(report["observed_points"], 0, "nothing reaches the grid")
    assert_equal(sum(sum(row) for row in report["cells"]), 0, "the grid stays empty")


def test_detector_source_names_still_count():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(ball(3000.0, 3000.0, source="player-detector-ball"))
    heatmap.observe(ball(3000.0, 3000.0, source="yolo"))
    assert_equal(heatmap.as_dict()["observed_points"], 2, "a detector name is an observed ball")


def test_ball_without_homography_is_kept_off_the_grid():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(ball(None, None))
    heatmap.observe(ball(float("nan"), 3000.0))
    report = heatmap.as_dict()
    assert_equal(report["frames_observed_ball"], 2, "the ball was seen")
    assert_equal(report["frames_without_field_point"], 2, "but had no field point")
    assert_equal(report["observed_points"], 0, "and is not drawn")


def test_touchline_jitter_is_clamped_and_far_points_are_dropped():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(ball(-50.0, -20.0))
    heatmap.observe(ball(12050.0, 7080.0))
    heatmap.observe(ball(-900.0, 3000.0))
    heatmap.observe(ball(5000.0, 9000.0))
    report = heatmap.as_dict()
    assert_equal(report["cells"][0][0], 1, "50 cm outside the corner is the corner cell")
    assert_equal(report["cells"][7][11], 1, "the far corner is the last cell")
    assert_equal(report["frames_outside_field"], 2, "9 m outside is not on the pitch")
    assert_equal(report["observed_points"], 2, "only on-pitch points are drawn")


def test_report_is_a_copy():
    heatmap = BallHeatmapAccumulator()
    heatmap.observe(ball(500.0, 400.0))
    report = heatmap.as_dict()
    report["cells"][0][0] = 99
    assert_equal(heatmap.as_dict()["cells"][0][0], 1, "the live grid is not shared")


if __name__ == "__main__":
    test_observed_ball_lands_in_its_cell()
    test_missing_and_predicted_ball_are_not_counted()
    test_detector_source_names_still_count()
    test_ball_without_homography_is_kept_off_the_grid()
    test_touchline_jitter_is_clamped_and_far_points_are_dropped()
    test_report_is_a_copy()
    print("ball heatmap tests passed")
