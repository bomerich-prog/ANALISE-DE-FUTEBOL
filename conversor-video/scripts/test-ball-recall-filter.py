#!/usr/bin/env python3
"""Checks that a weak ball box is kept only beside the current track."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from run_yolo_analysis import keep_recoverable_ball_candidates  # noqa: E402


def box(confidence, x):
    return {"confidence": confidence, "box": [x, 100.0, x + 16.0, 116.0]}


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def test_strong_box_hides_distant_weak_box():
    kept = keep_recoverable_ball_candidates(
        [box(0.56, 1200.0), box(0.17, 940.0)],
        predicted_center=(1208.0, 108.0),
    )
    assert_equal(len(kept), 1, "only the strong ball")
    assert_equal(kept[0]["confidence"], 0.56, "strong confidence")


def test_weak_box_continues_the_track():
    kept = keep_recoverable_ball_candidates(
        [box(0.18, 1200.0), box(0.17, 940.0)],
        predicted_center=(1208.0, 108.0),
    )
    assert_equal(len(kept), 1, "nearby weak ball")
    assert_equal(kept[0]["confidence"], 0.18, "recovered confidence")


def test_distant_weak_box_without_strong_ball_is_dropped():
    kept = keep_recoverable_ball_candidates(
        [box(0.17, 940.0)],
        predicted_center=(1208.0, 108.0),
    )
    assert_equal(kept, [], "distant false candidate")


def test_weak_box_cannot_open_a_track():
    kept = keep_recoverable_ball_candidates(
        [box(0.22, 400.0)],
        predicted_center=None,
    )
    assert_equal(kept, [], "no acquisition below 0.25")


def test_nearby_weak_box_beats_distant_strong_box():
    kept = keep_recoverable_ball_candidates(
        [box(0.70, 400.0), box(0.04, 1200.0)],
        predicted_center=(1208.0, 108.0),
    )
    assert_equal(len(kept), 1, "continuation stays")
    assert_equal(kept[0]["confidence"], 0.04, "weak continuation")


def test_nearby_strong_box_beats_weaker_continuation():
    kept = keep_recoverable_ball_candidates(
        [box(0.60, 1100.0), box(0.04, 1200.0)],
        predicted_center=(1208.0, 108.0),
    )
    assert_equal(len(kept), 1, "nearby strong ball")
    assert_equal(kept[0]["confidence"], 0.60, "strong confidence")


if __name__ == "__main__":
    test_strong_box_hides_distant_weak_box()
    test_weak_box_continues_the_track()
    test_distant_weak_box_without_strong_ball_is_dropped()
    test_weak_box_cannot_open_a_track()
    test_nearby_weak_box_beats_distant_strong_box()
    test_nearby_strong_box_beats_weaker_continuation()
    print("ball recall filter tests passed")
