#!/usr/bin/env python3
"""The player-detector ball class only continues an open track."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from run_yolo_analysis import select_player_ball_fallback  # noqa: E402


def box(confidence, x, y=100.0):
    return {"confidence": confidence, "box": [x, y, x + 16.0, y + 16.0]}


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_near_ball_continues_the_track():
    chosen, decision = select_player_ball_fallback(
        [box(0.40, 92.0)],
        predicted_center=(100.0, 108.0),
        last_valid_detection=box(0.5, 92.0),
    )
    assert_true(decision["used"], "nearby ball is used")
    assert_true(chosen["confidence"] == 0.40, "kept confidence")
    assert_true(decision["distance"] <= 80.0, "within the gate")


def test_sideline_ball_is_rejected():
    chosen, decision = select_player_ball_fallback(
        [box(0.75, 1176.0, 606.0)],
        predicted_center=(336.0, 482.0),
        last_valid_detection=box(0.6, 328.0, 474.0),
    )
    assert_true(chosen is None, "sideline ball is ignored")
    assert_true(decision["rejected_distance"], "distance gate")
    assert_true(not decision["used"], "not used")


def test_fallback_cannot_open_a_track():
    chosen, decision = select_player_ball_fallback(
        [box(0.82, 400.0)],
        predicted_center=None,
        last_valid_detection=None,
    )
    assert_true(chosen is None, "no new track")
    assert_true(decision["rejected_distance"], "no prediction is not a continuation")


def test_weak_nearby_ball_is_rejected():
    chosen, decision = select_player_ball_fallback(
        [box(0.21, 92.0)],
        predicted_center=(100.0, 108.0),
        last_valid_detection=box(0.5, 92.0),
    )
    assert_true(chosen is None, "below 0.25")
    assert_true(decision["rejected_confidence"], "confidence gate")


def test_near_ball_wins_over_a_sideline_ball():
    chosen, decision = select_player_ball_fallback(
        [box(0.90, 1176.0, 606.0), box(0.31, 92.0)],
        predicted_center=(100.0, 108.0),
        last_valid_detection=box(0.5, 92.0),
    )
    assert_true(decision["used"], "continuation wins")
    assert_true(chosen["confidence"] == 0.31, "the near ball")


def test_no_candidate_is_not_a_rejection_reason():
    chosen, decision = select_player_ball_fallback(
        [],
        predicted_center=(100.0, 108.0),
        last_valid_detection=box(0.5, 92.0),
    )
    assert_true(chosen is None, "nothing to use")
    assert_true(not decision["rejected_distance"], "no distance reject")
    assert_true(not decision["rejected_confidence"], "no confidence reject")


if __name__ == "__main__":
    test_near_ball_continues_the_track()
    test_sideline_ball_is_rejected()
    test_fallback_cannot_open_a_track()
    test_weak_nearby_ball_is_rejected()
    test_near_ball_wins_over_a_sideline_ball()
    test_no_candidate_is_not_a_rejection_reason()
    print("player ball fallback tests passed")
