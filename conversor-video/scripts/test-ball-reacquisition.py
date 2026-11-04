#!/usr/bin/env python3
"""A lost ball is recovered from the last touch, not from the old prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from run_yolo_analysis import (  # noqa: E402
    BALL_TRACK_LOST,
    BALL_TRACK_OBSERVED,
    BALL_TRACK_RECOVERED,
    BALL_TRACK_RECOVERING,
    ball_track_state_for_gap,
    note_rejected_ball_center,
    select_ball_reacquisition,
)


def box(confidence, x, y, size=10.0):
    return {"confidence": confidence, "box": [x, y, x + size, y + size]}


def center_box(confidence, x, y, size=10.0):
    half = size / 2.0
    return {"confidence": confidence, "box": [x - half, y - half, x + half, y + half]}


def person(x, y):
    return {"box": [x - 10.0, y - 40.0, x + 10.0, y]}


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_states_follow_the_gap():
    assert_true(
        ball_track_state_for_gap(0, None, True) == BALL_TRACK_OBSERVED,
        "a fresh ball is observed",
    )
    assert_true(
        ball_track_state_for_gap(5, None, True) == BALL_TRACK_OBSERVED,
        "five missing frames stay in memory",
    )
    assert_true(
        ball_track_state_for_gap(6, None, True) == BALL_TRACK_LOST,
        "the sixth missing frame is lost",
    )
    assert_true(
        ball_track_state_for_gap(6, {"center": (1.0, 1.0), "frame": 5}, True) == BALL_TRACK_RECOVERING,
        "a pending box is recovering",
    )
    assert_true(
        ball_track_state_for_gap(0, None, False) == BALL_TRACK_LOST,
        "no touch yet is lost",
    )


def test_frame_47_ball_is_recovered_away_from_the_prediction():
    # The real ball at frame 47 sits on the opening touch and 381 px from the
    # prediction that walked into the grass.
    chosen = select_ball_reacquisition(
        [center_box(0.60, 731.0, 232.0)],
        [person(700.0, 280.0)],
        last_touch_center=(712.0, 248.0),
        last_touch_area=100.0,
        rejected_centers=[(32, 480.3, 272.2)],
        frame_number=47,
        missing_frames=15,
        pending=None,
        predicted_center=(350.0, 290.0),
    )
    assert_true(chosen["used"], "frame 47 ball is recovered")
    assert_true(chosen["detection"]["confidence"] == 0.60, "kept the real box")
    assert_true(chosen["restart_history"], "the grass velocity is discarded")
    assert_true(chosen["pending"] is None, "one frame is enough at the last touch")


def test_grass_box_following_the_prediction_is_rejected():
    chosen = select_ball_reacquisition(
        [center_box(0.62, 480.0, 272.0)],
        [person(712.0, 248.0)],
        last_touch_center=(712.0, 248.0),
        last_touch_area=100.0,
        rejected_centers=[],
        frame_number=32,
        missing_frames=8,
        pending=None,
        predicted_center=(497.0, 265.0),
    )
    assert_true(not chosen["used"], "grass continuation is not the ball")
    assert_true(chosen["detection"] is None, "nothing published")
    assert_true(len(chosen["rejected"]) == 1, "the grass center is recorded")


def test_rejected_grass_center_cannot_return():
    rejected = note_rejected_ball_center([], 32, (480.0, 272.0))
    chosen = select_ball_reacquisition(
        [center_box(0.80, 490.0, 276.0)],
        [person(500.0, 290.0)],
        last_touch_center=(712.0, 248.0),
        last_touch_area=100.0,
        rejected_centers=rejected,
        frame_number=33,
        missing_frames=9,
        pending=None,
        predicted_center=None,
    )
    assert_true(not chosen["used"], "a box on the rejected grass trail stays out")


def test_lost_ball_at_a_players_feet_is_recovered():
    chosen = select_ball_reacquisition(
        [center_box(0.40, 400.0, 280.0)],
        [person(410.0, 300.0)],
        last_touch_center=(712.0, 248.0),
        last_touch_area=100.0,
        rejected_centers=[],
        frame_number=56,
        missing_frames=12,
        pending=None,
        predicted_center=None,
    )
    assert_true(chosen["used"], "a pass that reaches a player is recovered")
    assert_true(chosen["restart_history"], "lost track restarts here")


def test_feet_rule_waits_until_the_track_is_lost():
    chosen = select_ball_reacquisition(
        [center_box(0.40, 400.0, 280.0)],
        [person(410.0, 300.0)],
        last_touch_center=(712.0, 248.0),
        last_touch_area=100.0,
        rejected_centers=[],
        frame_number=30,
        missing_frames=3,
        pending=None,
        predicted_center=(700.0, 250.0),
    )
    assert_true(not chosen["used"], "during memory the old track still owns the ball")


def test_airborne_ball_needs_a_second_frame():
    first = select_ball_reacquisition(
        [center_box(0.61, 500.0, 260.0)],
        [person(712.0, 248.0)],
        last_touch_center=(712.0, 248.0),
        last_touch_area=100.0,
        rejected_centers=[],
        frame_number=40,
        missing_frames=10,
        pending=None,
        predicted_center=None,
    )
    assert_true(not first["used"], "the first airborne frame is not published")
    assert_true(first["pending"] is not None, "it waits one frame")
    assert_true(
        ball_track_state_for_gap(10, first["pending"], True) == BALL_TRACK_RECOVERING,
        "the wait is the recovering state",
    )
    second = select_ball_reacquisition(
        [center_box(0.63, 520.0, 266.0)],
        [person(712.0, 248.0)],
        last_touch_center=(712.0, 248.0),
        last_touch_area=100.0,
        rejected_centers=first["rejected"],
        frame_number=41,
        missing_frames=11,
        pending=first["pending"],
        predicted_center=None,
    )
    assert_true(second["used"], "the same place on the next frame is the ball")
    assert_true(second["pending"] is None, "the wait is over")


def test_open_grass_step_is_not_a_touch():
    chosen = select_ball_reacquisition(
        [center_box(0.70, 640.0, 250.0)],
        [person(712.0, 248.0)],
        last_touch_center=(712.0, 248.0),
        last_touch_area=100.0,
        rejected_centers=[],
        frame_number=24,
        missing_frames=6,
        pending=None,
        predicted_center=None,
    )
    assert_true(not chosen["used"], "70 px of open grass is not the last touch")
    assert_true(chosen["pending"] is not None, "it can only be confirmed next frame")


def test_oversized_box_is_rejected():
    chosen = select_ball_reacquisition(
        [center_box(0.90, 731.0, 232.0, size=40.0)],
        [person(712.0, 248.0)],
        last_touch_center=(712.0, 248.0),
        last_touch_area=100.0,
        rejected_centers=[],
        frame_number=47,
        missing_frames=15,
        pending=None,
        predicted_center=None,
    )
    assert_true(not chosen["used"], "a person-sized box is not the ball")


def test_window_requires_two_frames_at_the_feet():
    first = select_ball_reacquisition(
        [center_box(0.40, 400.0, 280.0)],
        [person(410.0, 300.0)],
        last_touch_center=(712.0, 248.0),
        last_touch_area=100.0,
        rejected_centers=[],
        frame_number=80,
        missing_frames=60,
        pending=None,
        predicted_center=None,
    )
    assert_true(not first["used"], "after the window the first foot box waits")
    second = select_ball_reacquisition(
        [center_box(0.42, 408.0, 284.0)],
        [person(410.0, 300.0)],
        last_touch_center=(712.0, 248.0),
        last_touch_area=100.0,
        rejected_centers=first["rejected"],
        frame_number=81,
        missing_frames=61,
        pending=first["pending"],
        predicted_center=None,
    )
    assert_true(second["used"], "the second frame at the feet is accepted")


if __name__ == "__main__":
    test_states_follow_the_gap()
    test_frame_47_ball_is_recovered_away_from_the_prediction()
    test_grass_box_following_the_prediction_is_rejected()
    test_rejected_grass_center_cannot_return()
    test_lost_ball_at_a_players_feet_is_recovered()
    test_feet_rule_waits_until_the_track_is_lost()
    test_airborne_ball_needs_a_second_frame()
    test_open_grass_step_is_not_a_touch()
    test_oversized_box_is_rejected()
    test_window_requires_two_frames_at_the_feet()
    print("ball reacquisition tests passed")
