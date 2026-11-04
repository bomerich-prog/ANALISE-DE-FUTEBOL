#!/usr/bin/env python3
"""The player-detector ball class only continues an open track."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from run_yolo_analysis import (  # noqa: E402
    ball_selection_keeps_player_support,
    nearest_supported_ball,
    select_player_ball_fallback,
)


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


def person(x, y):
    return {"box": [x - 10.0, y - 40.0, x + 10.0, y]}


def test_fallback_in_open_grass_is_rejected():
    chosen, decision = select_player_ball_fallback(
        [box(0.44, 480.0, 272.0)],
        predicted_center=(497.0, 265.0),
        last_valid_detection=box(0.47, 515.0, 262.0),
        person_detections=[person(700.0, 250.0)],
    )
    assert_true(chosen is None, "grass box is not a continuation")
    assert_true(decision["rejected_distance"], "no player near the box")
    assert_true(not decision["used"], "not used")


def test_fallback_next_to_a_player_continues():
    chosen, decision = select_player_ball_fallback(
        [box(0.44, 690.0, 240.0)],
        predicted_center=(700.0, 248.0),
        last_valid_detection=box(0.7, 700.0, 240.0),
        person_detections=[person(700.0, 260.0)],
    )
    assert_true(decision["used"], "ball by a player continues")
    assert_true(chosen is not None, "box kept")


def test_strong_flight_stays_near_the_last_player_ball():
    flying = box(0.55, 690.0, 250.0)
    assert_true(
        ball_selection_keeps_player_support(
            flying,
            predicted_center=(700.0, 248.0),
            person_detections=[person(200.0, 300.0)],
            last_supported_center=(700.0, 250.0),
        ),
        "short flight from the last touch",
    )
    far = box(0.55, 620.0, 250.0)
    assert_true(
        not ball_selection_keeps_player_support(
            far,
            predicted_center=(700.0, 248.0),
            person_detections=[person(200.0, 300.0)],
            last_supported_center=(700.0, 250.0),
        ),
        "a long unsupported jump is not a flight",
    )


def test_grass_walk_is_dropped():
    grass = box(0.50, 333.0, 294.0)
    assert_true(
        not ball_selection_keeps_player_support(
            grass,
            predicted_center=(350.0, 290.0),
            person_detections=[person(712.0, 248.0)],
            last_supported_center=(712.0, 248.0),
        ),
        "stable grass box is not the ball",
    )


def test_new_track_requires_a_player():
    grass = box(0.48, 333.0, 294.0)
    assert_true(
        not ball_selection_keeps_player_support(
            grass,
            predicted_center=None,
            person_detections=[person(712.0, 248.0)],
            last_supported_center=None,
        ),
        "open grass cannot open a track",
    )


def test_supported_ball_beats_the_grass_continuation():
    grass = box(0.62, 270.0, 295.0)
    play = box(0.48, 700.0, 240.0)
    chosen = nearest_supported_ball(
        [grass, play],
        [person(270.0, 310.0), person(710.0, 260.0)],
        last_supported_center=(712.0, 248.0),
    )
    assert_true(chosen is play, "the ball at the last touch wins")


def test_far_supported_ball_is_not_the_play():
    grass = box(0.62, 270.0, 295.0)
    chosen = nearest_supported_ball(
        [grass],
        [person(270.0, 310.0)],
        last_supported_center=(730.0, 233.0),
    )
    assert_true(chosen is None, "a defender across the pitch is not the ball")


def test_pass_to_a_nearby_player_is_kept():
    received = box(0.55, 400.0, 280.0)
    assert_true(
        ball_selection_keeps_player_support(
            received,
            predicted_center=None,
            person_detections=[person(410.0, 300.0)],
            last_supported_center=(712.0, 248.0),
        ),
        "a pass that reaches a player is the ball",
    )


def test_live_grass_track_cannot_reach_another_player():
    arrived = box(0.55, 270.0, 295.0)
    assert_true(
        not ball_selection_keeps_player_support(
            arrived,
            predicted_center=(300.0, 290.0),
            person_detections=[person(270.0, 310.0)],
            last_supported_center=(712.0, 248.0),
        ),
        "a live grass track does not arrive at another player",
    )


def test_far_player_does_not_reopen_the_track():
    grass = box(0.50, 270.0, 295.0)
    assert_true(
        not ball_selection_keeps_player_support(
            grass,
            predicted_center=None,
            person_detections=[person(270.0, 310.0)],
            last_supported_center=(730.0, 233.0),
        ),
        "a defender across the pitch does not take the ball",
    )


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
    test_fallback_in_open_grass_is_rejected()
    test_fallback_next_to_a_player_continues()
    test_strong_flight_stays_near_the_last_player_ball()
    test_grass_walk_is_dropped()
    test_new_track_requires_a_player()
    test_supported_ball_beats_the_grass_continuation()
    test_far_supported_ball_is_not_the_play()
    test_pass_to_a_nearby_player_is_kept()
    test_live_grass_track_cannot_reach_another_player()
    test_far_player_does_not_reopen_the_track()
    test_no_candidate_is_not_a_rejection_reason()
    print("player ball fallback tests passed")
