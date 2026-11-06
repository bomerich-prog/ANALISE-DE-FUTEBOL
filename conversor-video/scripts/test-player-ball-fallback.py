#!/usr/bin/env python3
"""The player-detector ball class only continues an open track."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from run_yolo_analysis import (  # noqa: E402
    BALL_FLIGHT_MAX_UNSUPPORTED_FRAMES,
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


def ball_at(confidence, x, y):
    return {"confidence": confidence, "box": [x - 6.0, y - 6.0, x + 6.0, y + 6.0]}


def test_fallback_continues_the_flight():
    # Frames 31 and 32 of the reference clip: the pass from 15 toward 11.
    chosen, decision = select_player_ball_fallback(
        [ball_at(0.44, 480.0, 272.0)],
        predicted_center=(497.0, 265.0),
        last_valid_detection=ball_at(0.47, 515.0, 262.0),
        person_detections=[person(700.0, 250.0)],
    )
    assert_true(decision["used"], "the next step of the pass is the ball")
    assert_true(chosen is not None, "box kept")


def test_fallback_jump_off_the_flight_is_rejected():
    chosen, decision = select_player_ball_fallback(
        [ball_at(0.44, 440.0, 272.0)],
        predicted_center=(497.0, 265.0),
        last_valid_detection=ball_at(0.47, 515.0, 262.0),
        person_detections=[person(700.0, 250.0)],
    )
    assert_true(chosen is None, "a 75 px jump is not one frame of flight")
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


def test_box_far_from_the_touch_with_no_ball_before_it_is_dropped():
    grass = ball_at(0.50, 333.0, 294.0)
    assert_true(
        not ball_selection_keeps_player_support(
            grass,
            predicted_center=(350.0, 290.0),
            person_detections=[person(712.0, 248.0)],
            last_supported_center=(712.0, 248.0),
        ),
        "with no published ball before it, a box in open grass is not the flight",
    )


def test_pass_flight_continues_frame_by_frame():
    # Frames 46 and 47 of the reference clip, 380 px from the passer.
    flying = ball_at(0.24, 333.0, 293.0)
    assert_true(
        ball_selection_keeps_player_support(
            flying,
            predicted_center=(340.0, 291.0),
            person_detections=[person(712.0, 248.0)],
            last_supported_center=(712.0, 248.0),
            last_ball_center=(353.0, 289.0),
        ),
        "a weak box on the predicted path continues the flight",
    )
    strong = ball_at(0.82, 325.0, 292.0)
    assert_true(
        ball_selection_keeps_player_support(
            strong,
            predicted_center=(326.0, 293.0),
            person_detections=[person(712.0, 248.0)],
            last_supported_center=(712.0, 248.0),
            last_ball_center=(333.0, 293.0),
        ),
        "a strong box one step on continues the flight",
    )


def test_flight_off_the_prediction_is_dropped():
    sideways = ball_at(0.82, 345.0, 365.0)
    assert_true(
        not ball_selection_keeps_player_support(
            sideways,
            predicted_center=(326.0, 293.0),
            person_detections=[person(712.0, 248.0)],
            last_supported_center=(712.0, 248.0),
            last_ball_center=(333.0, 293.0),
            frames_since_ball=1,
        ),
        "a box that leaves the predicted path is not the next step",
    )


def test_flight_cannot_outlast_a_pass():
    flying = ball_at(0.82, 325.0, 292.0)
    assert_true(
        not ball_selection_keeps_player_support(
            flying,
            predicted_center=(326.0, 293.0),
            person_detections=[person(712.0, 248.0)],
            last_supported_center=(712.0, 248.0),
            last_ball_center=(333.0, 293.0),
            unsupported_frames=BALL_FLIGHT_MAX_UNSUPPORTED_FRAMES,
        ),
        "four seconds away from every player is not a pass",
    )


def test_flight_reaches_the_receiver():
    arrived = ball_at(0.55, 263.0, 295.0)
    assert_true(
        ball_selection_keeps_player_support(
            arrived,
            predicted_center=(262.0, 296.0),
            person_detections=[person(255.0, 305.0)],
            last_supported_center=(712.0, 248.0),
            last_ball_center=(270.0, 295.0),
        ),
        "the flight ends at the receiver's feet",
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


def test_box_cannot_jump_to_another_player():
    arrived = box(0.55, 270.0, 295.0)
    assert_true(
        not ball_selection_keeps_player_support(
            arrived,
            predicted_center=(300.0, 290.0),
            person_detections=[person(270.0, 310.0)],
            last_supported_center=(712.0, 248.0),
        ),
        "without a flight before it, a box does not arrive at another player",
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
    test_fallback_continues_the_flight()
    test_fallback_jump_off_the_flight_is_rejected()
    test_fallback_next_to_a_player_continues()
    test_strong_flight_stays_near_the_last_player_ball()
    test_box_far_from_the_touch_with_no_ball_before_it_is_dropped()
    test_pass_flight_continues_frame_by_frame()
    test_flight_off_the_prediction_is_dropped()
    test_flight_cannot_outlast_a_pass()
    test_flight_reaches_the_receiver()
    test_new_track_requires_a_player()
    test_supported_ball_beats_the_grass_continuation()
    test_far_supported_ball_is_not_the_play()
    test_pass_to_a_nearby_player_is_kept()
    test_box_cannot_jump_to_another_player()
    test_far_player_does_not_reopen_the_track()
    test_no_candidate_is_not_a_rejection_reason()
    print("player ball fallback tests passed")
