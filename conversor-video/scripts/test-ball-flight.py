#!/usr/bin/env python3
"""A pass is followed from the release point after the ball track dies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from ball_flight import BallFlightPath  # noqa: E402


def ball(confidence, x, y):
    return {"confidence": confidence, "box": [x - 6.0, y - 6.0, x + 6.0, y + 6.0]}


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


# Player 15 releases the ball at frame 18 of the 401-frame clip. These are the
# boxes the detector emitted on the way to player 11, plus the white player's
# knee that the old recovery took at frame 47.
RELEASE = (18, 712.0, 248.0)
FLIGHT = [
    (21, 646.0, 248.0, 0.05),
    (22, 646.0, 248.0, 0.06),
    (25, 608.0, 249.0, 0.47),
    (29, 550.0, 256.0, 0.26),
    (31, 515.0, 262.0, 0.51),
    (37, 433.0, 281.0, 0.08),
    (39, 409.0, 279.0, 0.30),
    (40, 396.0, 280.0, 0.09),
    (43, 375.0, 282.0, 0.23),
    (44, 353.0, 289.0, 0.19),
    (47, 333.0, 294.0, 0.30),
]
KNEE = (731.0, 232.0, 0.60)


def released_path():
    path = BallFlightPath()
    frame, x, y = RELEASE
    path.note_published(frame, ball(0.73, x, y)["box"], at_player=True)
    return path


def test_pass_is_followed_to_the_receiver():
    path = released_path()
    published = {}
    for frame, x, y, confidence in FLIGHT:
        chosen = path.continue_flight(frame, [ball(confidence, x, y)])
        if chosen is not None:
            published[frame] = chosen
            path.note_published(frame, chosen["box"], at_player=False)
    assert_true(21 not in published and 22 not in published, "two boxes in place do not open a flight")
    assert_true(25 in published, "the flight is trusted once the ball has moved on")
    assert_true(all(frame in published for frame, *_ in FLIGHT if frame >= 25), "every later box is the ball")


def test_knee_behind_the_release_is_not_the_ball():
    path = released_path()
    for frame, x, y, confidence in FLIGHT[:-1]:
        chosen = path.continue_flight(frame, [ball(confidence, x, y)])
        if chosen is not None:
            path.note_published(frame, chosen["box"], at_player=False)
    knee_x, knee_y, knee_confidence = KNEE
    frame, x, y, confidence = FLIGHT[-1]
    chosen = path.continue_flight(
        frame,
        [ball(knee_confidence, knee_x, knee_y), ball(confidence, x, y)],
    )
    assert_true(chosen is not None, "the flight goes on")
    assert_true(chosen["confidence"] == confidence, "the ball beats the stronger knee box")


def test_knee_alone_cannot_open_a_flight():
    path = released_path()
    knee_x, knee_y, knee_confidence = KNEE
    for frame in (21, 22, 23):
        chosen = path.continue_flight(frame, [ball(knee_confidence, knee_x, knee_y)])
        assert_true(chosen is None, "a box that never moves is not a pass")


def test_false_start_does_not_block_the_real_flight():
    path = released_path()
    knee_x, knee_y, knee_confidence = KNEE
    for frame in (20, 21):
        path.continue_flight(frame, [ball(knee_confidence, knee_x, knee_y)])
    published = []
    for frame, x, y, confidence in FLIGHT[2:6]:
        chosen = path.continue_flight(frame, [ball(confidence, x, y)])
        if chosen is not None:
            published.append(frame)
            path.note_published(frame, chosen["box"], at_player=False)
    assert_true(published == [29, 31, 37], "the pass leaves from the release point, not from the knee")


def test_knee_is_out_of_reach_of_a_confirmed_flight():
    path = released_path()
    for frame, x, y, confidence in FLIGHT[:-1]:
        chosen = path.continue_flight(frame, [ball(confidence, x, y)])
        if chosen is not None:
            path.note_published(frame, chosen["box"], at_player=False)
    knee_x, knee_y, knee_confidence = KNEE
    assert_true(
        not path.reachable(47, ball(knee_confidence, knee_x, knee_y)["box"]),
        "the ball cannot fly back 380 px in three frames",
    )
    assert_true(path.reachable(47, ball(0.3, 333.0, 294.0)["box"]), "the next point of the flight is in reach")


def test_any_box_is_reachable_without_a_flight():
    path = released_path()
    assert_true(path.reachable(19, ball(0.6, 300.0, 600.0)["box"]), "only a flight limits the ball")


def test_far_box_after_a_gap_cannot_open_a_flight():
    path = BallFlightPath()
    path.note_published(28, ball(0.5, 296.0, 298.0)["box"], at_player=True)
    for frame, x in ((37, 824.0), (38, 831.0), (39, 831.0), (41, 839.0)):
        chosen = path.continue_flight(frame, [ball(0.3, x, 219.0)])
        assert_true(chosen is None, "a box 528 px away nine frames later is not the pass")


def test_faint_boxes_alone_do_not_confirm_a_flight():
    path = released_path()
    for frame, x, y, _confidence in FLIGHT[:6]:
        chosen = path.continue_flight(frame, [ball(0.03, x, y)])
        assert_true(chosen is None, "faint boxes may be grass marks")


def test_out_of_reach_only_right_after_the_flight():
    path = released_path()
    for frame, x, y, confidence in FLIGHT[:5]:
        chosen = path.continue_flight(frame, [ball(confidence, x, y)])
        if chosen is not None:
            path.note_published(frame, chosen["box"], at_player=False, confidence=confidence)
    far = ball(0.8, 1100.0, 500.0)["box"]
    assert_true(not path.reachable(32, far), "the next frame cannot be across the pitch")
    assert_true(path.reachable(36, far), "after a few missed frames the flight may have been lost")


def test_box_that_turns_back_breaks_the_flight():
    path = released_path()
    for frame, x, y, confidence in FLIGHT[:3]:
        path.continue_flight(frame, [ball(confidence, x, y)])
    chosen = path.continue_flight(26, [ball(0.6, 680.0, 249.0)])
    assert_true(chosen is None, "the ball does not fly back to the passer")


def test_box_off_the_line_is_not_the_ball():
    path = released_path()
    for frame, x, y, confidence in FLIGHT[:3]:
        path.continue_flight(frame, [ball(confidence, x, y)])
    chosen = path.continue_flight(27, [ball(0.6, 580.0, 330.0)])
    assert_true(chosen is None, "a box far off the line is another object")


def test_teleport_is_refused():
    path = released_path()
    chosen = path.continue_flight(19, [ball(0.9, 600.0, 248.0)])
    assert_true(chosen is None, "the ball cannot move 112 px in one frame")


def test_path_is_forgotten_after_a_long_gap():
    path = released_path()
    for frame, x, y, confidence in FLIGHT[:3]:
        path.continue_flight(frame, [ball(confidence, x, y)])
    chosen = path.continue_flight(60, [ball(0.6, 560.0, 252.0)])
    assert_true(chosen is None, "a box long after the last one does not continue it")
    assert_true(path.points == [], "the stale path is dropped")


def test_ball_at_a_player_is_the_new_release():
    path = released_path()
    for frame, x, y, confidence in FLIGHT[:4]:
        chosen = path.continue_flight(frame, [ball(confidence, x, y)])
        if chosen is not None:
            path.note_published(frame, chosen["box"], at_player=False)
    path.note_published(30, ball(0.7, 530.0, 258.0)["box"], at_player=True)
    assert_true(len(path.points) == 1, "the receiver starts the next path")
    assert_true(not path.confirmed, "the next flight must prove itself again")


if __name__ == "__main__":
    test_pass_is_followed_to_the_receiver()
    test_knee_behind_the_release_is_not_the_ball()
    test_knee_alone_cannot_open_a_flight()
    test_false_start_does_not_block_the_real_flight()
    test_knee_is_out_of_reach_of_a_confirmed_flight()
    test_any_box_is_reachable_without_a_flight()
    test_far_box_after_a_gap_cannot_open_a_flight()
    test_faint_boxes_alone_do_not_confirm_a_flight()
    test_out_of_reach_only_right_after_the_flight()
    test_box_that_turns_back_breaks_the_flight()
    test_box_off_the_line_is_not_the_ball()
    test_teleport_is_refused()
    test_path_is_forgotten_after_a_long_gap()
    test_ball_at_a_player_is_the_new_release()
    print("ball flight tests passed")
