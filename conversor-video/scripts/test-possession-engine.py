#!/usr/bin/env python3
"""Unit checks for the possession state machine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from possession import (  # noqa: E402
    POSSESSION_CONFIRM_FRAMES,
    POSSESSION_LOSE_FRAMES,
    POSSESSION_SWITCH_FRAMES,
    PossessionEngine,
)


def player(track_id, team_id, field_x, field_y, x=None, y=None):
    return {
        "track_id": track_id,
        "team_id": team_id,
        "field_x": field_x,
        "field_y": field_y,
        "x": x if x is not None else field_x,
        "y": y if y is not None else field_y,
    }


def ball(source, field_x=100, field_y=100, x=100, y=100):
    return {"source": source, "field_x": field_x, "field_y": field_y, "x": x, "y": y}


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def test_confirms_after_consecutive_observed_frames():
    engine = PossessionEngine()
    players = [player(7, 0, 110, 105), player(8, 1, 900, 900)]
    for _ in range(POSSESSION_CONFIRM_FRAMES - 1):
        snapshot = engine.update(players, ball("observed"))
        assert_equal(snapshot.current_possessor_track_id, None, "should still be pending")
    snapshot = engine.update(players, ball("observed"))
    assert_equal(snapshot.current_possessor_track_id, 7, "confirmed possessor")
    assert_equal(snapshot.current_possession_team, "Team 0", "confirmed team")


def test_memory_holds_but_does_not_open_possession():
    engine = PossessionEngine()
    players = [player(3, 1, 120, 110)]
    snapshot = engine.update(players, ball("remembered"))
    assert_equal(snapshot.current_possessor_track_id, None, "memory cannot open possession")

    for _ in range(POSSESSION_CONFIRM_FRAMES):
        engine.update(players, ball("observed"))
    assert_equal(engine.snapshot().current_possessor_track_id, 3, "opened on observed")

    for _ in range(POSSESSION_LOSE_FRAMES - 1):
        snapshot = engine.update(players, ball("predicted", field_x=140, field_y=130))
        assert_equal(snapshot.current_possessor_track_id, 3, "prediction should hold possession")


def test_does_not_flicker_between_nearby_players():
    engine = PossessionEngine()
    a = player(1, 0, 100, 100)
    far = player(2, 1, 400, 100)
    for _ in range(POSSESSION_CONFIRM_FRAMES):
        engine.update([a, far], ball("observed", field_x=105, field_y=100))
    assert_equal(engine.snapshot().current_possessor_track_id, 1, "A confirmed")
    b = player(2, 1, 130, 100)

    for _ in range(POSSESSION_SWITCH_FRAMES):
        snapshot = engine.update([a, b], ball("observed", field_x=118, field_y=100))
        assert_equal(snapshot.current_possessor_track_id, None, "a short lead is a duel, not a new owner")
        assert_equal(snapshot.current_possession_team, None, "neither team keeps the duel")


def test_short_hole_does_not_erase_a_pending_confirmation():
    engine = PossessionEngine()
    players = [player(11, 0, 100, 100, x=100, y=100)]
    observed = ball("observed", x=100, y=110)
    for _ in range(3):
        snapshot = engine.update(players, observed)
        assert_equal(snapshot.current_possessor_track_id, None, "three observations stay pending")
    for _ in range(2):
        snapshot = engine.update(players, ball("predicted", x=100, y=110))
        assert_equal(snapshot.current_possessor_track_id, None, "a predicted hole does not confirm")
    snapshot = engine.update(players, observed)
    assert_equal(snapshot.current_possessor_track_id, None, "the fourth observation is still short")
    snapshot = engine.update(players, observed)
    assert_equal(snapshot.current_possessor_track_id, 11, "the fifth observation confirms")
    assert_equal(snapshot.current_possession_team, "Team 0", "team stays with the track")


def test_a_long_hole_drops_the_pending_player():
    engine = PossessionEngine()
    players = [player(11, 0, 100, 100, x=100, y=100)]
    observed = ball("observed", x=100, y=110)
    for _ in range(4):
        engine.update(players, observed)
    for _ in range(POSSESSION_LOSE_FRAMES):
        snapshot = engine.update(players, None)
        assert_equal(snapshot.current_possessor_track_id, None, "a hole cannot confirm")
    snapshot = engine.update(players, observed)
    assert_equal(snapshot.current_possessor_track_id, None, "the old streak was dropped")
    assert_equal(engine.pending_count, 1, "the next observation starts again")


def test_an_observed_ball_away_from_everyone_drops_the_streak():
    engine = PossessionEngine()
    players = [player(11, 0, 100, 100, x=100, y=100)]
    for _ in range(4):
        engine.update(players, ball("observed", x=100, y=110))
    snapshot = engine.update(players, ball("observed", field_x=900, field_y=900, x=900, y=900))
    assert_equal(snapshot.current_possessor_track_id, None, "a far observation is not control")
    assert_equal(engine.pending_count, 0, "the streak was contradicted")


def test_loses_possession_after_enough_lost_frames():
    engine = PossessionEngine()
    players = [player(4, 0, 100, 100)]
    for _ in range(POSSESSION_CONFIRM_FRAMES):
        engine.update(players, ball("observed"))
    assert_equal(engine.snapshot().current_possessor_track_id, 4, "has possession")

    for index in range(POSSESSION_LOSE_FRAMES):
        snapshot = engine.update(players, None)
        if index < POSSESSION_LOSE_FRAMES - 1:
            assert_equal(snapshot.current_possessor_track_id, 4, "still holding")
        else:
            assert_equal(snapshot.current_possessor_track_id, None, "lost after lose_frames")
            assert_equal(snapshot.current_possession_team, None, "team cleared")


def test_a_fast_ball_is_taken_only_by_the_player_it_reaches():
    engine = PossessionEngine()
    receiver = player(10, 0, None, None, x=500, y=300)
    defender = player(4, 1, None, None, x=500, y=270)
    for ball_x in (445, 460, 475, 490, 505):
        snapshot = engine.update(
            [receiver, defender],
            ball("observed", field_x=None, field_y=None, x=ball_x, y=300),
        )
    assert_equal(snapshot.current_possessor_track_id, 10, "the ball arrived at 10")
    assert_equal(snapshot.current_possession_team, "Team 0", "10 keeps his team")


def test_a_fast_ball_sixty_pixels_away_opens_no_possession():
    engine = PossessionEngine()
    defender = player(4, 1, None, None, x=0, y=100)
    for ball_x in (70, 90, 110, 130, 150, 170):
        snapshot = engine.update(
            [defender],
            ball("observed", field_x=None, field_y=None, x=ball_x, y=100),
        )
    assert_equal(snapshot.current_possessor_track_id, None, "the ball never reached him")


def test_a_far_holder_is_not_renewed_by_a_neighbour_or_a_predicted_ball():
    engine = PossessionEngine()
    holder = player(13, 0, None, None, x=950, y=280)
    other = player(19, 1, None, None, x=735, y=340)
    close = ball("observed", field_x=None, field_y=None, x=980, y=285)
    for _ in range(POSSESSION_CONFIRM_FRAMES):
        engine.update([holder, other], close)
    assert_equal(engine.snapshot().current_possessor_track_id, 13, "13 earned the ball")

    at_the_other = ball("observed", field_x=None, field_y=None, x=746, y=337)
    snapshot = engine.update([holder, other], at_the_other)
    assert_equal(snapshot.current_possessor_track_id, None, "13 loses the ball the frame 19 is on it")
    assert_equal(engine.pending_id, 19, "19 has one frame at his feet")
    snapshot = engine.update([holder, other], at_the_other)
    assert_equal(snapshot.current_possessor_track_id, 19, "a clear touch takes the ball on the second frame")
    assert_equal(snapshot.current_possession_team, "Team 1", "19 keeps his team")

    ghost = ball("predicted", field_x=None, field_y=None, x=960, y=285)
    snapshot = engine.update([holder, other], ghost)
    assert_equal(snapshot.current_possessor_track_id, 19, "one predicted frame does not give the ball back to 13")
    assert_equal(snapshot.current_possession_team, "Team 1", "19 keeps the team he just took")


def test_predicted_ball_still_covers_a_short_hole_at_his_feet():
    engine = PossessionEngine()
    holder = player(4, 0, None, None, x=100, y=100)
    close = ball("observed", field_x=None, field_y=None, x=100, y=100)
    for _ in range(POSSESSION_CONFIRM_FRAMES):
        engine.update([holder], close)
    for _ in range(POSSESSION_LOSE_FRAMES):
        snapshot = engine.update([holder], ball("predicted", field_x=None, field_y=None, x=110, y=100))
        assert_equal(snapshot.current_possessor_track_id, 4, "prediction still at his feet keeps the touch")
    for index in range(POSSESSION_LOSE_FRAMES):
        snapshot = engine.update([holder], None)
        if index < POSSESSION_LOSE_FRAMES - 1:
            assert_equal(snapshot.current_possessor_track_id, 4, "the missing-ball window still holds")
        else:
            assert_equal(snapshot.current_possessor_track_id, None, "a real hole still ends the possession")


def test_switch_requires_consecutive_clear_leader():
    engine = PossessionEngine()
    a = player(11, 0, 100, 100)
    b = player(12, 1, 800, 100)
    for _ in range(POSSESSION_CONFIRM_FRAMES):
        engine.update([a, b], ball("observed", field_x=100, field_y=100))
    assert_equal(engine.snapshot().current_possession_team, "Team 0", "start team 0")

    snapshot = engine.update([a, b], ball("observed", field_x=800, field_y=100))
    assert_equal(snapshot.current_possessor_track_id, None, "11 is no longer the owner")
    snapshot = engine.update([a, b], ball("observed", field_x=800, field_y=100))
    assert_equal(snapshot.current_possessor_track_id, 12, "12 is on the ball for a second frame")
    assert_equal(snapshot.current_possession_team, "Team 1", "team became team 1")


def test_a_marker_does_not_inherit_a_missing_possessor():
    engine = PossessionEngine()
    holder = player(6, 1, None, None, x=400, y=300)
    marker = player(16, 0, None, None, x=450, y=300)
    close = ball("observed", field_x=None, field_y=None, x=400, y=300)
    for _ in range(POSSESSION_CONFIRM_FRAMES):
        engine.update([holder, marker], close)
    assert_equal(engine.snapshot().current_possessor_track_id, 6, "6 has the ball")

    still_there = ball("observed", field_x=None, field_y=None, x=400, y=300)
    for index in range(POSSESSION_LOSE_FRAMES):
        snapshot = engine.update([marker], still_there)
        assert_equal(snapshot.current_possessor_track_id != 16, True, "16 at about 50 px does not inherit")
        if index < POSSESSION_LOSE_FRAMES - 1:
            assert_equal(snapshot.current_possessor_track_id, 6, "the missing holder waits out the lose window")
        else:
            assert_equal(snapshot.current_possessor_track_id, None, "the missing holder is released")
    assert_equal(engine.snapshot().current_possession_team, None, "team 0 did not receive the ball")


def test_predicted_ball_does_not_keep_a_holder_the_other_team_has_passed():
    engine = PossessionEngine()
    holder = player(11, 0, None, None, x=200, y=200)
    other = player(19, 1, None, None, x=120, y=200)
    for _ in range(POSSESSION_CONFIRM_FRAMES):
        engine.update([holder, other], ball("observed", field_x=None, field_y=None, x=200, y=200))
    assert_equal(engine.snapshot().current_possessor_track_id, 11, "11 confirmed")
    snapshot = engine.update(
        [holder, other],
        ball("predicted", field_x=None, field_y=None, x=120, y=200),
    )
    assert_equal(snapshot.current_possessor_track_id, None, "the prediction is at 19, not at 11")
    assert_equal(snapshot.current_possession_team, None, "team 0 is not renewed")


if __name__ == "__main__":
    test_confirms_after_consecutive_observed_frames()
    test_short_hole_does_not_erase_a_pending_confirmation()
    test_a_long_hole_drops_the_pending_player()
    test_an_observed_ball_away_from_everyone_drops_the_streak()
    test_memory_holds_but_does_not_open_possession()
    test_does_not_flicker_between_nearby_players()
    test_loses_possession_after_enough_lost_frames()
    test_a_far_holder_is_not_renewed_by_a_neighbour_or_a_predicted_ball()
    test_predicted_ball_still_covers_a_short_hole_at_his_feet()
    test_switch_requires_consecutive_clear_leader()
    test_a_marker_does_not_inherit_a_missing_possessor()
    test_predicted_ball_does_not_keep_a_holder_the_other_team_has_passed()
    test_a_fast_ball_is_taken_only_by_the_player_it_reaches()
    test_a_fast_ball_sixty_pixels_away_opens_no_possession()
    print("possession engine tests passed")
