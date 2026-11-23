#!/usr/bin/env python3
"""Argentina v Spain, 32.5 s to 40.5 s: a duel and a ball that jumps to a boot.

The fixture is a slice of the event log of the demonstration clip. Spain's 117
plays to 121; an Argentine (143, no shirt vote yet) on top of him sends the
ball out to Argentina's 131, who plays to 91. Then the ball layer jumps to the
boot of Spain's 165 for seven frames while the real ball stays at 91's side.

Expected: Spain 117 to 121, Argentina 143 to 131 and 131 to 91, one turnover
(121 to 143, in the duel), and no turnover to 165.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from ball_jump import BallJumpGuard, jump_guard_ball  # noqa: E402
from event_replay import load_event_record, replay_event_record  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "event-duel-jump-argentina.json"


def test_duel_and_jump_replay():
    result = replay_event_record(load_event_record(FIXTURE))
    passes = result["passes"]
    pairs = [(event["from_track_id"], event["to_track_id"]) for event in passes["events"]]
    assert pairs == [(117, 121), (143, 131), (131, 91)], pairs
    turnovers = [(event["from_track_id"], event["to_track_id"]) for event in passes["turnover_events"]]
    assert turnovers == [(121, 143)], turnovers
    assert passes["duel_turnovers"] == 1
    assert result["ball_jump"]["ball_jumps_held_back"] == 1


class _Holder:
    def __init__(self, possessor_id):
        self.possessor_id = possessor_id


def _ball(x, y, source="observed"):
    return {"source": source, "x": x, "y": y}


def _players():
    return [
        {"track_id": 9, "x": 600.0, "y": 420.0},  # holder
        {"track_id": 16, "x": 532.0, "y": 402.0},  # rival with a white boot
    ]


def test_jump_onto_rival_is_held_while_someone_holds_the_ball():
    guard = BallJumpGuard()
    holder = _Holder(9)
    for x in (580.0, 586.0, 592.0):
        assert jump_guard_ball(guard, holder, _ball(x, 415.0), _players())["source"] == "observed"
    held = jump_guard_ball(guard, holder, _ball(532.0, 398.0), _players())
    assert held["source"] == "jump"
    # It crawls on: still held.
    assert jump_guard_ball(guard, holder, _ball(539.0, 400.0), _players())["source"] == "jump"
    # A repeated frame changes nothing.
    assert jump_guard_ball(guard, holder, _ball(539.0, 400.0), _players())["source"] == "jump"


def test_kick_is_not_held():
    guard = BallJumpGuard()
    holder = _Holder(9)
    for x in (580.0, 583.0, 586.0):
        jump_guard_ball(guard, holder, _ball(x, 415.0), _players())
    players = [{"track_id": 9, "x": 586.0, "y": 420.0}, {"track_id": 16, "x": 530.0, "y": 410.0}]
    # Lands next to 16 but keeps flying fast: a kick.
    assert jump_guard_ball(guard, holder, _ball(536.0, 412.0), players)["source"] == "jump"
    assert jump_guard_ball(guard, holder, _ball(536.0, 412.0), players)["source"] == "jump"  # repeated frame
    assert jump_guard_ball(guard, holder, _ball(490.0, 410.0), players)["source"] == "observed"


def test_jump_into_open_grass_is_not_held():
    guard = BallJumpGuard()
    holder = _Holder(9)
    for x in (580.0, 583.0, 586.0):
        jump_guard_ball(guard, holder, _ball(x, 415.0), _players())
    assert jump_guard_ball(guard, holder, _ball(700.0, 500.0), _players())["source"] == "observed"


def test_nobody_on_the_ball_sees_the_raw_ball():
    guard = BallJumpGuard()
    free = _Holder(None)
    for x in (580.0, 586.0, 592.0):
        jump_guard_ball(guard, free, _ball(x, 415.0), _players())
    assert jump_guard_ball(guard, free, _ball(532.0, 398.0), _players())["source"] == "observed"


if __name__ == "__main__":
    test_duel_and_jump_replay()
    test_jump_onto_rival_is_held_while_someone_holds_the_ball()
    test_kick_is_not_held()
    test_jump_into_open_grass_is_not_held()
    test_nobody_on_the_ball_sees_the_raw_ball()
    print("duel and jump tests passed")
