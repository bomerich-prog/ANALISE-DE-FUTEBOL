#!/usr/bin/env python3
"""Unit checks for pass detector refinements."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from pass_detector import PassDetector  # noqa: E402


FPS = 25.0


def snapshot(track_id, team=None):
    return SimpleNamespace(current_possessor_track_id=track_id, current_possession_team=team)


def player(track_id, team_id, field_x, field_y):
    return {
        "track_id": track_id,
        "team_id": team_id,
        "field_x": field_x,
        "field_y": field_y,
        "x": field_x,
        "y": field_y,
    }


def ball(field_x, field_y, source="observed"):
    return {"source": source, "field_x": field_x, "field_y": field_y, "x": field_x, "y": field_y}


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def run_frames(detector, frames):
    metrics = None
    for frame_number, possessor, team, players, ball_state in frames:
        metrics = detector.update(frame_number, FPS, snapshot(possessor, team), players, ball_state)
    return detector.finalize() if metrics is not None else detector.finalize()


def valid_pass_sequence(receiver_x=1500.0, ball_xs=None, gap_none_frames=4, hold_frames=5):
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, receiver_x, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(8):
        frames.append((frame, 1, "blue", players, ball(0.0, 0.0)))
        frame += 1
    lost_frame = frame
    ball_xs = list(ball_xs) if ball_xs is not None else [400.0, 800.0, 1100.0, 1400.0]
    for index in range(gap_none_frames):
        ball_x = ball_xs[min(index, len(ball_xs) - 1)]
        frames.append((frame, None, None, players, ball(ball_x, 0.0)))
        frame += 1
    for hold_index in range(hold_frames):
        ball_x = ball_xs[-1] if ball_xs else receiver_x
        if hold_index == hold_frames - 1:
            ball_x = receiver_x
        frames.append((frame, 2, "blue", players, ball(ball_x, 0.0)))
        frame += 1
    return frames, lost_frame


def test_confirms_stable_same_team_pass():
    detector = PassDetector()
    frames, _lost = valid_pass_sequence()
    metrics = run_frames(detector, frames)
    assert_equal(metrics["raw_candidate_passes"], 1, "naive candidate")
    assert_equal(metrics["total_passes"], 1, "refined pass")
    assert_equal(metrics["discarded_short_passes"], 0, "short")
    assert_equal(metrics["discarded_long_passes"], 0, "long")
    assert_equal(metrics["discarded_unstable_possession"], 0, "unstable")
    assert_equal(metrics["discarded_timeout"], 0, "timeout")
    assert_equal(metrics["discarded_direction_check"], 0, "direction")
    assert_equal(metrics["events"][0]["from_track_id"], 1, "from")
    assert_equal(metrics["events"][0]["to_track_id"], 2, "to")


def test_requires_five_stable_frames():
    detector = PassDetector()
    frames, _lost = valid_pass_sequence(hold_frames=4)
    metrics = run_frames(detector, frames)
    assert_equal(metrics["raw_candidate_passes"], 1, "naive candidate still counted")
    assert_equal(metrics["total_passes"], 0, "not confirmed")
    assert_equal(metrics["discarded_unstable_possession"], 1, "unstable hold")


def test_requires_ball_to_leave_hold_radius():
    detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1500.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "blue", players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "blue", players, ball(50.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 0, "ball stayed with donor")
    assert_equal(metrics["discarded_unstable_possession"], 1, "never left radius")


def test_discards_timeout_over_one_second():
    detector = PassDetector()
    frames, _lost = valid_pass_sequence(gap_none_frames=30, ball_xs=[400.0, 800.0, 1200.0, 1500.0])
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 0, "timed out")
    assert_equal(metrics["discarded_timeout"], 1, "timeout counter")


def test_discards_short_and_long_passes():
    short_detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 150.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "blue", players, ball(0.0, 0.0)))
        frame += 1
    for ball_x in (500.0, 700.0, 500.0, 200.0):
        frames.append((frame, None, None, players, ball(ball_x, 0.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "blue", players, ball(150.0, 0.0)))
        frame += 1
    short_metrics = run_frames(short_detector, frames)
    assert_equal(short_metrics["total_passes"], 0, "short pass rejected")
    assert_equal(short_metrics["discarded_short_passes"], 1, "short counter")

    long_detector = PassDetector()
    long_frames, _ = valid_pass_sequence(receiver_x=7000.0, ball_xs=[1500.0, 3000.0, 5000.0, 7000.0])
    long_metrics = run_frames(long_detector, long_frames)
    assert_equal(long_metrics["total_passes"], 0, "long pass rejected")
    assert_equal(long_metrics["discarded_long_passes"], 1, "long counter")


def test_discards_incompatible_ball_direction():
    detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1500.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "blue", players, ball(0.0, 0.0)))
        frame += 1
    for ball_x in (-200.0, -400.0, -700.0, -900.0):
        frames.append((frame, None, None, players, ball(ball_x, 0.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "blue", players, ball(-1100.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 0, "opposite direction rejected")
    assert_equal(metrics["discarded_direction_check"], 1, "direction counter")


def test_does_not_duplicate_same_pair():
    detector = PassDetector()
    frames, _ = valid_pass_sequence()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1500.0, 0.0)
    players = [a, b]
    frame = frames[-1][0] + 1
    frames.append((frame, None, None, players, ball(800.0, 0.0)))
    frame += 1
    frames.append((frame, 1, "blue", players, ball(0.0, 0.0)))
    frame += 1
    frames.append((frame, 1, "blue", players, ball(0.0, 0.0)))
    frame += 1
    for ball_x in (400.0, 800.0, 1100.0, 1400.0):
        frames.append((frame, None, None, players, ball(ball_x, 0.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "blue", players, ball(1500.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 1, "same A->B not counted twice")
    assert_equal(len(metrics["events"]), 1, "single event")


def test_first_possession_is_not_a_pass():
    detector = PassDetector()
    players = [player(4, 1, 100.0, 100.0)]
    frames = [(index, 4, "red", players, ball(100.0, 100.0)) for index in range(1, 10)]
    metrics = run_frames(detector, frames)
    assert_equal(metrics["raw_candidate_passes"], 0, "opening possession is not a pass")
    assert_equal(metrics["total_passes"], 0, "no pass")


def test_confirms_pass_when_ball_disappears_between_possessors():
    detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1500.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(8):
        frames.append((frame, 1, "blue", players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(10):
        frames.append((frame, None, None, players, None))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "blue", players, ball(1500.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 1, "gap without trajectory is a pass")
    assert_equal(metrics["possession_gap_passes"], 1, "counted from possession states")
    assert_equal(metrics["discarded_direction_check"], 0, "missing trajectory is not a direction reject")
    assert_equal(metrics["discarded_timeout"], 0, "gap stays inside one second")
    assert_equal(metrics["events"][0]["from_track_id"], 1, "donor")
    assert_equal(metrics["events"][0]["to_track_id"], 2, "receiver")


def test_predicted_ball_does_not_block_possession_pass():
    detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1800.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "blue", players, ball(0.0, 0.0)))
        frame += 1
    for ball_x in (-300.0, -700.0, -1200.0, -1600.0):
        frames.append((frame, None, None, players, ball(ball_x, 0.0, source="predicted")))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "blue", players, ball(1800.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 1, "predicted flight does not veto the pass")
    assert_equal(metrics["possession_gap_passes"], 1, "possession gap pass")
    assert_equal(metrics["discarded_direction_check"], 0, "prediction is not a trajectory")
    assert_equal(metrics["turnovers"], 0, "same team is not a turnover")


def test_missing_ball_still_respects_timeout_and_distance():
    timeout_detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1500.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "blue", players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(30):
        frames.append((frame, None, None, players, None))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "blue", players, ball(1500.0, 0.0)))
        frame += 1
    timeout_metrics = run_frames(timeout_detector, frames)
    assert_equal(timeout_metrics["total_passes"], 0, "long disappearance is not a pass")
    assert_equal(timeout_metrics["discarded_timeout"], 1, "timeout still applies")

    long_detector = PassDetector()
    far = player(2, 0, 7000.0, 0.0)
    long_frames = []
    frame = 1
    for _ in range(6):
        long_frames.append((frame, 1, "blue", [a, far], ball(0.0, 0.0)))
        frame += 1
    for _ in range(4):
        long_frames.append((frame, None, None, [a, far], None))
        frame += 1
    for _ in range(5):
        long_frames.append((frame, 2, "blue", [a, far], ball(7000.0, 0.0)))
        frame += 1
    long_metrics = run_frames(long_detector, long_frames)
    assert_equal(long_metrics["total_passes"], 0, "over 60m is still rejected")
    assert_equal(long_metrics["discarded_long_passes"], 1, "long counter")


def test_different_team_is_not_a_pass():
    detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(8, 1, 1500.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "blue", players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(3):
        frames.append((frame, None, None, players, ball(800.0, 0.0)))
        frame += 1
    for _ in range(6):
        frames.append((frame, 8, "red", players, ball(1500.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["raw_candidate_passes"], 0, "turnover is not a pass")
    assert_equal(metrics["total_passes"], 0, "no pass")
    assert_equal(metrics["turnovers"], 1, "other team is a turnover")


if __name__ == "__main__":
    test_confirms_stable_same_team_pass()
    test_requires_five_stable_frames()
    test_requires_ball_to_leave_hold_radius()
    test_discards_timeout_over_one_second()
    test_discards_short_and_long_passes()
    test_discards_incompatible_ball_direction()
    test_does_not_duplicate_same_pair()
    test_first_possession_is_not_a_pass()
    test_confirms_pass_when_ball_disappears_between_possessors()
    test_predicted_ball_does_not_block_possession_pass()
    test_missing_ball_still_respects_timeout_and_distance()
    test_different_team_is_not_a_pass()
    print("pass detector tests passed")
