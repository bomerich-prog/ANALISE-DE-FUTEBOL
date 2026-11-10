#!/usr/bin/env python3
"""Unit checks for pass detector refinements."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from pass_detector import PassDetector  # noqa: E402
from possession import POSSESSION_CONFIRM_FRAMES  # noqa: E402


# Low enough that the synthetic 15 m passes stay under the physical ball-speed ceiling.
FPS = 10.0


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
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
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
        frames.append((frame, 2, "Team 0", players, ball(ball_x, 0.0)))
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


def test_confirmed_snapshot_is_enough():
    detector = PassDetector()
    frames, _lost = valid_pass_sequence(hold_frames=1)
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 1, "snapshot confirmation is the pass")
    assert_equal(metrics["events"][0]["confidence"] >= 80, True, "score clears the threshold")


def test_ball_near_donor_does_not_veto_confirmed_reception():
    detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1500.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "Team 0", players, ball(50.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 1, "possession decides, not the stuck ball")
    assert_equal(metrics["events"][0]["trajectory"], "missing", "tiny motion is not a trajectory")
    assert_equal(metrics["events"][0]["confidence"], 90, "missing trajectory keeps the medium score")


def test_discards_flight_beyond_distance_budget():
    detector = PassDetector()
    frames, _lost = valid_pass_sequence(gap_none_frames=30, ball_xs=[400.0, 800.0, 1200.0, 1500.0])
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 0, "3 s for 15 m is a loose ball")
    assert_equal(metrics["discarded_timeout"], 1, "timeout counter")
    assert_equal(metrics["rejected_events"][0]["reason"], "timeout", "rejection reason")
    assert_equal(metrics["rejected_events"][0]["max_timing_seconds"], 2.375, "15 m budget")


def slow_pass_frames(distance=2300.0, flight_frames=16, ball_visible=True, field_jitter=None):
    """A holds, the ball travels for flight_frames, then B is confirmed."""

    def entity(track_id, team_id, x, frame):
        offset = field_jitter[frame % len(field_jitter)] if field_jitter else 0.0
        return {
            "track_id": track_id,
            "team_id": team_id,
            "field_x": x + offset,
            "field_y": offset * 0.5,
            "x": x / 10.0,
            "y": 0.0,
        }

    def ball_at(x, frame):
        offset = field_jitter[frame % len(field_jitter)] if field_jitter else 0.0
        return {"source": "observed", "field_x": x + offset, "field_y": offset * 0.5, "x": x / 10.0, "y": 0.0}

    frames = []
    frame = 1
    for _ in range(8):
        frames.append((frame, 1, "Team 0", [entity(1, 0, 0.0, frame), entity(2, 0, distance, frame)], ball_at(0.0, frame)))
        frame += 1
    for index in range(flight_frames):
        players = [entity(1, 0, 0.0, frame), entity(2, 0, distance, frame)]
        ball_state = ball_at(distance * (index + 1) / flight_frames, frame) if ball_visible else None
        frames.append((frame, None, None, players, ball_state))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "Team 0", [entity(1, 0, 0.0, frame), entity(2, 0, distance, frame)], ball_at(distance, frame)))
        frame += 1
    return frames


def test_extended_window_accepts_observed_slow_pass():
    detector = PassDetector()
    metrics = run_frames(detector, slow_pass_frames())
    assert_equal(metrics["total_passes"], 1, "23 m in 1.6 s with an observed flight is a pass")
    assert_equal(metrics["extended_window_passes"], 1, "accepted through the extended window")
    assert_equal(metrics["discarded_timeout"], 0, "not a timeout")
    event = metrics["events"][0]
    assert_equal(event["timing_tier"], "extended", "tier")
    assert_equal(event["trajectory"], "compatible", "in-frame progress toward B")
    assert_equal(event["confidence"], 90, "no short-interval bonus, trajectory bonus kept")
    assert_equal(event["score_components"]["unverified_extension"], 0, "flight was observed")
    assert_equal(event["distance_source"], "same_frame", "distance read inside one frame")


def test_extended_window_without_flight_evidence_stays_unconfirmed():
    detector = PassDetector()
    metrics = run_frames(detector, slow_pass_frames(ball_visible=False))
    assert_equal(metrics["total_passes"], 0, "1.6 s with no ball seen is not enough evidence")
    assert_equal(metrics["unconfirmed_candidates"], 1, "scored, then left unconfirmed")
    event = metrics["unconfirmed_events"][0]
    assert_equal(event["in_flight_samples"], 0, "no observed ball during the flight")
    assert_equal(event["score_components"]["unverified_extension"], -20, "extension penalty")
    assert_equal(event["confidence"], 70, "below the threshold")


def test_flight_budget_scales_with_distance():
    short_detector = PassDetector()
    short_metrics = run_frames(short_detector, slow_pass_frames(distance=400.0, flight_frames=20))
    assert_equal(short_metrics["total_passes"], 0, "4 m in 2 s is a loose ball")
    assert_equal(short_metrics["discarded_timeout"], 1, "short pass keeps the 1 s budget")

    long_detector = PassDetector()
    long_metrics = run_frames(long_detector, slow_pass_frames(distance=3000.0, flight_frames=20))
    assert_equal(long_metrics["total_passes"], 1, "30 m in 2 s is a pass")
    assert_equal(long_metrics["events"][0]["max_timing_seconds"], 4.0, "30 m budget hits the ceiling")


def test_reception_lag_does_not_count_as_flight():
    detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1500.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(8):
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for ball_x in (400.0, 800.0, 1100.0, 1400.0, 1450.0, 1480.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0):
        frames.append((frame, None, None, players, ball(ball_x, 0.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "Team 0", players, ball(1500.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 1, "possession confirmation delay is not flight time")
    event = metrics["events"][0]
    assert_equal(event["gap_seconds"] > 1.0, True, "empty possession exceeded one second")
    assert_equal(event["timing_tier"], "base", "the ball itself arrived in time")
    assert_equal(event["confidence"], 100, "full score")


def test_opponent_alone_on_the_ball_intercepts():
    detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1500.0, 0.0)
    opponent = player(7, 1, 750.0, 300.0)
    players = [a, b, opponent]
    frames = []
    frame = 1
    for _ in range(8):
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    path = [(300.0, 100.0), (600.0, 250.0), (750.0, 300.0), (750.0, 300.0), (750.0, 300.0), (750.0, 300.0), (1000.0, 200.0), (1300.0, 50.0)]
    for ball_x, ball_y in path:
        frames.append((frame, None, None, players, ball(ball_x, ball_y)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "Team 0", players, ball(1500.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 0, "the opponent touched the ball")
    assert_equal(metrics["disputes"], 0, "one team on the ball is not a dispute")
    assert_equal(metrics["discarded_intercepted"], 1, "interception counter")
    assert_equal(metrics["rejected_events"][0]["interceptor_track_id"], 7, "interceptor")


def test_ball_faster_than_physics_is_rejected():
    detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 6000.0, 0.0)
    players = [a, b]
    frames = [(frame, 1, "Team 0", players, ball(0.0, 0.0)) for frame in range(1, 7)]
    frames.append((7, 2, "Team 0", players, ball(6000.0, 0.0)))
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 0, "60 m in one frame is not a pass")
    assert_equal(metrics["discarded_implausible_speed"], 1, "speed counter")


def test_homography_jitter_does_not_break_trajectory_or_distance():
    detector = PassDetector()
    jitter = [0.0, 3000.0, -2500.0, 1800.0, -900.0, 2600.0, -3100.0]
    metrics = run_frames(detector, slow_pass_frames(field_jitter=jitter))
    assert_equal(metrics["total_passes"], 1, "per-frame jitter cancels inside each frame")
    event = metrics["events"][0]
    assert_equal(event["trajectory"], "compatible", "trajectory from in-frame distances")
    assert_equal(round(event["distance_cm"]), 2300, "distance from one frame")


def test_discards_short_and_long_passes():
    short_detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 80.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for ball_x in (500.0, 700.0, 500.0, 200.0):
        frames.append((frame, None, None, players, ball(ball_x, 0.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "Team 0", players, ball(80.0, 0.0)))
        frame += 1
    short_metrics = run_frames(short_detector, frames)
    assert_equal(short_metrics["total_passes"], 0, "short pass rejected")
    assert_equal(short_metrics["discarded_short_passes"], 1, "short counter")

    long_detector = PassDetector()
    long_frames, _ = valid_pass_sequence(receiver_x=8000.0, ball_xs=[1500.0, 3000.0, 5000.0, 8000.0])
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
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for ball_x in (-200.0, -400.0, -700.0, -900.0):
        frames.append((frame, None, None, players, ball(ball_x, 0.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "Team 0", players, ball(-1100.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 0, "incompatible trajectory stays below the threshold")
    assert_equal(metrics["unconfirmed_candidates"], 1, "scored, then left unconfirmed")
    assert_equal(metrics["discarded_direction_check"], 0, "direction is a score term, not a veto")
    assert_equal(metrics["unconfirmed_events"][0]["trajectory"], "incompatible", "trajectory label")
    assert_equal(metrics["unconfirmed_events"][0]["confidence"], 70, "penalty keeps the pass under 80")


def test_each_possession_transfer_is_its_own_pass():
    detector = PassDetector()
    frames, _ = valid_pass_sequence()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1500.0, 0.0)
    players = [a, b]
    frame = frames[-1][0] + 1
    for ball_x in (1100.0, 700.0, 400.0, 100.0):
        frames.append((frame, None, None, players, ball(ball_x, 0.0)))
        frame += 1
    frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
    frame += 1
    frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
    frame += 1
    for ball_x in (400.0, 800.0, 1100.0, 1400.0):
        frames.append((frame, None, None, players, ball(ball_x, 0.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "Team 0", players, ball(1500.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    pairs = [(event["from_track_id"], event["to_track_id"]) for event in metrics["events"]]
    assert_equal(pairs, [(1, 2), (2, 1), (1, 2)], "return and the next pass both count")
    assert_equal(metrics["total_passes"], 3, "three possession transfers")


def test_first_possession_is_not_a_pass():
    detector = PassDetector()
    players = [player(4, 1, 100.0, 100.0)]
    frames = [(index, 4, "Team 1", players, ball(100.0, 100.0)) for index in range(1, 10)]
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
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(10):
        frames.append((frame, None, None, players, None))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "Team 0", players, ball(1500.0, 0.0)))
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
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for ball_x in (-300.0, -700.0, -1200.0, -1600.0):
        frames.append((frame, None, None, players, ball(ball_x, 0.0, source="predicted")))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "Team 0", players, ball(1800.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 1, "predicted flight does not veto the pass")
    assert_equal(metrics["possession_gap_passes"], 1, "possession gap pass")
    assert_equal(metrics["discarded_direction_check"], 0, "prediction is not a trajectory")
    assert_equal(metrics["turnovers"], 0, "same team is not a turnover")


def test_predicted_drift_during_reception_does_not_cancel_pass():
    detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1500.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(4):
        frames.append((frame, None, None, players, None))
        frame += 1
    frames.append((frame, 2, "Team 0", players, ball(1500.0, 0.0)))
    frame += 1
    for ball_x in (900.0, 200.0, -400.0, -900.0):
        frames.append((frame, 2, "Team 0", players, ball(ball_x, 0.0, source="predicted")))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 1, "predicted drift does not cancel the pass")
    assert_equal(metrics["possession_gap_passes"], 1, "accepted from the possession gap")
    assert_equal(metrics["discarded_direction_check"], 0, "prediction is not the trajectory")


def test_missing_ball_still_respects_timeout_and_distance():
    timeout_detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    b = player(2, 0, 1500.0, 0.0)
    players = [a, b]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(30):
        frames.append((frame, None, None, players, None))
        frame += 1
    for _ in range(5):
        frames.append((frame, 2, "Team 0", players, ball(1500.0, 0.0)))
        frame += 1
    timeout_metrics = run_frames(timeout_detector, frames)
    assert_equal(timeout_metrics["total_passes"], 0, "long disappearance is not a pass")
    assert_equal(timeout_metrics["discarded_timeout"], 1, "timeout still applies")

    long_detector = PassDetector()
    far = player(2, 0, 8000.0, 0.0)
    long_frames = []
    frame = 1
    for _ in range(6):
        long_frames.append((frame, 1, "Team 0", [a, far], ball(0.0, 0.0)))
        frame += 1
    for _ in range(4):
        long_frames.append((frame, None, None, [a, far], None))
        frame += 1
    for _ in range(5):
        long_frames.append((frame, 2, "Team 0", [a, far], ball(8000.0, 0.0)))
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
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(3):
        frames.append((frame, None, None, players, ball(800.0, 0.0)))
        frame += 1
    for _ in range(6):
        frames.append((frame, 8, "Team 1", players, ball(1500.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["raw_candidate_passes"], 0, "turnover is not a pass")
    assert_equal(metrics["total_passes"], 0, "no pass")
    assert_equal(metrics["turnovers"], 1, "other team is a turnover")
    assert_equal(metrics["turnover_events"][0]["confidence"] > 70, True, "observed flight raises turnover confidence")


def test_dispute_on_the_frame_before_the_switch():
    detector = PassDetector()
    donor = player(1, 0, 0.0, 0.0)
    opponent = player(7, 1, 80.0, 0.0)
    receiver = player(2, 0, 1500.0, 0.0)
    players = [donor, opponent, receiver]
    frames = []
    frame = 1
    for _ in range(5):
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    frames.append((frame, 1, "Team 0", players, ball(40.0, 0.0)))
    frame += 1
    frames.append((frame, 2, "Team 0", players, ball(1500.0, 0.0)))
    metrics = run_frames(detector, frames)
    assert_equal(metrics["disputes"], 1, "the duel before confirmation blocks the pass")
    assert_equal(metrics["total_passes"], 0, "not a pass")


def test_same_player_return_is_a_carry():
    detector = PassDetector()
    a = player(1, 0, 0.0, 0.0)
    players = [a]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(4):
        frames.append((frame, None, None, players, None))
        frame += 1
    for _ in range(3):
        frames.append((frame, 1, "Team 0", players, ball(40.0, 0.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["conductions"], 1, "A to empty to A is a carry")
    assert_equal(metrics["total_passes"], 0, "a carry is not a pass")
    assert_equal(metrics["turnovers"], 0, "a carry is not a turnover")


def one_two_frames(wall_x, ball_xs, opponent=None):
    carrier = player(1, 0, 0.0, 0.0)
    wall = player(2, 0, wall_x, 0.0)
    players = [carrier, wall]
    if opponent is not None:
        players.append(opponent)
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for ball_x in ball_xs:
        frames.append((frame, None, None, players, ball(ball_x, 0.0)))
        frame += 1
    for _ in range(4):
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    return frames


def test_wall_pass_becomes_two_passes():
    detector = PassDetector()
    metrics = run_frames(detector, one_two_frames(800.0, [400.0, 800.0, 800.0, 400.0]))
    pairs = [(event["from_track_id"], event["to_track_id"]) for event in metrics["events"]]
    assert_equal(pairs, [(1, 2), (2, 1)], "a wall is a pass each way")
    assert_equal(metrics["total_passes"], 2, "two passes")
    assert_equal(metrics["wall_passes"], 2, "both legs are wall passes")
    assert_equal(metrics["conductions"], 0, "the return is not a carry")
    assert_equal(all(event.get("wall") for event in metrics["events"]), True, "events are marked")
    assert_equal(all(event["confidence"] >= 80 for event in metrics["events"]), True, "both legs clear the score")
    assert_equal(POSSESSION_CONFIRM_FRAMES, 5, "possession confirmation stays at five frames")


def test_same_player_recovery_without_a_wall_stays_a_carry():
    detector = PassDetector()
    metrics = run_frames(detector, one_two_frames(5000.0, [400.0, 700.0, 400.0]))
    assert_equal(metrics["total_passes"], 0, "a far teammate is not a wall")
    assert_equal(metrics["wall_passes"], 0, "no wall pass")
    assert_equal(metrics["conductions"], 1, "A recovering the ball is a carry")
    assert_equal(metrics["events"], [], "no invented pass")


def test_opponent_on_the_ball_is_not_a_wall():
    detector = PassDetector()
    opponent = player(7, 1, 800.0, 0.0)
    metrics = run_frames(
        detector,
        one_two_frames(1600.0, [800.0, 800.0, 800.0, 800.0], opponent=opponent),
    )
    assert_equal(metrics["total_passes"], 0, "an opponent on the ball is not a wall")
    assert_equal(metrics["wall_passes"], 0, "no wall pass")
    assert_equal(metrics["conductions"], 1, "the return stays a carry")
    assert_equal(metrics["events"], [], "a dispute does not create passes")


def test_teammate_outside_contest_radius_is_not_a_wall():
    detector = PassDetector()
    metrics = run_frames(detector, one_two_frames(800.0, [300.0, 650.0, 300.0]))
    assert_equal(metrics["total_passes"], 0, "150 cm from the ball is not a touch")
    assert_equal(metrics["wall_passes"], 0, "no wall pass")
    assert_equal(metrics["conductions"], 1, "a pass without a wall stays a carry")


def test_wall_leg_below_the_score_is_not_created():
    detector = PassDetector(score_threshold=101)
    metrics = run_frames(detector, one_two_frames(800.0, [400.0, 800.0, 400.0]))
    assert_equal(metrics["total_passes"], 0, "a leg under the score is not a pass")
    assert_equal(metrics["events"], [], "nothing is invented")
    assert_equal(metrics["unconfirmed_candidates"], 2, "both legs stay unconfirmed")
    assert_equal(metrics["conductions"], 0, "the wall was still recognized")


def test_one_frame_wall_touch_is_still_two_passes():
    detector = PassDetector()
    metrics = run_frames(detector, one_two_frames(800.0, [400.0, 800.0, 400.0]))
    pairs = [(event["from_track_id"], event["to_track_id"]) for event in metrics["events"]]
    assert_equal(pairs, [(1, 2), (2, 1)], "one observed touch is enough")
    assert_equal(metrics["total_passes"], 2, "the short touch still makes two passes")
    assert_equal(metrics["conductions"], 0, "the short touch is not a carry")
    assert_equal(metrics["events"][0]["in_flight_samples"] >= 1, True, "the ball was seen between the players")


def test_a_clear_receiver_is_not_a_dispute():
    detector = PassDetector()
    donor = {
        "track_id": 32,
        "team_id": 0,
        "field_x": None,
        "field_y": None,
        "x": 0.0,
        "y": 0.0,
    }
    receiver = {
        "track_id": 44,
        "team_id": 0,
        "field_x": None,
        "field_y": None,
        "x": 200.0,
        "y": 0.0,
    }
    opponent = {
        "track_id": 45,
        "team_id": 1,
        "field_x": None,
        "field_y": None,
        "x": 240.0,
        "y": 0.0,
    }
    players = [donor, receiver, opponent]

    def pixel_ball(x):
        return {"source": "observed", "field_x": None, "field_y": None, "x": x, "y": 0.0}

    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 32, "Team 0", players, pixel_ball(0.0)))
        frame += 1
    frames.append((frame, None, None, players, pixel_ball(100.0)))
    frame += 1
    frames.append((frame, 44, "Team 0", players, pixel_ball(200.0)))
    metrics = run_frames(detector, frames)
    assert_equal(metrics["disputes"], 0, "an opponent a margin behind is not a dispute")
    assert_equal(metrics["total_passes"], 1, "the clear reception is a pass")
    assert_equal(metrics["events"][0]["from_track_id"], 32, "donor")
    assert_equal(metrics["events"][0]["to_track_id"], 44, "receiver")


def image_player(track_id, team_id, x, y):
    return {
        "track_id": track_id,
        "team_id": team_id,
        "field_x": None,
        "field_y": None,
        "x": x,
        "y": y,
    }


def image_ball(x, y):
    return {"source": "observed", "field_x": None, "field_y": None, "x": x, "y": y}


def _seen_pixel_flight(cruise_frames):
    """The ball leaves 11, travels in the open, then reaches 30."""
    donor = image_player(11, 0, 0.0, 300.0)
    receiver = image_player(30, 0, 400.0, 300.0)
    players = [donor, receiver]
    frames = []
    frame = 1
    for _ in range(8):
        frames.append((frame, 11, "Team 0", players, image_ball(0.0, 300.0)))
        frame += 1
    for x in [100.0] * cruise_frames + [220.0, 270.0, 320.0, 400.0]:
        frames.append((frame, None, None, players, image_ball(x, 300.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 30, "Team 0", players, image_ball(400.0, 300.0)))
        frame += 1
    return frames


def test_verified_pixel_flight_can_run_past_one_second():
    detector = PassDetector()
    metrics = run_frames(detector, _seen_pixel_flight(cruise_frames=8))
    assert_equal(metrics["total_passes"], 1, "a seen flight of 1.1 s is a pass")
    event = metrics["events"][0]
    assert_equal(event["from_track_id"], 11, "donor")
    assert_equal(event["to_track_id"], 30, "receiver")
    assert_equal(event["timing_tier"], "extended", "past the one second gap")
    assert_equal(event["max_timing_seconds"], 2.0, "verified pixel budget")
    assert_equal(event["trajectory"], "compatible", "the ball approaches the receiver")


def test_hidden_pixel_flight_past_one_second_is_rejected():
    detector = PassDetector()
    donor = image_player(11, 0, 0.0, 300.0)
    receiver = image_player(30, 0, 400.0, 300.0)
    players = [donor, receiver]
    frames = []
    frame = 1
    for _ in range(8):
        frames.append((frame, 11, "Team 0", players, image_ball(0.0, 300.0)))
        frame += 1
    for _ in range(12):
        frames.append((frame, None, None, players, None))
        frame += 1
    for _ in range(5):
        frames.append((frame, 30, "Team 0", players, image_ball(400.0, 300.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 0, "an unseen 1.2 s gap is not a pass")
    assert_equal(metrics["rejected_events"][0]["reason"], "timeout", "the one second window holds")
    assert_equal(metrics["rejected_events"][0]["max_timing_seconds"], 1.0, "no verified flight")


def test_verified_pixel_flight_beyond_two_seconds_is_rejected():
    detector = PassDetector()
    metrics = run_frames(detector, _seen_pixel_flight(cruise_frames=21))
    assert_equal(metrics["total_passes"], 0, "2.5 s is a loose ball even when the ball is seen")
    assert_equal(metrics["rejected_events"][0]["reason"], "timeout", "the verified budget is two seconds")
    assert_equal(metrics["rejected_events"][0]["max_timing_seconds"], 2.0, "verified pixel budget")


def pixel_player(track_id, team_id, x, y=300.0):
    return {"track_id": track_id, "team_id": team_id, "x": x, "y": y, "field_x": None, "field_y": None}


def pixel_ball(x, y=300.0, source="observed"):
    return {"source": source, "x": x, "y": y, "field_x": None, "field_y": None}


def test_a_one_touch_is_a_pass_without_a_long_possession():
    detector = PassDetector()
    holder = pixel_player(11, 0, 0.0)
    toucher = pixel_player(19, 1, 200.0)
    passer_by = pixel_player(17, 0, 230.0)
    receiver = pixel_player(7, 1, 400.0)
    players = [holder, toucher, passer_by, receiver]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 11, "Team 0", players, pixel_ball(0.0)))
        frame += 1
    for _ in range(2):
        frames.append((frame, 11, "Team 0", players, pixel_ball(200.0)))
        frame += 1
    frames.append((frame, 11, "Team 0", players, pixel_ball(200.0, source="predicted")))
    frame += 1
    for _ in range(2):
        frames.append((frame, 11, "Team 0", players, pixel_ball(230.0)))
        frame += 1
    for _ in range(4):
        frames.append((frame, None, None, players, pixel_ball(320.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 7, "Team 1", players, pixel_ball(400.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 1, "the one-touch is a pass")
    assert_equal(metrics["short_touch_passes"], 1, "it was opened by the short touch")
    assert_equal(metrics["events"][0]["from_track_id"], 19, "19 is the origin")
    assert_equal(metrics["events"][0]["to_track_id"], 7, "7 is the receiver")
    assert_equal(metrics["events"][0]["team"], "Team 1", "same team")
    assert_equal(metrics["disputes"], 0, "the passer-by does not make a dispute")
    assert_equal(metrics["turnovers"], 0, "the passer-by does not make a turnover")


def test_a_player_sixty_pixels_away_does_not_open_a_short_touch():
    detector = PassDetector()
    holder = pixel_player(11, 0, 0.0)
    nearby = pixel_player(4, 0, 60.0)
    receiver = pixel_player(10, 0, 400.0)
    players = [holder, nearby, receiver]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 11, "Team 0", players, pixel_ball(0.0)))
        frame += 1
    for _ in range(4):
        frames.append((frame, None, None, players, pixel_ball(0.0)))
        frame += 1
    for _ in range(5):
        frames.append((frame, 10, "Team 0", players, pixel_ball(400.0)))
        frame += 1
    metrics = run_frames(detector, frames)
    assert_equal(metrics["short_touch_passes"], 0, "60 px is not a one-touch")
    assert_equal(metrics["events"][0]["from_track_id"], 11, "the confirmed possessor stays the origin")


def test_a_possessor_without_a_team_is_not_a_pass_or_a_dispute():
    detector = PassDetector()
    donor = player(13, None, 0.0, 0.0)
    receiver = player(7, 1, 1500.0, 0.0)
    players = [donor, receiver]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 13, None, players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(3):
        frames.append((frame, None, None, players, ball(700.0, 0.0)))
        frame += 1
    frames.append((frame, 7, "Team 1", players, ball(1500.0, 0.0)))
    metrics = run_frames(detector, frames)
    assert_equal(metrics["total_passes"], 0, "no team does not start a pass")
    assert_equal(metrics["disputes"], 0, "no team does not start a dispute")
    assert_equal(metrics["turnovers"], 0, "no team does not start a turnover")
    assert_equal(metrics["discarded_unstable_possession"], 1, "the link is discarded")


def test_both_teams_on_the_ball_is_a_dispute():
    detector = PassDetector()
    donor = player(1, 0, 0.0, 0.0)
    opponent = player(7, 1, 100.0, 0.0)
    receiver = player(2, 0, 1500.0, 0.0)
    players = [donor, opponent, receiver]
    frames = []
    frame = 1
    for _ in range(6):
        frames.append((frame, 1, "Team 0", players, ball(0.0, 0.0)))
        frame += 1
    for _ in range(3):
        frames.append((frame, None, None, players, ball(50.0, 0.0)))
        frame += 1
    frames.append((frame, 2, "Team 0", players, ball(1500.0, 0.0)))
    metrics = run_frames(detector, frames)
    assert_equal(metrics["disputes"], 1, "two teams on the ball is a dispute")
    assert_equal(metrics["total_passes"], 0, "a dispute is not a pass")
    assert_equal(metrics["turnovers"], 0, "a dispute replaces a clean turnover")
    assert_equal(metrics["dispute_events"][0]["from_track_id"], 1, "donor")
    assert_equal(metrics["dispute_events"][0]["to_track_id"], 2, "next possessor")


if __name__ == "__main__":
    test_confirms_stable_same_team_pass()
    test_confirmed_snapshot_is_enough()
    test_ball_near_donor_does_not_veto_confirmed_reception()
    test_discards_flight_beyond_distance_budget()
    test_extended_window_accepts_observed_slow_pass()
    test_extended_window_without_flight_evidence_stays_unconfirmed()
    test_flight_budget_scales_with_distance()
    test_reception_lag_does_not_count_as_flight()
    test_opponent_alone_on_the_ball_intercepts()
    test_ball_faster_than_physics_is_rejected()
    test_homography_jitter_does_not_break_trajectory_or_distance()
    test_discards_short_and_long_passes()
    test_discards_incompatible_ball_direction()
    test_each_possession_transfer_is_its_own_pass()
    test_first_possession_is_not_a_pass()
    test_confirms_pass_when_ball_disappears_between_possessors()
    test_predicted_ball_does_not_block_possession_pass()
    test_predicted_drift_during_reception_does_not_cancel_pass()
    test_missing_ball_still_respects_timeout_and_distance()
    test_different_team_is_not_a_pass()
    test_dispute_on_the_frame_before_the_switch()
    test_same_player_return_is_a_carry()
    test_wall_pass_becomes_two_passes()
    test_same_player_recovery_without_a_wall_stays_a_carry()
    test_opponent_on_the_ball_is_not_a_wall()
    test_teammate_outside_contest_radius_is_not_a_wall()
    test_wall_leg_below_the_score_is_not_created()
    test_one_frame_wall_touch_is_still_two_passes()
    test_a_clear_receiver_is_not_a_dispute()
    test_a_one_touch_is_a_pass_without_a_long_possession()
    test_a_player_sixty_pixels_away_does_not_open_a_short_touch()
    test_a_possessor_without_a_team_is_not_a_pass_or_a_dispute()
    test_both_teams_on_the_ball_is_a_dispute()
    test_verified_pixel_flight_can_run_past_one_second()
    test_hidden_pixel_flight_past_one_second_is_rejected()
    test_verified_pixel_flight_beyond_two_seconds_is_rejected()
    print("pass detector tests passed")
