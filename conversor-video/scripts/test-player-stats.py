#!/usr/bin/env python3
"""Per-track totals only count tracks, possession, and passes the pipeline already produced."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from player_stats import PlayerStatsAccumulator  # noqa: E402


def assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def player(track_id, team_id, field_x=None, field_y=None):
    return {
        "track_id": track_id,
        "team_id": team_id,
        "field_x": field_x,
        "field_y": field_y,
    }


def by_id(report):
    return {track["track_id"]: track for track in report["tracks"]}


def test_counts_presence_possession_and_a_new_spell_after_a_gap():
    stats = PlayerStatsAccumulator(fps=10)
    stats.observe([player(11, 1, 1000, 2000)], 11, "Team 1", [])
    stats.observe([player(11, 1, 3000, 4000)], 11, "Team 1", [])
    stats.observe([player(11, 1, 1000, 2000)], None, None, [])
    report = by_id(stats.observe([player(11, 1, 3000, 4000)], 11, "Team 1", []))
    row = report[11]
    assert_equal(row["frames_seen"], 4, "every visible frame")
    assert_equal(row["frames_with_ball"], 3, "confirmed possession frames")
    assert_equal(row["possession_spells"], 2, "a gap starts a new spell")
    assert_equal(row["seconds_with_ball"], 0.3, "seconds from frames and fps")
    assert_equal(row["team"], "Team 1", "team from the confirmed possessor")
    assert_equal(row["mean_x_cm"], 2000.0, "mean length")
    assert_equal(row["mean_y_cm"], 3000.0, "mean width")
    assert_equal(row["percent_of_team"], 100.0, "only holder on Team 1")


def test_pass_events_credit_maker_and_receiver_once():
    stats = PlayerStatsAccumulator(fps=30)
    first = [{"type": "pass", "from_track_id": 11, "to_track_id": 15, "team": "Team 1"}]
    stats.observe([player(11, 1), player(15, 1)], 15, "Team 1", first)
    second = first + [{"type": "pass", "from_track_id": 15, "to_track_id": 11, "team": "Team 1"}]
    report = by_id(stats.observe([player(11, 1), player(15, 1)], 11, "Team 1", second))
    assert_equal(report[11]["passes_made"], 1, "11 made the first pass")
    assert_equal(report[11]["passes_received"], 1, "11 received the second pass")
    assert_equal(report[15]["passes_made"], 1, "15 made the second pass")
    assert_equal(report[15]["passes_received"], 1, "15 received the first pass")
    assert_equal(report[11]["frames_with_ball"], 1, "possession frames stay on the snapshot")
    assert_equal(report[15]["frames_with_ball"], 1, "15 held the ball on the first frame")


def test_percent_is_share_of_the_same_team_only():
    stats = PlayerStatsAccumulator(fps=10)
    for _ in range(6):
        stats.observe([player(11, 1, 1000, 1000)], 11, "Team 1", [])
    for _ in range(4):
        stats.observe([player(15, 1, 2000, 1000)], 15, "Team 1", [])
    stats.observe([player(3, 0, 9000, 1000)], 3, "Team 0", [])
    report = by_id(stats.as_dict())
    assert_equal(report[11]["percent_of_team"], 60.0, "11 has 6 of Team 1's 10 frames")
    assert_equal(report[15]["percent_of_team"], 40.0, "15 has the rest of Team 1")
    assert_equal(report[3]["percent_of_team"], 100.0, "Team 0 is a separate total")


def test_ignores_referee_ids_the_caller_does_not_pass_and_bad_positions():
    stats = PlayerStatsAccumulator(fps=10)
    stats.observe(
        [
            player(7, 0, 1000, 1000),
            player(7, 0, 5000, 5000),
            player(-1, 0, 1000, 1000),
            player(8, 1, None, None),
            player(9, 1, 20000, 1000),
        ],
        None,
        None,
        [],
    )
    report = by_id(stats.as_dict())
    assert_equal(set(report), {7, 8, 9}, "negative ids are dropped")
    assert_equal(report[7]["frames_seen"], 1, "a repeated id in one frame counts once")
    assert_equal(report[7]["position_samples"], 1, "one accepted foot point")
    assert_equal(report[8]["mean_x_cm"], None, "missing homography stays empty")
    assert_equal(report[9]["position_samples"], 0, "a point far outside the pitch is ignored")
    assert_equal(report[9]["percent_of_team"], None, "no possession means no team share")


def test_same_player_return_is_two_spells():
    stats = PlayerStatsAccumulator(fps=30)
    stats.observe([player(11, 1)], 11, "Team 1", [])
    stats.observe([], None, None, [])
    report = by_id(stats.observe([player(11, 1)], 11, "Team 1", []))
    assert_equal(report[11]["possession_spells"], 2, "the ball coming back starts another spell")
    assert_equal(report[11]["frames_with_ball"], 2, "both confirmed frames count")


if __name__ == "__main__":
    test_counts_presence_possession_and_a_new_spell_after_a_gap()
    test_pass_events_credit_maker_and_receiver_once()
    test_percent_is_share_of_the_same_team_only()
    test_ignores_referee_ids_the_caller_does_not_pass_and_bad_positions()
    test_same_player_return_is_two_spells()
    print("player stats tests passed")
