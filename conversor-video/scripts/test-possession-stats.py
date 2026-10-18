#!/usr/bin/env python3
"""The team possession tally only counts snapshots PossessionEngine already confirmed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from possession_stats import PossessionTeamAccumulator  # noqa: E402


def assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def test_splits_confirmed_teams_and_empty_frames():
    stats = PossessionTeamAccumulator(fps=10)
    for _ in range(10):
        stats.observe("Team 0")
    for _ in range(5):
        stats.observe("Team 1")
    for _ in range(5):
        stats.observe(None)
    report = stats.as_dict()
    assert_equal(report["frames_total"], 20, "every observed frame is counted")
    assert_equal(report["frames_team_0"], 10, "Team 0 frames")
    assert_equal(report["frames_team_1"], 5, "Team 1 frames")
    assert_equal(report["frames_without_possessor"], 5, "frames with no possessor")
    assert_equal(report["seconds_team_0"], 1.0, "Team 0 time")
    assert_equal(report["seconds_team_1"], 0.5, "Team 1 time")
    assert_equal(report["seconds_without_possessor"], 0.5, "time without a possessor")
    assert_equal(report["percent_team_0"], 50.0, "Team 0 share of the video")
    assert_equal(report["percent_team_1"], 25.0, "Team 1 share of the video")
    assert_equal(report["percent_without_possessor"], 25.0, "share without a possessor")


def test_unknown_team_stays_without_possessor():
    stats = PossessionTeamAccumulator(fps=30)
    stats.observe("blue")
    stats.observe("")
    report = stats.as_dict()
    assert_equal(report["frames_team_0"], 0, "unknown names are not Team 0")
    assert_equal(report["frames_team_1"], 0, "unknown names are not Team 1")
    assert_equal(report["frames_without_possessor"], 2, "unknown names have no possessor")


def test_same_team_pass_reassigns_the_open_interval_once():
    stats = PossessionTeamAccumulator(fps=10)
    stats.observe("Team 1")
    for _ in range(10):
        during = stats.observe(None)
    assert_equal(during["frames_without_possessor"], 10, "the flight stays open while the pass is unresolved")
    assert_equal(during["frames_team_1"], 1, "only the confirmed frame counts before the pass")
    confirmed = {
        "type": "pass",
        "team": "Team 1",
        "from_track_id": 15,
        "to_track_id": 11,
    }
    after = stats.observe("Team 1", [confirmed])
    assert_equal(after["frames_team_1"], 12, "the 10 open frames join the confirmation frame and the start")
    assert_equal(after["frames_without_possessor"], 0, "the open interval left the empty bucket")
    assert_equal(after["frames_reassigned_team_1"], 10, "exactly the open interval was moved")
    assert_equal(after["frames_total"], 12, "no frame was added twice")
    again = stats.observe("Team 1", [confirmed])
    assert_equal(again["frames_reassigned_team_1"], 10, "the same pass is not applied twice")
    assert_equal(again["frames_without_possessor"], 0, "a repeated event list does not move frames again")


def test_conduction_turnover_and_unconfirmed_keep_the_gap_empty():
    for label, closing_team, events in (
        ("conduction", "Team 1", None),
        ("turnover", "Team 0", None),
        ("unconfirmed", "Team 1", []),
    ):
        stats = PossessionTeamAccumulator(fps=10)
        stats.observe("Team 1")
        for _ in range(4):
            stats.observe(None)
        report = stats.observe(closing_team, events)
        assert_equal(report["frames_reassigned_team_1"], 0, label)
        assert_equal(report["frames_reassigned_team_0"], 0, label)
        assert_equal(report["frames_without_possessor"], 4, label)


def test_a_later_pass_does_not_reclaim_an_earlier_gap():
    stats = PossessionTeamAccumulator(fps=10)
    stats.observe("Team 1")
    for _ in range(10):
        stats.observe(None)
    stats.observe("Team 1")
    for _ in range(5):
        stats.observe(None)
    report = stats.observe("Team 1", [{"type": "pass", "team": "Team 1"}])
    assert_equal(report["frames_reassigned_team_1"], 5, "only the gap closed by the pass moves")
    assert_equal(report["frames_without_possessor"], 10, "the earlier conduction gap stays empty")


def test_video_that_ends_open_keeps_the_gap_empty():
    stats = PossessionTeamAccumulator(fps=30)
    stats.observe("Team 0")
    for _ in range(3):
        report = stats.observe(None)
    assert_equal(report["frames_team_0"], 1, "the opening possession stays")
    assert_equal(report["frames_without_possessor"], 3, "an unfinished gap is not given to a team")
    assert_equal(report["frames_reassigned_team_0"], 0, "nothing was reassigned")


def test_missing_fps_still_counts_frames():
    stats = PossessionTeamAccumulator(fps=0)
    stats.observe("Team 1")
    report = stats.as_dict()
    assert_equal(report["frames_team_1"], 1, "a frame is counted without a usable fps")
    assert_equal(report["seconds_team_1"], round(1 / 30, 3), "fallback fps is 30")


if __name__ == "__main__":
    test_splits_confirmed_teams_and_empty_frames()
    test_unknown_team_stays_without_possessor()
    test_same_team_pass_reassigns_the_open_interval_once()
    test_conduction_turnover_and_unconfirmed_keep_the_gap_empty()
    test_a_later_pass_does_not_reclaim_an_earlier_gap()
    test_video_that_ends_open_keeps_the_gap_empty()
    test_missing_fps_still_counts_frames()
    print("possession stats tests passed")
