#!/usr/bin/env python3
"""The team possession tally only counts snapshots PossessionEngine already confirmed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from possession_stats import (  # noqa: E402
    SAME_TEAM_GAP_FILL_SECONDS,
    PossessionTeamAccumulator,
)


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


def test_same_team_return_within_one_second_fills_the_gap():
    for label, events in (
        ("conduction", None),
        ("no pass event", []),
    ):
        stats = PossessionTeamAccumulator(fps=10)
        stats.observe("Team 1")
        for _ in range(4):
            stats.observe(None)
        report = stats.observe("Team 1", events)
        assert_equal(report["frames_reassigned_team_1"], 4, label)
        assert_equal(report["frames_reassigned_team_0"], 0, label)
        assert_equal(report["frames_without_possessor"], 0, label)
        assert_equal(report["frames_team_1"], 6, label)
        assert_equal(report["frames_total"], 6, label)


def test_same_team_return_at_one_second_still_fills():
    fps = 30
    gap_frames = int(fps * SAME_TEAM_GAP_FILL_SECONDS)
    stats = PossessionTeamAccumulator(fps=fps)
    stats.observe("Team 0")
    for _ in range(gap_frames):
        stats.observe(None)
    report = stats.observe("Team 0")
    assert_equal(report["frames_reassigned_team_0"], gap_frames, "one second joins the team")
    assert_equal(report["frames_without_possessor"], 0, "the one-second gap left the empty bucket")


def test_same_team_return_past_one_second_stays_empty():
    stats = PossessionTeamAccumulator(fps=10)
    stats.observe("Team 1")
    for _ in range(11):
        stats.observe(None)
    report = stats.observe("Team 1")
    assert_equal(report["frames_reassigned_team_1"], 0, "1.1 seconds is not continuity")
    assert_equal(report["frames_without_possessor"], 11, "the long gap stays empty")
    assert_equal(report["frames_team_1"], 2, "only the confirmed frames count")


def test_other_team_does_not_inherit_the_gap():
    stats = PossessionTeamAccumulator(fps=10)
    stats.observe("Team 1")
    for _ in range(4):
        stats.observe(None)
    report = stats.observe(
        "Team 0",
        [{"type": "turnover", "team": "Team 0"}],
    )
    assert_equal(report["frames_reassigned_team_0"], 0, "the new team does not receive the gap")
    assert_equal(report["frames_reassigned_team_1"], 0, "the old team does not keep the gap")
    assert_equal(report["frames_without_possessor"], 4, "the change of team leaves the gap empty")
    assert_equal(report["frames_team_0"], 1, "only the confirming frame is Team 0")
    assert_equal(report["frames_team_1"], 1, "Team 1 keeps the frame it had")


def test_long_gap_still_moves_when_a_pass_closes_it():
    stats = PossessionTeamAccumulator(fps=10)
    stats.observe("Team 0")
    for _ in range(15):
        stats.observe(None)
    report = stats.observe("Team 0", [{"type": "pass", "team": "Team 0"}])
    assert_equal(report["frames_reassigned_team_0"], 15, "a confirmed pass still closes 1.5 seconds")
    assert_equal(report["frames_without_possessor"], 0, "the flight left the empty bucket")
    again = stats.observe("Team 0", [{"type": "pass", "team": "Team 0"}])
    assert_equal(again["frames_reassigned_team_0"], 15, "that pass is not applied twice")


def test_dispute_event_does_not_fill_a_long_gap():
    stats = PossessionTeamAccumulator(fps=10)
    stats.observe("Team 0")
    for _ in range(15):
        stats.observe(None)
    report = stats.observe("Team 0", [{"type": "dispute", "team": "Team 0"}])
    assert_equal(report["frames_reassigned_team_0"], 0, "a dispute is not a pass")
    assert_equal(report["frames_without_possessor"], 15, "the long gap stays empty")


def test_a_later_pass_does_not_reclaim_an_earlier_gap():
    stats = PossessionTeamAccumulator(fps=10)
    stats.observe("Team 1")
    for _ in range(11):
        stats.observe(None)
    stats.observe("Team 1")
    for _ in range(5):
        stats.observe(None)
    report = stats.observe("Team 1", [{"type": "pass", "team": "Team 1"}])
    assert_equal(report["frames_reassigned_team_1"], 5, "only the gap closed by the pass moves")
    assert_equal(report["frames_without_possessor"], 11, "the earlier gap past one second stays empty")


def test_a_short_gap_is_not_moved_again_by_a_later_pass():
    stats = PossessionTeamAccumulator(fps=10)
    stats.observe("Team 1")
    for _ in range(4):
        stats.observe(None)
    returned = stats.observe("Team 1")
    assert_equal(returned["frames_reassigned_team_1"], 4, "the return itself closes the short gap")
    for _ in range(5):
        stats.observe(None)
    report = stats.observe("Team 1", [{"type": "pass", "team": "Team 1"}])
    assert_equal(report["frames_reassigned_team_1"], 9, "the pass adds only its own gap")
    assert_equal(report["frames_without_possessor"], 0, "both closed gaps left the empty bucket")


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
    test_same_team_return_within_one_second_fills_the_gap()
    test_same_team_return_at_one_second_still_fills()
    test_same_team_return_past_one_second_stays_empty()
    test_other_team_does_not_inherit_the_gap()
    test_long_gap_still_moves_when_a_pass_closes_it()
    test_dispute_event_does_not_fill_a_long_gap()
    test_a_later_pass_does_not_reclaim_an_earlier_gap()
    test_a_short_gap_is_not_moved_again_by_a_later_pass()
    test_video_that_ends_open_keeps_the_gap_empty()
    test_missing_fps_still_counts_frames()
    print("possession stats tests passed")
