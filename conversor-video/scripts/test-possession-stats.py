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


def test_missing_fps_still_counts_frames():
    stats = PossessionTeamAccumulator(fps=0)
    stats.observe("Team 1")
    report = stats.as_dict()
    assert_equal(report["frames_team_1"], 1, "a frame is counted without a usable fps")
    assert_equal(report["seconds_team_1"], round(1 / 30, 3), "fallback fps is 30")


if __name__ == "__main__":
    test_splits_confirmed_teams_and_empty_frames()
    test_unknown_team_stays_without_possessor()
    test_missing_fps_still_counts_frames()
    print("possession stats tests passed")
