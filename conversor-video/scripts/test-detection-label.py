#!/usr/bin/env python3
"""Overlay labels show the player id and team, never the confidence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from run_yolo_analysis import detection_label  # noqa: E402


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def assert_no_confidence(label, message):
    if "%" in label:
        raise AssertionError(f"{message}: label still shows a percentage: {label!r}")


def test_player_label_is_id_and_team():
    label = detection_label(("Player", 12, 0))
    assert_equal(label, "Player #12 [Team 0]", "blue shirt")
    assert_no_confidence(label, "player")
    other = detection_label(("Player", 4, 1))
    assert_equal(other, "Player #4 [Team 1]", "red shirt")
    assert_no_confidence(other, "other team")


def test_goalkeeper_label_is_id_and_team():
    label = detection_label(("Goalkeeper", 7, 1))
    assert_equal(label, "Goalkeeper #7 [Team 1]", "goalkeeper")
    assert_no_confidence(label, "goalkeeper")


def test_missing_team_omits_confidence():
    label = detection_label(("Player", 9, None))
    assert_equal(label, "Player #9", "team not assigned yet")
    assert_no_confidence(label, "missing team")


def test_referee_and_ball_omit_confidence():
    referee = detection_label(("Referee", 25))
    assert_equal(referee, "Referee #25", "referee has no team")
    assert_no_confidence(referee, "referee")
    ball = detection_label("Ball")
    assert_equal(ball, "Ball", "ball")
    assert_no_confidence(ball, "ball")


def main():
    test_player_label_is_id_and_team()
    test_goalkeeper_label_is_id_and_team()
    test_missing_team_omits_confidence()
    test_referee_and_ball_omit_confidence()
    print("detection label tests passed")


if __name__ == "__main__":
    main()
