#!/usr/bin/env python3
"""Stability checks for shirt-color anchoring and per-track team locks."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from roboflow_team import (  # noqa: E402
    TEAM_LOCK_CONFIRM_STREAK,
    TEAM_LOCK_UNLOCK_STREAK,
    TeamTrackLock,
    anchor_cluster_labels,
    jersey_blueness,
    jersey_color_profile,
    reject_training_crop,
)


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def solid_bgr(hue, saturation=200, value=180, width=40, height=80):
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = saturation
    hsv[:, :, 2] = value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_blue_and_red_shirts_have_opposite_scores():
    blue = jersey_color_profile(solid_bgr(110))
    red = jersey_color_profile(solid_bgr(0))
    assert jersey_blueness(blue) > jersey_blueness(red), (blue, red)
    assert blue["blue"] > blue["red"], blue
    assert red["red"] > red["blue"], red


def test_anchor_swaps_raw_kmeans_ids_to_shirt_color():
    blue = solid_bgr(110)
    red = solid_bgr(0)
    crops = [red, red, blue, blue]
    raw_labels = [0, 0, 1, 1]
    mapping, report = anchor_cluster_labels(crops, raw_labels)
    assert_equal(mapping[0], 1, "raw red cluster becomes team red")
    assert_equal(mapping[1], 0, "raw blue cluster becomes team blue")
    assert_equal(report["labels_were_remapped"], True, "permutation recorded")

    identity_labels = [0, 0, 1, 1]
    identity_crops = [blue, blue, red, red]
    identity_mapping, identity_report = anchor_cluster_labels(identity_crops, identity_labels)
    assert_equal(identity_mapping[0], 0, "blue cluster stays team blue")
    assert_equal(identity_mapping[1], 1, "red cluster stays team red")
    assert_equal(identity_report["labels_were_remapped"], False, "identity mapping")


def test_two_runs_with_swapped_labels_share_semantic_ids():
    blue = solid_bgr(115)
    red = solid_bgr(5)
    first_mapping, _first = anchor_cluster_labels([blue, blue, red, red], [0, 0, 1, 1])
    second_mapping, _second = anchor_cluster_labels([blue, blue, red, red], [1, 1, 0, 0])
    first_semantic = [first_mapping[label] for label in (0, 0, 1, 1)]
    second_semantic = [second_mapping[label] for label in (1, 1, 0, 0)]
    assert_equal(first_semantic, second_semantic, "same shirts stay the same teams")
    assert_equal(first_semantic, [0, 0, 1, 1], "blue then red")


def test_rejects_flag_post_and_grass_but_keeps_colored_shirt():
    assert_equal(reject_training_crop(solid_bgr(110)), None, "blue shirt is a player crop")
    thin = solid_bgr(0, width=20, height=140)
    assert_equal(reject_training_crop(thin), "thin_non_player", "post")
    grass = solid_bgr(60, saturation=160, value=140, width=40, height=80)
    assert_equal(reject_training_crop(grass), "mostly_grass", "grass")
    flag = solid_bgr(15, saturation=220, value=220, width=22, height=110)
    assert_equal(reject_training_crop(flag), "corner_flag", "corner flag")
    official = solid_bgr(0, saturation=10, value=30, width=36, height=90)
    assert_equal(reject_training_crop(official), "low_chroma_official", "dark official")


def test_lock_survives_single_refresh_and_flips_only_after_streak():
    lock = TeamTrackLock()
    for frame in range(1, TEAM_LOCK_CONFIRM_STREAK + 1):
        team = lock.assign(7, 0, frame, verifying=False)
    assert_equal(team, 0, "confirmed blue")
    assert_equal(lock.is_locked(7), True, "locked")

    for frame in range(10, 10 + TEAM_LOCK_UNLOCK_STREAK - 1):
        team = lock.assign(7, 1, frame, verifying=True)
        assert_equal(team, 0, "one or two disagreements do not overwrite")
        assert_equal(lock.is_locked(7), True, "still locked")

    team = lock.assign(7, 1, 40, verifying=True)
    assert_equal(team, 1, "third consecutive disagreement unlocks")
    assert_equal(lock.metrics()["team_changes_after_lock"], 1, "one post-lock change")
    assert_equal(lock.metrics()["refresh_contradictions"], TEAM_LOCK_UNLOCK_STREAK, "contradiction count")


def test_refresh_agreement_resets_contradiction_streak():
    lock = TeamTrackLock()
    for frame in range(1, TEAM_LOCK_CONFIRM_STREAK + 1):
        lock.assign(4, 1, frame, verifying=False)
    lock.assign(4, 0, 20, verifying=True)
    lock.assign(4, 0, 21, verifying=True)
    team = lock.assign(4, 1, 22, verifying=True)
    assert_equal(team, 1, "agreement clears the streak")
    team = lock.assign(4, 0, 23, verifying=True)
    assert_equal(team, 1, "streak restarted")
    assert_equal(lock.metrics()["team_changes_after_lock"], 0, "no unlock")


def test_cached_interval_skips_locked_tracks():
    lock = TeamTrackLock()
    lock.assign(9, 0, 1, verifying=False)
    lock.assign(9, 0, 2, verifying=False)
    lock.assign(9, 0, 3, verifying=False)
    assert_equal(lock.needs_observation(9, 33, 30), False, "inside refresh window")
    assert_equal(lock.needs_observation(9, 34, 30), True, "refresh due")
    assert_equal(lock.needs_observation(3, 1, 30), True, "unknown track")


if __name__ == "__main__":
    test_blue_and_red_shirts_have_opposite_scores()
    test_anchor_swaps_raw_kmeans_ids_to_shirt_color()
    test_two_runs_with_swapped_labels_share_semantic_ids()
    test_rejects_flag_post_and_grass_but_keeps_colored_shirt()
    test_lock_survives_single_refresh_and_flips_only_after_streak()
    test_refresh_agreement_resets_contradiction_streak()
    test_cached_interval_skips_locked_tracks()
    print("team classifier stability tests passed")
