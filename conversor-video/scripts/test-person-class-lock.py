#!/usr/bin/env python3
"""Checks for non-person rejection and per-track class locks."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from person_class import (  # noqa: E402
    PERSON_CLASS_CONFIRM_STREAK,
    PERSON_CLASS_UNLOCK_STREAK,
    PersonClassLock,
    non_person_rejection_reason,
)

PLAYER = 2
REFEREE = 3
GOALKEEPER = 1


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def solid_bgr(hue, saturation=200, value=180, width=40, height=120):
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = saturation
    hsv[:, :, 2] = value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_rejects_pole_flag_and_wide_box():
    player_box = [10, 10, 50, 130]
    assert_equal(non_person_rejection_reason(player_box), None, "upright person")
    assert_equal(non_person_rejection_reason([0, 0, 8, 140]), "thin_non_person", "pole")
    assert_equal(non_person_rejection_reason([0, 0, 200, 40]), "wide_non_person", "flat object")
    flag = solid_bgr(15, saturation=220, value=220, width=18, height=110)
    flag_box = [0, 0, 18, 110]
    assert_equal(non_person_rejection_reason(flag_box, flag), "corner_flag", "corner flag")


def test_player_does_not_become_referee_on_flicker():
    lock = PersonClassLock()
    for frame in range(1, PERSON_CLASS_CONFIRM_STREAK + 1):
        label = lock.assign(4, PLAYER, frame, verifying=False)
    assert_equal(label, PLAYER, "player confirmed")
    assert_equal(lock.is_locked(4), True, "locked")
    for frame in range(10, 10 + PERSON_CLASS_UNLOCK_STREAK - 1):
        label = lock.assign(4, REFEREE, frame, verifying=True)
        assert_equal(label, PLAYER, "flicker stays player")
    assert_equal(lock.metrics()["class_changes_after_lock"], 0, "no post-lock change yet")


def test_referee_does_not_become_player_on_flicker():
    lock = PersonClassLock()
    for frame in range(1, PERSON_CLASS_CONFIRM_STREAK + 1):
        lock.assign(8, REFEREE, frame, verifying=False)
    label = lock.assign(8, PLAYER, 12, verifying=True)
    assert_equal(label, REFEREE, "referee stays referee")
    assert_equal(lock.is_locked(8), True, "still locked")


def test_goalkeeper_stays_goalkeeper():
    lock = PersonClassLock()
    for frame in range(1, PERSON_CLASS_CONFIRM_STREAK + 1):
        lock.assign(2, GOALKEEPER, frame, verifying=False)
    label = lock.assign(2, PLAYER, 9, verifying=True)
    assert_equal(label, GOALKEEPER, "goalkeeper stays")


def test_unlocks_only_after_consecutive_disagreements():
    lock = PersonClassLock()
    for frame in range(1, PERSON_CLASS_CONFIRM_STREAK + 1):
        lock.assign(6, PLAYER, frame, verifying=False)
    lock.assign(6, REFEREE, 20, verifying=True)
    lock.assign(6, PLAYER, 21, verifying=True)
    label = lock.assign(6, REFEREE, 22, verifying=True)
    assert_equal(label, PLAYER, "agreement resets the streak")
    for offset in range(PERSON_CLASS_UNLOCK_STREAK):
        label = lock.assign(6, REFEREE, 30 + offset, verifying=True)
    assert_equal(label, REFEREE, "streak unlocks")
    assert_equal(lock.metrics()["class_changes_after_lock"], 1, "one post-lock change")


if __name__ == "__main__":
    test_rejects_pole_flag_and_wide_box()
    test_player_does_not_become_referee_on_flicker()
    test_referee_does_not_become_player_on_flicker()
    test_goalkeeper_stays_goalkeeper()
    test_unlocks_only_after_consecutive_disagreements()
    print("person class lock tests passed")
