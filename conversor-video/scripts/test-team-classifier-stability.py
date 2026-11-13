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
    TeamClassifier,
    TeamTrackLock,
    anchor_cluster_labels,
    crops_form_two_teams,
    jersey_blueness,
    jersey_color_profile,
    measure_torso,
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


def test_lock_survives_a_run_of_opposite_votes():
    lock = TeamTrackLock()
    for frame in range(1, TEAM_LOCK_CONFIRM_STREAK + 1):
        team = lock.assign(7, 0, frame, verifying=False)
    assert_equal(team, 0, "confirmed dark shirt")
    assert_equal(lock.is_locked(7), True, "locked")

    for frame in range(10, 10 + TEAM_LOCK_UNLOCK_STREAK + 4):
        team = lock.assign(7, 1, frame, verifying=True)
        assert_equal(team, 0, "a locked shirt does not change")
        assert_equal(lock.is_locked(7), True, "still locked")
    assert_equal(lock.metrics()["team_changes_after_lock"], 0, "no post-lock change")
    assert_equal(lock.metrics()["unlocks"], 0, "the lock does not open")
    assert_equal(
        lock.metrics()["refresh_contradictions"],
        TEAM_LOCK_UNLOCK_STREAK + 4,
        "contradictions are still counted",
    )


def test_flicker_before_lock_keeps_the_first_clear_team():
    lock = TeamTrackLock()
    votes = [0, 1, 0, 1, 0, 1, 0]
    published = []
    for frame, vote in enumerate(votes, start=1):
        published.append(lock.assign(49, vote, frame, verifying=False))
    assert_equal(published, [0, 0, 0, 0, 0, 0, 0], "alternating votes do not rename the track")
    assert_equal(lock.is_locked(49), True, "the repeated first team locks")
    assert_equal(lock.metrics()["prelock_team_switches"], 0, "the team never moved")


def test_three_opposite_votes_before_lock_can_correct_the_first_guess():
    lock = TeamTrackLock()
    assert_equal(lock.assign(31, 1, 1, verifying=False), 1, "first clear vote")
    assert_equal(lock.assign(31, 0, 2, verifying=False), 1, "one opposite vote waits")
    assert_equal(lock.assign(31, 0, 3, verifying=False), 1, "two opposite votes wait")
    assert_equal(lock.assign(31, 0, 4, verifying=False), 0, "three opposite votes correct the guess")
    assert_equal(lock.is_locked(31), True, "the corrected team locks")
    assert_equal(lock.assign(31, 1, 5, verifying=True), 0, "the correction stays")


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


def fit_light_and_dark():
    light = [solid_bgr(0, saturation=12, value=210) for _ in range(6)]
    dark = [solid_bgr(170, saturation=150, value=90) for _ in range(6)]
    classifier = TeamClassifier()
    classifier.fit(light + dark)
    return classifier


def test_light_shirt_and_dark_shirt_get_stable_teams():
    classifier = fit_light_and_dark()
    assert_equal(classifier.color_anchor["mode"], "value", "brightness separates this match")
    assert_equal(classifier.color_anchor["team_0"], "dark", "team 0 is the dark shirt")
    assert_equal(classifier.color_anchor["team_1"], "light", "team 1 is the light shirt")
    swapped = TeamClassifier()
    swapped.fit(
        [solid_bgr(170, saturation=140, value=80) for _ in range(6)]
        + [solid_bgr(40, saturation=8, value=220) for _ in range(6)]
    )
    white = solid_bgr(90, saturation=10, value=205)
    garnet = solid_bgr(172, saturation=160, value=95)
    assert_equal(int(classifier.predict([white])[0]), 1, "white shirt")
    assert_equal(int(classifier.predict([garnet])[0]), 0, "dark shirt")
    assert_equal(int(swapped.predict([white])[0]), 1, "white stays team 1 after a swapped fit")
    assert_equal(int(swapped.predict([garnet])[0]), 0, "dark stays team 0 after a swapped fit")


def test_gradual_broadcast_shirts_still_split():
    dark = []
    light = []
    for value, saturation in ((90, 140), (110, 110), (125, 95), (140, 80)):
        dark.extend(solid_bgr(160, saturation=saturation, value=value) for _ in range(2))
    for value, saturation in ((175, 28), (200, 18), (230, 12), (245, 10)):
        light.extend(solid_bgr(110, saturation=saturation, value=value) for _ in range(2))
    classifier = TeamClassifier()
    classifier.fit(dark + light)
    assert_equal(classifier.color_anchor["mode"], "value", "gradual kits still use brightness")
    assert_equal(int(classifier.predict([solid_bgr(164, saturation=100, value=115)])[0]), 0, "mid dark shirt")
    assert_equal(int(classifier.predict([solid_bgr(112, saturation=16, value=225)])[0]), 1, "bright shirt")


def test_shadowed_white_stays_light_and_sunlit_dark_stays_dark():
    classifier = fit_light_and_dark()
    shadow = solid_bgr(0, saturation=18, value=150)
    sunlit = solid_bgr(168, saturation=130, value=145)
    assert_equal(int(classifier.predict([shadow])[0]), 1, "dim white shirt")
    assert_equal(int(classifier.predict([sunlit])[0]), 0, "lit dark shirt")


def test_a_small_clear_shirt_votes_and_a_small_gray_shirt_does_not():
    green = solid_bgr(46, saturation=110, value=215, width=14, height=26)
    gray = solid_bgr(20, saturation=20, value=140, width=14, height=26)
    red = solid_bgr(8, saturation=140, value=160, width=40, height=80)
    neon = solid_bgr(46, saturation=100, value=220, width=40, height=80)
    classifier = TeamClassifier()
    classifier.fit([red, red, red, red, neon, neon, neon, neon])
    assert_equal(int(classifier.predict([green])[0]), 1, "a small bright green shirt is team 1")
    assert_equal(int(classifier.predict([gray])[0]), -1, "a small dull crop still abstains")
    assert_equal(measure_torso(green)["abstain"], False, "the green measurement is usable")
    assert_equal(measure_torso(gray)["reason"], "small", "dull color does not open the small gate")


def test_broadcast_lime_yellow_abstains_and_team_shirts_still_vote():
    dark = [solid_bgr(112, saturation=160, value=110) for _ in range(6)]
    light = [solid_bgr(0, saturation=12, value=230) for _ in range(6)]
    classifier = TeamClassifier()
    classifier.fit(dark + light)
    lime = solid_bgr(35, saturation=220, value=230)
    blue = solid_bgr(112, saturation=170, value=115)
    white = solid_bgr(0, saturation=10, value=235)
    assert_equal(measure_torso(lime)["reason"], "yellow", "fluorescent yellow abstains")
    assert_equal(int(classifier.predict([lime])[0]), -1, "the official receives no team")
    assert_equal(classifier.matches_team(lime), False, "yellow does not match a team shirt")
    assert_equal(int(classifier.predict([blue])[0]), 0, "blue stays team 0")
    assert_equal(int(classifier.predict([white])[0]), 1, "white stays team 1")
    assert_equal(classifier.matches_team(blue), True, "blue still matches a team")
    assert_equal(classifier.matches_team(white), True, "white still matches a team")


def test_weak_crops_do_not_vote():
    classifier = fit_light_and_dark()
    tiny = solid_bgr(0, saturation=10, value=210, width=8, height=20)
    readable_small = solid_bgr(0, saturation=12, value=214, width=16, height=48)
    yellow = solid_bgr(30, saturation=210, value=220)
    grass = solid_bgr(60, saturation=160, value=140)
    ambiguous = solid_bgr(20, saturation=80, value=155)
    predicted = classifier.predict([tiny, yellow, grass, ambiguous])
    assert_equal([int(team) for team in predicted], [-1, -1, -1, -1], "weak crops abstain")
    assert_equal(int(classifier.predict([readable_small])[0]), 1, "small but clear white shirt still votes")


def test_red_and_neon_green_shirts_split():
    red = [solid_bgr(8, saturation=130, value=155) for _ in range(6)]
    green = [solid_bgr(46, saturation=100, value=215) for _ in range(6)]
    classifier = TeamClassifier()
    classifier.fit(red + green)
    assert_equal(classifier.color_anchor["mode"], "value", "neon green is brighter than red")
    assert_equal(int(classifier.predict([solid_bgr(8, saturation=130, value=155)])[0]), 0, "red shirt")
    assert_equal(int(classifier.predict([solid_bgr(46, saturation=100, value=215)])[0]), 1, "neon green shirt")
    grass = solid_bgr(60, saturation=160, value=140)
    assert_equal(int(classifier.predict([grass])[0]), -1, "pitch grass still abstains")


def test_one_shirt_color_does_not_count_as_two_teams():
    red = [solid_bgr(8, saturation=140, value=160) for _ in range(15)]
    green = [solid_bgr(46, saturation=100, value=230, width=20, height=40) for _ in range(8)]
    assert_equal(crops_form_two_teams(red), False, "fifteen red shirts are one team")
    assert_equal(crops_form_two_teams(red + green), True, "distant neon shirts complete the pair")
    classifier = TeamClassifier()
    classifier.fit(red + green)
    assert_equal(classifier.color_anchor["mode"], "value", "merged set still splits by brightness")


def test_similar_brightness_uses_hue_for_green_and_red():
    red = [solid_bgr(6, saturation=150, value=200) for _ in range(6)]
    green = [solid_bgr(48, saturation=140, value=205) for _ in range(6)]
    classifier = TeamClassifier()
    classifier.fit(red + green)
    assert_equal(classifier.color_anchor["mode"], "hue", "close brightness falls through to hue")
    assert_equal(int(classifier.predict([solid_bgr(4, saturation=150, value=198)])[0]), 1, "red is team 1")
    assert_equal(int(classifier.predict([solid_bgr(50, saturation=130, value=202)])[0]), 0, "green is team 0")


def test_two_saturated_kits_split_by_hue():
    blue = [solid_bgr(112, saturation=180, value=100) for _ in range(6)]
    red = [solid_bgr(4, saturation=180, value=100) for _ in range(6)]
    classifier = TeamClassifier()
    classifier.fit(blue + red)
    assert_equal(classifier.color_anchor["mode"], "hue", "similar brightness uses hue")
    assert_equal(int(classifier.predict([solid_bgr(115, saturation=170, value=105)])[0]), 0, "blue kit")
    assert_equal(int(classifier.predict([solid_bgr(2, saturation=170, value=105)])[0]), 1, "red kit")
    white = solid_bgr(0, saturation=8, value=220)
    assert_equal(int(classifier.predict([white])[0]), -1, "white has no saturated hue")


def test_mixed_torso_abstains():
    crop = np.zeros((80, 40, 3), dtype=np.uint8)
    left = solid_bgr(0, saturation=10, value=220, width=20, height=80)
    right = solid_bgr(170, saturation=160, value=80, width=20, height=80)
    crop[:, :20] = left
    crop[:, 20:] = right
    classifier = fit_light_and_dark()
    assert_equal(int(classifier.predict([crop])[0]), -1, "two shirts in one box")


def test_weak_vote_does_not_create_or_unlock_a_team():
    lock = TeamTrackLock()
    assert_equal(lock.assign(3, None, 1, verifying=False), None, "no team yet")
    assert_equal(lock.is_locked(3), False, "weak vote does not open a track")
    for frame in range(1, TEAM_LOCK_CONFIRM_STREAK + 1):
        lock.assign(3, 1, frame, verifying=False)
    assert_equal(lock.is_locked(3), True, "three clear votes lock")
    for frame in range(10, 10 + TEAM_LOCK_UNLOCK_STREAK):
        team = lock.assign(3, None, frame, verifying=True)
        assert_equal(team, 1, "weak refresh keeps the team")
        assert_equal(lock.is_locked(3), True, "weak refresh stays locked")
    assert_equal(lock.metrics()["refresh_contradictions"], 0, "weak votes are not disagreements")
    assert_equal(lock.metrics()["ignored_weak_votes"] > 0, True, "weak votes were counted")


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
    test_lock_survives_a_run_of_opposite_votes()
    test_flicker_before_lock_keeps_the_first_clear_team()
    test_three_opposite_votes_before_lock_can_correct_the_first_guess()
    test_refresh_agreement_resets_contradiction_streak()
    test_light_shirt_and_dark_shirt_get_stable_teams()
    test_gradual_broadcast_shirts_still_split()
    test_shadowed_white_stays_light_and_sunlit_dark_stays_dark()
    test_a_small_clear_shirt_votes_and_a_small_gray_shirt_does_not()
    test_broadcast_lime_yellow_abstains_and_team_shirts_still_vote()
    test_weak_crops_do_not_vote()
    test_red_and_neon_green_shirts_split()
    test_one_shirt_color_does_not_count_as_two_teams()
    test_similar_brightness_uses_hue_for_green_and_red()
    test_two_saturated_kits_split_by_hue()
    test_mixed_torso_abstains()
    test_weak_vote_does_not_create_or_unlock_a_team()
    test_cached_interval_skips_locked_tracks()
    print("team classifier stability tests passed")
