#!/usr/bin/env python3
"""A short dropout reuses the player id. A different neighbour does not."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from player_track_stitch import (  # noqa: E402
    LOST_TRACK_MEMORY_FRAMES,
    STABLE_TRACK_MEMORY_FRAMES,
    PlayerTrackStitch,
)


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def box_at(foot_x, foot_y=400, width=30, height=70):
    return [foot_x - width / 2.0, foot_y - height, foot_x + width / 2.0, foot_y]


def see(stitch, frame, tracks):
    ids = [track["id"] for track in tracks]
    boxes = [track["box"] for track in tracks]
    hues = [track.get("hue", 110) for track in tracks]
    teams = [track.get("team", 0) for track in tracks]
    return list(stitch.apply(frame, ids, boxes, hues=hues, teams=teams))


def test_gap_along_the_motion_reuses_the_same_id():
    stitch = PlayerTrackStitch()
    for frame, foot_x in enumerate((100, 108, 116, 124, 132), start=1):
        see(stitch, frame, [{"id": 4, "box": box_at(foot_x)}])
    returned = see(stitch, 10, [{"id": 19, "box": box_at(172)}])
    assert_equal(returned, [4], "the same player keeps id 4")
    assert_equal(stitch.metrics()["stitches"], 1, "one stitch")
    assert_equal(stitch.metrics()["stitch_events"][0]["gap_frames"], 5, "five frames between sightings")
    assert_equal(stitch.metrics()["memory_frames"], LOST_TRACK_MEMORY_FRAMES, "memory window")


def test_same_id_coming_back_is_not_a_stitch():
    stitch = PlayerTrackStitch()
    see(stitch, 1, [{"id": 4, "box": box_at(100)}])
    see(stitch, 2, [{"id": 4, "box": box_at(108)}])
    assert_equal(stitch.metrics()["stitches"], 0, "ByteTrack already kept the id")


def test_far_reappearance_keeps_the_new_id():
    stitch = PlayerTrackStitch()
    for frame, foot_x in enumerate((100, 108, 116), start=1):
        see(stitch, frame, [{"id": 4, "box": box_at(foot_x)}])
    returned = see(stitch, 7, [{"id": 19, "box": box_at(420)}])
    assert_equal(returned, [19], "a far box is a different player")
    assert_equal(stitch.metrics()["stitches"], 0, "no stitch")


def test_memory_expires_after_twelve_frames():
    stitch = PlayerTrackStitch()
    see(stitch, 1, [{"id": 4, "box": box_at(100)}])
    see(stitch, 2, [{"id": 4, "box": box_at(104)}])
    returned = see(stitch, 2 + LOST_TRACK_MEMORY_FRAMES + 1, [{"id": 19, "box": box_at(108)}])
    assert_equal(returned, [19], "the lost id is gone after the window")
    assert_equal(stitch.metrics()["expired"] >= 1, True, "the memory dropped the track")


def test_wrong_shirt_and_wrong_team_stay_split():
    stitch = PlayerTrackStitch()
    for frame, foot_x in enumerate((100, 108, 116, 124), start=1):
        see(stitch, frame, [{"id": 4, "box": box_at(foot_x), "hue": 110, "team": 0}])
    hue_clash = see(stitch, 8, [{"id": 19, "box": box_at(150), "hue": 10, "team": 0}])
    assert_equal(hue_clash, [19], "a red shirt does not inherit the blue track")

    other = PlayerTrackStitch()
    for frame, foot_x in enumerate((100, 108, 116, 124), start=1):
        see(other, frame, [{"id": 4, "box": box_at(foot_x), "hue": 110, "team": 0}])
    team_clash = see(other, 8, [{"id": 21, "box": box_at(150), "hue": 110, "team": 1}])
    assert_equal(team_clash, [21], "the other team does not inherit the track")


def test_different_body_size_stays_split():
    stitch = PlayerTrackStitch()
    for frame, foot_x in enumerate((100, 108, 116), start=1):
        see(stitch, frame, [{"id": 4, "box": box_at(foot_x, width=30, height=70)}])
    returned = see(stitch, 6, [{"id": 19, "box": box_at(140, width=80, height=160)}])
    assert_equal(returned, [19], "a much larger box is not the same player")


def test_reverse_of_the_run_stays_split():
    stitch = PlayerTrackStitch()
    for frame, foot_x in enumerate((100, 108, 116, 124, 132), start=1):
        see(stitch, frame, [{"id": 4, "box": box_at(foot_x)}])
    returned = see(stitch, 7, [{"id": 19, "box": box_at(106)}])
    assert_equal(returned, [19], "a box behind the run is not the same player")
    assert_equal(stitch.metrics()["rejected_motion"] >= 1, True, "the direction rejected it")


def test_neighbour_blocks_the_reuse():
    stitch = PlayerTrackStitch()
    for frame in range(1, 5):
        see(
            stitch,
            frame,
            [
                {"id": 4, "box": box_at(100)},
                {"id": 8, "box": box_at(150)},
            ],
        )
    returned = see(
        stitch,
        8,
        [
            {"id": 8, "box": box_at(150)},
            {"id": 19, "box": box_at(132)},
        ],
    )
    assert_equal(returned, [8, 19], "the new box is too close to the player who stayed")
    assert_equal(stitch.metrics()["stitches"], 0, "no merge")


def test_parallel_player_who_stays_visible_does_not_take_the_id():
    stitch = PlayerTrackStitch()
    for frame, step in enumerate((0, 8, 16, 24), start=1):
        see(
            stitch,
            frame,
            [
                {"id": 4, "box": box_at(100 + step)},
                {"id": 8, "box": box_at(220 + step)},
            ],
        )
    returned = see(
        stitch,
        8,
        [
            {"id": 8, "box": box_at(252)},
            {"id": 19, "box": box_at(156)},
        ],
    )
    assert_equal(sorted(returned), [4, 8], "only the missing player is sewn back")


def test_two_lost_players_one_box_between_them_stays_new():
    stitch = PlayerTrackStitch()
    for frame in range(1, 4):
        see(
            stitch,
            frame,
            [
                {"id": 4, "box": box_at(100)},
                {"id": 5, "box": box_at(140)},
            ],
        )
    returned = see(stitch, 6, [{"id": 19, "box": box_at(120)}])
    assert_equal(returned, [19], "a box between two lost players starts a new id")
    assert_equal(stitch.metrics()["rejected_ambiguous"] >= 1, True, "the pair was ambiguous")


def test_still_player_returns_beside_the_last_foot():
    stitch = PlayerTrackStitch()
    for frame in range(1, 5):
        see(stitch, frame, [{"id": 4, "box": box_at(200)}])
    returned = see(stitch, 8, [{"id": 30, "box": box_at(208)}])
    assert_equal(returned, [4], "a player who barely moved keeps the id")


def test_goalkeeper_does_not_absorb_a_nearby_player():
    stitch = PlayerTrackStitch()
    for frame in range(1, 4):
        ids = stitch.apply(
            frame,
            [3],
            [box_at(100)],
            hues=[30],
            teams=[0],
            classes=[1],
        )
    assert_equal(list(ids), [3], "goalkeeper stays himself")
    returned = list(
        stitch.apply(
            6,
            [19],
            [box_at(106)],
            hues=[30],
            teams=[0],
            classes=[2],
        )
    )
    assert_equal(returned, [19], "an outfield player does not take the goalkeeper id")


def hold_stable(stitch, track_id=28, frames=8, foot=200, team=0, class_id=2):
    for frame in range(1, frames + 1):
        see(stitch, frame, [{"id": track_id, "box": box_at(foot), "team": team, "hue": 110}])
    stitch.note_class_lock(track_id, True, class_id)


def test_stable_identity_survives_ids_36_and_54():
    stitch = PlayerTrackStitch()
    hold_stable(stitch)
    first_return = see(stitch, 38, [{"id": 36, "box": box_at(208), "team": 0, "hue": 110}])
    still_same = see(stitch, 39, [{"id": 36, "box": box_at(212), "team": 0, "hue": 110}])
    second_return = see(stitch, 55, [{"id": 54, "box": box_at(216), "team": 0, "hue": 110}])
    assert_equal(first_return, [28], "36 is the same player as 28")
    assert_equal(still_same, [28], "the new ByteTrack id keeps being rewritten")
    assert_equal(second_return, [28], "54 is still the same player")
    assert_equal(stitch.metrics()["stitch_events"][0]["memory"], "long", "the long memory did the work")
    assert_equal(stitch.metrics()["stitch_events"][0]["gap_frames"], 30, "thirty frames between sightings")


def test_stable_identity_expires_after_sixty_frames():
    stitch = PlayerTrackStitch()
    hold_stable(stitch)
    returned = see(
        stitch,
        8 + STABLE_TRACK_MEMORY_FRAMES + 1,
        [{"id": 54, "box": box_at(208), "team": 0, "hue": 110}],
    )
    assert_equal(returned, [54], "beyond the long window this is a new identity")
    assert_equal(stitch.metrics()["expired"] >= 1, True, "the stable memory dropped the track")


def test_unlocked_track_does_not_keep_the_long_memory():
    stitch = PlayerTrackStitch()
    for frame in range(1, 9):
        see(stitch, frame, [{"id": 28, "box": box_at(200), "team": 0, "hue": 110}])
    returned = see(
        stitch,
        8 + LOST_TRACK_MEMORY_FRAMES + 1,
        [{"id": 36, "box": box_at(204), "team": 0, "hue": 110}],
    )
    assert_equal(returned, [36], "without a locked class the short window still expires")


def test_long_gap_rejects_the_other_team_and_a_close_neighbour():
    team_clash = PlayerTrackStitch()
    hold_stable(team_clash, team=0)
    returned = see(team_clash, 30, [{"id": 36, "box": box_at(208), "team": 1, "hue": 110}])
    assert_equal(returned, [36], "a different team does not inherit the stable id")

    crowded = PlayerTrackStitch()
    for frame in range(1, 9):
        see(
            crowded,
            frame,
            [
                {"id": 28, "box": box_at(200), "team": 0, "hue": 110},
                {"id": 8, "box": box_at(230), "team": 1, "hue": 20},
            ],
        )
    crowded.note_class_lock(28, True, 2)
    crowded.note_class_lock(8, True, 2)
    returned = see(
        crowded,
        30,
        [
            {"id": 8, "box": box_at(230), "team": 1, "hue": 20},
            {"id": 36, "box": box_at(214), "team": 0, "hue": 110},
        ],
    )
    assert_equal(returned, [8, 36], "the player who stayed blocks the reuse")


def test_two_stable_players_leave_an_ambiguous_box_new():
    stitch = PlayerTrackStitch()
    for frame in range(1, 9):
        see(
            stitch,
            frame,
            [
                {"id": 28, "box": box_at(100), "team": 0, "hue": 110},
                {"id": 29, "box": box_at(140), "team": 0, "hue": 110},
            ],
        )
    stitch.note_class_lock(28, True, 2)
    stitch.note_class_lock(29, True, 2)
    returned = see(stitch, 30, [{"id": 36, "box": box_at(120), "team": 0, "hue": 110}])
    assert_equal(returned, [36], "a box between two stable players stays a new id")


def test_goalkeeper_role_survives_into_the_long_memory():
    stitch = PlayerTrackStitch()
    for frame in range(1, 9):
        stitch.apply(frame, [7], [box_at(100)], hues=[30], teams=[0], classes=[1])
    stitch.note_class_lock(7, True, 1)
    returned = list(
        stitch.apply(30, [36], [box_at(108)], hues=[30], teams=[0], classes=[2])
    )
    assert_equal(returned, [36], "an outfield player does not inherit the goalkeeper")


def test_note_team_is_remembered_for_the_next_gap():
    stitch = PlayerTrackStitch()
    see(stitch, 1, [{"id": 4, "box": box_at(100), "team": None, "hue": None}])
    stitch.note_team(4, 1)
    see(stitch, 2, [{"id": 4, "box": box_at(106), "team": None, "hue": None}])
    returned = see(stitch, 5, [{"id": 19, "box": box_at(118), "team": 0, "hue": None}])
    assert_equal(returned, [19], "a team stored after the fact still rejects the other shirt")


if __name__ == "__main__":
    test_gap_along_the_motion_reuses_the_same_id()
    test_same_id_coming_back_is_not_a_stitch()
    test_far_reappearance_keeps_the_new_id()
    test_memory_expires_after_twelve_frames()
    test_wrong_shirt_and_wrong_team_stay_split()
    test_different_body_size_stays_split()
    test_reverse_of_the_run_stays_split()
    test_neighbour_blocks_the_reuse()
    test_parallel_player_who_stays_visible_does_not_take_the_id()
    test_two_lost_players_one_box_between_them_stays_new()
    test_still_player_returns_beside_the_last_foot()
    test_goalkeeper_does_not_absorb_a_nearby_player()
    test_stable_identity_survives_ids_36_and_54()
    test_stable_identity_expires_after_sixty_frames()
    test_unlocked_track_does_not_keep_the_long_memory()
    test_long_gap_rejects_the_other_team_and_a_close_neighbour()
    test_two_stable_players_leave_an_ambiguous_box_new()
    test_goalkeeper_role_survives_into_the_long_memory()
    test_note_team_is_remembered_for_the_next_gap()
    print("player track stitch tests passed")
