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
    TrackAppearanceGate,
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


def observe(gate, entries):
    gate.begin_frame()
    for entry in entries:
        gate.consider(
            track_id=entry["track_id"],
            box=entry["box"],
            frame=entry["frame"],
            raw_class_id=entry.get("raw_class_id", PLAYER),
            locked_class_id=entry.get("locked_class_id", PLAYER),
            foot_field=entry.get("foot_field"),
        )
    return gate.finish_frame()


def corner_flag_image(hue=20, width=40, height=120):
    """Yellow or red cloth on top of a thin pole, with grass around the pole."""
    image = solid_bgr(55, saturation=140, value=110, width=width, height=height)
    cloth_height = max(1, height // 3)
    image[:cloth_height] = solid_bgr(hue, saturation=220, value=220, width=width, height=cloth_height)
    pole_width = max(2, width // 10)
    pole_x = (width - pole_width) // 2
    image[cloth_height:, pole_x : pole_x + pole_width] = (80, 80, 80)
    return image


def yellow_jersey_image(width=48, height=120):
    """Shirt fills the torso. Dark shorts fill the bottom, so this is not a pole."""
    image = solid_bgr(22, saturation=200, value=180, width=width, height=height)
    image[int(height * 0.62) :] = (25, 25, 30)
    return image


def fill(width, height, color):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def place(canvas, box, patch):
    x1, y1, x2, y2 = box
    canvas[y1:y2, x1:x2] = patch
    return canvas


def grass_canvas(width=320, height=220):
    return solid_bgr(55, saturation=140, value=110, width=width, height=height)


def test_wide_corner_flag_is_dropped_after_two_frames_and_jersey_stays():
    gate = TrackAppearanceGate()
    flag = corner_flag_image()
    flag_box = [0, 0, flag.shape[1], flag.shape[0]]
    assert_equal(
        non_person_rejection_reason(flag_box, flag),
        None,
        "pre-track filter misses a flag once the cloth widens the box",
    )
    stand = fill(14, 80, (18, 18, 18))
    assert_equal(
        non_person_rejection_reason([0, 0, 14, 80], stand),
        None,
        "pre-track filter misses a dark stand between the thin and flag cuts",
    )
    first = observe(
        gate,
        [{"track_id": 27, "box": flag_box, "frame": flag, "foot_field": (40, 20)}],
    )
    second = observe(
        gate,
        [{"track_id": 27, "box": flag_box, "frame": flag, "foot_field": (40, 20)}],
    )
    assert_equal(first[0]["action"], "keep", "one flag frame is not enough")
    assert_equal(second[0]["action"], "drop", "second flag frame drops it")
    assert_equal(second[0]["reason"], "corner_flag_track", "reason is the flag")

    red = corner_flag_image(hue=175)
    red_gate = TrackAppearanceGate()
    red_box = [0, 0, red.shape[1], red.shape[0]]
    observe(red_gate, [{"track_id": 3, "box": red_box, "frame": red, "foot_field": (40, 20)}])
    red_verdict = observe(
        red_gate,
        [{"track_id": 3, "box": red_box, "frame": red, "foot_field": (40, 20)}],
    )
    assert_equal(red_verdict[0]["action"], "drop", "red flag drops too")

    jersey = yellow_jersey_image()
    jersey_box = [0, 0, jersey.shape[1], jersey.shape[0]]
    jersey_gate = TrackAppearanceGate()
    verdicts = []
    for _ in range(6):
        verdict = observe(
            jersey_gate,
            [{"track_id": 11, "box": jersey_box, "frame": jersey, "foot_field": (3000, 40)}],
        )[0]
        verdicts.append(verdict)
    assert_equal(
        [item["action"] for item in verdicts[:2]],
        ["keep", "keep"],
        "yellow shirt is not dropped as a corner flag",
    )
    assert_equal(
        [item["action"] for item in verdicts[2:]],
        ["referee"] * 4,
        "neon yellow shirt becomes the referee",
    )
    assert_equal(verdicts[2]["reason"], "yellow_kit", "reason is the yellow kit")


def test_flag_returns_only_after_the_color_is_gone():
    gate = TrackAppearanceGate()
    flag = corner_flag_image()
    box = [0, 0, flag.shape[1], flag.shape[0]]
    observe(gate, [{"track_id": 27, "box": box, "frame": flag}])
    assert_equal(
        observe(gate, [{"track_id": 27, "box": box, "frame": flag}])[0]["action"],
        "drop",
        "flag latched",
    )
    player = solid_bgr(100, saturation=180, value=160, width=flag.shape[1], height=flag.shape[0])
    held = [
        observe(gate, [{"track_id": 27, "box": box, "frame": player}])[0]["action"]
        for _ in range(2)
    ]
    released = observe(gate, [{"track_id": 27, "box": box, "frame": player}])[0]["action"]
    assert_equal(held, ["drop", "drop"], "flag stays dropped for two ordinary frames")
    assert_equal(released, "keep", "three ordinary frames release the flag")


def test_untracked_flag_drops_immediately_and_jersey_does_not():
    gate = TrackAppearanceGate()
    flag = corner_flag_image()
    verdict = observe(
        gate,
        [{"track_id": -1, "box": [0, 0, flag.shape[1], flag.shape[0]], "frame": flag}],
    )
    assert_equal(verdict[0]["action"], "drop", "untracked flag has no id to wait for")
    jersey = yellow_jersey_image()
    kept = observe(
        gate,
        [{"track_id": -1, "box": [0, 0, jersey.shape[1], jersey.shape[0]], "frame": jersey}],
    )
    assert_equal(kept[0]["action"], "keep", "untracked jersey stays")


def test_still_sideline_stand_is_dropped_and_real_players_stay():
    def dark_scene(box):
        canvas = fill(240, 160, (18, 18, 18))
        x1, y1, x2, y2 = box
        canvas[y1:y2, x1:x2] = (18, 18, 18)
        return canvas

    gate = TrackAppearanceGate()
    box = [20, 30, 34, 110]
    sideline = (400.0, 40.0)
    early = []
    for _ in range(11):
        early.append(
            observe(
                gate,
                [{"track_id": 9, "box": box, "frame": dark_scene(box), "foot_field": sideline}],
            )[0]["action"]
        )
    latched = observe(
        gate,
        [{"track_id": 9, "box": box, "frame": dark_scene(box), "foot_field": sideline}],
    )[0]
    assert_equal(early, ["keep"] * 11, "eleven still frames do not drop the stand")
    assert_equal(latched["action"], "drop", "twelve still frames drop the stand")
    assert_equal(latched["reason"], "static_sideline", "reason is static sideline gear")

    moved = [70, 30, 84, 110]
    released = observe(
        gate,
        [{"track_id": 9, "box": moved, "frame": dark_scene(moved), "foot_field": sideline}],
    )[0]
    assert_equal(released["action"], "keep", "a 50px step releases the stand")

    blue = solid_bgr(100, saturation=180, value=160, width=40, height=110)
    blue_gate = TrackAppearanceGate()
    blue_box = [0, 0, 40, 110]
    blue_actions = [
        observe(
            blue_gate,
            [{"track_id": 4, "box": blue_box, "frame": blue, "foot_field": sideline}],
        )[0]["action"]
        for _ in range(16)
    ]
    assert_equal(blue_actions, ["keep"] * 16, "shoulder-width player on the sideline stays")

    thin_blue = solid_bgr(100, saturation=180, value=160, width=22, height=80)
    thin_gate = TrackAppearanceGate()
    thin_box = [0, 0, 22, 80]
    thin_actions = [
        observe(
            thin_gate,
            [{"track_id": 5, "box": thin_box, "frame": thin_blue, "foot_field": sideline}],
        )[0]["action"]
        for _ in range(16)
    ]
    assert_equal(thin_actions, ["keep"] * 16, "thin blue player stays because the shirt is colorful")

    midfield_gate = TrackAppearanceGate()
    midfield_box = [0, 0, 24, 80]
    midfield = fill(24, 80, (20, 20, 20))
    midfield_actions = [
        observe(
            midfield_gate,
            [
                {
                    "track_id": 6,
                    "box": midfield_box,
                    "frame": midfield,
                    "foot_field": (6000.0, 3500.0),
                }
            ],
        )[0]["action"]
        for _ in range(16)
    ]
    assert_equal(midfield_actions, ["keep"] * 16, "still dark player in midfield stays")

    wide_dark = fill(36, 90, (20, 20, 20))
    wide_gate = TrackAppearanceGate()
    wide_actions = [
        observe(
            wide_gate,
            [
                {
                    "track_id": 7,
                    "box": [0, 0, 36, 90],
                    "frame": wide_dark,
                    "foot_field": sideline,
                }
            ],
        )[0]["action"]
        for _ in range(16)
    ]
    assert_equal(wide_actions, ["keep"] * 16, "dark player with shoulders stays on the sideline")

    walker = TrackAppearanceGate()
    walker_actions = []
    for step in range(16):
        walking_box = [20 + step * 6, 30, 34 + step * 6, 110]
        walker_actions.append(
            observe(
                walker,
                [
                    {
                        "track_id": 19,
                        "box": walking_box,
                        "frame": dark_scene(walking_box),
                        "foot_field": sideline,
                    }
                ],
            )[0]["action"]
        )
    assert_equal(walker_actions, ["keep"] * 16, "a thin dark player who is actually moving stays")


def test_static_without_field_only_rejects_a_very_thin_stand():
    thin = fill(16, 80, (16, 16, 16))
    gate = TrackAppearanceGate()
    box = [0, 0, 16, 80]
    for _ in range(11):
        observe(gate, [{"track_id": 12, "box": box, "frame": thin}])
    verdict = observe(gate, [{"track_id": 12, "box": box, "frame": thin}])[0]
    assert_equal(verdict["action"], "drop", "very thin still stand drops without homography")

    person = fill(22, 80, (16, 16, 16))
    person_gate = TrackAppearanceGate()
    person_box = [0, 0, 22, 80]
    actions = [
        observe(person_gate, [{"track_id": 13, "box": person_box, "frame": person}])[0]["action"]
        for _ in range(16)
    ]
    assert_equal(actions, ["keep"] * 16, "a slightly wider dark box stays when field position is unknown")


def test_referee_vote_needs_three_labels_and_does_not_rewrite_goalkeeper():
    body = fill(80, 160, (40, 40, 40))
    box = [0, 0, 80, 160]
    gate = TrackAppearanceGate()
    raw_labels = [PLAYER, PLAYER, REFEREE, PLAYER, REFEREE, REFEREE]
    actions = []
    for raw_label in raw_labels:
        verdict = observe(
            gate,
            [
                {
                    "track_id": 25,
                    "box": box,
                    "frame": body,
                    "raw_class_id": raw_label,
                    "locked_class_id": PLAYER,
                    "foot_field": (6000.0, 3500.0),
                }
            ],
        )[0]
        actions.append(verdict["action"])
    assert_equal(
        actions,
        ["keep", "keep", "keep", "keep", "keep", "referee"],
        "only the third referee label inside the window reroutes",
    )
    assert_equal(
        observe(
            gate,
            [
                {
                    "track_id": 25,
                    "box": box,
                    "frame": body,
                    "raw_class_id": REFEREE,
                    "locked_class_id": PLAYER,
                    "foot_field": (6000.0, 3500.0),
                }
            ],
        )[0]["reason"],
        "referee_vote",
        "route reason is the vote",
    )

    keeper = TrackAppearanceGate()
    keeper_actions = []
    for _ in range(4):
        verdict = observe(
            keeper,
            [
                {
                    "track_id": 1,
                    "box": box,
                    "frame": body,
                    "raw_class_id": REFEREE,
                    "locked_class_id": GOALKEEPER,
                    "foot_field": (6000.0, 3500.0),
                }
            ],
        )[0]
        keeper_actions.append(verdict["action"])
    assert_equal(keeper_actions, ["keep"] * 4, "goalkeeper is not rewritten by referee votes")

    colored = solid_bgr(120, saturation=180, value=90, width=80, height=160)
    colored_gate = TrackAppearanceGate()
    colored_actions = []
    for _ in range(6):
        verdict = observe(
            colored_gate,
            [
                {
                    "track_id": 30,
                    "box": [0, 0, 80, 160],
                    "frame": colored,
                    "raw_class_id": REFEREE,
                    "locked_class_id": PLAYER,
                    "foot_field": (6000.0, 3500.0),
                }
            ],
        )[0]
        colored_actions.append(verdict["action"])
    assert_equal(
        colored_actions,
        ["keep"] * 6,
        "a blue shirt the detector calls referee stays a player",
    )


def test_linesman_beside_flag_becomes_referee_and_players_do_not():
    flag_box = [40, 20, 80, 140]
    person_box = [110, 30, 160, 150]

    def scene(person_patch):
        canvas = grass_canvas()
        place(canvas, flag_box, corner_flag_image())
        place(canvas, person_box, person_patch)
        return canvas

    dark_person = fill(50, 120, (20, 20, 20))
    gate = TrackAppearanceGate()
    actions = []
    for _ in range(5):
        verdicts = observe(
            gate,
            [
                {
                    "track_id": 27,
                    "box": flag_box,
                    "frame": scene(dark_person),
                    "foot_field": (80.0, 30.0),
                },
                {
                    "track_id": 25,
                    "box": person_box,
                    "frame": scene(dark_person),
                    "foot_field": (160.0, 50.0),
                },
            ],
        )
        actions.append((verdicts[0]["action"], verdicts[1]["action"]))
    assert_equal(actions[1][0], "drop", "flag is already dropped on the second frame")
    assert_equal(actions[3][1], "keep", "three frames beside the flag are not enough")
    assert_equal(actions[4][1], "referee", "fourth frame beside the latched flag reroutes")
    assert_equal(
        observe(
            gate,
            [
                {
                    "track_id": 27,
                    "box": flag_box,
                    "frame": scene(dark_person),
                    "foot_field": (80.0, 30.0),
                },
                {
                    "track_id": 25,
                    "box": person_box,
                    "frame": scene(dark_person),
                    "foot_field": (160.0, 50.0),
                },
            ],
        )[1]["reason"],
        "linesman_near_flag",
        "reason is the assistant beside the flag",
    )

    midfield = TrackAppearanceGate()
    midfield_actions = []
    for _ in range(6):
        verdicts = observe(
            midfield,
            [
                {
                    "track_id": 27,
                    "box": flag_box,
                    "frame": scene(dark_person),
                    "foot_field": (80.0, 30.0),
                },
                {
                    "track_id": 25,
                    "box": person_box,
                    "frame": scene(dark_person),
                    "foot_field": (160.0, 3500.0),
                },
            ],
        )
        midfield_actions.append(verdicts[1]["action"])
    assert_equal(midfield_actions, ["keep"] * 6, "dark player in midfield stays a player")

    distant = TrackAppearanceGate()
    distant_actions = []
    for _ in range(6):
        verdicts = observe(
            distant,
            [
                {
                    "track_id": 27,
                    "box": flag_box,
                    "frame": scene(dark_person),
                    "foot_field": (80.0, 30.0),
                },
                {
                    "track_id": 8,
                    "box": person_box,
                    "frame": scene(dark_person),
                    "foot_field": (900.0, 40.0),
                },
            ],
        )
        distant_actions.append(verdicts[1]["action"])
    assert_equal(distant_actions, ["keep"] * 6, "dark player further down the touchline stays")

    blue_person = solid_bgr(100, saturation=180, value=160, width=50, height=120)
    colorful = TrackAppearanceGate()
    colorful_actions = []
    for _ in range(6):
        verdicts = observe(
            colorful,
            [
                {
                    "track_id": 27,
                    "box": flag_box,
                    "frame": scene(blue_person),
                    "foot_field": (80.0, 30.0),
                },
                {
                    "track_id": 15,
                    "box": person_box,
                    "frame": scene(blue_person),
                    "foot_field": (160.0, 50.0),
                },
            ],
        )
        colorful_actions.append(verdicts[1]["action"])
    assert_equal(colorful_actions, ["keep"] * 6, "colorful player beside the flag stays")

    keeper = TrackAppearanceGate()
    keeper_actions = []
    for _ in range(6):
        verdicts = observe(
            keeper,
            [
                {
                    "track_id": 27,
                    "box": flag_box,
                    "frame": scene(dark_person),
                    "foot_field": (80.0, 30.0),
                },
                {
                    "track_id": 1,
                    "box": person_box,
                    "frame": scene(dark_person),
                    "raw_class_id": GOALKEEPER,
                    "locked_class_id": GOALKEEPER,
                    "foot_field": (160.0, 50.0),
                },
            ],
        )
        keeper_actions.append(verdicts[1]["action"])
    assert_equal(keeper_actions, ["keep"] * 6, "goalkeeper beside the flag is not sent to referees")


def test_yellow_kit_referee_spares_goalkeeper_and_outfield_kits():
    jersey = yellow_jersey_image()
    box = [0, 0, jersey.shape[1], jersey.shape[0]]
    midfield = (6000.0, 3500.0)

    gate = TrackAppearanceGate()
    early = []
    for _ in range(2):
        early.append(
            observe(
                gate,
                [
                    {
                        "track_id": 16,
                        "box": box,
                        "frame": jersey,
                        "raw_class_id": PLAYER,
                        "locked_class_id": PLAYER,
                        "foot_field": midfield,
                    }
                ],
            )[0]["action"]
        )
    latched = observe(
        gate,
        [
            {
                "track_id": 16,
                "box": box,
                "frame": jersey,
                "raw_class_id": PLAYER,
                "locked_class_id": PLAYER,
                "foot_field": midfield,
            }
        ],
    )[0]
    assert_equal(early, ["keep", "keep"], "two yellow frames stay a player")
    assert_equal(latched["action"], "referee", "third yellow frame is the referee")
    assert_equal(latched["reason"], "yellow_kit", "reason is the yellow kit")

    white = fill(jersey.shape[1], jersey.shape[0], (240, 240, 240))
    held = [
        observe(
            gate,
            [
                {
                    "track_id": 16,
                    "box": box,
                    "frame": white,
                    "raw_class_id": PLAYER,
                    "locked_class_id": PLAYER,
                    "foot_field": midfield,
                }
            ],
        )[0]["action"]
        for _ in range(5)
    ]
    released = observe(
        gate,
        [
            {
                "track_id": 16,
                "box": box,
                "frame": white,
                "raw_class_id": PLAYER,
                "locked_class_id": PLAYER,
                "foot_field": midfield,
            }
        ],
    )[0]
    assert_equal(held, ["referee"] * 5, "a short dip keeps the referee")
    assert_equal(released["action"], "keep", "six ordinary frames release the yellow latch")

    blue = solid_bgr(100, saturation=180, value=160, width=jersey.shape[1], height=jersey.shape[0])
    garnet = solid_bgr(175, saturation=180, value=120, width=jersey.shape[1], height=jersey.shape[0])
    for patch, label in ((white, "white"), (blue, "blue"), (garnet, "garnet")):
        kit_gate = TrackAppearanceGate()
        actions = [
            observe(
                kit_gate,
                [
                    {
                        "track_id": 8,
                        "box": box,
                        "frame": patch,
                        "raw_class_id": PLAYER,
                        "locked_class_id": PLAYER,
                        "foot_field": midfield,
                    }
                ],
            )[0]["action"]
            for _ in range(6)
        ]
        assert_equal(actions, ["keep"] * 6, f"{label} kit stays a player")

    keeper = TrackAppearanceGate()
    keeper_actions = []
    for raw_label, locked_label in (
        (GOALKEEPER, GOALKEEPER),
        (GOALKEEPER, PLAYER),
        (PLAYER, GOALKEEPER),
    ):
        for _ in range(4):
            verdict = observe(
                keeper,
                [
                    {
                        "track_id": 1,
                        "box": box,
                        "frame": jersey,
                        "raw_class_id": raw_label,
                        "locked_class_id": locked_label,
                        "foot_field": midfield,
                    }
                ],
            )[0]
            keeper_actions.append(verdict["action"])
    assert_equal(keeper_actions, ["keep"] * 12, "yellow goalkeeper stays a goalkeeper")
    assert_equal(keeper.metrics()["routed_yellow_kit"], 0, "goalkeeper never counts as yellow referee")
    assert_equal(gate.metrics()["routed_yellow_kit"] > 0, True, "the center referee was counted")


if __name__ == "__main__":
    test_rejects_pole_flag_and_wide_box()
    test_player_does_not_become_referee_on_flicker()
    test_referee_does_not_become_player_on_flicker()
    test_goalkeeper_stays_goalkeeper()
    test_unlocks_only_after_consecutive_disagreements()
    test_wide_corner_flag_is_dropped_after_two_frames_and_jersey_stays()
    test_flag_returns_only_after_the_color_is_gone()
    test_untracked_flag_drops_immediately_and_jersey_does_not()
    test_still_sideline_stand_is_dropped_and_real_players_stay()
    test_static_without_field_only_rejects_a_very_thin_stand()
    test_referee_vote_needs_three_labels_and_does_not_rewrite_goalkeeper()
    test_linesman_beside_flag_becomes_referee_and_players_do_not()
    test_yellow_kit_referee_spares_goalkeeper_and_outfield_kits()
    print("person class lock tests passed")
