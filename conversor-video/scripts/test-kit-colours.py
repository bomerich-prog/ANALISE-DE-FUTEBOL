"""Kit colours read under warm stadium light, and the map's grass probe."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server" / "services" / "analysis" / "yolo"))

from person_class import YELLOW_TORSO_MIN, appearance_profile  # noqa: E402
from pitch_map import grass_inside_pitch  # noqa: E402


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def _player_on_grass(shirt_hsv):
    """A 40x100 box: yellow-green grass (Bayern v Real Madrid) around a shirt."""
    hsv = np.zeros((100, 40, 3), dtype=np.uint8)
    hsv[:, :] = (40, 180, 125)
    hsv[20:65, 12:28] = shirt_hsv
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return appearance_profile(frame, [0, 0, 40, 100])


def test_yellow_green_grass_is_not_a_yellow_kit():
    red_player = _player_on_grass((5, 190, 190))
    assert_equal(red_player["torso_yellow"] < YELLOW_TORSO_MIN, True, "grass beside a red shirt")
    white_player = _player_on_grass((30, 25, 235))
    assert_equal(white_player["torso_flag"] < 0.3, True, "grass around a white shirt is not flag cloth")


def test_a_yellow_referee_stays_yellow():
    referee = _player_on_grass((28, 200, 220))
    assert_equal(referee["torso_yellow"] >= YELLOW_TORSO_MIN, True, "a yellow shirt on that grass")


def test_the_grass_probe_ignores_the_sign_of_the_matrix():
    grass = np.full((200, 400), 255, dtype=np.uint8)
    matrix = np.array([[10.0, 0.0, 4000.0], [0.0, 10.0, 2000.0], [0.0, 0.0, 1.0]])
    assert_equal(grass_inside_pitch(matrix, grass), grass_inside_pitch(-matrix, grass), "H and -H are one mapping")
    assert_equal(grass_inside_pitch(-matrix, grass) > 0.9, True, "the grass is on the pitch")


if __name__ == "__main__":
    test_yellow_green_grass_is_not_a_yellow_kit()
    test_a_yellow_referee_stays_yellow()
    test_the_grass_probe_ignores_the_sign_of_the_matrix()
    print("kit colour tests passed")
