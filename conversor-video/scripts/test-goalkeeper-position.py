"""Goalkeeper by position: the last man of the line in a kit of his own."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server" / "services" / "analysis" / "yolo"))

from goalkeeper_position import PositionKeeper  # noqa: E402


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def frame(keeper_x=1450.0, keeper_odd=True, winger_x=20.0):
    def man(track_id, x, y, odd):
        return {"track_id": track_id, "x": x, "y": y, "height": 60.0, "kit_odd": odd}

    return [
        man(1, keeper_x, 350.0, keeper_odd),
        man(2, 1200.0, 420.0, False),
        man(3, 900.0, 500.0, False),
        man(4, 600.0, 450.0, False),
        man(5, winger_x, 400.0, False),
    ]


def test_the_last_man_in_a_kit_of_his_own_is_the_keeper():
    keeper = PositionKeeper()
    found = set()
    for _ in range(20):
        found = keeper.update(frame())
    assert_equal(found, {1}, "the isolated odd kit at the end of the line")


def test_a_winger_alone_on_the_wing_in_his_team_shirt_is_not():
    keeper = PositionKeeper()
    found = set()
    for _ in range(40):
        found = keeper.update(frame(keeper_odd=False))
    assert_equal(found, set(), "team shirts never become keepers")


def test_an_odd_kit_in_the_middle_of_the_play_is_not():
    keeper = PositionKeeper()
    found = set()
    for _ in range(40):
        found = keeper.update(frame(keeper_x=750.0))
    assert_equal(found, set(), "not the last man")


def test_an_odd_kit_at_the_end_but_in_a_crowd_is_not():
    keeper = PositionKeeper()
    found = set()
    for _ in range(40):
        players = frame()
        players[1]["x"] = 1440.0
        players[1]["y"] = 380.0
        found = keeper.update(players)
    assert_equal(found, set(), "a goalmouth crowd is not the keeper alone")


def test_his_new_id_after_an_occlusion_is_quick():
    keeper = PositionKeeper()
    for _ in range(15):
        keeper.update(frame())
    found = set()
    for _ in range(3):
        players = frame()
        players[0]["track_id"] = 9
        found = keeper.update(players)
    assert_equal(found, {9}, "three frames once a keeper is known")


def test_the_team_is_the_majority_of_the_centroid_votes():
    keeper = PositionKeeper()
    assert_equal(keeper.vote_team(1, 1), None, "too few votes")
    keeper.vote_team(1, 0)
    assert_equal(keeper.vote_team(1, 0), 0, "two of three")
    keeper.vote_team(1, 1)
    assert_equal(keeper.vote_team(1, 0), 0, "three of five")
    assert_equal(keeper.vote_team(1, 1), None, "a tie decides nothing")


if __name__ == "__main__":
    test_the_last_man_in_a_kit_of_his_own_is_the_keeper()
    test_a_winger_alone_on_the_wing_in_his_team_shirt_is_not()
    test_an_odd_kit_in_the_middle_of_the_play_is_not()
    test_an_odd_kit_at_the_end_but_in_a_crowd_is_not()
    test_his_new_id_after_an_occlusion_is_quick()
    test_the_team_is_the_majority_of_the_centroid_votes()
    print("goalkeeper position tests passed")
