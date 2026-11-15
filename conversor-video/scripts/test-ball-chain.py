#!/usr/bin/env python3
"""A strong box rolling on the grass away from the boots is the ball; a boot is not."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from ball_chain import BallChains, BallReach  # noqa: E402


def assert_true(condition, label):
    if not condition:
        raise AssertionError(label)


def box(x, y, confidence, half=10.0):
    return {"box": [x - half, y - half, x + half, y + half], "confidence": confidence}


def at_feet(feet, limit=45.0):
    def check(candidate):
        x1, y1, x2, y2 = candidate["box"]
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return any(np.hypot(center[0] - fx, center[1] - fy) <= limit for fx, fy in feet)

    return check


def test_a_rolling_ball_beats_a_boot():
    chains = BallChains()
    rolling = None
    for frame in range(1, 7):
        ball = box(800.0 - 12.0 * frame, 620.0 + 3.0 * frame, 0.85)
        boot = box(830.0 + 2.0 * frame, 340.0, 0.82)
        feet = [(830.0 + 2.0 * frame, 345.0)]
        chains.update(frame, [ball, boot], at_feet(feet))
        rolling = chains.rolling_ball(frame)
        if frame < 4:
            assert_true(rolling is None, "a chain needs four boxes")
    assert_true(rolling is not None, "the rolling ball is found")
    assert_true(rolling["box"][1] > 500.0, "the ball, not the boot")


def test_a_box_at_the_feet_is_not_a_rolling_ball():
    chains = BallChains()
    for frame in range(1, 8):
        boot = box(500.0 + 4.0 * frame, 400.0, 0.9)
        chains.update(frame, [boot], at_feet([(500.0 + 4.0 * frame, 405.0)]))
    assert_true(chains.rolling_ball(7) is None, "a boot that moves with its player is not the ball")


def test_a_jump_breaks_the_chain():
    chains = BallChains()
    for frame in range(1, 4):
        chains.update(frame, [box(300.0 + 5.0 * frame, 500.0, 0.8)], at_feet([]))
    chains.update(4, [box(900.0, 200.0, 0.8)], at_feet([]))
    assert_true(chains.rolling_ball(4) is None, "a box far away does not continue the chain")


def test_faint_boxes_and_the_stands_do_not_count():
    chains = BallChains()
    grass = np.zeros((1080, 1920), dtype=np.uint8)
    grass[300:, :] = 255
    for frame in range(1, 8):
        chains.update(
            frame,
            [box(400.0 + 5.0 * frame, 600.0, 0.3), box(900.0 + 5.0 * frame, 150.0, 0.9)],
            at_feet([]),
            grass,
        )
    assert_true(chains.rolling_ball(7) is None, "faint grass boxes and boxes in the stands are not the ball")


def test_a_boot_across_the_picture_is_out_of_reach_of_a_rolling_ball():
    """Reference clip: the ball rolls down-left at about 8 px per frame, a boot sits 290 px above."""
    reach = BallReach()
    for frame in range(165, 172):
        reach.note(frame, (794.0 - 8.0 * (frame - 171), 621.0 + 3.0 * (frame - 171)), False)
    assert_true(not reach.allows(177, (828.0, 340.0)), "the boot six frames later is out of reach")
    assert_true(not reach.allows(185, (749.0, 352.0)), "and still out of reach fourteen frames later")
    assert_true(reach.allows(187, (730.0, 643.0)), "the ball further along its path is in reach")
    assert_true(reach.allows(210, (100.0, 100.0)), "after a second the ball may be anywhere")


def test_a_pass_from_the_feet_stays_in_reach():
    reach = BallReach()
    for frame in range(1, 6):
        reach.note(frame, (500.0, 400.0), True)
    assert_true(reach.allows(12, (780.0, 400.0)), "a pass kicked while hidden reaches the receiver")
    assert_true(not reach.allows(7, (900.0, 400.0)), "but not 400 px in two frames")


def main():
    test_a_rolling_ball_beats_a_boot()
    test_a_box_at_the_feet_is_not_a_rolling_ball()
    test_a_jump_breaks_the_chain()
    test_faint_boxes_and_the_stands_do_not_count()
    test_a_boot_across_the_picture_is_out_of_reach_of_a_rolling_ball()
    test_a_pass_from_the_feet_stays_in_reach()
    print("ball chain tests passed")


if __name__ == "__main__":
    main()
