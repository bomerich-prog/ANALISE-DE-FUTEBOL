#!/usr/bin/env python3
"""The 2D map solved on one frame in three follows the camera on the other two."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from pitch_map import VisiblePitchMap, carry_decision  # noqa: E402
from pitch_motion import FrameMotion, plausible_turn  # noqa: E402
from roboflow_view import ViewTransformer  # noqa: E402

WIDTH, HEIGHT = 1280, 720


def _world(seed=7, line_offset=0):
    """A wide picture of grass with painted lines and speckles, larger than the camera view."""
    rng = np.random.default_rng(seed)
    world = np.zeros((HEIGHT + 400, WIDTH + 400, 3), dtype=np.uint8)
    world[:] = (40, 120, 40)
    noise = rng.integers(-18, 18, size=world.shape[:2], dtype=np.int16)
    world[:, :, 1] = np.clip(world[:, :, 1].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for x in range(60 + line_offset, world.shape[1], 170):
        cv2.line(world, (x, 0), (x + 90, world.shape[0]), (235, 235, 235), 3)
    for y in range(80 + line_offset, world.shape[0], 150):
        cv2.line(world, (0, y), (world.shape[1], y + 40), (235, 235, 235), 3)
    for _ in range(400):
        x, y = int(rng.integers(0, world.shape[1])), int(rng.integers(0, world.shape[0]))
        cv2.circle(world, (x, y), int(rng.integers(2, 5)), (200, 220, 200), -1)
    return world


def _view(world, dx, dy):
    """The camera picture after panning dx, dy pixels."""
    turn = np.array([[1.0, 0.0, -200.0 - dx], [0.0, 1.0, -200.0 - dy], [0.0, 0.0, 1.0]])
    return cv2.warpPerspective(world, turn, (WIDTH, HEIGHT))


def _grass(frame):
    return np.full(frame.shape[:2], 255, dtype=np.uint8)


def test_the_camera_pan_is_read_from_the_grass():
    world = _world()
    motion = FrameMotion()
    first = _view(world, 0, 0)
    assert motion.anchor(first, _grass(first))
    turn = motion.since_anchor(_view(world, 14, -6))
    assert turn is not None
    moved = cv2.perspectiveTransform(np.array([[[640.0, 360.0]]]), turn).reshape(2)
    # The picture moves opposite to the pan.
    assert abs(moved[0] - (640.0 - 14.0)) < 1.0, moved
    assert abs(moved[1] - (360.0 + 6.0)) < 1.0, moved


def test_players_running_on_their_own_do_not_move_the_camera():
    world = _world()
    first = _view(world, 0, 0)
    second = _view(world, 10, 0)
    rng = np.random.default_rng(3)
    boxes = []
    for _ in range(12):
        x, y = int(rng.integers(50, WIDTH - 120)), int(rng.integers(200, HEIGHT - 160))
        boxes.append((x, y, x + 50, y + 120))
    for frame, step in ((first, 0), (second, 40)):
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1 + step // 2, y1), (x2 - step // 2 + 20, y2), (30, 30, 200), -1)
            cv2.putText(frame, "9", (x1 + 10, y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    motion = FrameMotion()
    assert motion.anchor(first, _grass(first), boxes)
    turn = motion.since_anchor(second)
    assert turn is not None
    moved = cv2.perspectiveTransform(np.array([[[640.0, 360.0]]]), turn).reshape(2)
    assert abs(moved[0] - 630.0) < 1.0, moved
    assert abs(moved[1] - 360.0) < 1.0, moved


def test_a_cut_to_another_picture_is_not_read_as_a_turn():
    motion = FrameMotion()
    first = _view(_world(7), 0, 0)
    assert motion.anchor(first, _grass(first))
    assert motion.since_anchor(_view(_world(99, line_offset=75), 0, 0)) is None
    assert motion.failed == 1


def test_a_zoom_jump_is_not_a_turn():
    huge = np.diag([1.6, 1.6, 1.0])
    assert not plausible_turn(huge, (HEIGHT, WIDTH))
    assert plausible_turn(np.eye(3), (HEIGHT, WIDTH))


def test_the_carried_matrix_maps_the_same_field_point():
    image_to_field = np.array([[2.0, 0.1, 100.0], [0.05, 3.0, 50.0], [0.0, 0.0001, 1.0]])
    turn = np.array([[1.0, 0.0, -12.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]])
    decision = {
        "accepted": True,
        "matrix": image_to_field,
        "camera_matrix": None,
        "corners": ((100.0, 400.0), (900.0, 420.0)),
        "corner_perp": (0.0, 1.0),
        "side_px": (80.0, 0.0),
        "side_lines": [((900.0, 420.0), (950.0, 600.0))],
        "front_line": ((100.0, 400.0), (900.0, 420.0)),
    }
    carried = carry_decision(decision, turn)
    point = np.array([[[300.0, 500.0]]])
    then = cv2.perspectiveTransform(point, image_to_field).reshape(2)
    now_point = cv2.perspectiveTransform(point, turn)
    now = cv2.perspectiveTransform(now_point, carried["matrix"]).reshape(2)
    assert np.allclose(then, now, atol=1e-6)
    assert np.allclose(carried["corners"][0], (88.0, 405.0))
    assert np.allclose(carried["front_line"][1], (888.0, 425.0))
    # The second corner already has its painted side; the first gets its drawn side as a segment.
    assert carried["side_px"] == 0.0
    assert len(carried["side_lines"]) == 2
    assert np.allclose(carried["side_lines"][1][1], (88.0, 485.0))
    # The solved decision is left alone.
    assert decision["side_px"] == (80.0, 0.0)


class _StubMap(VisiblePitchMap):
    """A map whose solve always lands on the same picture-to-field matrix."""

    def __init__(self, every):
        super().__init__(every=every, fixed_camera=False)
        self.solves = 0
        self.lines_built = 0

    def _update_fit(self, white_line_mask, frame=None, grass_mask=None):
        self.solves += 1
        self.matrix = np.eye(3)
        self.transformer = ViewTransformer.from_matrix(self.matrix)
        self.positive_x = True
        self.region = "penalty"
        self.accepted_frames += 1
        return self._decision(True, False, "accepted")

    def _through_camera(self, decision, frame, grass_mask):
        return decision


def _lines(stub):
    def build():
        stub.lines_built += 1
        return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    return build


def test_one_frame_in_three_is_solved_and_the_others_follow_the_pan():
    world = _world()
    stub = _StubMap(every=3)
    centres = []
    for step in range(9):
        frame = _view(world, 6 * step, 0)
        decision = stub.update(_lines(stub), frame, _grass(frame))
        assert decision["accepted"]
        centres.append(stub._to_anchor(np.array([[640.0, 360.0]], dtype=np.float32))[0])
    assert stub.solves == 3, stub.solves
    # The painted lines are built only on the solved frames.
    assert stub.lines_built == 3
    assert stub.carried_frames == 6
    metrics = stub.metrics()
    assert metrics["frames_with_map"] == 9
    assert metrics["map_every_frames"] == 3
    # A followed frame reads its point where the solved frame saw it: 6 px further on per frame.
    for step, centre in enumerate(centres):
        offset = 6.0 * (step % 3)
        assert abs(centre[0] - (640.0 + offset)) < 1.0, (step, centre)


def test_a_turn_that_cannot_be_read_solves_the_frame():
    stub = _StubMap(every=3)
    first = _view(_world(7), 0, 0)
    stub.update(_lines(stub), first, _grass(first))
    other = _view(_world(99, line_offset=75), 0, 0)
    stub.update(_lines(stub), other, _grass(other))
    assert stub.solves == 2
    assert stub.carry_failed_frames == 1


def test_with_no_map_every_frame_is_solved():
    """A fit that lands on one frame alone is not skipped over."""

    class Empty(_StubMap):
        def _update_fit(self, white_line_mask, frame=None, grass_mask=None):
            self.solves += 1
            self.rejected_frames += 1
            return self._decision(False, False, "no_visible_penalty")

    stub = Empty(every=3)
    for step in range(6):
        frame = _view(_world(), step, 0)
        assert not stub.update(_lines(stub), frame, _grass(frame))["accepted"]
    assert stub.solves == 6
    metrics = stub.metrics()
    assert metrics["frames_with_map"] == 0
    assert metrics["frames_without_map"] == 6


def test_every_one_solves_every_frame():
    stub = _StubMap(every=1)
    for step in range(4):
        frame = _view(_world(), step, 0)
        stub.update(_lines(stub), frame, _grass(frame))
    assert stub.solves == 4
    assert stub.carried_frames == 0


def test_the_hold_lasts_as_long_in_frames():
    """A bridged matrix spends a frame of its hold for every frame since the last solve."""
    pitch_map = VisiblePitchMap(every=3)
    assert pitch_map.hold_frames == 15
    assert pitch_map.camera.hold_frames == 15  # the camera hold counts frames too (step)
    pitch_map.matrix = np.eye(3)
    pitch_map.positive_x = True
    pitch_map.hold_left = 15
    pitch_map._step = 3
    pitch_map._geometry_holds = lambda *args, **kwargs: False
    blank = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    bridged = 0
    for _ in range(10):
        decision = pitch_map._update_fit(blank, None, None)
        if not decision["accepted"]:
            break
        bridged += 1
    assert bridged == 5, bridged


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
    print("pitch map stride tests passed")
