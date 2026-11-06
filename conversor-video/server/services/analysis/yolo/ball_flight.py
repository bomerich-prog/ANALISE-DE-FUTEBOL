"""Follow a pass from the release point when the ball track has died.

The detector misses a fast ball in many frames of a pass, so the frame-to-frame
track dies within a few frames of the kick. The boxes it does emit lie on one
line from the player who released the ball. This path keeps the published ball
and the boxes that continue it, and hands back the box that is the next point
of the flight.
"""

from __future__ import annotations

import math

# A pass moves the ball about 10 to 40 px per frame on a broadcast view.
FLIGHT_STEP_PX = 60.0
# Longest run of frames without a box on the path before it is forgotten.
FLIGHT_MAX_GAP_FRAMES = 10
# A missed stretch does not widen the reach without limit: across a gap the
# ball still travels at most this many frames' worth of steps.
FLIGHT_MAX_STEP_FRAMES = 4
# A box that is out of reach of the flight is only refused this soon after
# the last point of it. Later, the flight may simply have been missed.
FLIGHT_REACH_FRAMES = 3
# The first box must leave the release point; a ball that stays put is still
# with the player and is not this path's concern.
FLIGHT_MIN_LAUNCH_PX = 12.0
# A box may sit still or slow down, but not move back toward the release.
FLIGHT_MAX_BACKTRACK_PX = 8.0
FLIGHT_LATERAL_BASE_PX = 20.0
FLIGHT_LATERAL_RATIO = 0.35
# Net progress beyond the first box before an unseen flight is trusted.
FLIGHT_CONFIRM_PROGRESS_PX = 15.0
# A flight nobody published needs one box the detector is sure of. Faint
# boxes alone are grass marks and sideline gear as often as the ball.
FLIGHT_CONFIRM_CONFIDENCE = 0.25
FLIGHT_MIN_DIRECTION_PX = 10.0


def _center(box):
    return ((float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5)


class BallFlightPath:
    def __init__(
        self,
        step_px=FLIGHT_STEP_PX,
        max_gap_frames=FLIGHT_MAX_GAP_FRAMES,
    ):
        self.step_px = float(step_px)
        self.max_gap_frames = int(max_gap_frames)
        self.points = []
        self.confirmed = False

    def reset(self):
        self.points = []
        self.confirmed = False

    def _expire(self, frame):
        if self.points and int(frame) - self.points[-1][0] > self.max_gap_frames:
            self.reset()

    def _direction(self):
        """Unit direction of travel from the latest points, or None at launch."""
        last = self.points[-1]
        for earlier in reversed(self.points[:-1]):
            dx = last[1] - earlier[1]
            dy = last[2] - earlier[2]
            length = math.hypot(dx, dy)
            if length >= FLIGHT_MIN_DIRECTION_PX:
                return dx / length, dy / length
        return None

    def _deviation(self, frame, center):
        """How far a box strays from the path, or None when it cannot continue it."""
        last_frame, last_x, last_y = self.points[-1][:3]
        elapsed = int(frame) - int(last_frame)
        if elapsed < 1 or elapsed > self.max_gap_frames:
            return None
        dx = center[0] - last_x
        dy = center[1] - last_y
        step = math.hypot(dx, dy)
        if step > self.step_px * min(elapsed, FLIGHT_MAX_STEP_FRAMES):
            return None
        direction = self._direction()
        if direction is None:
            if step < FLIGHT_MIN_LAUNCH_PX:
                return None
            return 0.0
        along = dx * direction[0] + dy * direction[1]
        lateral = abs(dx * direction[1] - dy * direction[0])
        if along < -FLIGHT_MAX_BACKTRACK_PX:
            return None
        if lateral > FLIGHT_LATERAL_BASE_PX + FLIGHT_LATERAL_RATIO * max(0.0, along):
            return None
        return lateral

    def reachable(self, frame, box):
        """False when a box is farther from the ball in flight than it can travel."""
        self._expire(frame)
        if not self.confirmed or not self.points:
            return True
        last_frame, last_x, last_y = self.points[-1][:3]
        elapsed = max(1, int(frame) - int(last_frame))
        if elapsed > FLIGHT_REACH_FRAMES:
            return True
        center = _center(box)
        return math.hypot(center[0] - last_x, center[1] - last_y) <= self.step_px * elapsed

    def _closest(self, frame, candidates):
        best = None
        best_deviation = None
        for candidate in candidates:
            box = candidate.get("box") or candidate.get("bounding_box")
            if not box:
                continue
            deviation = self._deviation(frame, _center(box))
            if deviation is None:
                continue
            if best_deviation is None or deviation < best_deviation:
                best = candidate
                best_deviation = deviation
        return best

    def _progress(self):
        anchor = self.points[0]
        first = self.points[1]
        last = self.points[-1]
        first_distance = math.hypot(first[1] - anchor[1], first[2] - anchor[2])
        last_distance = math.hypot(last[1] - anchor[1], last[2] - anchor[2])
        return last_distance - first_distance

    def _sure(self):
        return any(point[3] >= FLIGHT_CONFIRM_CONFIDENCE for point in self.points[1:])

    def note_published(self, frame, box, at_player, confidence=1.0):
        """Record the ball the pipeline published this frame.

        A ball at a player's feet is a new release point. Any other ball either
        continues the path or starts a new one from where it is.
        """
        self._expire(frame)
        center = _center(box)
        point = (int(frame), center[0], center[1], float(confidence))
        if at_player or not self.points:
            self.points = [point]
            self.confirmed = False
            return
        if self.points[-1][0] == int(frame):
            return
        if self._deviation(frame, center) is None:
            self.points = [point]
            self.confirmed = False
            return
        self.points.append(point)

    def continue_flight(self, frame, candidates):
        """The box that continues the flight this frame, once the flight is real.

        Every box on the path is remembered, but a flight nobody published is
        only handed back after two boxes have carried the ball forward.
        """
        self._expire(frame)
        if not self.points or not candidates:
            return None
        best = self._closest(frame, candidates)
        if best is None and not self.confirmed and len(self.points) > 1:
            # A box beside the release point may have opened a false start.
            # The real flight can still leave from the release point itself.
            tentative = self.points
            self.points = tentative[:1]
            best = self._closest(frame, candidates)
            if best is None:
                self.points = tentative
        if best is None:
            return None
        center = _center(best.get("box") or best.get("bounding_box"))
        self.points.append((int(frame), center[0], center[1], float(best.get("confidence", 0.0))))
        if not self.confirmed:
            if (
                len(self.points) < 3
                or self._progress() < FLIGHT_CONFIRM_PROGRESS_PX
                or not self._sure()
            ):
                return None
            self.confirmed = True
        return best
