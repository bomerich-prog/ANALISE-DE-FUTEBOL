"""A rolling ball: one strong box, frame after frame, away from every player's boot.

The ball pipeline only opens a track, or takes the ball back after a gap, at a
player's feet. A boot is also a small round box at a player's feet, and the
detector often scores it as high as the ball. When the real ball is rolling in
open grass, a boot can win that way and the published ball jumps across the
pitch.

This module follows the strong boxes from frame to frame. A chain of boxes
that keeps moving in short steps, stays on the grass and is mostly away from
the players' feet is the ball itself: a boot moves with its player and is
always at a foot.
"""

from __future__ import annotations

import math

# Only a box the detector is fairly sure of can start or carry a chain.
CHAIN_MIN_CONFIDENCE = 0.50
CHAIN_MIN_MEAN_CONFIDENCE = 0.60
# A box continues a chain within this many pixels per frame of gap.
CHAIN_STEP_PX = 60.0
# A chain survives this many frames without a box.
CHAIN_MAX_GAP_FRAMES = 2
# Boxes needed before a chain is the ball.
CHAIN_CONFIRM_POINTS = 4
# At most this share of the chain may sit at a player's feet.
CHAIN_MAX_AT_FOOT_SHARE = 0.5


def _center(box):
    return ((float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5)


def _on_grass(grass_mask, center):
    if grass_mask is None:
        return True
    height, width = grass_mask.shape[:2]
    x = int(round(center[0]))
    y = int(round(center[1]))
    if not (0 <= x < width and 0 <= y < height):
        return False
    top = max(0, y - 6)
    bottom = min(height, y + 7)
    left = max(0, x - 6)
    right = min(width, x + 7)
    return bool((grass_mask[top:bottom, left:right] > 0).any())


class BallChains:
    def __init__(self):
        self.chains = []

    def update(self, frame, candidates, at_foot, grass_mask=None):
        """Link this frame's strong boxes to the chains. at_foot(box_dict) -> bool."""
        frame = int(frame)
        self.chains = [chain for chain in self.chains if frame - chain[-1]["frame"] <= CHAIN_MAX_GAP_FRAMES + 1]
        boxes = []
        for candidate in candidates or []:
            box = candidate.get("box") or candidate.get("bounding_box")
            if not box or float(candidate.get("confidence", 0.0)) < CHAIN_MIN_CONFIDENCE:
                continue
            center = _center(box)
            if not _on_grass(grass_mask, center):
                continue
            boxes.append((candidate, center))
        pairs = []
        for box_index, (_candidate, center) in enumerate(boxes):
            for chain_index, chain in enumerate(self.chains):
                last = chain[-1]
                gap = frame - last["frame"]
                if gap < 1:
                    continue
                distance = math.hypot(center[0] - last["center"][0], center[1] - last["center"][1])
                if distance <= CHAIN_STEP_PX * gap:
                    pairs.append((distance, box_index, chain_index))
        pairs.sort()
        used_boxes = set()
        used_chains = set()
        for _distance, box_index, chain_index in pairs:
            if box_index in used_boxes or chain_index in used_chains:
                continue
            used_boxes.add(box_index)
            used_chains.add(chain_index)
            candidate, center = boxes[box_index]
            self.chains[chain_index].append(self._point(frame, candidate, center, at_foot))
        for box_index, (candidate, center) in enumerate(boxes):
            if box_index not in used_boxes:
                self.chains.append([self._point(frame, candidate, center, at_foot)])
        for chain in self.chains:
            del chain[:-12]

    @staticmethod
    def _point(frame, candidate, center, at_foot):
        return {
            "frame": int(frame),
            "center": center,
            "confidence": float(candidate.get("confidence", 0.0)),
            "at_foot": bool(at_foot(candidate)),
            "candidate": candidate,
        }

    def rolling_ball(self, frame):
        """This frame's box of a chain that is the ball rolling in open grass, or None."""
        best = None
        best_score = None
        for chain in self.chains:
            if chain[-1]["frame"] != int(frame) or len(chain) < CHAIN_CONFIRM_POINTS:
                continue
            recent = chain[-CHAIN_CONFIRM_POINTS:]
            at_foot_share = sum(1 for point in recent if point["at_foot"]) / float(len(recent))
            if at_foot_share > CHAIN_MAX_AT_FOOT_SHARE:
                continue
            mean_confidence = sum(point["confidence"] for point in recent) / float(len(recent))
            if mean_confidence < CHAIN_MIN_MEAN_CONFIDENCE:
                continue
            score = (len(chain), mean_confidence)
            if best_score is None or score > best_score:
                best = chain[-1]["candidate"]
                best_score = score
        return best


# A published ball is followed for this long. After a longer gap the ball may
# be anywhere: the camera and the play have moved on.
REACH_WINDOW_FRAMES = 30
REACH_BASE_PX = 60.0
# The ball left a player's feet, or its motion is unknown: a kick may have
# happened while it was hidden, so it may have travelled a pass.
REACH_KICK_PX_PER_FRAME = 40.0
# The ball was rolling on its own: it keeps going that way, give or take this.
REACH_ROLLING_PX_PER_FRAME = 12.0
REACH_MOTION_FRAMES = 5


class BallReach:
    """Where the ball can be now, from the ball published last.

    A box beyond that reach is another object, most often a boot at a player's
    feet on the other side of the picture. It is not published, however the
    rest of the pipeline scored it.
    """

    def __init__(self):
        self.points = []

    def note(self, frame, center, at_foot):
        self.points.append((int(frame), float(center[0]), float(center[1]), bool(at_foot)))
        del self.points[:-REACH_MOTION_FRAMES]

    def _velocity(self):
        """Pixels per frame of a ball rolling on its own, or None."""
        last = self.points[-1]
        if last[3]:
            return None
        recent = [point for point in self.points if last[0] - point[0] <= REACH_MOTION_FRAMES]
        if len(recent) < 3 or any(point[3] for point in recent):
            return None
        first = recent[0]
        frames = last[0] - first[0]
        if frames <= 0:
            return None
        return ((last[1] - first[1]) / frames, (last[2] - first[2]) / frames)

    def allows(self, frame, center):
        if not self.points:
            return True
        last = self.points[-1]
        gap = int(frame) - last[0]
        if gap <= 0 or gap > REACH_WINDOW_FRAMES:
            return True
        x, y = float(center[0]), float(center[1])
        velocity = self._velocity()
        if velocity is None:
            return math.hypot(x - last[1], y - last[2]) <= REACH_BASE_PX + REACH_KICK_PX_PER_FRAME * gap
        limit = REACH_BASE_PX + REACH_ROLLING_PX_PER_FRAME * gap
        predicted = (last[1] + velocity[0] * gap, last[2] + velocity[1] * gap)
        if math.hypot(x - predicted[0], y - predicted[1]) <= limit:
            return True
        return math.hypot(x - last[1], y - last[2]) <= limit
