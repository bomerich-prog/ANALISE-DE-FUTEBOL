"""Reuse a player track id after a ByteTrack break.

ByteTrack sometimes retires an id and mints a new one when the same player
returns a few frames later. This memory gives the old id back before possession,
passes and stats ever see the new number. A short window covers any track. A
longer window is only for a track that was already stable. Position, size,
motion, shirt, team, role and the surrounding players must agree. Distance by
itself never accepts a match, and an ambiguous pair stays split.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


LOST_TRACK_MEMORY_FRAMES = 12
STABLE_TRACK_MEMORY_FRAMES = 60
STABLE_MIN_FRAMES = 8
GOALKEEPER_CLASS_ID = 1
SEARCH_RADIUS_CAP_PX = 64.0
SEARCH_RADIUS_FLOOR_PX = 24.0
SEARCH_RADIUS_GAP_PX = 5.0
NEARBY_PLAYER_PX = 22.0
MOVING_SPEED_PX = 3.0
AREA_RATIO_MIN = 0.50
AREA_RATIO_MAX = 2.00
ASPECT_RATIO_MIN = 0.62
ASPECT_RATIO_MAX = 1.60
HUE_MAX_DISTANCE = 28.0
AMBIGUITY_RATIO = 0.55
VELOCITY_BLEND = 0.6
STITCH_EVENT_LIMIT = 12


def _hypot(left, right) -> float:
    return float(np.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def _foot(box) -> tuple:
    return ((float(box[0]) + float(box[2])) * 0.5, float(box[3]))


def _size(box) -> tuple:
    width = max(1.0, float(box[2]) - float(box[0]))
    height = max(1.0, float(box[3]) - float(box[1]))
    return width, height, width * height, width / height


def _role_conflict(left, right) -> bool:
    """A goalkeeper and an outfield player are never the same track."""
    if left is None or right is None:
        return False
    return (int(left) == GOALKEEPER_CLASS_ID) != (int(right) == GOALKEEPER_CLASS_ID)


def _hue_distance(left: float, right: float) -> float:
    gap = abs(float(left) - float(right))
    return min(gap, 180.0 - gap)


def torso_hue(frame, box) -> Optional[float]:
    """Median OpenCV hue of the saturated torso. White and empty crops stay unknown."""
    if frame is None or box is None:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    left = int(max(0, min(width - 1, x1)))
    top = int(max(0, min(height - 1, y1)))
    right = int(max(left + 1, min(width, x2)))
    bottom = int(max(top + 1, min(height, y2)))
    crop = frame[top:bottom, left:right]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    crop_height, crop_width = crop.shape[:2]
    y_start = int(crop_height * 0.22)
    y_end = max(y_start + 1, int(crop_height * 0.62))
    x_start = int(crop_width * 0.22)
    x_end = max(x_start + 1, int(crop_width * 0.78))
    torso = crop[y_start:y_end, x_start:x_end]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    saturated = (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 70)
    if float(saturated.mean()) < 0.22:
        return None
    return float(np.median(hsv[:, :, 0][saturated]))


class PlayerTrackStitch:
    """Give a brand-new ByteTrack id back to the player who just went missing."""

    def __init__(
        self,
        memory_frames: int = LOST_TRACK_MEMORY_FRAMES,
        stable_memory_frames: int = STABLE_TRACK_MEMORY_FRAMES,
        stable_min_frames: int = STABLE_MIN_FRAMES,
    ):
        self.memory_frames = max(1, int(memory_frames))
        self.stable_memory_frames = max(self.memory_frames, int(stable_memory_frames))
        self.stable_min_frames = max(1, int(stable_min_frames))
        self._tracks: Dict[int, dict] = {}
        self.stitches = 0
        self.rejected_ambiguous = 0
        self.rejected_nearby = 0
        self.rejected_size = 0
        self.rejected_motion = 0
        self.rejected_hue = 0
        self.rejected_team = 0
        self.rejected_role = 0
        self.expired = 0
        self.stitch_events: List[dict] = []

    def note_team(self, track_id, team_id) -> None:
        """Remember a team the classifier already wrote. Unknown teams are ignored."""
        if track_id is None or team_id is None:
            return
        try:
            track_key = int(track_id)
            team_value = int(team_id)
        except (TypeError, ValueError):
            return
        state = self._tracks.get(track_key)
        if state is None:
            return
        state["team_id"] = team_value

    def note_class_lock(self, track_id, locked, class_id=None) -> None:
        """Remember whether this identity already has a locked role."""
        if track_id is None:
            return
        try:
            track_key = int(track_id)
        except (TypeError, ValueError):
            return
        state = self._tracks.get(track_key)
        if state is None:
            return
        state["class_locked"] = bool(locked)
        if locked and class_id is not None:
            try:
                state["class_id"] = int(class_id)
            except (TypeError, ValueError):
                return

    def apply(
        self,
        frame_number: int,
        tracker_ids,
        boxes,
        frame=None,
        teams: Optional[Sequence] = None,
        hues: Optional[Sequence] = None,
        classes: Optional[Sequence] = None,
    ):
        """Return tracker ids, with a lost id reused when every gate agrees."""
        frame_number = int(frame_number)
        ids = [int(value) for value in list(tracker_ids)]
        box_list = [list(box)[:4] for box in list(boxes)]
        observations = [
            self._observation(index, ids, box_list, frame, teams, hues, classes)
            for index in range(len(ids))
        ]
        present_known = []
        candidates = []
        for observation in observations:
            if observation["track_id"] < 0:
                continue
            if observation["track_id"] in self._tracks:
                present_known.append(observation)
            else:
                candidates.append(observation)

        present_ids = {observation["track_id"] for observation in present_known}
        lost_states = []
        expired_ids = []
        for track_id, state in self._tracks.items():
            if track_id in present_ids:
                continue
            gap = frame_number - int(state["last_frame"])
            if gap > self._memory_limit(state):
                expired_ids.append(track_id)
                continue
            if gap >= 1:
                lost_states.append(state)
        for track_id in expired_ids:
            self._tracks.pop(track_id, None)
            self.expired += 1

        active_feet = [observation["foot"] for observation in present_known]
        claims = self._claims(candidates, lost_states, active_feet, frame_number)
        rewritten = list(ids)
        stitched_observations = []
        claimed_candidates = set()
        for candidate, lost_state, _distance in claims:
            replaced_track_id = int(candidate["track_id"])
            rewritten[candidate["index"]] = int(lost_state["track_id"])
            candidate["track_id"] = int(lost_state["track_id"])
            stitched_observations.append(candidate)
            claimed_candidates.add(candidate["index"])
            self._record_stitch(frame_number, lost_state, replaced_track_id)

        for observation in present_known:
            self._update_state(self._tracks[observation["track_id"]], observation, frame_number)
        for observation in stitched_observations:
            self._update_state(self._tracks[observation["track_id"]], observation, frame_number)
        for observation in candidates:
            if observation["index"] in claimed_candidates:
                continue
            self._tracks[observation["track_id"]] = self._new_state(observation, frame_number)
        return np.asarray(rewritten, dtype=int)

    def metrics(self) -> Dict[str, object]:
        return {
            "memory_frames": self.memory_frames,
            "stable_memory_frames": self.stable_memory_frames,
            "stable_min_frames": self.stable_min_frames,
            "stitches": self.stitches,
            "rejected_ambiguous": self.rejected_ambiguous,
            "rejected_nearby": self.rejected_nearby,
            "rejected_size": self.rejected_size,
            "rejected_motion": self.rejected_motion,
            "rejected_hue": self.rejected_hue,
            "rejected_team": self.rejected_team,
            "rejected_role": self.rejected_role,
            "expired": self.expired,
            "tracks_remembered": len(self._tracks),
            "stitch_events": list(self.stitch_events),
        }

    def _observation(self, index, ids, boxes, frame, teams, hues, classes) -> dict:
        box = boxes[index] if index < len(boxes) else [0.0, 0.0, 1.0, 1.0]
        width, height, area, aspect = _size(box)
        hue = None
        if hues is not None and index < len(hues):
            hue = None if hues[index] is None else float(hues[index])
        elif frame is not None:
            hue = torso_hue(frame, box)
        team = None
        if teams is not None and index < len(teams) and teams[index] is not None:
            team = int(teams[index])
        class_id = None
        if classes is not None and index < len(classes) and classes[index] is not None:
            class_id = int(classes[index])
        return {
            "index": index,
            "track_id": int(ids[index]),
            "box": box,
            "foot": _foot(box),
            "width": width,
            "height": height,
            "area": area,
            "aspect": aspect,
            "hue": hue,
            "team": team,
            "class_id": class_id,
        }

    def _claims(self, candidates, lost_states, active_feet, frame_number):
        candidate_options = []
        for candidate in candidates:
            options = []
            nearest_reason = None
            nearest_distance = None
            for state in lost_states:
                accepted, distance, reason = self._score(candidate, state, frame_number, active_feet)
                if nearest_distance is None or (distance is not None and distance < nearest_distance):
                    nearest_distance = distance
                    nearest_reason = reason
                if accepted:
                    options.append((distance, state))
            options.sort(key=lambda item: item[0])
            if len(options) >= 2 and options[1][0] * AMBIGUITY_RATIO <= options[0][0]:
                self.rejected_ambiguous += 1
                continue
            if options:
                candidate_options.append((candidate, options[0][1], options[0][0]))
                continue
            if nearest_reason in {"nearby", "size", "motion", "hue", "team", "role"}:
                self._count_rejection(nearest_reason)

        by_lost: Dict[int, list] = {}
        for candidate, state, distance in candidate_options:
            by_lost.setdefault(int(state["track_id"]), []).append((distance, candidate, state))

        claims = []
        for entries in by_lost.values():
            entries.sort(key=lambda item: item[0])
            if len(entries) >= 2 and entries[1][0] * AMBIGUITY_RATIO <= entries[0][0]:
                self.rejected_ambiguous += 1
                continue
            distance, candidate, state = entries[0]
            claims.append((candidate, state, distance))
        return claims

    def _score(self, candidate, state, frame_number, active_feet):
        gap = int(frame_number) - int(state["last_frame"])
        if gap < 1 or gap > self._memory_limit(state):
            return False, None, "expired"
        last = state["foot"]
        foot = candidate["foot"]
        velocity = (float(state["vx"]), float(state["vy"]))
        speed = float(np.hypot(velocity[0], velocity[1]))
        predicted = (last[0] + velocity[0] * gap, last[1] + velocity[1] * gap)
        distance_last = _hypot(foot, last)
        distance_predicted = _hypot(foot, predicted)
        height = max(8.0, float(state["height"]))
        radius = min(
            SEARCH_RADIUS_CAP_PX,
            max(SEARCH_RADIUS_FLOOR_PX, 0.50 * height) + SEARCH_RADIUS_GAP_PX * gap,
        )
        anchor_distance = distance_predicted if speed >= MOVING_SPEED_PX else distance_last
        close_to_last = distance_last <= max(18.0, 0.35 * height)
        if close_to_last:
            anchor_distance = min(anchor_distance, distance_last)
        if anchor_distance > radius:
            return False, anchor_distance, "far"
        if speed >= MOVING_SPEED_PX and not close_to_last:
            displacement = (foot[0] - last[0], foot[1] - last[1])
            forward = displacement[0] * velocity[0] + displacement[1] * velocity[1]
            if forward < 0.0:
                return False, anchor_distance, "motion"
        area_ratio = float(candidate["area"]) / max(1.0, float(state["area"]))
        aspect_ratio = float(candidate["aspect"]) / max(0.05, float(state["aspect"]))
        if area_ratio < AREA_RATIO_MIN or area_ratio > AREA_RATIO_MAX:
            return False, anchor_distance, "size"
        if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
            return False, anchor_distance, "size"
        if candidate["hue"] is not None and state["hue"] is not None:
            if _hue_distance(candidate["hue"], state["hue"]) > HUE_MAX_DISTANCE:
                return False, anchor_distance, "hue"
        if candidate["team"] is not None and state["team_id"] is not None:
            if int(candidate["team"]) != int(state["team_id"]):
                return False, anchor_distance, "team"
        if _role_conflict(candidate.get("class_id"), state.get("class_id")):
            return False, anchor_distance, "role"
        for other in active_feet:
            if _hypot(foot, other) < max(NEARBY_PLAYER_PX, 0.90 * anchor_distance):
                return False, anchor_distance, "nearby"
        return True, anchor_distance, None

    def _count_rejection(self, reason: str) -> None:
        if reason == "nearby":
            self.rejected_nearby += 1
        elif reason == "size":
            self.rejected_size += 1
        elif reason == "motion":
            self.rejected_motion += 1
        elif reason == "hue":
            self.rejected_hue += 1
        elif reason == "team":
            self.rejected_team += 1
        elif reason == "role":
            self.rejected_role += 1

    def _record_stitch(self, frame_number, state, replaced_track_id: int) -> None:
        self.stitches += 1
        if len(self.stitch_events) >= STITCH_EVENT_LIMIT:
            return
        self.stitch_events.append(
            {
                "frame": int(frame_number),
                "track_id": int(state["track_id"]),
                "replaced_track_id": int(replaced_track_id),
                "gap_frames": int(frame_number) - int(state["last_frame"]),
                "memory": "long" if self._is_stable(state) else "short",
            }
        )

    def _new_state(self, observation, frame_number) -> dict:
        return {
            "track_id": int(observation["track_id"]),
            "last_frame": int(frame_number),
            "foot": observation["foot"],
            "width": observation["width"],
            "height": observation["height"],
            "area": observation["area"],
            "aspect": observation["aspect"],
            "vx": 0.0,
            "vy": 0.0,
            "hue": observation["hue"],
            "team_id": observation["team"],
            "class_id": observation.get("class_id"),
            "class_locked": False,
            "frames_seen": 1,
        }

    def _is_stable(self, state) -> bool:
        """A long memory is only for someone already seen, locked and identified."""
        if int(state.get("frames_seen", 0)) < self.stable_min_frames:
            return False
        if not state.get("class_locked"):
            return False
        return state.get("team_id") is not None or state.get("class_id") is not None

    def _memory_limit(self, state) -> int:
        if self._is_stable(state):
            return self.stable_memory_frames
        return self.memory_frames

    def _update_state(self, state, observation, frame_number) -> None:
        gap = max(1, int(frame_number) - int(state["last_frame"]))
        instant_vx = (observation["foot"][0] - state["foot"][0]) / gap
        instant_vy = (observation["foot"][1] - state["foot"][1]) / gap
        state["vx"] = VELOCITY_BLEND * instant_vx + (1.0 - VELOCITY_BLEND) * float(state["vx"])
        state["vy"] = VELOCITY_BLEND * instant_vy + (1.0 - VELOCITY_BLEND) * float(state["vy"])
        state["last_frame"] = int(frame_number)
        state["foot"] = observation["foot"]
        state["width"] = observation["width"]
        state["height"] = observation["height"]
        state["area"] = observation["area"]
        state["aspect"] = observation["aspect"]
        state["frames_seen"] = int(state.get("frames_seen", 0)) + 1
        if observation["hue"] is not None:
            state["hue"] = observation["hue"]
        if observation["team"] is not None:
            state["team_id"] = int(observation["team"])
        if observation.get("class_id") is not None:
            state["class_id"] = int(observation["class_id"])
