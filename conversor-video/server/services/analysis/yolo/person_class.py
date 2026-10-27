"""Lock Player, Referee and Goalkeeper labels per track.

Thin posts, corner flags and incompatible boxes are rejected before a track
exists. A confirmed class stays until several consecutive frames disagree.

Flags and sideline equipment whose box is wide enough to pass that first
cut are removed after ByteTrack, once color, motion and pitch position agree.
A neon yellow torso that holds for a few frames is the center referee,
even when the detector calls that track a player. A goalkeeper stays a
goalkeeper. The track id stays alive; only the emitted class changes.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
import supervision as sv


PERSON_CLASS_CONFIRM_STREAK = 3
PERSON_CLASS_UNLOCK_STREAK = 3


def non_person_rejection_reason(box, frame=None) -> Optional[str]:
    """Reject poles, corner flags and boxes that cannot be a person."""
    if box is None or len(box) < 4:
        return "empty"
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    if not all(np.isfinite([x1, y1, x2, y2])):
        return "empty"
    width = x2 - x1
    height = y2 - y1
    if width <= 1.0 or height <= 1.0:
        return "empty"
    aspect = width / height
    if aspect < 0.16:
        return "thin_non_person"
    if aspect > 1.8:
        return "wide_non_person"
    if frame is not None and aspect < 0.24 and _top_is_corner_flag(frame, (x1, y1, x2, y2)):
        return "corner_flag"
    return None


def _top_is_corner_flag(frame, box) -> bool:
    height, width = frame.shape[:2]
    x1 = int(max(0, min(width - 1, box[0])))
    y1 = int(max(0, min(height - 1, box[1])))
    x2 = int(max(x1 + 1, min(width, box[2])))
    y2 = int(max(y1 + 1, min(height, box[3])))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    top = crop[: max(1, crop.shape[0] // 3)]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    flag = ((hue <= 25) | ((hue >= 15) & (hue <= 45))) & (saturation > 80) & (value > 60)
    return float(flag.mean()) > 0.45


def reject_non_person_detections(detections: sv.Detections, frame):
    """Drop non-person boxes before ByteTrack assigns ids."""
    reasons: Dict[str, int] = {}
    if detections is None or len(detections) == 0:
        return detections, reasons
    keep = []
    for index, box in enumerate(detections.xyxy):
        reason = non_person_rejection_reason(box, frame)
        if reason is None:
            keep.append(index)
            continue
        reasons[reason] = int(reasons.get(reason, 0)) + 1
    if len(keep) == len(detections):
        return detections, reasons
    if not keep:
        return sv.Detections.empty(), reasons
    return detections[np.array(keep, dtype=int)], reasons


class PersonClassLock:
    """Confirm a class, then keep it until consecutive frames disagree."""

    def __init__(
        self,
        confirm_streak: int = PERSON_CLASS_CONFIRM_STREAK,
        unlock_streak: int = PERSON_CLASS_UNLOCK_STREAK,
    ):
        self.confirm_streak = max(1, int(confirm_streak))
        self.unlock_streak = max(2, int(unlock_streak))
        self.tracks = {}
        self.prelock_class_switches = 0
        self.refresh_checks = 0
        self.refresh_agreements = 0
        self.refresh_contradictions = 0
        self.unlocks = 0
        self.class_changes_after_lock = []

    def is_locked(self, track_id: int) -> bool:
        state = self.tracks.get(int(track_id))
        return bool(state and state["locked"])

    def current_class(self, track_id: int):
        state = self.tracks.get(int(track_id))
        if state is None:
            return None
        return int(state["class_id"])

    def assign(self, track_id: int, raw_class_id: int, frame_number: int, verifying: bool) -> int:
        track_id = int(track_id)
        raw_class_id = int(raw_class_id)
        frame_number = int(frame_number)
        state = self.tracks.get(track_id)
        if state is None:
            state = {
                "class_id": raw_class_id,
                "locked": self.confirm_streak <= 1,
                "confirm_streak": 1,
                "contradiction_streak": 0,
                "last_frame": frame_number,
            }
            self.tracks[track_id] = state
            return int(state["class_id"])

        state["last_frame"] = frame_number
        if not state["locked"]:
            if raw_class_id != int(state["class_id"]):
                self.prelock_class_switches += 1
                state["class_id"] = raw_class_id
                state["confirm_streak"] = 1
            else:
                state["confirm_streak"] += 1
            if state["confirm_streak"] >= self.confirm_streak:
                state["locked"] = True
                state["contradiction_streak"] = 0
            return int(state["class_id"])

        if not verifying:
            return int(state["class_id"])

        self.refresh_checks += 1
        if raw_class_id == int(state["class_id"]):
            self.refresh_agreements += 1
            state["contradiction_streak"] = 0
            return int(state["class_id"])

        self.refresh_contradictions += 1
        state["contradiction_streak"] += 1
        if state["contradiction_streak"] < self.unlock_streak:
            return int(state["class_id"])

        previous_class = int(state["class_id"])
        self.unlocks += 1
        self.class_changes_after_lock.append(
            {
                "track_id": track_id,
                "from_class": previous_class,
                "to_class": raw_class_id,
                "frame": frame_number,
            }
        )
        state["locked"] = False
        state["class_id"] = raw_class_id
        state["confirm_streak"] = 1
        state["contradiction_streak"] = 0
        return int(state["class_id"])

    def metrics(self) -> Dict[str, object]:
        final_class_by_track = {
            str(track_id): int(state["class_id"])
            for track_id, state in sorted(self.tracks.items(), key=lambda item: item[0])
        }
        return {
            "confirm_streak_required": self.confirm_streak,
            "unlock_streak_required": self.unlock_streak,
            "tracks_seen": len(self.tracks),
            "tracks_locked": sum(1 for state in self.tracks.values() if state["locked"]),
            "prelock_class_switches": self.prelock_class_switches,
            "class_changes_after_lock": len(self.class_changes_after_lock),
            "class_change_events": list(self.class_changes_after_lock),
            "refresh_checks": self.refresh_checks,
            "refresh_agreements": self.refresh_agreements,
            "refresh_contradictions": self.refresh_contradictions,
            "unlocks": self.unlocks,
            "final_class_by_track": final_class_by_track,
        }


GATE_PLAYER_CLASS_ID = 2
GATE_GOALKEEPER_CLASS_ID = 1
GATE_REFEREE_CLASS_ID = 3

FLAG_ASPECT_MAX = 0.62
FLAG_TOP_MIN = 0.28
FLAG_TORSO_MAX = 0.55
FLAG_CONFIRM_FRAMES = 2
FLAG_RELEASE_FRAMES = 3
FLAG_STRONG_ASPECT_MAX = 0.50
FLAG_STRONG_TOP_MIN = 0.50
POLE_COLUMN_MAX = 0.28

STATIC_WINDOW = 12
STATIC_SPREAD_PX = 5.0
STATIC_RELEASE_PX = 18.0
STATIC_ASPECT_MAX = 0.34
STATIC_ASPECT_MAX_NO_FIELD = 0.22
LOW_CHROMA_MIN = 0.62

SIDELINE_STATIC_CM = 180.0
SIDELINE_LINESMAN_CM = 150.0
LINESMAN_FLAG_CM = 250.0
LINESMAN_FLAG_PX = 140.0
LINESMAN_CONFIRM_FRAMES = 4
LINESMAN_ASPECT_MIN = 0.28
LINESMAN_ASPECT_MAX = 0.90

REFEREE_VOTE_WINDOW = 8
REFEREE_VOTE_MIN = 3
REFEREE_CLEAR_FRAMES = 8

YELLOW_HUE_MIN = 16
YELLOW_HUE_MAX = 45
YELLOW_SATURATION_MIN = 120
YELLOW_VALUE_MIN = 120
YELLOW_TORSO_MIN = 0.40
YELLOW_CONFIRM_FRAMES = 3
YELLOW_CLEAR_FRAMES = 6

PITCH_LENGTH_CM = 12000.0
PITCH_WIDTH_CM = 7000.0


def _clip_crop(frame, box):
    if frame is None or box is None or len(box) < 4:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    if not all(np.isfinite([x1, y1, x2, y2])):
        return None
    left = int(max(0, min(width - 1, x1)))
    top = int(max(0, min(height - 1, y1)))
    right = int(max(left + 1, min(width, x2)))
    bottom = int(max(top + 1, min(height, y2)))
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    return crop


def _flag_hue_mask(hsv) -> np.ndarray:
    """Yellow, orange and red cloth. OpenCV hue stops at 179."""
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored = (hue <= 42) | (hue >= 158)
    return colored & (saturation > 80) & (value > 60)


def _torso_yellow_fraction(hsv) -> float:
    """Neon yellow shirt. OpenCV hue stops at 179, and this band is the kit."""
    if hsv.size == 0:
        return 0.0
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    yellow = (
        (hue >= YELLOW_HUE_MIN)
        & (hue <= YELLOW_HUE_MAX)
        & (saturation >= YELLOW_SATURATION_MIN)
        & (value >= YELLOW_VALUE_MIN)
    )
    return float(yellow.mean())


def _bottom_is_pole(hsv) -> bool:
    """A corner flag is cloth over a thin pole. A torso fills the box width."""
    height, width = hsv.shape[:2]
    if width < 2 or height < 4:
        return False
    bottom = hsv[int(height * 0.55) :]
    hue = bottom[:, :, 0]
    saturation = bottom[:, :, 1]
    value = bottom[:, :, 2]
    grass = (hue >= 35) & (hue <= 90) & (saturation > 30) & (value > 25)
    occupied = ~grass
    column_fraction = occupied.mean(axis=0)
    return float((column_fraction > 0.25).mean()) < POLE_COLUMN_MAX


def appearance_profile(frame, box) -> Dict[str, float]:
    """Color and shape cues for one box. Missing crops stay neutral."""
    neutral = {
        "aspect": 1.0,
        "top_flag": 0.0,
        "torso_flag": 0.0,
        "low_chroma": 0.0,
        "bottom_pole": 0.0,
        "torso_yellow": 0.0,
    }
    crop = _clip_crop(frame, box)
    if crop is None:
        return neutral
    crop_height, crop_width = crop.shape[:2]
    if crop_height < 2 or crop_width < 2:
        return neutral
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    top = hsv[: max(1, crop_height // 3)]
    torso_y1 = int(crop_height * 0.22)
    torso_y2 = max(torso_y1 + 1, int(crop_height * 0.62))
    torso_x1 = int(crop_width * 0.22)
    torso_x2 = max(torso_x1 + 1, int(crop_width * 0.78))
    torso = hsv[torso_y1:torso_y2, torso_x1:torso_x2]
    if torso.size == 0:
        torso = hsv
    dark_neutral = (torso[:, :, 1] < 42) & (torso[:, :, 2] < 85)
    return {
        "aspect": float(crop_width) / float(crop_height),
        "top_flag": float(_flag_hue_mask(top).mean()),
        "torso_flag": float(_flag_hue_mask(torso).mean()),
        "low_chroma": float(dark_neutral.mean()),
        "bottom_pole": 1.0 if _bottom_is_pole(hsv) else 0.0,
        "torso_yellow": _torso_yellow_fraction(torso),
    }


def _is_flag_shape(profile: Dict[str, float], strong: bool = False) -> bool:
    aspect_max = FLAG_STRONG_ASPECT_MAX if strong else FLAG_ASPECT_MAX
    top_min = FLAG_STRONG_TOP_MIN if strong else FLAG_TOP_MIN
    if float(profile["aspect"]) >= aspect_max:
        return False
    if float(profile["top_flag"]) <= top_min:
        return False
    if float(profile["torso_flag"]) >= FLAG_TORSO_MAX:
        return False
    return float(profile["bottom_pole"]) >= 1.0


def _near_touchline(foot_field, band_cm: float) -> bool:
    if foot_field is None:
        return False
    x_cm, y_cm = float(foot_field[0]), float(foot_field[1])
    if not (np.isfinite(x_cm) and np.isfinite(y_cm)):
        return False
    outside = (
        y_cm < 0.0
        or y_cm > PITCH_WIDTH_CM
        or x_cm < 0.0
        or x_cm > PITCH_LENGTH_CM
    )
    if outside:
        return True
    return y_cm <= band_cm or y_cm >= (PITCH_WIDTH_CM - band_cm)


def _point_distance(left, right) -> float:
    if left is None or right is None:
        return float("inf")
    return float(np.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def _foot_spread_px(feet) -> float:
    if len(feet) < 2:
        return 0.0
    xs = np.array([point[0] for point in feet], dtype=np.float32)
    ys = np.array([point[1] for point in feet], dtype=np.float32)
    origin = np.array([float(np.median(xs)), float(np.median(ys))], dtype=np.float32)
    return float(np.max(np.hypot(xs - origin[0], ys - origin[1])))


class TrackAppearanceGate:
    """Drop flags and still sideline gear after the track already exists.

    The pre-track filter misses a corner flag once the cloth makes the box
    wider than aspect 0.24, and it blinks when only some frames are thin.
    ByteTrack then keeps the id and PersonClassLock freezes Player.
    This gate does not delete the track. It only withholds the emission.
    """

    def __init__(self):
        self._pending: List[dict] = []
        self._tracks: Dict[int, dict] = {}
        self.dropped_corner_flag = 0
        self.dropped_static_sideline = 0
        self.routed_referee_vote = 0
        self.routed_linesman = 0
        self.routed_yellow_kit = 0

    def begin_frame(self) -> None:
        self._pending = []

    def consider(
        self,
        track_id: int,
        box,
        frame,
        raw_class_id: int,
        locked_class_id: int,
        foot_field=None,
    ) -> None:
        box_values = [float(value) for value in list(box)[:4]]
        foot = ((box_values[0] + box_values[2]) * 0.5, box_values[3])
        field_point = None
        if foot_field is not None:
            field_point = (float(foot_field[0]), float(foot_field[1]))
        self._pending.append(
            {
                "track_id": int(track_id),
                "raw_class_id": int(raw_class_id),
                "locked_class_id": int(locked_class_id),
                "foot": foot,
                "foot_field": field_point,
                "profile": appearance_profile(frame, box_values),
            }
        )

    def finish_frame(self) -> List[dict]:
        items = self._pending
        self._pending = []
        anchors = []
        for item in items:
            if item["track_id"] < 0:
                continue
            state = self._state(item["track_id"])
            self._update_flag(state, _is_flag_shape(item["profile"]))
            if state["flag_latched"]:
                anchors.append(item)
        return [self._verdict(item, anchors) for item in items]

    def metrics(self) -> Dict[str, int]:
        return {
            "dropped_corner_flag": self.dropped_corner_flag,
            "dropped_static_sideline": self.dropped_static_sideline,
            "routed_referee_vote": self.routed_referee_vote,
            "routed_linesman": self.routed_linesman,
            "routed_yellow_kit": self.routed_yellow_kit,
            "tracks_seen": len(self._tracks),
        }

    def _state(self, track_id: int) -> dict:
        state = self._tracks.get(int(track_id))
        if state is None:
            state = {
                "flag_streak": 0,
                "flag_clear": 0,
                "flag_latched": False,
                "feet": deque(maxlen=STATIC_WINDOW),
                "static_latched": False,
                "referee_votes": deque(maxlen=REFEREE_VOTE_WINDOW),
                "referee_latched": False,
                "referee_clear": 0,
                "linesman_streak": 0,
                "yellow_streak": 0,
                "yellow_clear": 0,
                "yellow_latched": False,
            }
            self._tracks[int(track_id)] = state
        return state

    def _update_flag(self, state: dict, flag_now: bool) -> None:
        if flag_now:
            state["flag_streak"] += 1
            state["flag_clear"] = 0
            if state["flag_streak"] >= FLAG_CONFIRM_FRAMES:
                state["flag_latched"] = True
            return
        state["flag_streak"] = 0
        if not state["flag_latched"]:
            return
        state["flag_clear"] += 1
        if state["flag_clear"] >= FLAG_RELEASE_FRAMES:
            state["flag_latched"] = False
            state["flag_clear"] = 0

    def _update_static(self, state: dict, item: dict) -> bool:
        state["feet"].append(item["foot"])
        spread = _foot_spread_px(state["feet"])
        if state["static_latched"] and spread > STATIC_RELEASE_PX:
            state["static_latched"] = False
        profile = item["profile"]
        low_chroma = float(profile["low_chroma"]) > LOW_CHROMA_MIN
        aspect = float(profile["aspect"])
        enough = len(state["feet"]) >= STATIC_WINDOW and spread <= STATIC_SPREAD_PX
        if item["foot_field"] is None:
            shape_ok = aspect <= STATIC_ASPECT_MAX_NO_FIELD
            on_sideline = True
        else:
            shape_ok = aspect <= STATIC_ASPECT_MAX
            on_sideline = _near_touchline(item["foot_field"], SIDELINE_STATIC_CM)
        if enough and low_chroma and shape_ok and on_sideline:
            state["static_latched"] = True
        return bool(state["static_latched"])

    def _update_referee(self, state: dict, raw_class_id: int, profile: Dict[str, float]) -> bool:
        """Count a referee label only when the torso is yellow or a dark kit.

        The detector often calls a coloured outfield player a referee. A
        neon shirt and a black kit are the two official uniforms; a blue,
        garnet or white shirt is not.
        """
        referee_kit = (
            float(profile.get("torso_yellow", 0.0)) >= YELLOW_TORSO_MIN
            or float(profile.get("low_chroma", 0.0)) > LOW_CHROMA_MIN
        )
        vote = 1 if int(raw_class_id) == GATE_REFEREE_CLASS_ID and referee_kit else 0
        state["referee_votes"].append(vote)
        if sum(state["referee_votes"]) >= REFEREE_VOTE_MIN:
            state["referee_latched"] = True
            state["referee_clear"] = 0
            return True
        if not state["referee_latched"]:
            return False
        state["referee_clear"] = state["referee_clear"] + 1 if vote == 0 else 0
        if state["referee_clear"] >= REFEREE_CLEAR_FRAMES:
            state["referee_latched"] = False
            state["referee_clear"] = 0
        return bool(state["referee_latched"])

    def _update_yellow_referee(self, state: dict, profile: Dict[str, float]) -> bool:
        yellow = float(profile.get("torso_yellow", 0.0)) >= YELLOW_TORSO_MIN
        if yellow:
            state["yellow_streak"] += 1
            state["yellow_clear"] = 0
            if state["yellow_streak"] >= YELLOW_CONFIRM_FRAMES:
                state["yellow_latched"] = True
            return bool(state["yellow_latched"])
        state["yellow_streak"] = 0
        if not state["yellow_latched"]:
            return False
        state["yellow_clear"] += 1
        if state["yellow_clear"] >= YELLOW_CLEAR_FRAMES:
            state["yellow_latched"] = False
            state["yellow_clear"] = 0
        return bool(state["yellow_latched"])

    def _linesman_now(self, item: dict, anchors: List[dict]) -> bool:
        profile = item["profile"]
        aspect = float(profile["aspect"])
        if aspect < LINESMAN_ASPECT_MIN or aspect > LINESMAN_ASPECT_MAX:
            return False
        if float(profile["low_chroma"]) <= LOW_CHROMA_MIN:
            return False
        if not _near_touchline(item["foot_field"], SIDELINE_LINESMAN_CM):
            return False
        for anchor in anchors:
            if int(anchor["track_id"]) == int(item["track_id"]):
                continue
            if item["foot_field"] is not None and anchor["foot_field"] is not None:
                if _point_distance(item["foot_field"], anchor["foot_field"]) <= LINESMAN_FLAG_CM:
                    return True
                continue
            if _point_distance(item["foot"], anchor["foot"]) <= LINESMAN_FLAG_PX:
                return True
        return False

    def _verdict(self, item: dict, anchors: List[dict]) -> dict:
        if item["track_id"] < 0:
            if _is_flag_shape(item["profile"], strong=True):
                self.dropped_corner_flag += 1
                return {"action": "drop", "reason": "corner_flag_track"}
            return {"action": "keep", "reason": None}

        state = self._state(item["track_id"])
        if state["flag_latched"]:
            self.dropped_corner_flag += 1
            return {"action": "drop", "reason": "corner_flag_track"}
        if self._update_static(state, item):
            self.dropped_static_sideline += 1
            return {"action": "drop", "reason": "static_sideline"}
        goalkeeper = (
            int(item["locked_class_id"]) == GATE_GOALKEEPER_CLASS_ID
            or int(item["raw_class_id"]) == GATE_GOALKEEPER_CLASS_ID
        )
        if goalkeeper:
            state["linesman_streak"] = 0
            state["yellow_streak"] = 0
            state["yellow_clear"] = 0
            state["yellow_latched"] = False
            return {"action": "keep", "reason": None}
        if int(item["locked_class_id"]) == GATE_REFEREE_CLASS_ID:
            return {"action": "keep", "reason": None}
        if self._update_referee(state, item["raw_class_id"], item["profile"]):
            self.routed_referee_vote += 1
            return {"action": "referee", "reason": "referee_vote"}
        if self._linesman_now(item, anchors):
            state["linesman_streak"] += 1
        else:
            state["linesman_streak"] = 0
        if state["linesman_streak"] >= LINESMAN_CONFIRM_FRAMES:
            self.routed_linesman += 1
            return {"action": "referee", "reason": "linesman_near_flag"}
        if self._update_yellow_referee(state, item["profile"]):
            self.routed_yellow_kit += 1
            return {"action": "referee", "reason": "yellow_kit"}
        return {"action": "keep", "reason": None}
