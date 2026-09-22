"""Lock Player, Referee and Goalkeeper labels per track.

Thin posts, corner flags and incompatible boxes are rejected before a track
exists. A confirmed class stays until several consecutive frames disagree.
"""

from __future__ import annotations

from typing import Dict, Optional

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
