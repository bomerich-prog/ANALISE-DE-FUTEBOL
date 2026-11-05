"""Record the player identity inputs the event log does not keep.

Each row stores the ByteTrack id, the id after stitching, the raw class, the
box, the torso hue, the shirt vote and the published team. That is enough to
replay stitching and the team lock without YOLO. The record has no manual
label of who the person is. A team change after a gap still needs the shirt
looked at on those frames.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


RECORD_VERSION = 1


def _number(value, digits=2):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def record_player(
    raw_track_id,
    detection,
    class_id,
    team_vote,
    team_locked,
    raw_class_id=None,
    hue=None,
    accepted=True,
):
    box = detection.get("box") or detection.get("bounding_box") or [None, None, None, None]
    team_id = detection.get("team_id")
    if team_id not in (0, 1):
        team_id = None
    vote = team_vote if team_vote in (0, 1) else None
    return {
        "raw_track_id": None if raw_track_id is None else int(raw_track_id),
        "track_id": int(detection.get("track_id", -1)),
        "raw_class_id": None if raw_class_id is None else int(raw_class_id),
        "class_id": None if class_id is None else int(class_id),
        "box": [_number(value) for value in list(box)[:4]],
        "hue": _number(hue, 1),
        "team_vote": vote,
        "team_id": None if team_id is None else int(team_id),
        "team_locked": bool(team_locked),
        "accepted": bool(accepted),
    }


def write_player_record(path, clip_name, fps, frames):
    payload = {
        "version": RECORD_VERSION,
        "clip_name": clip_name,
        "fps": float(fps),
        "frame_count": len(frames),
        "missing": [
            "manual identity of a person across track ids",
            "shirt label for a track whose team changes after a gap",
        ],
        "frames": frames,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return destination
