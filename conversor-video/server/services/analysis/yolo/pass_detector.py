"""Pass detector refinements on top of the existing possession snapshots.

This module does not change PossessionEngine. It only consumes each frame's
confirmed possessor and decides whether a same-team transfer is a real pass.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

from possession import HOLD_RADIUS_CM, HOLD_RADIUS_PX


PASS_CONFIRM_FRAMES = 5
PASS_MAX_GAP_SECONDS = 1.0
PASS_MIN_DISTANCE_CM = 200.0
PASS_MAX_DISTANCE_CM = 6000.0
PASS_DIRECTION_MIN_MOTION_CM = 80.0
PASS_BALL_HISTORY = 32


def _safe_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _point_from_entity(entity):
    if not entity:
        return None
    field_x = _safe_float(entity.get("field_x"))
    field_y = _safe_float(entity.get("field_y"))
    if field_x is not None and field_y is not None:
        return {"x": field_x, "y": field_y, "space": "field"}
    pixel_x = _safe_float(entity.get("x"))
    pixel_y = _safe_float(entity.get("y"))
    if pixel_x is not None and pixel_y is not None:
        return {"x": pixel_x, "y": pixel_y, "space": "pixel"}
    return None


def _find_player(players, track_id):
    if track_id is None:
        return None
    for player in players or []:
        try:
            if int(player.get("track_id", -1)) == int(track_id):
                return player
        except (TypeError, ValueError):
            continue
    return None


def _distance(point_a, point_b):
    if not point_a or not point_b or point_a.get("space") != point_b.get("space"):
        return None, None
    return math.hypot(point_a["x"] - point_b["x"], point_a["y"] - point_b["y"]), point_a["space"]


def _hold_radius(space):
    return HOLD_RADIUS_CM if space == "field" else HOLD_RADIUS_PX


def _team_of(snapshot, players, track_id):
    team = snapshot.current_possession_team if snapshot is not None else None
    if team in {"blue", "red"} and snapshot is not None and snapshot.current_possessor_track_id == track_id:
        return team
    player = _find_player(players, track_id)
    if not player:
        return team if team in {"blue", "red"} else None
    team_id = player.get("team_id")
    if team_id in (0, "0", "blue"):
        return "blue"
    if team_id in (1, "1", "red"):
        return "red"
    return team if team in {"blue", "red"} else None


class PassDetector:
    """Confirm same-team possession transfers that look like real passes."""

    def __init__(
        self,
        confirm_frames=PASS_CONFIRM_FRAMES,
        max_gap_seconds=PASS_MAX_GAP_SECONDS,
        min_distance_cm=PASS_MIN_DISTANCE_CM,
        max_distance_cm=PASS_MAX_DISTANCE_CM,
    ):
        self.confirm_frames = int(confirm_frames)
        self.max_gap_seconds = float(max_gap_seconds)
        self.min_distance_cm = float(min_distance_cm)
        self.max_distance_cm = float(max_distance_cm)
        self.ball_history = deque(maxlen=PASS_BALL_HISTORY)
        self.last_possessor_id = None
        self.last_possessor_team = None
        self.last_possessor_pos = None
        self.last_possessor_frame = None
        self.pending = None
        self.last_event_pair = None
        self.total_passes = 0
        self.raw_candidate_passes = 0
        self.discarded_short_passes = 0
        self.discarded_long_passes = 0
        self.discarded_unstable_possession = 0
        self.discarded_timeout = 0
        self.discarded_direction_check = 0
        self.events = []

    def metrics(self):
        return {
            "total_passes": self.total_passes,
            "raw_candidate_passes": self.raw_candidate_passes,
            "discarded_short_passes": self.discarded_short_passes,
            "discarded_long_passes": self.discarded_long_passes,
            "discarded_unstable_possession": self.discarded_unstable_possession,
            "discarded_timeout": self.discarded_timeout,
            "discarded_direction_check": self.discarded_direction_check,
            "events": list(self.events),
        }

    def update(self, frame_number, fps, snapshot, players, ball):
        frame_number = int(frame_number)
        fps = float(fps) if fps and float(fps) > 0 else 25.0
        possessor_id = snapshot.current_possessor_track_id if snapshot is not None else None
        possessor_team = _team_of(snapshot, players, possessor_id)
        possessor_player = _find_player(players, possessor_id)
        possessor_pos = _point_from_entity(possessor_player)
        ball_point = _point_from_entity(ball)
        if ball_point is not None:
            self.ball_history.append((frame_number, ball_point))

        if self.pending is not None:
            self._refresh_pending_radius(players, ball_point)

        if possessor_id is None:
            if self.last_possessor_id is not None:
                self._handle_loss(frame_number, ball_point)
            self.last_possessor_id = None
            self.last_possessor_team = None
            return self.metrics()

        if self.last_possessor_id is None:
            self._handle_gain_from_empty(frame_number, fps, possessor_id, possessor_team, possessor_pos)
        elif possessor_id != self.last_possessor_id:
            self._handle_direct_switch(
                frame_number,
                fps,
                possessor_id,
                possessor_team,
                possessor_pos,
                ball_point,
            )
        else:
            self._handle_hold(frame_number, fps, possessor_id, possessor_team, possessor_pos)

        self.last_possessor_id = possessor_id
        self.last_possessor_team = possessor_team
        if possessor_pos is not None:
            self.last_possessor_pos = possessor_pos
        self.last_possessor_frame = frame_number
        return self.metrics()

    def finalize(self):
        if self.pending is not None and self.pending.get("to_id") is not None:
            self._discard_pending("unstable")
        self.pending = None
        return self.metrics()

    def _handle_loss(self, frame_number, ball_point):
        if self.pending is not None and self.pending.get("to_id") is not None:
            self._discard_pending("unstable")
        self.pending = {
            "from_id": self.last_possessor_id,
            "from_team": self.last_possessor_team,
            "from_pos": self.last_possessor_pos,
            "lost_frame": int(frame_number),
            "to_id": None,
            "to_team": None,
            "gained_frame": None,
            "hold_frames": 0,
            "ball_left_radius": self._ball_left_donor_radius(self.last_possessor_pos, ball_point),
            "raw_counted": False,
        }

    def _handle_gain_from_empty(self, frame_number, fps, possessor_id, possessor_team, possessor_pos):
        if self.pending is None or self.pending.get("from_id") is None:
            return
        if self.pending.get("to_id") is not None:
            self._discard_pending("unstable")
            return
        if possessor_id == self.pending["from_id"]:
            self.pending = None
            return
        if self.pending.get("from_team") != possessor_team or possessor_team is None:
            self.pending = None
            return
        self.raw_candidate_passes += 1
        self.pending["to_id"] = possessor_id
        self.pending["to_team"] = possessor_team
        self.pending["gained_frame"] = int(frame_number)
        self.pending["hold_frames"] = 1
        self.pending["raw_counted"] = True
        if possessor_pos is not None:
            self.pending["to_pos"] = possessor_pos
        self._try_confirm(frame_number, fps)

    def _handle_direct_switch(self, frame_number, fps, possessor_id, possessor_team, possessor_pos, ball_point):
        if self.pending is not None and self.pending.get("to_id") is not None:
            self._discard_pending("unstable")
        same_team = (
            self.last_possessor_team is not None
            and self.last_possessor_team == possessor_team
        )
        if not same_team:
            self.pending = None
            return
        self.raw_candidate_passes += 1
        self.pending = {
            "from_id": self.last_possessor_id,
            "from_team": self.last_possessor_team,
            "from_pos": self.last_possessor_pos,
            "lost_frame": int(frame_number),
            "to_id": possessor_id,
            "to_team": possessor_team,
            "to_pos": possessor_pos,
            "gained_frame": int(frame_number),
            "hold_frames": 1,
            "ball_left_radius": self._ball_left_donor_radius(self.last_possessor_pos, ball_point),
            "raw_counted": True,
        }
        self._try_confirm(frame_number, fps)

    def _handle_hold(self, frame_number, fps, possessor_id, possessor_team, possessor_pos):
        if self.pending is None or self.pending.get("to_id") != possessor_id:
            return
        self.pending["hold_frames"] = int(self.pending.get("hold_frames") or 0) + 1
        self.pending["to_team"] = possessor_team or self.pending.get("to_team")
        if possessor_pos is not None:
            self.pending["to_pos"] = possessor_pos
        self._try_confirm(frame_number, fps)

    def _refresh_pending_radius(self, players, ball_point):
        pending = self.pending
        if pending is None:
            return
        donor = _find_player(players, pending.get("from_id"))
        donor_pos = _point_from_entity(donor) or pending.get("from_pos")
        if donor_pos is not None:
            pending["from_pos"] = donor_pos
        if self._ball_left_donor_radius(pending.get("from_pos"), ball_point):
            pending["ball_left_radius"] = True

    def _ball_left_donor_radius(self, donor_pos, ball_point):
        distance, space = _distance(donor_pos, ball_point)
        if distance is None:
            return False
        return distance > _hold_radius(space)

    def _try_confirm(self, frame_number, fps):
        pending = self.pending
        if pending is None or pending.get("to_id") is None:
            return
        if int(pending.get("hold_frames") or 0) < self.confirm_frames:
            return
        if not pending.get("ball_left_radius"):
            self._discard_pending("unstable")
            return
        gained_frame = int(pending.get("gained_frame") or frame_number)
        lost_frame = int(pending.get("lost_frame") or gained_frame)
        gap_seconds = max(0, gained_frame - lost_frame) / fps
        if gap_seconds > self.max_gap_seconds:
            self._discard_pending("timeout")
            return
        distance, space = _distance(pending.get("from_pos"), pending.get("to_pos"))
        if distance is None or space != "field":
            self._discard_pending("unstable")
            return
        if distance < self.min_distance_cm:
            self._discard_pending("short")
            return
        if distance > self.max_distance_cm:
            self._discard_pending("long")
            return
        if not self._direction_compatible(pending):
            self._discard_pending("direction")
            return
        event_pair = (int(pending["from_id"]), int(pending["to_id"]))
        if self.last_event_pair == event_pair:
            self.pending = None
            return
        self.total_passes += 1
        self.last_event_pair = event_pair
        self.events.append(
            {
                "from_track_id": event_pair[0],
                "to_track_id": event_pair[1],
                "team": pending.get("to_team") or pending.get("from_team"),
                "frame": int(frame_number),
                "distance_cm": float(distance),
                "gap_seconds": float(gap_seconds),
            }
        )
        self.pending = None

    def _direction_compatible(self, pending):
        from_pos = pending.get("from_pos")
        to_pos = pending.get("to_pos")
        if not from_pos or not to_pos or from_pos.get("space") != to_pos.get("space"):
            return False
        lost_frame = int(pending.get("lost_frame") or 0)
        window = [
            point
            for frame, point in self.ball_history
            if frame >= lost_frame and point.get("space") == from_pos["space"]
        ]
        if len(window) < 2:
            window = [point for _, point in self.ball_history if point.get("space") == from_pos["space"]]
        if len(window) < 2:
            return True
        motion_x = window[-1]["x"] - window[0]["x"]
        motion_y = window[-1]["y"] - window[0]["y"]
        motion_len = math.hypot(motion_x, motion_y)
        to_receiver_x = to_pos["x"] - window[-1]["x"]
        to_receiver_y = to_pos["y"] - window[-1]["y"]
        to_receiver_len = math.hypot(to_receiver_x, to_receiver_y)
        min_motion = PASS_DIRECTION_MIN_MOTION_CM if from_pos["space"] == "field" else 20.0
        if to_receiver_len <= min_motion:
            return True
        if motion_len < min_motion:
            return True
        return (motion_x * to_receiver_x + motion_y * to_receiver_y) >= 0.0

    def _discard_pending(self, reason):
        pending = self.pending
        self.pending = None
        if not pending or not pending.get("raw_counted"):
            return
        if reason == "short":
            self.discarded_short_passes += 1
        elif reason == "long":
            self.discarded_long_passes += 1
        elif reason == "timeout":
            self.discarded_timeout += 1
        elif reason == "direction":
            self.discarded_direction_check += 1
        else:
            self.discarded_unstable_possession += 1
