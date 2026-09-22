"""Ball possession state machine for the current analysis pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


POSSESSION_CONFIRM_FRAMES = 5
POSSESSION_LOSE_FRAMES = 8
POSSESSION_SWITCH_FRAMES = 8
CONTROL_RADIUS_CM = 250.0
HOLD_RADIUS_CM = 400.0
CONTROL_RADIUS_PX = 90.0
HOLD_RADIUS_PX = 140.0
CONTEST_MARGIN_CM = 70.0
CONTEST_MARGIN_PX = 25.0
TEAM_NAME_BY_ID = {0: "Team 0", 1: "Team 1"}


@dataclass
class PossessionSnapshot:
    current_possessor_track_id: Optional[int] = None
    current_possession_team: Optional[str] = None

    def as_dict(self):
        return {
            "current_possessor_track_id": self.current_possessor_track_id,
            "current_possession_team": self.current_possession_team,
        }


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


def _hypot(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def _player_distance(player, ball):
    player_field_x = _safe_float(player.get("field_x"))
    player_field_y = _safe_float(player.get("field_y"))
    ball_field_x = _safe_float(ball.get("field_x"))
    ball_field_y = _safe_float(ball.get("field_y"))
    if None not in (player_field_x, player_field_y, ball_field_x, ball_field_y):
        return _hypot(player_field_x, player_field_y, ball_field_x, ball_field_y), "field"

    player_x = _safe_float(player.get("x"))
    player_y = _safe_float(player.get("y"))
    ball_x = _safe_float(ball.get("x"))
    ball_y = _safe_float(ball.get("y"))
    if None not in (player_x, player_y, ball_x, ball_y):
        return _hypot(player_x, player_y, ball_x, ball_y), "pixel"
    return None, None


def _radii_for_space(space):
    if space == "field":
        return CONTROL_RADIUS_CM, HOLD_RADIUS_CM, CONTEST_MARGIN_CM
    return CONTROL_RADIUS_PX, HOLD_RADIUS_PX, CONTEST_MARGIN_PX


def build_possession_players(detections):
    players = []
    for detection in detections or []:
        box = detection.get("box") or detection.get("bounding_box")
        foot_x = None
        foot_y = None
        if box and len(box) >= 4:
            foot_x = (float(box[0]) + float(box[2])) / 2.0
            foot_y = float(box[3])
        players.append(
            {
                "track_id": int(detection.get("track_id", -1)),
                "team_id": detection.get("team_id"),
                "field_x": detection.get("field_x"),
                "field_y": detection.get("field_y"),
                "x": foot_x,
                "y": foot_y,
            }
        )
    return players


def build_ball_state(observed_ball, remembered_ball=None, predicted_field=None, predicted_pixel=None, internal_alive=False):
    if observed_ball:
        box = observed_ball.get("box") or observed_ball.get("bounding_box")
        center_x = observed_ball.get("center_x")
        center_y = observed_ball.get("center_y")
        if (center_x is None or center_y is None) and box and len(box) >= 4:
            center_x = (float(box[0]) + float(box[2])) / 2.0
            center_y = (float(box[1]) + float(box[3])) / 2.0
        return {
            "source": "observed",
            "field_x": observed_ball.get("field_x"),
            "field_y": observed_ball.get("field_y"),
            "x": center_x,
            "y": center_y,
        }

    if not internal_alive:
        return None

    remembered = remembered_ball or {}
    remembered_box = remembered.get("box") or remembered.get("bounding_box")
    remembered_x = remembered.get("x")
    remembered_y = remembered.get("y")
    if (remembered_x is None or remembered_y is None) and remembered_box and len(remembered_box) >= 4:
        remembered_x = (float(remembered_box[0]) + float(remembered_box[2])) / 2.0
        remembered_y = (float(remembered_box[1]) + float(remembered_box[3])) / 2.0

    predicted_x = None
    predicted_y = None
    if predicted_pixel is not None and len(predicted_pixel) >= 2:
        predicted_x = float(predicted_pixel[0])
        predicted_y = float(predicted_pixel[1])

    predicted_field_x = None
    predicted_field_y = None
    if predicted_field is not None and len(predicted_field) >= 2:
        predicted_field_x = float(predicted_field[0])
        predicted_field_y = float(predicted_field[1])
        source = "predicted"
    elif predicted_x is not None:
        source = "predicted"
    else:
        source = "remembered"

    return {
        "source": source,
        "field_x": predicted_field_x if predicted_field_x is not None else remembered.get("field_x"),
        "field_y": predicted_field_y if predicted_field_y is not None else remembered.get("field_y"),
        "x": predicted_x if predicted_x is not None else remembered_x,
        "y": predicted_y if predicted_y is not None else remembered_y,
    }


class PossessionEngine:
    """Confirm/hold/switch/lose possession using track_id and field coordinates."""

    def __init__(
        self,
        confirm_frames=POSSESSION_CONFIRM_FRAMES,
        lose_frames=POSSESSION_LOSE_FRAMES,
        switch_frames=POSSESSION_SWITCH_FRAMES,
    ):
        self.confirm_frames = confirm_frames
        self.lose_frames = lose_frames
        self.switch_frames = switch_frames
        self.possessor_id = None
        self.possessor_team_id = None
        self.pending_id = None
        self.pending_count = 0
        self.switch_id = None
        self.switch_count = 0
        self.no_evidence_count = 0
        self.last_team_by_track = {}
        self.frames_with_possessor = 0
        self.confirmed_gains = 0
        self.losses = 0
        self.switches = 0

    def snapshot(self):
        team = TEAM_NAME_BY_ID.get(self.possessor_team_id) if self.possessor_id is not None else None
        return PossessionSnapshot(self.possessor_id, team)

    def metrics(self):
        return {
            "confirm_frames": self.confirm_frames,
            "lose_frames": self.lose_frames,
            "switch_frames": self.switch_frames,
            "frames_with_possessor": self.frames_with_possessor,
            "confirmed_gains": self.confirmed_gains,
            "losses": self.losses,
            "switches": self.switches,
        }

    def update(self, players, ball):
        previous_id = self.possessor_id
        self._remember_teams(players)

        if not ball or ball.get("source") == "lost":
            self._lose_if_unsupported()
            return self._finish_update(previous_id)

        ranked = self._rank_players(players, ball)
        allow_new = ball.get("source") == "observed"
        possessor_entry = next((item for item in ranked if item["track_id"] == self.possessor_id), None)
        nearest = ranked[0] if ranked else None
        control_radius, hold_radius, contest_margin = _radii_for_space(
            (possessor_entry or nearest or {}).get("space") or "pixel"
        )

        if self.possessor_id is not None:
            if possessor_entry and possessor_entry["distance"] <= hold_radius:
                self.no_evidence_count = 0
                contested = (
                    nearest is not None
                    and nearest["track_id"] != self.possessor_id
                    and nearest["distance"] + contest_margin >= possessor_entry["distance"]
                )
                if (
                    nearest is None
                    or nearest["track_id"] == self.possessor_id
                    or contested
                    or not allow_new
                    or nearest["distance"] > control_radius
                ):
                    self._clear_switch()
                    self._refresh_possessor_team()
                    return self._finish_update(previous_id)
                self._advance_switch(nearest["track_id"])
                return self._finish_update(previous_id)

            if allow_new and nearest is not None and nearest["distance"] <= control_radius:
                self.no_evidence_count = 0
                self._advance_switch(nearest["track_id"])
                return self._finish_update(previous_id)

            self._lose_if_unsupported()
            return self._finish_update(previous_id)

        if allow_new and nearest is not None and nearest["distance"] <= control_radius:
            self.no_evidence_count = 0
            self._advance_confirm(nearest["track_id"])
            return self._finish_update(previous_id)

        self._reset_pending()
        self._lose_if_unsupported()
        return self._finish_update(previous_id)

    def _remember_teams(self, players):
        for player in players or []:
            track_id = int(player.get("track_id", -1))
            team_id = player.get("team_id")
            if track_id >= 0 and team_id is not None:
                try:
                    self.last_team_by_track[track_id] = int(team_id)
                except (TypeError, ValueError):
                    continue

    def _rank_players(self, players, ball):
        ranked = []
        for player in players or []:
            track_id = int(player.get("track_id", -1))
            if track_id < 0:
                continue
            distance, space = _player_distance(player, ball)
            if distance is None:
                continue
            ranked.append({"track_id": track_id, "distance": distance, "space": space})
        ranked.sort(key=lambda item: (0 if item["space"] == "field" else 1, item["distance"]))
        return ranked

    def _advance_confirm(self, track_id):
        if self.pending_id == track_id:
            self.pending_count += 1
        else:
            self.pending_id = track_id
            self.pending_count = 1
        if self.pending_count >= self.confirm_frames:
            self._set_possessor(track_id)
            self.confirmed_gains += 1
            self._reset_pending()
            self._clear_switch()

    def _advance_switch(self, track_id):
        if track_id == self.possessor_id:
            self._clear_switch()
            return
        if self.switch_id == track_id:
            self.switch_count += 1
        else:
            self.switch_id = track_id
            self.switch_count = 1
        if self.switch_count >= self.switch_frames:
            self._set_possessor(track_id)
            self.switches += 1
            self._clear_switch()
            self._reset_pending()

    def _lose_if_unsupported(self):
        self.no_evidence_count += 1
        self._reset_pending()
        self._clear_switch()
        if self.possessor_id is not None and self.no_evidence_count >= self.lose_frames:
            self.possessor_id = None
            self.possessor_team_id = None
            self.losses += 1
            self.no_evidence_count = 0

    def _set_possessor(self, track_id):
        self.possessor_id = track_id
        self.possessor_team_id = self.last_team_by_track.get(track_id)
        self.no_evidence_count = 0

    def _refresh_possessor_team(self):
        if self.possessor_id is None:
            return
        team_id = self.last_team_by_track.get(self.possessor_id)
        if team_id is not None:
            self.possessor_team_id = team_id

    def _reset_pending(self):
        self.pending_id = None
        self.pending_count = 0

    def _clear_switch(self):
        self.switch_id = None
        self.switch_count = 0

    def _finish_update(self, previous_id):
        if self.possessor_id is not None:
            self.frames_with_possessor += 1
            self._refresh_possessor_team()
        return self.snapshot()
