"""Tally each track from frames the pipeline already accepted.

This does not decide possession, passes, teams, or field position. It counts
the confirmed possessor, the passes already recorded, and the foot point the
homography already wrote on each detection.
"""

from __future__ import annotations

import math

from roboflow_soccer_config import SoccerPitchConfiguration

PLAYER_STATS_EDGE_TOLERANCE_CM = 100.0


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _track_id(value):
    if isinstance(value, dict):
        value = value.get("track_id")
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _team_name(value):
    if value in {"Team 0", "Team 1"}:
        return value
    number = _finite(value)
    if number == 0:
        return "Team 0"
    if number == 1:
        return "Team 1"
    return None


class PlayerStatsAccumulator:
    def __init__(self, fps, pitch=None, edge_tolerance_cm=PLAYER_STATS_EDGE_TOLERANCE_CM):
        try:
            parsed = float(fps)
        except (TypeError, ValueError):
            parsed = 0.0
        pitch = pitch or SoccerPitchConfiguration()
        self.fps = parsed if parsed > 0 else 30.0
        self.field_length_cm = float(pitch.length)
        self.field_width_cm = float(pitch.width)
        self.edge_tolerance_cm = float(edge_tolerance_cm)
        self._tracks = {}
        self._open_possessor = None
        self._seen_pass_events = 0

    def observe(self, players, possessor_track_id=None, possessor_team=None, pass_events=None):
        seen = set()
        for player in players or []:
            track_id = _track_id(player)
            if track_id is None or track_id in seen:
                continue
            seen.add(track_id)
            row = self._row(track_id)
            row["frames_seen"] += 1
            team = _team_name(player.get("team_id") if isinstance(player, dict) else None)
            if team:
                row["team"] = team
            self._add_position(row, player.get("field_x"), player.get("field_y"))

        possessor = _track_id(possessor_track_id)
        if possessor is None:
            self._open_possessor = None
        else:
            row = self._row(possessor)
            row["frames_with_ball"] += 1
            team = _team_name(possessor_team)
            if team:
                row["team"] = team
            if self._open_possessor != possessor:
                row["possession_spells"] += 1
                self._open_possessor = possessor

        self._count_new_passes(pass_events)
        return self.as_dict()

    def _row(self, track_id):
        row = self._tracks.get(track_id)
        if row is None:
            row = {
                "team": None,
                "frames_seen": 0,
                "frames_with_ball": 0,
                "possession_spells": 0,
                "passes_made": 0,
                "passes_received": 0,
                "position_samples": 0,
                "sum_x_cm": 0.0,
                "sum_y_cm": 0.0,
            }
            self._tracks[track_id] = row
        return row

    def _add_position(self, row, field_x, field_y):
        x = _finite(field_x)
        y = _finite(field_y)
        if x is None or y is None:
            return
        tolerance = self.edge_tolerance_cm
        if not (-tolerance <= x <= self.field_length_cm + tolerance):
            return
        if not (-tolerance <= y <= self.field_width_cm + tolerance):
            return
        row["position_samples"] += 1
        row["sum_x_cm"] += min(max(x, 0.0), self.field_length_cm)
        row["sum_y_cm"] += min(max(y, 0.0), self.field_width_cm)

    def _count_new_passes(self, pass_events):
        events = list(pass_events or [])
        fresh = events[self._seen_pass_events :]
        self._seen_pass_events = len(events)
        for event in fresh:
            if not isinstance(event, dict) or event.get("type", "pass") != "pass":
                continue
            team = _team_name(event.get("team"))
            made = _track_id(event.get("from_track_id"))
            received = _track_id(event.get("to_track_id"))
            if made is not None:
                row = self._row(made)
                row["passes_made"] += 1
                if team and not row["team"]:
                    row["team"] = team
            if received is not None:
                row = self._row(received)
                row["passes_received"] += 1
                if team and not row["team"]:
                    row["team"] = team

    def as_dict(self):
        team_ball_frames = {"Team 0": 0, "Team 1": 0}
        for row in self._tracks.values():
            if row["team"] in team_ball_frames:
                team_ball_frames[row["team"]] += row["frames_with_ball"]

        tracks = []
        for track_id in sorted(self._tracks):
            row = self._tracks[track_id]
            samples = row["position_samples"]
            team_frames = team_ball_frames.get(row["team"], 0)
            if row["team"] in team_ball_frames and team_frames > 0:
                percent = round((100.0 * row["frames_with_ball"]) / team_frames, 1)
            else:
                percent = None
            tracks.append(
                {
                    "track_id": int(track_id),
                    "team": row["team"],
                    "frames_seen": int(row["frames_seen"]),
                    "frames_with_ball": int(row["frames_with_ball"]),
                    "possession_spells": int(row["possession_spells"]),
                    "seconds_with_ball": round(row["frames_with_ball"] / self.fps, 3),
                    "passes_made": int(row["passes_made"]),
                    "passes_received": int(row["passes_received"]),
                    "percent_of_team": percent,
                    "position_samples": int(samples),
                    "mean_x_cm": None if samples <= 0 else round(row["sum_x_cm"] / samples, 1),
                    "mean_y_cm": None if samples <= 0 else round(row["sum_y_cm"] / samples, 1),
                }
            )
        return {"tracks": tracks}
