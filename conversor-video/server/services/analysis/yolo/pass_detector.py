"""Possession-oriented pass detector.

The confirmed possessor is the event. Player A had the ball, possession may go
empty, then player B is confirmed. That sequence is a pass, a turnover, a
dispute, or a carry. An observed ball trajectory only raises or lowers the
pass score. A missing trajectory does not veto the pass.

PossessionEngine, the detectors, and the trackers are not modified. This
module only reads each frame's possession snapshot.
"""

from __future__ import annotations

import math
from collections import deque

from possession import TEAM_NAME_BY_ID


PASS_MAX_GAP_SECONDS = 1.0
PASS_MIN_DISTANCE_CM = 150.0
PASS_MAX_DISTANCE_CM = 7000.0
PASS_SCORE_SAME_TEAM = 40
PASS_SCORE_POSSESSOR_CHANGE = 30
PASS_SCORE_SHORT_INTERVAL = 10
PASS_SCORE_PLAUSIBLE_DISTANCE = 10
PASS_SCORE_TRAJECTORY_COMPATIBLE = 10
PASS_SCORE_TRAJECTORY_INCOMPATIBLE = -20
PASS_SCORE_THRESHOLD = 80
PASS_DIRECTION_MIN_MOTION_CM = 80.0
PASS_DIRECTION_MIN_MOTION_PX = 20.0
PASS_CONTEST_RADIUS_CM = 120.0
PASS_CONTEST_RADIUS_PX = 45.0
PASS_BALL_HISTORY = 32
TURNOVER_BASE_SCORE = 70
TURNOVER_TRAJECTORY_BONUS = 10
TURNOVER_TRAJECTORY_PENALTY = -10


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


def _team_name_from_player(player):
    if not player:
        return None
    team_id = player.get("team_id")
    if team_id in TEAM_NAME_BY_ID:
        return TEAM_NAME_BY_ID[team_id]
    if team_id in (0, "0", "Team 0"):
        return "Team 0"
    if team_id in (1, "1", "Team 1"):
        return "Team 1"
    if team_id in {"Team 0", "Team 1"}:
        return team_id
    return None


def _team_of(snapshot, players, track_id):
    if snapshot is not None and snapshot.current_possessor_track_id == track_id:
        team = snapshot.current_possession_team
        if team in {"Team 0", "Team 1"}:
            return team
    return _team_name_from_player(_find_player(players, track_id))


def _observed_ball_point(ball):
    """Use a ball point only when this frame detected the ball.

    Predicted and remembered positions keep possession alive elsewhere. They are
    not evidence of the pass trajectory.
    """
    if not ball:
        return None
    source = ball.get("source")
    if source is not None and source != "observed":
        return None
    return _point_from_entity(ball)


def _contest_radius(space):
    return PASS_CONTEST_RADIUS_CM if space == "field" else PASS_CONTEST_RADIUS_PX


def _min_motion(space):
    return PASS_DIRECTION_MIN_MOTION_CM if space == "field" else PASS_DIRECTION_MIN_MOTION_PX


class PassDetector:
    """Classify a confirmed possession change without requiring the ball path."""

    def __init__(
        self,
        max_gap_seconds=PASS_MAX_GAP_SECONDS,
        min_distance_cm=PASS_MIN_DISTANCE_CM,
        max_distance_cm=PASS_MAX_DISTANCE_CM,
        score_threshold=PASS_SCORE_THRESHOLD,
    ):
        self.max_gap_seconds = float(max_gap_seconds)
        self.min_distance_cm = float(min_distance_cm)
        self.max_distance_cm = float(max_distance_cm)
        self.score_threshold = int(score_threshold)
        self.ball_history = deque(maxlen=PASS_BALL_HISTORY)
        self.last_possessor_id = None
        self.last_possessor_team = None
        self.last_possessor_pos = None
        self.last_possessor_frame = None
        self.recent_contest = False
        self.pending = None
        self.total_passes = 0
        self.raw_candidate_passes = 0
        self.discarded_short_passes = 0
        self.discarded_long_passes = 0
        self.discarded_unstable_possession = 0
        self.discarded_timeout = 0
        self.discarded_direction_check = 0
        self.possession_gap_passes = 0
        self.turnovers = 0
        self.disputes = 0
        self.conductions = 0
        self.unconfirmed_candidates = 0
        self.events = []
        self.unconfirmed_events = []
        self.turnover_events = []
        self.dispute_events = []

    def metrics(self):
        return {
            "total_passes": self.total_passes,
            "raw_candidate_passes": self.raw_candidate_passes,
            "discarded_short_passes": self.discarded_short_passes,
            "discarded_long_passes": self.discarded_long_passes,
            "discarded_unstable_possession": self.discarded_unstable_possession,
            "discarded_timeout": self.discarded_timeout,
            "discarded_direction_check": self.discarded_direction_check,
            "possession_gap_passes": self.possession_gap_passes,
            "turnovers": self.turnovers,
            "disputes": self.disputes,
            "conductions": self.conductions,
            "unconfirmed_candidates": self.unconfirmed_candidates,
            "pass_score_threshold": self.score_threshold,
            "events": list(self.events),
            "unconfirmed_events": list(self.unconfirmed_events),
            "turnover_events": list(self.turnover_events),
            "dispute_events": list(self.dispute_events),
        }

    def update(self, frame_number, fps, snapshot, players, ball):
        frame_number = int(frame_number)
        fps = float(fps) if fps and float(fps) > 0 else 25.0
        possessor_id = snapshot.current_possessor_track_id if snapshot is not None else None
        possessor_team = _team_of(snapshot, players, possessor_id)
        possessor_player = _find_player(players, possessor_id)
        possessor_pos = _point_from_entity(possessor_player)
        observed_ball = _observed_ball_point(ball)
        if observed_ball is not None:
            self.ball_history.append((frame_number, observed_ball))

        if self.pending is not None:
            if observed_ball is None:
                self.pending["ball_missing_frames"] += 1
            elif _both_teams_on_ball(players, observed_ball):
                self.pending["contested"] = True

        if possessor_id is None:
            if self.last_possessor_id is not None and self.pending is None:
                self._open_pending(frame_number, observed_ball, players)
            return self.metrics()

        if self.pending is None and self.last_possessor_id is None:
            self._remember(possessor_id, possessor_team, possessor_pos, frame_number)
            return self.metrics()

        if self.pending is None and possessor_id == self.last_possessor_id:
            self.recent_contest = bool(observed_ball is not None and _both_teams_on_ball(players, observed_ball))
            self._remember(possessor_id, possessor_team, possessor_pos, frame_number)
            return self.metrics()

        if self.pending is None:
            self._open_pending(frame_number, observed_ball, players)

        if possessor_id == self.pending["from_id"]:
            self.conductions += 1
            self.pending = None
            self._remember(possessor_id, possessor_team, possessor_pos, frame_number)
            return self.metrics()

        self._resolve(
            frame_number,
            fps,
            possessor_id,
            possessor_team,
            possessor_pos,
        )
        self._remember(possessor_id, possessor_team, possessor_pos, frame_number)
        return self.metrics()

    def finalize(self):
        self.pending = None
        return self.metrics()

    def _remember(self, possessor_id, possessor_team, possessor_pos, frame_number):
        self.last_possessor_id = possessor_id
        self.last_possessor_team = possessor_team
        if possessor_pos is not None:
            self.last_possessor_pos = possessor_pos
        self.last_possessor_frame = frame_number

    def _open_pending(self, frame_number, observed_ball, players):
        self.pending = {
            "from_id": self.last_possessor_id,
            "from_team": self.last_possessor_team,
            "from_pos": self.last_possessor_pos,
            "from_frame": int(self.last_possessor_frame or frame_number),
            "lost_frame": int(frame_number),
            "ball_missing_frames": 0 if observed_ball is not None else 1,
            "contested": bool(self.recent_contest)
            or bool(observed_ball is not None and _both_teams_on_ball(players, observed_ball)),
        }
        self.recent_contest = False

    def _resolve(self, frame_number, fps, possessor_id, possessor_team, possessor_pos):
        pending = self.pending
        self.pending = None
        if pending is None or pending.get("from_id") is None:
            return

        gained_frame = int(frame_number)
        lost_frame = int(pending["lost_frame"])
        gap_seconds = max(0, gained_frame - lost_frame) / fps
        distance, space = _distance(pending.get("from_pos"), possessor_pos)
        distance_cm = distance if space == "field" else None
        trajectory = self._trajectory_label(
            int(pending["from_frame"]),
            gained_frame,
            pending.get("from_pos"),
            possessor_pos,
        )
        contested = bool(pending.get("contested"))

        if contested:
            self.disputes += 1
            self.dispute_events.append(
                {
                    "type": "dispute",
                    "from_track_id": int(pending["from_id"]),
                    "to_track_id": int(possessor_id),
                    "from_team": pending.get("from_team"),
                    "to_team": possessor_team,
                    "frame": gained_frame,
                    "gap_seconds": float(gap_seconds),
                }
            )
            return

        from_team = pending.get("from_team")
        if from_team in {"Team 0", "Team 1"} and possessor_team in {"Team 0", "Team 1"} and from_team != possessor_team:
            confidence = _turnover_confidence(trajectory)
            self.turnovers += 1
            self.turnover_events.append(
                {
                    "type": "turnover",
                    "from_track_id": int(pending["from_id"]),
                    "to_track_id": int(possessor_id),
                    "from_team": from_team,
                    "to_team": possessor_team,
                    "frame": gained_frame,
                    "gap_seconds": float(gap_seconds),
                    "distance_cm": None if distance_cm is None else float(distance_cm),
                    "trajectory": trajectory,
                    "confidence": confidence,
                }
            )
            return

        if from_team not in {"Team 0", "Team 1"} or possessor_team not in {"Team 0", "Team 1"} or from_team != possessor_team:
            self.discarded_unstable_possession += 1
            return

        self.raw_candidate_passes += 1
        if gap_seconds > self.max_gap_seconds:
            self.discarded_timeout += 1
            return
        if distance_cm is not None and distance_cm < self.min_distance_cm:
            self.discarded_short_passes += 1
            return
        if distance_cm is not None and distance_cm > self.max_distance_cm:
            self.discarded_long_passes += 1
            return

        distance_plausible = distance_cm is not None
        score = _pass_score(
            short_interval=True,
            distance_plausible=distance_plausible,
            trajectory=trajectory,
        )
        event = {
            "type": "pass",
            "from_track_id": int(pending["from_id"]),
            "to_track_id": int(possessor_id),
            "team": possessor_team,
            "frame": gained_frame,
            "distance_cm": None if distance_cm is None else float(distance_cm),
            "gap_seconds": float(gap_seconds),
            "trajectory": trajectory,
            "confidence": int(score),
        }
        if score < self.score_threshold:
            self.unconfirmed_candidates += 1
            self.unconfirmed_events.append(event)
            return

        self.total_passes += 1
        if int(pending.get("ball_missing_frames") or 0) > 0:
            self.possession_gap_passes += 1
        self.events.append(event)

    def _trajectory_label(self, from_frame, gained_frame, from_pos, to_pos):
        if not from_pos or not to_pos or from_pos.get("space") != to_pos.get("space"):
            return "missing"
        space = from_pos["space"]
        window = [
            point
            for frame, point in self.ball_history
            if from_frame <= int(frame) <= gained_frame and point.get("space") == space
        ]
        if len(window) < 2:
            return "missing"
        motion_x = window[-1]["x"] - window[0]["x"]
        motion_y = window[-1]["y"] - window[0]["y"]
        motion_len = math.hypot(motion_x, motion_y)
        if motion_len < _min_motion(space):
            return "missing"
        to_receiver_x = to_pos["x"] - window[-1]["x"]
        to_receiver_y = to_pos["y"] - window[-1]["y"]
        if math.hypot(to_receiver_x, to_receiver_y) <= _min_motion(space):
            return "compatible"
        if (motion_x * to_receiver_x) + (motion_y * to_receiver_y) >= 0.0:
            return "compatible"
        return "incompatible"


def _both_teams_on_ball(players, ball_point):
    if ball_point is None:
        return False
    radius = _contest_radius(ball_point["space"])
    teams = set()
    for player in players or []:
        point = _point_from_entity(player)
        distance, space = _distance(point, ball_point)
        if distance is None or space != ball_point["space"] or distance > radius:
            continue
        team = _team_name_from_player(player)
        if team in {"Team 0", "Team 1"}:
            teams.add(team)
    return "Team 0" in teams and "Team 1" in teams


def _pass_score(short_interval, distance_plausible, trajectory):
    score = PASS_SCORE_SAME_TEAM + PASS_SCORE_POSSESSOR_CHANGE
    if short_interval:
        score += PASS_SCORE_SHORT_INTERVAL
    if distance_plausible:
        score += PASS_SCORE_PLAUSIBLE_DISTANCE
    if trajectory == "compatible":
        score += PASS_SCORE_TRAJECTORY_COMPATIBLE
    elif trajectory == "incompatible":
        score += PASS_SCORE_TRAJECTORY_INCOMPATIBLE
    return score


def _turnover_confidence(trajectory):
    score = TURNOVER_BASE_SCORE
    if trajectory == "compatible":
        score += TURNOVER_TRAJECTORY_BONUS
    elif trajectory == "incompatible":
        score += TURNOVER_TRAJECTORY_PENALTY
    return score
