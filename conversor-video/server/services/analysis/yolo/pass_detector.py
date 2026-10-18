"""Possession-oriented pass detector.

The confirmed possessor is the event. Player A had the ball, possession may go
empty, then player B is confirmed. That sequence is a pass, a turnover, a
dispute, or a carry. An observed ball trajectory only raises or lowers the
pass score. A missing trajectory does not veto the pass.

Timing is judged on the ball flight, not on the empty-possession interval. A
short flight log keeps, per frame, the observed ball and every player. Release
is the last frame where A was within control radius of the observed ball, and
arrival is the first later frame where B was. Every relation is measured inside
one frame because the homography jitters between frames and moves players and
ball together.

PossessionEngine, the detectors, and the trackers are not modified. This
module only reads each frame's possession snapshot.

A return to the same player is a carry. If a teammate met the observed ball
inside the contest radius and the ball then left that teammate toward the
original player, the carry is split into two passes. Each leg still has to
clear the existing score. Possession confirmation is not involved.
"""

from __future__ import annotations

import math
from collections import deque

from possession import CONTROL_RADIUS_CM, CONTROL_RADIUS_PX, TEAM_NAME_BY_ID


PASS_MAX_GAP_SECONDS = 1.0
PASS_MAX_FLIGHT_SECONDS = 4.0
PASS_MIN_BALL_SPEED_CM_S = 800.0
PASS_MAX_BALL_SPEED_CM_S = 4000.0
PASS_RECEPTION_ALLOWANCE_SECONDS = 0.5
PASS_MAX_RECEPTION_LAG_SECONDS = 1.0
PASS_FLIGHT_LOG_SECONDS = 8.0
PASS_INTERCEPT_MIN_SECONDS = 0.25
PASS_INTERCEPT_MIN_FRAMES = 3
PASS_MIN_DISTANCE_CM = 150.0
PASS_MAX_DISTANCE_CM = 7000.0
PASS_SCORE_SAME_TEAM = 40
PASS_SCORE_POSSESSOR_CHANGE = 30
PASS_SCORE_SHORT_INTERVAL = 10
PASS_SCORE_PLAUSIBLE_DISTANCE = 10
PASS_SCORE_TRAJECTORY_COMPATIBLE = 10
PASS_SCORE_TRAJECTORY_INCOMPATIBLE = -20
PASS_SCORE_UNVERIFIED_EXTENSION = -20
PASS_EXTENDED_MIN_FLIGHT_SAMPLES = 2
PASS_SCORE_THRESHOLD = 80
PASS_DIRECTION_MIN_MOTION_CM = 80.0
PASS_DIRECTION_MIN_MOTION_PX = 20.0
PASS_TRAJECTORY_MIN_PROGRESS = 0.5
PASS_CONTEST_RADIUS_CM = 120.0
PASS_CONTEST_RADIUS_PX = 45.0
TURNOVER_BASE_SCORE = 70
TURNOVER_TRAJECTORY_BONUS = 10
TURNOVER_TRAJECTORY_PENALTY = -10

TIMING_BASE = "base"
TIMING_EXTENDED = "extended"
_WALL_PHYSICAL_REJECTIONS = {"timeout", "short", "long", "implausible_speed", "intercepted"}


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


def _both_points(entity):
    """Field and pixel point of one entity in one frame. Either may be None."""
    if not entity:
        return None
    field_x = _safe_float(entity.get("field_x"))
    field_y = _safe_float(entity.get("field_y"))
    pixel_x = _safe_float(entity.get("x"))
    pixel_y = _safe_float(entity.get("y"))
    field = (field_x, field_y) if field_x is not None and field_y is not None else None
    pixel = (pixel_x, pixel_y) if pixel_x is not None and pixel_y is not None else None
    if field is None and pixel is None:
        return None
    return {"field": field, "pixel": pixel}


def _inframe_distance(points_a, points_b):
    """Distance between two entities of the same frame, field first, then pixel."""
    if not points_a or not points_b:
        return None, None
    for space in ("field", "pixel"):
        a = points_a.get(space)
        b = points_b.get(space)
        if a is not None and b is not None:
            return math.hypot(a[0] - b[0], a[1] - b[1]), space
    return None, None


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


def _is_observed(ball):
    """Predicted and remembered positions keep possession alive elsewhere. They are
    not evidence of the pass trajectory.
    """
    if not ball:
        return False
    source = ball.get("source")
    return source is None or source == "observed"


def _observed_ball_point(ball):
    if not _is_observed(ball):
        return None
    return _point_from_entity(ball)


def _contest_radius(space):
    return PASS_CONTEST_RADIUS_CM if space == "field" else PASS_CONTEST_RADIUS_PX


def _control_radius(space):
    return CONTROL_RADIUS_CM if space == "field" else CONTROL_RADIUS_PX


def _min_motion(space):
    return PASS_DIRECTION_MIN_MOTION_CM if space == "field" else PASS_DIRECTION_MIN_MOTION_PX


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def max_flight_seconds(distance_cm):
    """Longest plausible release-to-arrival time for a pass of this length.

    A pass slower than PASS_MIN_BALL_SPEED_CM_S on average, plus the time the
    receiver needs to settle the ball, is a loose ball rather than a pass.
    """
    if distance_cm is None:
        return PASS_MAX_GAP_SECONDS
    budget = distance_cm / PASS_MIN_BALL_SPEED_CM_S + PASS_RECEPTION_ALLOWANCE_SECONDS
    return min(PASS_MAX_FLIGHT_SECONDS, max(PASS_MAX_GAP_SECONDS, budget))


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
        self.flight_log = deque()
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
        self.discarded_intercepted = 0
        self.discarded_implausible_speed = 0
        self.extended_window_passes = 0
        self.possession_gap_passes = 0
        self.turnovers = 0
        self.disputes = 0
        self.conductions = 0
        self.wall_passes = 0
        self.unconfirmed_candidates = 0
        self.events = []
        self.unconfirmed_events = []
        self.rejected_events = []
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
            "discarded_intercepted": self.discarded_intercepted,
            "discarded_implausible_speed": self.discarded_implausible_speed,
            "extended_window_passes": self.extended_window_passes,
            "possession_gap_passes": self.possession_gap_passes,
            "turnovers": self.turnovers,
            "disputes": self.disputes,
            "conductions": self.conductions,
            "wall_passes": self.wall_passes,
            "unconfirmed_candidates": self.unconfirmed_candidates,
            "pass_score_threshold": self.score_threshold,
            "events": list(self.events),
            "unconfirmed_events": list(self.unconfirmed_events),
            "rejected_events": list(self.rejected_events),
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
        self._log_frame(frame_number, fps, players, ball)

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
            if self._apply_wall(frame_number, fps, possessor_team, possessor_pos):
                self.pending = None
                self._remember(possessor_id, possessor_team, possessor_pos, frame_number)
                return self.metrics()
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

    def _log_frame(self, frame_number, fps, players, ball):
        roster = {}
        for player in players or []:
            try:
                track_id = int(player.get("track_id"))
            except (TypeError, ValueError):
                continue
            points = _both_points(player)
            if points is None:
                continue
            roster[track_id] = {"team": _team_name_from_player(player), "points": points}
        self.flight_log.append(
            {
                "frame": frame_number,
                "ball": _both_points(ball) if _is_observed(ball) else None,
                "players": roster,
            }
        )
        horizon = frame_number - int(math.ceil(PASS_FLIGHT_LOG_SECONDS * fps))
        while self.flight_log and self.flight_log[0]["frame"] < horizon:
            self.flight_log.popleft()

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

        from_id = int(pending["from_id"])
        to_id = int(possessor_id)
        gained_frame = int(frame_number)
        lost_frame = int(pending["lost_frame"])
        gap_seconds = max(0, gained_frame - lost_frame) / fps
        flight = self._flight_evidence(from_id, to_id, int(pending["from_frame"]), lost_frame, gained_frame, fps)
        distance_cm, distance_source = self._pass_distance(
            from_id, to_id, flight, pending.get("from_pos"), possessor_pos
        )
        trajectory = self._trajectory_label(
            to_id,
            flight["window_start"],
            flight["window_end"],
            pending.get("from_pos"),
            possessor_pos,
        )
        contested = bool(pending.get("contested"))

        if contested:
            self.disputes += 1
            self.dispute_events.append(
                {
                    "type": "dispute",
                    "from_track_id": from_id,
                    "to_track_id": to_id,
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
                    "from_track_id": from_id,
                    "to_track_id": to_id,
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

        prepared = self._prepare_pass(
            from_id,
            to_id,
            from_team,
            possessor_team,
            pending.get("from_pos"),
            possessor_pos,
            int(pending["from_frame"]),
            lost_frame,
            gained_frame,
            fps,
            wall=False,
        )
        self._record_prepared(prepared, pending.get("ball_missing_frames"))

    def _apply_wall(self, frame_number, fps, possessor_team, possessor_pos):
        """Split a same-player return into two passes when a teammate was the wall."""
        pending = self.pending
        if pending is None or pending.get("from_id") is None:
            return False
        wall = self._find_wall(pending, int(frame_number), fps)
        if wall is None:
            return False
        from_id = int(pending["from_id"])
        from_team = pending.get("from_team")
        legs = (
            self._prepare_pass(
                from_id,
                wall["id"],
                from_team,
                wall["team"],
                pending.get("from_pos"),
                wall["pos"],
                int(pending["from_frame"]),
                int(pending["lost_frame"]),
                wall["frame"],
                fps,
                wall=True,
            ),
            self._prepare_pass(
                wall["id"],
                from_id,
                wall["team"],
                possessor_team,
                wall["pos"],
                possessor_pos,
                wall["frame"],
                wall["frame"],
                int(frame_number),
                fps,
                wall=True,
            ),
        )
        if any(leg["_status"] in _WALL_PHYSICAL_REJECTIONS for leg in legs):
            return False
        for leg in legs:
            self._record_prepared(leg, pending.get("ball_missing_frames"))
        return True

    def _find_wall(self, pending, gained_frame, fps):
        from_id = int(pending["from_id"])
        from_team = pending.get("from_team")
        if from_team not in {"Team 0", "Team 1"} or pending.get("contested"):
            return None
        release_frame = self._release_frame(from_id, int(pending["lost_frame"]), int(pending["from_frame"]))
        if self._opponent_owns_ball(from_team, release_frame, gained_frame, fps):
            return None

        closest = {}
        for entry in self._entries(release_frame + 1, gained_frame - 1):
            if entry["ball"] is None or _entry_both_teams(entry):
                if _entry_both_teams(entry):
                    return None
                continue
            nearest_id, nearest_team = _nearest_within_contest(entry)
            if nearest_id is None or nearest_id == from_id or nearest_team != from_team:
                continue
            player = entry["players"].get(nearest_id)
            distance, _space = _inframe_distance(player["points"], entry["ball"])
            if distance is None:
                continue
            current = closest.get(nearest_id)
            if current is None or distance < current["distance"]:
                closest[nearest_id] = {
                    "frame": entry["frame"],
                    "distance": distance,
                    "points": player["points"],
                }
        if not closest:
            return None
        wall_id, info = min(closest.items(), key=lambda item: item[1]["distance"])
        if not self._ball_approaches_and_leaves(from_id, wall_id, release_frame, info["frame"], gained_frame):
            return None
        return {
            "id": int(wall_id),
            "team": from_team,
            "frame": int(info["frame"]),
            "pos": _point_from_logged(info["points"]),
        }

    def _release_frame(self, from_id, lost_frame, from_frame):
        for entry in reversed(self.flight_log):
            if entry["frame"] > lost_frame:
                continue
            if _within_control(entry, from_id):
                return entry["frame"]
        return from_frame

    def _opponent_owns_ball(self, from_team, first_frame, last_frame, fps):
        needed = max(PASS_INTERCEPT_MIN_FRAMES, int(math.ceil(PASS_INTERCEPT_MIN_SECONDS * fps)))
        run_id = None
        run_length = 0
        for entry in self._entries(first_frame, last_frame):
            if entry["ball"] is None:
                continue
            nearest_id, nearest_team = _nearest_within_contest(entry)
            opponent = nearest_id is not None and nearest_team in {"Team 0", "Team 1"} and nearest_team != from_team
            if opponent:
                run_length = run_length + 1 if nearest_id == run_id else 1
                run_id = nearest_id
                if run_length >= needed:
                    return True
            else:
                run_id = None
                run_length = 0
        return False

    def _ball_approaches_and_leaves(self, from_id, wall_id, release_frame, touch_frame, gained_frame):
        to_wall = self._distance_series(wall_id, release_frame, gained_frame)
        to_carrier = self._distance_series(from_id, touch_frame, gained_frame)
        touch = [sample for sample in to_wall if sample[0] == touch_frame]
        before = [sample[1] for sample in to_wall if sample[0] < touch_frame]
        after = [sample[1] for sample in to_wall if sample[0] > touch_frame]
        if not touch or not before or not after or len(to_carrier) < 2:
            return False
        motion = _min_motion(touch[0][2])
        approached = before[0] - touch[0][1] >= motion
        left_wall = after[-1] - touch[0][1] >= motion
        toward_carrier = to_carrier[0][1] - to_carrier[-1][1] >= motion
        return approached and left_wall and toward_carrier

    def _distance_series(self, track_id, first_frame, last_frame):
        samples = []
        for entry in self._entries(first_frame, last_frame):
            player = entry["players"].get(track_id)
            if entry["ball"] is None or player is None:
                continue
            distance, space = _inframe_distance(player["points"], entry["ball"])
            if distance is None:
                continue
            samples.append((entry["frame"], distance, space))
        field = [sample for sample in samples if sample[2] == "field"]
        return field or samples

    def _prepare_pass(
        self,
        from_id,
        to_id,
        from_team,
        possessor_team,
        from_pos,
        possessor_pos,
        from_frame,
        lost_frame,
        gained_frame,
        fps,
        wall,
    ):
        gap_seconds = max(0, int(gained_frame) - int(lost_frame)) / fps
        flight = self._flight_evidence(from_id, to_id, int(from_frame), int(lost_frame), int(gained_frame), fps)
        distance_cm, distance_source = self._pass_distance(
            from_id, to_id, flight, from_pos, possessor_pos
        )
        trajectory = self._trajectory_label(
            to_id,
            flight["window_start"],
            flight["window_end"],
            from_pos,
            possessor_pos,
        )
        timing_seconds = gap_seconds
        if flight["arrival_observed"] and flight["reception_lag_seconds"] <= PASS_MAX_RECEPTION_LAG_SECONDS:
            timing_seconds = min(gap_seconds, flight["flight_seconds"])
        max_timing = max_flight_seconds(distance_cm)
        tier = TIMING_BASE if timing_seconds <= self.max_gap_seconds else TIMING_EXTENDED
        implied_speed = self._implied_speed(flight, distance_cm, int(gained_frame), fps)
        event = {
            "type": "pass",
            "from_track_id": from_id,
            "to_track_id": to_id,
            "team": possessor_team,
            "frame": int(gained_frame),
            "distance_cm": None if distance_cm is None else float(distance_cm),
            "distance_source": distance_source,
            "gap_seconds": float(gap_seconds),
            "release_frame": flight["release_frame"],
            "arrival_frame": flight["arrival_frame"],
            "flight_seconds": float(flight["flight_seconds"]),
            "timing_seconds": float(timing_seconds),
            "max_timing_seconds": float(max_timing),
            "timing_tier": tier,
            "in_flight_samples": flight["in_flight_samples"],
            "implied_speed_mps": None if implied_speed is None else round(implied_speed / 100.0, 2),
            "trajectory": trajectory,
        }
        if wall:
            event["wall"] = True

        if timing_seconds > max_timing:
            event["_status"] = "timeout"
            return event
        if distance_cm is not None and distance_cm < self.min_distance_cm:
            event["_status"] = "short"
            return event
        if distance_cm is not None and distance_cm > self.max_distance_cm:
            event["_status"] = "long"
            return event
        if implied_speed is not None and implied_speed > PASS_MAX_BALL_SPEED_CM_S:
            event["_status"] = "implausible_speed"
            return event
        interceptor = self._interceptor(from_team, flight, fps)
        if interceptor is not None:
            event["interceptor_track_id"] = interceptor
            event["_status"] = "intercepted"
            return event

        components = _pass_score_components(
            short_interval=tier == TIMING_BASE,
            distance_plausible=distance_cm is not None,
            trajectory=trajectory,
            flight_verified=flight["in_flight_samples"] >= PASS_EXTENDED_MIN_FLIGHT_SAMPLES,
        )
        score = sum(components.values())
        event["score_components"] = components
        event["confidence"] = int(score)
        event["_status"] = "pass" if score >= self.score_threshold else "unconfirmed"
        return event

    def _record_prepared(self, prepared, ball_missing_frames):
        status = prepared.pop("_status")
        self.raw_candidate_passes += 1
        if status == "pass":
            self.total_passes += 1
            if prepared.get("timing_tier") == TIMING_EXTENDED:
                self.extended_window_passes += 1
            if int(ball_missing_frames or 0) > 0:
                self.possession_gap_passes += 1
            if prepared.get("wall"):
                self.wall_passes += 1
            self.events.append(prepared)
            return status
        if status == "unconfirmed":
            self.unconfirmed_candidates += 1
            self.unconfirmed_events.append(prepared)
            return status
        if status == "timeout":
            self.discarded_timeout += 1
        elif status == "short":
            self.discarded_short_passes += 1
        elif status == "long":
            self.discarded_long_passes += 1
        elif status == "implausible_speed":
            self.discarded_implausible_speed += 1
        elif status == "intercepted":
            self.discarded_intercepted += 1
        self._reject(prepared, status)
        return status

    def _reject(self, event, reason):
        rejected = dict(event)
        rejected["type"] = "rejected_pass"
        rejected["reason"] = reason
        self.rejected_events.append(rejected)

    def _entries(self, first_frame, last_frame):
        return [entry for entry in self.flight_log if first_frame <= entry["frame"] <= last_frame]

    def _flight_evidence(self, from_id, to_id, from_frame, lost_frame, gained_frame, fps):
        release_frame = None
        for entry in reversed(self.flight_log):
            if entry["frame"] > lost_frame:
                continue
            if _within_control(entry, from_id):
                release_frame = entry["frame"]
                break
        release_observed = release_frame is not None
        if release_frame is None:
            release_frame = from_frame

        arrival_frame = None
        for entry in self._entries(release_frame + 1, gained_frame):
            if _within_control(entry, to_id):
                arrival_frame = entry["frame"]
                break
        arrival_observed = arrival_frame is not None
        if arrival_frame is None:
            arrival_frame = gained_frame
        in_flight_samples = sum(
            1 for entry in self._entries(release_frame + 1, arrival_frame - 1) if entry["ball"] is not None
        )

        return {
            "release_frame": int(release_frame),
            "arrival_frame": int(arrival_frame),
            "release_observed": release_observed,
            "arrival_observed": arrival_observed,
            "flight_seconds": max(0, arrival_frame - release_frame) / fps,
            "reception_lag_seconds": max(0, gained_frame - arrival_frame) / fps,
            "in_flight_samples": int(in_flight_samples),
            "window_start": int(min(release_frame, from_frame)),
            "window_end": int(arrival_frame),
        }

    def _pass_distance(self, from_id, to_id, flight, from_pos, to_pos):
        """A-to-B distance read inside one frame, so homography jitter cancels out."""
        for frame in (flight["release_frame"], flight["arrival_frame"]):
            entry = self._entry_at(frame)
            if entry is None:
                continue
            a = entry["players"].get(from_id)
            b = entry["players"].get(to_id)
            distance, space = _inframe_distance(a and a["points"], b and b["points"])
            if space == "field":
                return distance, "same_frame"
        distance, space = _distance(from_pos, to_pos)
        if space == "field":
            return distance, "cross_frame"
        return None, None

    def _entry_at(self, frame):
        for entry in reversed(self.flight_log):
            if entry["frame"] == frame:
                return entry
            if entry["frame"] < frame:
                return None
        return None

    def _implied_speed(self, flight, distance_cm, gained_frame, fps):
        """Ball speed needed to cover the pass, using the most generous time span."""
        if distance_cm is None or not flight["release_observed"] or not flight["arrival_observed"]:
            return None
        frames = max(1, gained_frame - flight["release_frame"])
        return distance_cm / (frames / fps)

    def _interceptor(self, from_team, flight, fps):
        """Opponent that stayed alone on the ball during the flight, if any."""
        needed = max(PASS_INTERCEPT_MIN_FRAMES, int(math.ceil(PASS_INTERCEPT_MIN_SECONDS * fps)))
        run_id = None
        run_length = 0
        for entry in self._entries(flight["release_frame"] + 1, flight["arrival_frame"] - 1):
            nearest_id, nearest_team = _nearest_within_contest(entry)
            if nearest_id is not None and nearest_team in {"Team 0", "Team 1"} and nearest_team != from_team:
                run_length = run_length + 1 if nearest_id == run_id else 1
                run_id = nearest_id
                if run_length >= needed:
                    return int(run_id)
            elif entry["ball"] is not None:
                run_id = None
                run_length = 0
        return None

    def _trajectory_label(self, to_id, window_start, window_end, from_pos, to_pos):
        """Did the observed ball get closer to the receiver during the flight?

        Each sample is the ball-to-B distance inside one frame. The first and last
        samples are medians, so one jittered frame cannot flip the label.
        """
        series = {"field": [], "pixel": []}
        for entry in self._entries(window_start, window_end):
            receiver = entry["players"].get(to_id)
            if entry["ball"] is None or receiver is None:
                continue
            for space in ("field", "pixel"):
                ball_point = entry["ball"].get(space)
                receiver_point = receiver["points"].get(space)
                if ball_point is not None and receiver_point is not None:
                    series[space].append(
                        math.hypot(ball_point[0] - receiver_point[0], ball_point[1] - receiver_point[1])
                    )
        for space in ("field", "pixel"):
            samples = series[space]
            if len(samples) < 2:
                continue
            k = max(1, min(3, len(samples) // 2))
            start = _median(samples[:k])
            end = _median(samples[-k:])
            if start - end >= max(_min_motion(space), PASS_TRAJECTORY_MIN_PROGRESS * start):
                return "compatible"
            if end - start >= _min_motion(space):
                return "incompatible"
            return "missing"
        return self._cross_frame_trajectory(window_start, window_end, from_pos, to_pos)

    def _cross_frame_trajectory(self, window_start, window_end, from_pos, to_pos):
        """Fallback when the receiver is not visible next to the ball."""
        if not from_pos or not to_pos or from_pos.get("space") != to_pos.get("space"):
            return "missing"
        space = from_pos["space"]
        window = [
            entry["ball"][space]
            for entry in self._entries(window_start, window_end)
            if entry["ball"] is not None and entry["ball"].get(space) is not None
        ]
        if len(window) < 2:
            return "missing"
        motion_x = window[-1][0] - window[0][0]
        motion_y = window[-1][1] - window[0][1]
        if math.hypot(motion_x, motion_y) < _min_motion(space):
            return "missing"
        to_receiver_x = to_pos["x"] - window[-1][0]
        to_receiver_y = to_pos["y"] - window[-1][1]
        if math.hypot(to_receiver_x, to_receiver_y) <= _min_motion(space):
            return "compatible"
        if (motion_x * to_receiver_x) + (motion_y * to_receiver_y) >= 0.0:
            return "compatible"
        return "incompatible"


def _within_control(entry, track_id):
    player = entry["players"].get(track_id)
    if entry["ball"] is None or player is None:
        return False
    distance, space = _inframe_distance(player["points"], entry["ball"])
    return distance is not None and distance <= _control_radius(space)


def _point_from_logged(points):
    if not points:
        return None
    field = points.get("field")
    if field is not None:
        return {"x": field[0], "y": field[1], "space": "field"}
    pixel = points.get("pixel")
    if pixel is not None:
        return {"x": pixel[0], "y": pixel[1], "space": "pixel"}
    return None


def _entry_both_teams(entry):
    if not entry or entry.get("ball") is None:
        return False
    teams = set()
    for player in entry["players"].values():
        distance, space = _inframe_distance(player["points"], entry["ball"])
        if distance is None or distance > _contest_radius(space):
            continue
        if player.get("team") in {"Team 0", "Team 1"}:
            teams.add(player["team"])
    return "Team 0" in teams and "Team 1" in teams


def _nearest_within_contest(entry):
    if entry["ball"] is None:
        return None, None
    best = None
    for track_id, player in entry["players"].items():
        distance, space = _inframe_distance(player["points"], entry["ball"])
        if distance is None or distance > _contest_radius(space):
            continue
        if best is None or distance < best[0]:
            best = (distance, track_id, player["team"])
    if best is None:
        return None, None
    return best[1], best[2]


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


def _pass_score_components(short_interval, distance_plausible, trajectory, flight_verified=False):
    components = {
        "same_team": PASS_SCORE_SAME_TEAM,
        "possessor_change": PASS_SCORE_POSSESSOR_CHANGE,
        "short_interval": PASS_SCORE_SHORT_INTERVAL if short_interval else 0,
        "plausible_distance": PASS_SCORE_PLAUSIBLE_DISTANCE if distance_plausible else 0,
        "trajectory": 0,
        "unverified_extension": 0,
    }
    if trajectory == "compatible":
        components["trajectory"] = PASS_SCORE_TRAJECTORY_COMPATIBLE
    elif trajectory == "incompatible":
        components["trajectory"] = PASS_SCORE_TRAJECTORY_INCOMPATIBLE
    if not short_interval and not (trajectory == "compatible" and flight_verified):
        components["unverified_extension"] = PASS_SCORE_UNVERIFIED_EXTENSION
    return components


def _turnover_confidence(trajectory):
    score = TURNOVER_BASE_SCORE
    if trajectory == "compatible":
        score += TURNOVER_TRAJECTORY_BONUS
    elif trajectory == "incompatible":
        score += TURNOVER_TRAJECTORY_PENALTY
    return score
