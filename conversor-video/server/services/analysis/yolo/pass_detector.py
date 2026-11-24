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

from possession import (
    CONTEST_MARGIN_CM,
    CONTEST_MARGIN_PX,
    CONTROL_RADIUS_CM,
    CONTROL_RADIUS_PX,
    TEAM_NAME_BY_ID,
    pixel_ground_distance,
)


PASS_MAX_GAP_SECONDS = 1.0
PASS_MAX_FLIGHT_SECONDS = 4.0
# A pass measured in image pixels has no field length, so the one-second gap
# is the normal budget. A flight the detector actually followed, moving toward
# the receiver, can run longer than that gap without being a loose ball: a long
# ball across the box took 2.2 s (France v Spain, Olympic final).
PASS_VERIFIED_PIXEL_FLIGHT_SECONDS = 2.5
PASS_MIN_BALL_SPEED_CM_S = 800.0
PASS_MAX_BALL_SPEED_CM_S = 4000.0
PASS_RECEPTION_ALLOWANCE_SECONDS = 0.5
PASS_MAX_RECEPTION_LAG_SECONDS = 1.0
PASS_FLIGHT_LOG_SECONDS = 8.0
# A receiver whose shirt has not voted yet (covered by an opponent) holds the
# decision this long, waiting for his team, instead of dropping the pass.
PASS_TEAM_WAIT_SECONDS = 2.0
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
# A flight with the ball seen in fewer than one frame in four was mostly hidden.
HIDDEN_FLIGHT_SEEN_RATIO = 4
PASS_SCORE_THRESHOLD = 80
PASS_DIRECTION_MIN_MOTION_CM = 80.0
PASS_DIRECTION_MIN_MOTION_PX = 20.0
PASS_TRAJECTORY_MIN_PROGRESS = 0.5
PASS_CONTEST_RADIUS_CM = 120.0
PASS_CONTEST_RADIUS_PX = 45.0
# A one-touch is closer than a ball rolling past a defender. Two observed
# frames at this range can start a pass without a confirmed possession.
TOUCH_RADIUS_CM = 80.0
TOUCH_RADIUS_PX = 30.0
TOUCH_MARGIN_CM = 30.0
TOUCH_MARGIN_PX = 12.0
TOUCH_MIN_OBSERVED = 2
# First-time ball: the ball of an open pass reaches a teammate of the passer
# (within the contest radius, no teamed opponent closer) and is next seen going
# back the other way, away from him, within this many frames.
FIRST_TIME_MIN_FRAMES = 3
FIRST_TIME_MAX_FRAMES = 20
FIRST_TIME_MIN_MOTION_PX = 20.0
FIRST_TIME_MIN_MOTION_CM = 60.0
# A teammate who takes the ball afterwards controlled it (normal pass); the
# first-time reading waits this long for that before it counts.
FIRST_TIME_CONFIRM_FRAMES = 15
# A lookalike visit: the ball layer leaves the ball, settles on a white boot or
# sock for a few frames, then finds the real ball again back where it was.
# Spain v Cape Verde, 3.1 s and 7.8 s: Spain's 19 stood still by the touchline,
# the ball sat on his boot without moving a pixel, and the next ball was 160 to
# 220 px away after three or four frames. No pass leaves a still ball that fast.
LOOKALIKE_STILL_PX = 6.0
LOOKALIKE_STILL_CM = 30.0
LOOKALIKE_MIN_STILL_FRAMES = 2
LOOKALIKE_EXIT_PX_PER_FRAME = 40.0
# A new track id that starts where the player on the ball just vanished is the
# same player under a new number (ByteTrack retires ids in a crowd). Spain v
# Cape Verde: 12 became 36 at 9.5 s, 84 became 88 at 29.4 s and 36 became 120
# at 41.5 s, each time on the ball, and each change read as a pass.
HANDOFF_MAX_GAP_FRAMES = 12
HANDOFF_RADIUS_PX = 64.0
HANDOFF_RADIUS_CM = 250.0
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
            return _space_distance(a[0], a[1], b[0], b[1], space), space
    return None, None


def _space_distance(x1, y1, x2, y2, space):
    if space == "pixel":
        return pixel_ground_distance(x1, y1, x2, y2)
    return math.hypot(x1 - x2, y1 - y2)


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
    space = point_a["space"]
    return _space_distance(point_a["x"], point_a["y"], point_b["x"], point_b["y"], space), space


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


def _touch_radius(space):
    return TOUCH_RADIUS_CM if space == "field" else TOUCH_RADIUS_PX


def _touch_margin(space):
    return TOUCH_MARGIN_CM if space == "field" else TOUCH_MARGIN_PX


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


def max_flight_seconds(distance_cm, verified_pixel_flight=False):
    """Longest plausible release-to-arrival time for a pass of this length.

    A pass slower than PASS_MIN_BALL_SPEED_CM_S on average, plus the time the
    receiver needs to settle the ball, is a loose ball rather than a pass.
    Without a field length, a flight is that one-second gap unless the ball
    was seen moving toward the receiver the whole way.
    """
    if distance_cm is None:
        if verified_pixel_flight:
            return PASS_VERIFIED_PIXEL_FLIGHT_SECONDS
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
        self.known_teams = {}
        self.waiting_for_team = []
        self.team_wait_resolved = 0
        self.first_time_receptions = 0
        self.duel_turnovers = 0
        self.duel = None
        self.possession_start = None
        self.deflection = None
        self.first_time = None
        self._last_ball = None
        # Ball step that ended at _last_ball (the step before the current one).
        self._ball_step = None
        self._fps = 25.0
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
        self.short_touch_passes = 0
        self.touch = None
        self.touch_origin = None
        self.lookalike_touches = 0
        self.identity_handoffs = 0
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
            "team_wait_resolved": self.team_wait_resolved,
            "first_time_receptions": self.first_time_receptions,
            "duel_turnovers": self.duel_turnovers,
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
            "short_touch_passes": self.short_touch_passes,
            "lookalike_touches": self.lookalike_touches,
            "identity_handoffs": self.identity_handoffs,
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
        self._fps = fps
        possessor_id = snapshot.current_possessor_track_id if snapshot is not None else None
        possessor_team = _team_of(snapshot, players, possessor_id)
        possessor_player = _find_player(players, possessor_id)
        possessor_pos = _point_from_entity(possessor_player)
        observed_ball = _observed_ball_point(ball)
        self._note_teams(players)
        if self.first_time is not None:
            candidate = self.first_time
            if possessor_id is not None:
                self.first_time = None
                if possessor_id not in {candidate["id"], (self.pending or {}).get("from_id")}:
                    # Someone else took the ball after the first-time touch:
                    # the touch was a pass in, and the ball went on from him.
                    self._commit_first_time(candidate)
                    self._open_pending(candidate["frame"], None, players)
                    self._duel_turnover(candidate, fps, possessor_id, possessor_team, players)
            elif frame_number - candidate["seen"] >= FIRST_TIME_CONFIRM_FRAMES:
                self.first_time = None
                self._commit_first_time(candidate)
                if candidate.get("rival") is not None:
                    self.duel = candidate
        if possessor_id is None and self.pending is not None and self.first_time is None:
            self._watch_first_time(frame_number, fps, players, observed_ball)
        else:
            self.deflection = None
        if observed_ball is not None:
            last = self._last_ball
            self._ball_step = (
                (observed_ball["x"] - last["x"], observed_ball["y"] - last["y"])
                if last is not None and last["space"] == observed_ball["space"]
                else None
            )
            self._last_ball = observed_ball
        self._retry_waiting(frame_number, fps)
        self._log_frame(frame_number, fps, players, ball)
        self._observe_touch(frame_number, players, ball, possessor_id)

        if self.pending is not None:
            if observed_ball is None:
                self.pending["ball_missing_frames"] += 1
            elif _teams_contesting(players, observed_ball):
                self.pending["contested"] = True

        if possessor_id is None:
            if self.last_possessor_id is not None and self.pending is None:
                self._open_pending(frame_number, observed_ball, players)
            return self.metrics()

        if self.pending is None and self.last_possessor_id is None:
            self._remember(possessor_id, possessor_team, possessor_pos, frame_number)
            return self.metrics()

        if self.pending is None and possessor_id == self.last_possessor_id:
            self.recent_contest = bool(observed_ball is not None and _teams_contesting(players, observed_ball))
            self._remember(possessor_id, possessor_team, possessor_pos, frame_number)
            return self.metrics()

        if self.pending is None:
            self._open_pending(frame_number, observed_ball, players)

        duel = self.duel
        self.duel = None
        if duel is not None and self.pending.get("from_id") == duel["id"]:
            self._duel_turnover(duel, fps, possessor_id, possessor_team, players)

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
        if self.first_time is not None:
            candidate = self.first_time
            self.first_time = None
            self._commit_first_time(candidate)
        self.pending = None
        self.waiting_for_team = []
        return self.metrics()

    def _note_teams(self, players):
        for player in players or []:
            team = _team_name_from_player(player)
            if team not in {"Team 0", "Team 1"}:
                continue
            try:
                self.known_teams[int(player.get("track_id"))] = team
            except (TypeError, ValueError):
                continue

    def _retry_waiting(self, frame_number, fps):
        """Settle the passes that waited for a receiver's shirt, once it voted."""
        still = []
        for item in self.waiting_for_team:
            team = self.known_teams.get(item["to_id"])
            if team is None:
                if frame_number - item["frame"] <= PASS_TEAM_WAIT_SECONDS * fps:
                    still.append(item)
                else:
                    self.discarded_unstable_possession += 1
                continue
            saved = self.pending
            self.pending = item["pending"]
            self.team_wait_resolved += 1
            self._resolve(item["frame"], item["fps"], item["to_id"], team, item["to_pos"], waited=True)
            self.pending = saved
        self.waiting_for_team = still

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
        if possessor_id != self.last_possessor_id or self.possession_start is None:
            self.possession_start = int(frame_number)
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
            "possession_start": int(self.possession_start if self.possession_start is not None else frame_number),
            "lost_frame": int(frame_number),
            "ball_missing_frames": 0 if observed_ball is not None else 1,
            "contested": bool(self.recent_contest)
            or bool(observed_ball is not None and _teams_contesting(players, observed_ball)),
        }
        self.recent_contest = False

    def _observe_touch(self, frame_number, players, ball, possessor_id):
        """A tight contact can start a pass. It does not change possession.

        Predicted frames neither open nor erase the contact. A player the ball
        only passes does not replace a touch that already left someone's foot.
        """
        if not ball or ball.get("source") not in (None, "observed"):
            return
        ball_point = _observed_ball_point(ball)
        if ball_point is None:
            return
        ranked = []
        for player in players or []:
            team = _team_name_from_player(player)
            if team not in {"Team 0", "Team 1"}:
                continue
            point = _point_from_entity(player)
            distance, space = _distance(point, ball_point)
            if distance is None or space != ball_point["space"]:
                continue
            try:
                track_id = int(player.get("track_id"))
            except (TypeError, ValueError):
                continue
            ranked.append({"id": track_id, "team": team, "distance": distance, "space": space, "pos": point})
        ranked.sort(key=lambda item: item["distance"])
        leader = ranked[0] if ranked else None
        radius = _touch_radius(leader["space"]) if leader else None
        margin = _touch_margin(leader["space"]) if leader else None
        unique = (
            leader is not None
            and leader["distance"] <= radius
            and (len(ranked) == 1 or ranked[1]["distance"] >= leader["distance"] + margin)
        )
        if not unique or int(leader["id"]) == int(possessor_id or -1):
            # The confirmed possessor standing on the ball is possession, not a
            # one-touch. A passer-by does not replace a touch already released.
            if self.touch is not None and int(self.touch["frames"]) >= TOUCH_MIN_OBSERVED:
                self._release_touch(frame_number, ball_point)
            else:
                self.touch = None
            return
        touch = self.touch
        if touch is None or int(touch["id"]) != int(leader["id"]):
            if touch is not None and int(touch["frames"]) >= TOUCH_MIN_OBSERVED:
                self._release_touch(frame_number, ball_point)
            if self.touch_origin is not None:
                self.touch = None
                return
            self.touch = {
                "id": leader["id"],
                "team": leader["team"],
                "frames": 1,
                "pos": leader["pos"],
                "first_frame": int(frame_number),
                "frame": int(frame_number),
            }
            return
        touch["frames"] = int(touch["frames"]) + 1
        touch["pos"] = leader["pos"]
        touch["frame"] = int(frame_number)

    def _commit_first_time(self, candidate):
        if self.pending is None:
            return
        self.first_time_receptions += 1
        self._resolve(candidate["frame"], candidate["fps"], candidate["id"], candidate["team"], candidate["pos"])
        self.last_possessor_id = candidate["id"]
        self.last_possessor_team = candidate["team"]
        self.last_possessor_pos = candidate["pos"]
        self.last_possessor_frame = candidate["frame"]

    def _duel_turnover(self, candidate, fps, possessor_id, possessor_team, players):
        """The ball came back out of a duel and went to the other team.

        Argentina v Spain, 35 s: Spain's pass reached its man with an Argentine
        on top of him, and the ball came straight back out to Argentina's 20.
        The Argentine in the duel had no shirt vote yet, so the reply was given
        to the Spaniard and the move read as Spain losing the ball to the 20.
        The man who sent the ball out of the duel is the rival: the ball was won
        there, and his touch is a pass to the teammate who took it.
        """
        rival = candidate.get("rival")
        if rival is None or self.pending is None:
            return
        if possessor_team not in {"Team 0", "Team 1"} or possessor_team == candidate.get("team"):
            return
        rival_id = int(rival["id"])
        if rival_id == int(possessor_id):
            return
        rival_team = rival.get("team") if rival.get("team") in {"Team 0", "Team 1"} else self.known_teams.get(rival_id)
        if rival_team not in {None, possessor_team}:
            return
        frame = int(candidate["frame"])
        # The touch that left the duel belongs to the pass that follows it.
        self._resolve(frame, fps, rival_id, possessor_team, rival.get("pos"), short_touch=False)
        self.duel_turnovers += 1
        self.last_possessor_id = rival_id
        self.last_possessor_team = possessor_team
        self.last_possessor_pos = rival.get("pos")
        self.last_possessor_frame = frame
        self.possession_start = frame
        self._open_pending(frame, None, players)

    def _watch_first_time(self, frame_number, fps, players, ball_point):
        """Close an open pass on a teammate who played it first time.

        Final Spain v Argentina: 7 passed to 14, who sent it straight back with
        an opponent at his side. The ball was not seen at his foot, and nobody
        held it afterwards, so the open pass used to expire.
        """
        pending = self.pending
        if ball_point is None or pending is None or pending.get("contested"):
            return
        from_id = int(pending["from_id"]) if pending.get("from_id") is not None else None
        from_team = pending.get("from_team") or (self.known_teams.get(from_id) if from_id is not None else None)
        if from_team not in {"Team 0", "Team 1"}:
            return
        min_motion = FIRST_TIME_MIN_MOTION_PX if ball_point["space"] == "pixel" else FIRST_TIME_MIN_MOTION_CM
        approach = self.deflection
        if approach is not None:
            age = frame_number - approach["frame"]
            if age > FIRST_TIME_MAX_FRAMES:
                self.deflection = approach = None
            elif age >= FIRST_TIME_MIN_FRAMES:
                out_x = ball_point["x"] - approach["ball"]["x"]
                out_y = ball_point["y"] - approach["ball"]["y"]
                in_x, in_y = approach["incoming"]
                before, _space = _distance(approach["pos"], approach["ball"])
                after, _space = _distance(approach["pos"], ball_point)
                if (
                    math.hypot(out_x, out_y) >= min_motion
                    and out_x * in_x + out_y * in_y < 0
                    and before is not None
                    and after is not None
                    and after >= before + min_motion
                ):
                    self.deflection = None
                    if self._lookalike_exit(frame_number, ball_point, fps):
                        self.lookalike_touches += 1
                        return
                    self.first_time = dict(approach, seen=int(frame_number), fps=fps)
                    return
        last = self._last_ball
        if last is None or last["space"] != ball_point["space"]:
            return
        incoming = (ball_point["x"] - last["x"], ball_point["y"] - last["y"])
        if math.hypot(*incoming) < 1.0:
            return
        ranked = []
        for player in players or []:
            team = _team_name_from_player(player)
            if team not in {"Team 0", "Team 1"}:
                continue
            point = _point_from_entity(player)
            distance, space = _distance(point, ball_point)
            if distance is None or space != ball_point["space"]:
                continue
            try:
                track_id = int(player.get("track_id"))
            except (TypeError, ValueError):
                continue
            ranked.append((distance, track_id, team, point))
        if not ranked:
            return
        ranked.sort(key=lambda item: item[0])
        distance, track_id, team, point = ranked[0]
        radius = PASS_CONTEST_RADIUS_PX if ball_point["space"] == "pixel" else PASS_CONTEST_RADIUS_CM
        if track_id == from_id or team != from_team or distance > radius:
            return
        if approach is None or distance <= approach["distance"]:
            # The frame of the touch can already move the ball back out. That
            # step is the reply, not the approach: keep the approach direction,
            # from the earlier approach or from the step before this one.
            prior = self._ball_step
            if (
                approach is not None
                and approach["id"] == track_id
                and incoming[0] * approach["incoming"][0] + incoming[1] * approach["incoming"][1] < 0
            ):
                incoming = approach["incoming"]
            elif (
                prior is not None
                and math.hypot(*prior) >= 1.0
                and incoming[0] * prior[0] + incoming[1] * prior[1] < 0
            ):
                incoming = prior
            self.deflection = {
                "id": track_id,
                "team": team,
                "pos": point,
                "frame": int(frame_number),
                "ball": ball_point,
                "incoming": incoming,
                "distance": distance,
                "rival": _duel_rival(players, ball_point, track_id, from_team, distance, radius),
            }

    def _release_touch(self, frame_number, ball_point=None):
        touch = self.touch
        self.touch = None
        if touch is None or int(touch["frames"]) < TOUCH_MIN_OBSERVED:
            return
        if ball_point is not None and self._lookalike_exit(frame_number, ball_point, self._fps):
            # The ball never was at that foot: no touch, no pass origin.
            self.lookalike_touches += 1
            return
        if touch.get("team") not in {"Team 0", "Team 1"} or self.touch_origin is not None:
            return
        self.touch_origin = {
            "id": touch["id"],
            "team": touch["team"],
            "pos": touch["pos"],
            "from_frame": int(touch["first_frame"]),
            "release_frame": int(frame_number),
        }

    def _lookalike_exit(self, frame_number, ball_point, fps):
        """The ball sat still, then reappeared back where it came from too fast.

        The still run is the observed balls before this frame that stay within a
        few pixels of the last one. The exit is the step from that last still
        ball to this one, at a pace no kick gives a resting ball, and it lands
        closer to the ball seen before the still run than to the still spot.
        """
        space = ball_point["space"]
        observed = []
        for entry in self.flight_log:
            if entry["frame"] >= frame_number or entry["ball"] is None:
                continue
            point = _point_from_logged(entry["ball"])
            if point is not None and point["space"] == space:
                observed.append((int(entry["frame"]), point))
        if len(observed) < LOOKALIKE_MIN_STILL_FRAMES + 1:
            return False
        still = LOOKALIKE_STILL_PX if space == "pixel" else LOOKALIKE_STILL_CM
        last_frame, last = observed[-1]
        index = len(observed) - 2
        run = 1
        while index >= 0 and _plain_distance(observed[index][1], last) <= still:
            run += 1
            index -= 1
        if run < LOOKALIKE_MIN_STILL_FRAMES or index < 0:
            return False
        before = observed[index][1]
        gap = max(1, int(frame_number) - last_frame)
        if space == "pixel":
            limit = LOOKALIKE_EXIT_PX_PER_FRAME * gap
        else:
            limit = PASS_MAX_BALL_SPEED_CM_S * gap / max(1.0, float(fps))
        exit_step = _plain_distance(ball_point, last)
        if exit_step < limit:
            return False
        return _plain_distance(ball_point, before) < exit_step

    def _identity_handoff(self, from_id, to_id):
        """The receiver's id was born where the passer had just vanished."""
        born = None
        for entry in self.flight_log:
            if to_id in entry["players"]:
                born = entry
                break
        if born is None or born is self.flight_log[0]:
            # Already there when the log starts: not a new id.
            return False
        born_frame = int(born["frame"])
        if from_id in born["players"]:
            return False
        last_seen = None
        for entry in self.flight_log:
            if entry["frame"] >= born_frame:
                break
            if from_id in entry["players"]:
                last_seen = entry
        if last_seen is None or born_frame - int(last_seen["frame"]) > HANDOFF_MAX_GAP_FRAMES:
            return False
        before = _point_from_logged(last_seen["players"][from_id]["points"])
        after = _point_from_logged(born["players"][to_id]["points"])
        if before is None or after is None or before["space"] != after["space"]:
            return False
        radius = HANDOFF_RADIUS_PX if before["space"] == "pixel" else HANDOFF_RADIUS_CM
        return _plain_distance(before, after) <= radius

    def _claim_short_touch(self, to_id, to_team, from_id):
        """Use a released one-touch as the pass origin when the receiver is a teammate."""
        if self.touch is not None and int(self.touch["frames"]) >= TOUCH_MIN_OBSERVED:
            self._release_touch(int(self.touch["frame"]))
        origin = self.touch_origin
        if origin is None:
            return None
        self.touch_origin = None
        if int(origin["id"]) in {int(to_id), int(from_id or -1)}:
            return None
        if origin.get("team") != to_team or to_team not in {"Team 0", "Team 1"}:
            return None
        return origin

    def _resolve(self, frame_number, fps, possessor_id, possessor_team, possessor_pos, waited=False, short_touch=True):
        pending = self.pending
        self.pending = None
        if pending is None or pending.get("from_id") is None:
            return
        if pending.get("from_team") not in {"Team 0", "Team 1"}:
            # The passer held the ball before his shirt voted; it has voted since.
            known = self.known_teams.get(int(pending["from_id"]))
            if known is not None:
                pending = dict(pending)
                pending["from_team"] = known
        if possessor_team not in {"Team 0", "Team 1"} and not waited and not pending.get("contested"):
            self.waiting_for_team.append(
                {
                    "pending": dict(pending),
                    "frame": int(frame_number),
                    "fps": fps,
                    "to_id": int(possessor_id),
                    "to_pos": possessor_pos,
                }
            )
            return

        from_id = int(pending["from_id"])
        to_id = int(possessor_id)
        teams_differ = (
            pending.get("from_team") in {"Team 0", "Team 1"}
            and possessor_team in {"Team 0", "Team 1"}
            and pending.get("from_team") != possessor_team
        )
        if not teams_differ and self._identity_handoff(from_id, to_id):
            # Same player under a new number: the ball never left him.
            self.identity_handoffs += 1
            self.conductions += 1
            self.touch_origin = None
            return
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
        origin = self._claim_short_touch(to_id, possessor_team, from_id) if short_touch else None
        if origin is not None and int(origin["release_frame"]) < int(pending.get("possession_start", -1)):
            # The touch left the foot before the passer even had the ball: it
            # belongs to an older move, not to this pass.
            origin = None
        if origin is not None:
            prepared = self._prepare_pass(
                origin["id"],
                to_id,
                origin["team"],
                possessor_team,
                origin["pos"],
                possessor_pos,
                origin["from_frame"],
                origin["release_frame"],
                gained_frame,
                fps,
                wall=False,
                short_touch=True,
            )
            prepared["short_touch"] = True
            if prepared.get("_status") != "timeout":
                status = self._record_prepared(prepared, 0)
                if status == "pass":
                    self.short_touch_passes += 1
                return

        contested = bool(pending.get("contested"))
        from_team = pending.get("from_team")
        if from_team not in {"Team 0", "Team 1"}:
            # A possessor whose shirt never voted is not the origin of a pass
            # or a dispute. The next player can still take the ball.
            self.discarded_unstable_possession += 1
            return

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
        short_touch=False,
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
        if short_touch and flight["arrival_observed"]:
            # The one-touch is the flight to the receiver's feet. Possession
            # may confirm later; that wait is not the pass.
            timing_seconds = flight["flight_seconds"]
        elif flight["arrival_observed"] and (
            flight["reception_lag_seconds"] <= PASS_MAX_RECEPTION_LAG_SECONDS
            or flight["lost_at_receiver"]
            or flight["kept_by_receiver"]
        ):
            # The ball reached the receiver. If the detector then lost it at his
            # feet, the wait for possession to confirm is not the ball rolling loose.
            timing_seconds = min(gap_seconds, flight["flight_seconds"])
        verified_pixel_flight = (
            distance_cm is None
            and trajectory == "compatible"
            and flight["arrival_observed"]
            and flight["in_flight_samples"] >= PASS_EXTENDED_MIN_FLIGHT_SAMPLES
        )
        max_timing = max_flight_seconds(distance_cm, verified_pixel_flight=verified_pixel_flight)
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
        if arrival_observed and in_flight_samples * HIDDEN_FLIGHT_SEEN_RATIO < arrival_frame - release_frame - 1:
            # The ball was hidden for most of the "flight": the last time it was
            # seen at the passer's feet is not when it left them. Argentina v
            # Spain, 5.5 s to 7.5 s: Spain's 10 carried it for two seconds with
            # the ball behind his legs, then played it to the man beside him.
            # If it shows up at his feet again before the receiver has it, the
            # pass left from there.
            for entry in self._entries(release_frame + 1, arrival_frame):
                if _within_control(entry, from_id):
                    release_frame = entry["frame"]
            if release_frame >= arrival_frame:
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
        after_arrival = self._entries(arrival_frame + 1, gained_frame)
        seen_after = [entry for entry in after_arrival if entry["ball"] is not None]
        # The wait after the arrival is the detector losing the ball at the
        # receiver's feet when the ball is mostly unseen and never seen elsewhere.
        lost_at_receiver = (
            arrival_observed
            and len(after_arrival) - len(seen_after) >= len(seen_after)
            # One stray box elsewhere is not the ball leaving him.
            and sum(1 for entry in seen_after if not _within_control(entry, to_id)) < TOUCH_MIN_OBSERVED
        )

        # The ball reached him and stayed with him until possession confirmed:
        # whenever it was seen it was at his feet, or he was not detected in
        # that frame. A contested wait (an opponent nearer the ball while he is
        # in view) is not this.
        with_him = sum(1 for entry in seen_after if _within_control(entry, to_id))
        elsewhere = sum(
            1
            for entry in seen_after
            if not _within_control(entry, to_id) and to_id in entry["players"]
        )
        # A ball seen at his feet most of the wait without possession
        # confirming is a loose ball resting near him, not a reception.
        kept_by_receiver = (
            arrival_observed
            and with_him >= TOUCH_MIN_OBSERVED
            and elsewhere < TOUCH_MIN_OBSERVED
            and with_him * 2 < len(after_arrival)
        )

        return {
            "release_frame": int(release_frame),
            "arrival_frame": int(arrival_frame),
            "release_observed": release_observed,
            "arrival_observed": arrival_observed,
            "flight_seconds": max(0, arrival_frame - release_frame) / fps,
            "reception_lag_seconds": max(0, gained_frame - arrival_frame) / fps,
            "in_flight_samples": int(in_flight_samples),
            "lost_at_receiver": bool(lost_at_receiver),
            "kept_by_receiver": bool(kept_by_receiver),
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
                        _space_distance(
                            ball_point[0], ball_point[1], receiver_point[0], receiver_point[1], space
                        )
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
        # Judge the flight against where the receiver met the ball. By the
        # time possession confirms he may have carried it back past the last
        # ball seen (Argentina v Spain, 3 s: 8 played it 150 px to a man coming
        # in from the touchline, who turned back with it).
        arrival = self._entry_at(window_end)
        receiver = arrival["players"].get(to_id) if arrival is not None else None
        met_at = _point_from_logged(receiver["points"]) if receiver is not None else None
        if met_at is None or to_pos is None or met_at["space"] != to_pos.get("space"):
            met_at = to_pos
        return self._cross_frame_trajectory(window_start, window_end, from_pos, met_at)

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
        # The ball ended much closer to the receiver than it started: it went to
        # him, even if he then carried it back past the last ball seen.
        start_distance = math.hypot(to_pos["x"] - window[0][0], to_pos["y"] - window[0][1])
        end_distance = math.hypot(to_receiver_x, to_receiver_y)
        if start_distance - end_distance >= max(
            _min_motion(space), PASS_TRAJECTORY_MIN_PROGRESS * start_distance
        ):
            return "compatible"
        if (motion_x * to_receiver_x) + (motion_y * to_receiver_y) >= 0.0:
            return "compatible"
        return "incompatible"


def _plain_distance(point_a, point_b):
    return math.hypot(point_a["x"] - point_b["x"], point_a["y"] - point_b["y"])


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


def _duel_rival(players, ball_point, mate_id, mate_team, mate_distance, radius):
    """A non-teammate as close to the ball as the teammate who met it.

    Players whose shirt has not voted yet count: a duel often starts the moment
    a new track appears on top of the man with the ball.
    """
    best = None
    for player in players or []:
        try:
            track_id = int(player.get("track_id"))
        except (TypeError, ValueError):
            continue
        if track_id == int(mate_id):
            continue
        team = _team_name_from_player(player)
        if team == mate_team:
            continue
        point = _point_from_entity(player)
        distance, space = _distance(point, ball_point)
        if distance is None or space != ball_point["space"] or distance > radius:
            continue
        if distance > mate_distance + _touch_margin(space):
            continue
        if best is None or distance < best["distance"]:
            best = {"id": track_id, "team": team, "pos": point, "distance": distance}
    return best


def _contest_margin(space):
    return CONTEST_MARGIN_CM if space == "field" else CONTEST_MARGIN_PX


def _teams_contesting(players, ball_point):
    """Both teams are on the ball and the leader is not clear of the other team.

    An opponent inside the wide contest radius but a full margin behind the
    player who has the ball is not a dispute. Equal distances are.
    """
    if ball_point is None:
        return False
    radius = _contest_radius(ball_point["space"])
    nearest = {}
    for player in players or []:
        point = _point_from_entity(player)
        distance, space = _distance(point, ball_point)
        if distance is None or space != ball_point["space"] or distance > radius:
            continue
        team = _team_name_from_player(player)
        if team not in {"Team 0", "Team 1"}:
            continue
        if team not in nearest or distance < nearest[team]:
            nearest[team] = distance
    if "Team 0" not in nearest or "Team 1" not in nearest:
        return False
    leader = min(nearest.values())
    second = max(nearest.values())
    return leader + _contest_margin(ball_point["space"]) >= second


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
