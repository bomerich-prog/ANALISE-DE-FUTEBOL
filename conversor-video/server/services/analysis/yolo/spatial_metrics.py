"""Width, length, compactness, defensive line and speed from map coordinates.

Nothing here chooses a possessor or a pass. A sample counts only when the
visible-third map published map_x and map_y for that detection.
"""

from __future__ import annotations

import math

from ball_heatmap import BallHeatmapAccumulator


PLAYER_SPEED_MAX_M_S = 12.0
BALL_SPEED_MAX_M_S = 40.0


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


def _team_name(value):
    if value in {"Team 0", "Team 1"}:
        return value
    number = _finite(value)
    if number == 0:
        return "Team 0"
    if number == 1:
        return "Team 1"
    return None


def _hull(points):
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) < 3:
        return []

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def hull_area_m2(points):
    hull = _hull(points)
    if len(hull) < 3:
        return None
    total = 0.0
    for index, (x, y) in enumerate(hull):
        next_x, next_y = hull[(index + 1) % len(hull)]
        total += x * next_y - next_x * y
    return abs(total) / 2.0 / 10000.0


def _round(value):
    if value is None:
        return None
    return round(float(value), 2)


class SpatialMetrics:
    def __init__(self, fps=30.0, goal_at_high_x=None):
        try:
            parsed = float(fps)
        except (TypeError, ValueError):
            parsed = 30.0
        self.fps = parsed if parsed > 0 else 30.0
        self.goal_at_high_x = goal_at_high_x
        self.player_heatmap = BallHeatmapAccumulator()
        self._sums = {
            "Team 0": self._blank_team(),
            "Team 1": self._blank_team(),
        }
        self._speeds = {"Team 0": [], "Team 1": [], "ball": []}
        self._last = {}
        self._last_ball = None
        self.frames = 0

    def observe(self, frame_number, players, ball=None, generation=None, goal_at_high_x=None):
        self.frames += 1
        if goal_at_high_x is not None:
            self.goal_at_high_x = bool(goal_at_high_x)
        self.player_heatmap.frames_total += 1
        by_team = {"Team 0": [], "Team 1": []}
        seen = set()
        for player in players or []:
            track_id = player.get("track_id")
            team = _team_name(player.get("team_id"))
            x = _finite(player.get("map_x"))
            y = _finite(player.get("map_y"))
            if team is None or x is None or y is None:
                continue
            try:
                track_key = int(track_id)
            except (TypeError, ValueError):
                continue
            if track_key < 0 or track_key in seen:
                continue
            seen.add(track_key)
            by_team[team].append((track_key, x, y))
            self.player_heatmap.count_point(x, y)
        if seen:
            self.player_heatmap.frames_observed_ball += 1
        else:
            self.player_heatmap.frames_without_field_point += 1

        for team, samples in by_team.items():
            self._note_shape(team, samples)
            self._note_speed(team, samples, frame_number, generation)

        ball_x = _finite(ball.get("map_x")) if isinstance(ball, dict) else None
        ball_y = _finite(ball.get("map_y")) if isinstance(ball, dict) else None
        self._note_ball_speed(ball_x, ball_y, frame_number, generation)
        return self.as_dict()

    def as_dict(self):
        teams = {name: self._team_report(name) for name in ("Team 0", "Team 1")}
        defending = self._defending_team(teams)
        defensive_line = None if defending is None else teams[defending]["line_from_visible_goal_m"]
        ball_speeds = self._speeds["ball"]
        return {
            "feeds_events": False,
            "frames": self.frames,
            "goal_end": None if self.goal_at_high_x is None else ("high_x" if self.goal_at_high_x else "low_x"),
            "visible_third_m": 35.0,
            "teams": teams,
            "defending_visible_goal": defending,
            "defensive_line_m": defensive_line,
            "ball_speed_m_s": None if not ball_speeds else _round(sum(ball_speeds) / len(ball_speeds)),
            "ball_speed_samples": len(ball_speeds),
            "player_heatmap": self.player_heatmap.as_dict(),
        }

    def _blank_team(self):
        return {
            "width": [],
            "length": [],
            "compactness": [],
            "line": [],
        }

    def _note_shape(self, team, samples):
        if len(samples) < 2:
            return
        xs = [sample[1] for sample in samples]
        ys = [sample[2] for sample in samples]
        row = self._sums[team]
        row["width"].append((max(ys) - min(ys)) / 100.0)
        row["length"].append((max(xs) - min(xs)) / 100.0)
        area = hull_area_m2([(x, y) for _track, x, y in samples])
        if area is not None:
            row["compactness"].append(area)
        if self.goal_at_high_x is None:
            return
        if self.goal_at_high_x:
            depths = [(12000.0 - x) / 100.0 for x in xs]
        else:
            depths = [x / 100.0 for x in xs]
        row["line"].append(min(depths))

    def _note_speed(self, team, samples, frame_number, generation):
        if generation is None:
            for track_id, _x, _y in samples:
                self._last.pop((team, track_id), None)
            return
        for track_id, x, y in samples:
            key = (team, track_id)
            previous = self._last.get(key)
            self._last[key] = (frame_number, x, y, generation)
            if previous is None:
                continue
            prev_frame, prev_x, prev_y, prev_generation = previous
            if prev_generation != generation or frame_number - prev_frame != 1:
                continue
            speed = math.hypot(x - prev_x, y - prev_y) / 100.0 * self.fps
            if 0.0 <= speed <= PLAYER_SPEED_MAX_M_S:
                self._speeds[team].append(speed)

    def _note_ball_speed(self, x, y, frame_number, generation):
        if x is None or y is None or generation is None:
            self._last_ball = None
            return
        previous = self._last_ball
        self._last_ball = (frame_number, x, y, generation)
        if previous is None:
            return
        prev_frame, prev_x, prev_y, prev_generation = previous
        if prev_generation != generation or frame_number - prev_frame != 1:
            return
        speed = math.hypot(x - prev_x, y - prev_y) / 100.0 * self.fps
        if 0.0 <= speed <= BALL_SPEED_MAX_M_S:
            self._speeds["ball"].append(speed)

    def _team_report(self, name):
        row = self._sums[name]
        speeds = self._speeds[name]
        return {
            "width_m": _mean(row["width"]),
            "length_m": _mean(row["length"]),
            "compactness_m2": _mean(row["compactness"]),
            "line_from_visible_goal_m": _mean(row["line"]),
            "speed_m_s": None if not speeds else _round(sum(speeds) / len(speeds)),
            "speed_samples": len(speeds),
            "shape_samples": len(row["width"]),
        }

    def _defending_team(self, teams):
        ranked = []
        for name, report in teams.items():
            line = report["line_from_visible_goal_m"]
            if line is not None:
                ranked.append((line, name))
        if not ranked:
            return None
        ranked.sort()
        return ranked[0][1]


def _mean(values):
    if not values:
        return None
    return _round(sum(values) / len(values))
