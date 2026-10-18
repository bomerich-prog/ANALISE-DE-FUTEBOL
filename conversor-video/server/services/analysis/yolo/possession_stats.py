"""Count frames already confirmed by PossessionEngine, split by team.

This does not decide possession. It only tallies the team name on each
snapshot the engine already returned.
"""

from __future__ import annotations


class PossessionTeamAccumulator:
    def __init__(self, fps):
        try:
            parsed = float(fps)
        except (TypeError, ValueError):
            parsed = 0.0
        self.fps = parsed if parsed > 0 else 30.0
        self.frames_team_0 = 0
        self.frames_team_1 = 0
        self.frames_without_possessor = 0

    def observe(self, team):
        if team == "Team 0":
            self.frames_team_0 += 1
        elif team == "Team 1":
            self.frames_team_1 += 1
        else:
            self.frames_without_possessor += 1
        return self.as_dict()

    def as_dict(self):
        frames_total = (
            self.frames_team_0 + self.frames_team_1 + self.frames_without_possessor
        )

        def percent(frames):
            if frames_total <= 0:
                return 0.0
            return round((100.0 * frames) / frames_total, 1)

        def seconds(frames):
            return round(frames / self.fps, 3)

        return {
            "frames_total": int(frames_total),
            "frames_team_0": int(self.frames_team_0),
            "frames_team_1": int(self.frames_team_1),
            "frames_without_possessor": int(self.frames_without_possessor),
            "seconds_team_0": seconds(self.frames_team_0),
            "seconds_team_1": seconds(self.frames_team_1),
            "seconds_without_possessor": seconds(self.frames_without_possessor),
            "percent_team_0": percent(self.frames_team_0),
            "percent_team_1": percent(self.frames_team_1),
            "percent_without_possessor": percent(self.frames_without_possessor),
        }
