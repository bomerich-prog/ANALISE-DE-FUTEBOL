"""Count frames already confirmed by PossessionEngine, split by team.

This does not decide who has the ball. It tallies the team already on
each snapshot. A gap with no possessor stays empty until it closes:

- a confirmed pass of the team that opened the gap moves the whole gap;
- the same team coming back after a long gap, with no pass, moves only
  the first and last GAP_EDGE_SECONDS;
- a different team, or a clip that ends while the gap is open, moves
  nothing.

The middle of a long gap is never given to a team.
"""

from __future__ import annotations


# A same-team hole longer than this keeps only its edges. A pass still
# closes the whole flight, whatever its length.
LONG_GAP_SECONDS = 1.0
GAP_EDGE_SECONDS = 0.5


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
        self.frames_reassigned_team_0 = 0
        self.frames_reassigned_team_1 = 0
        self._last_team = None
        self._open_team = None
        self._open_frames = 0
        self._seen_pass_events = 0

    def observe(self, team, pass_events=None):
        confirmed_pass_team = self._take_new_pass_team(pass_events)
        if team == "Team 0":
            self.frames_team_0 += 1
        elif team == "Team 1":
            self.frames_team_1 += 1
        else:
            self.frames_without_possessor += 1
            if self._last_team in {"Team 0", "Team 1"}:
                if self._open_team is None:
                    self._open_team = self._last_team
                    self._open_frames = 0
                self._open_frames += 1
            return self.as_dict()

        self._close_open_gap(team, confirmed_pass_team)
        self._open_team = None
        self._open_frames = 0
        self._last_team = team
        return self.as_dict()

    def _close_open_gap(self, closing_team, confirmed_pass_team):
        if self._open_frames <= 0 or self._open_team not in {"Team 0", "Team 1"}:
            return
        if confirmed_pass_team == self._open_team:
            self._reassign(self._open_team, self._open_frames)
            return
        if closing_team != self._open_team:
            return
        self._reassign(self._open_team, self._edge_frames(self._open_frames))

    def _edge_frames(self, frames):
        if frames <= 0:
            return 0
        if (frames / self.fps) <= LONG_GAP_SECONDS:
            return 0
        edge = max(0, int(round(self.fps * GAP_EDGE_SECONDS)))
        if edge <= 0 or edge * 2 >= frames:
            return 0 if edge <= 0 else frames
        return edge * 2

    def _take_new_pass_team(self, pass_events):
        events = list(pass_events or [])
        fresh = events[self._seen_pass_events :]
        self._seen_pass_events = len(events)
        teams = []
        for event in fresh:
            if not isinstance(event, dict) or event.get("type", "pass") != "pass":
                continue
            team = event.get("team")
            if team in {"Team 0", "Team 1"}:
                teams.append(team)
        if not teams or any(team != teams[0] for team in teams):
            return None
        return teams[0]

    def _reassign(self, team, frames):
        frames = min(int(frames), int(self.frames_without_possessor))
        if frames <= 0:
            return
        self.frames_without_possessor -= frames
        if team == "Team 0":
            self.frames_team_0 += frames
            self.frames_reassigned_team_0 += frames
        else:
            self.frames_team_1 += frames
            self.frames_reassigned_team_1 += frames

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
            "frames_reassigned_team_0": int(self.frames_reassigned_team_0),
            "frames_reassigned_team_1": int(self.frames_reassigned_team_1),
        }
