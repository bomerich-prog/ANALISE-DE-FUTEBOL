"""Goalkeeper by position: the last man of the line in a kit of his own.

The detector learnt goalkeepers from few examples and often calls one a
player. The team classifier then puts him in the team whose shirt is closest
in brightness: Neuer's neon green reads as Real Madrid's white. Passes to and
from him are lost and his touches become turnovers.

Two general cues decide it, and both must hold over the track's life:

* position: he is one of the two outermost players along the length of the
  image (the near-side man can stand further out in perspective) and nobody
  stands near him: the nearest player is at least 2.5 of his own heights
  away. In the broadcast side view the image x runs along the pitch, and the
  goalkeeper stays behind his line;
* kit: his shirt matches neither fitted team (or is neon, or the detector
  itself called him a goalkeeper). A winger alone on the touchline wears his
  team's shirt, so he is never taken.

His team is the Roboflow heuristic (the team whose players' centroid is
nearer), counted over every frame instead of locked by the first votes.
"""

from __future__ import annotations

import math
from collections import Counter

KEEPER_MIN_SEEN = 12
# Once a keeper is known, his new id after an occlusion needs only a few frames.
KEEPER_MIN_SEEN_AFTER_FIRST = 3
KEEPER_OUTERMOST = 2
KEEPER_ALONE_HEIGHTS = 2.5
KEEPER_MIN_LAST_SHARE = 0.5
KEEPER_MIN_KIT_MEASURES = 4
KEEPER_MIN_ODD_SHARE = 0.5
KEEPER_TEAM_MIN_VOTES = 3


class PositionKeeper:
    def __init__(self, frame_width=None):
        self.tracks = {}
        self.confirmed = set()
        self.team_votes = {}

    def update(self, players):
        """players: dicts with track_id, x and y (foot), height and kit_odd (True/False/None).

        Returns the confirmed keeper track ids among this frame's players.
        """
        usable = [
            p
            for p in players
            if int(p.get("track_id", -1)) >= 0 and p.get("x") is not None and p.get("y") is not None
        ]
        last = set()
        if len(usable) >= 2:
            ordered = sorted(usable, key=lambda p: float(p["x"]))
            ends = {int(p["track_id"]) for p in ordered[:KEEPER_OUTERMOST] + ordered[-KEEPER_OUTERMOST:]}
            for player in usable:
                if int(player["track_id"]) not in ends:
                    continue
                nearest = min(
                    math.hypot(float(player["x"]) - float(other["x"]), float(player["y"]) - float(other["y"]))
                    for other in usable
                    if other is not player
                )
                if nearest >= KEEPER_ALONE_HEIGHTS * max(1.0, float(player.get("height") or 0.0)):
                    last.add(int(player["track_id"]))
        for player in usable:
            track_id = int(player["track_id"])
            state = self.tracks.setdefault(track_id, {"seen": 0, "last": 0, "measured": 0, "odd": 0})
            state["seen"] += 1
            if track_id in last:
                state["last"] += 1
            kit_odd = player.get("kit_odd")
            if kit_odd is not None:
                state["measured"] += 1
                if kit_odd:
                    state["odd"] += 1
            if track_id not in self.confirmed and self._qualifies(state, bool(self.confirmed)):
                self.confirmed.add(track_id)
        return {int(p["track_id"]) for p in usable if int(p["track_id"]) in self.confirmed}

    @staticmethod
    def _qualifies(state, keeper_known=False):
        min_seen = KEEPER_MIN_SEEN_AFTER_FIRST if keeper_known else KEEPER_MIN_SEEN
        min_measured = min(KEEPER_MIN_KIT_MEASURES, min_seen)
        return (
            state["seen"] >= min_seen
            and state["last"] >= KEEPER_MIN_LAST_SHARE * state["seen"]
            and state["measured"] >= min_measured
            and state["odd"] >= KEEPER_MIN_ODD_SHARE * state["measured"]
        )

    def vote_team(self, track_id, team_id):
        """Record one centroid vote; return the majority team or None."""
        votes = self.team_votes.setdefault(int(track_id), Counter())
        if team_id in (0, 1):
            votes[int(team_id)] += 1
        if sum(votes.values()) < KEEPER_TEAM_MIN_VOTES:
            return None
        ranked = votes.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return int(ranked[0][0])

    def metrics(self):
        return {
            "confirmed_tracks": sorted(self.confirmed),
            "team_votes": {str(k): dict(v) for k, v in sorted(self.team_votes.items())},
        }
