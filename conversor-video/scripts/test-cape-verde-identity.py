#!/usr/bin/env python3
"""Spain v Cape Verde, 44.5 s of Spain keeping the ball: 14 passes, no turnover.

The full analysis read 17 passes and one turnover. The fixture holds three
slices of the event log of that clip, each replayed on its own.

1. 1.3 s to 10.7 s. Spain's 19 stands still by the touchline and the ball
   layer twice settles on his boot for a few frames while the real ball is
   elsewhere. Read before: 14 to 19, 19 to 6, and 19 to 12. The real moves are
   14 to 6 and 6 to 12. Then 12 gets the new id 36 while he dribbles in a duel,
   which read as a pass 12 to 36.
2. 27.3 s to 31.0 s. 84 gets the new id 88 on the ball: a pass 84 to 88.
3. 38.3 s to 44.5 s. Spain's 20 (track 36) dribbles into a crowd and comes out
   as 120: a pass 36 to 120. At 43.5 s the id 120 slides onto a Cape Verde
   defender, its shirt votes flip it to Cape Verde, and Spain's lay-off from
   the 20 to the 7 (track 20) read as a turnover.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from event_replay import replay_event_record  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "event-cape-verde-identity.json"

EXPECTED = [
    # first frame, passes, lookalike touches held back, identity handoffs
    (40, [(14, 6), (6, 12)], 3, 1),
    (820, [(53, 84), (88, 36)], 0, 1),
    (1150, [(88, 36), (120, 20)], 0, 1),
]


def _slice_record(fixture, piece):
    record = {key: value for key, value in fixture.items() if key != "slices"}
    record["frames"] = piece["frames"]
    record["frame_count"] = len(piece["frames"])
    return record


def test_cape_verde_slices():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert [piece["first_frame"] for piece in fixture["slices"]] == [row[0] for row in EXPECTED]
    for piece, (first_frame, passes, lookalikes, handoffs) in zip(fixture["slices"], EXPECTED):
        result = replay_event_record(_slice_record(fixture, piece))
        detected = result["passes"]
        pairs = [(event["from_track_id"], event["to_track_id"]) for event in detected["events"]]
        assert pairs == passes, (first_frame, pairs)
        assert detected["turnover_events"] == [], (first_frame, detected["turnover_events"])
        assert detected["lookalike_touches"] == lookalikes, (first_frame, detected["lookalike_touches"])
        assert detected["identity_handoffs"] == handoffs, (first_frame, detected["identity_handoffs"])
        # Cape Verde never had the ball: the defender under id 120 does not get it.
        assert result["possession_by_team"]["frames_team_1"] == 0, first_frame


def main():
    test_cape_verde_slices()
    print("cape verde identity tests passed")


if __name__ == "__main__":
    main()
