#!/usr/bin/env python3
"""Gremio x Cruzeiro: two slices of the event log.

1. 4.3 s to 7.7 s (clip 926a375f). Gremio's 8 passes to his 21 and the ball
   rolls 21 px in front of Cruzeiro's 10. Possession gave the ball to 10, then
   21 took it: read as two turnovers (8 to 10, 10 to 21). The ball kept its
   line and pace across 10, so it is one pass, 8 to 21. Then 21 loses it to 23.
   Track 8 is Gremio here: the old log locked him as Cruzeiro after one turn
   of his striped shirt (see the team classifier stability test).
2. 8.3 s to 11.5 s (clip ee338f4f). Cruzeiro's 7 and Gremio's 9 fight for the
   ball, which is first seen at 9's feet already moving out to Cruzeiro's 12.
   The ball never came to 9 from outside, so the two turnovers stay.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from event_replay import replay_event_record  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "event-gremio-roll-past.json"

EXPECTED = [
    # first frame, passes (from, to, rolled past), turnovers
    (130, [(8, 21, 10)], [(21, 23)]),
    (250, [], [(7, 9), (9, 12)]),
]


def _slice_record(fixture, piece):
    record = {key: value for key, value in fixture.items() if key != "slices"}
    record["clip_name"] = piece["clip_name"]
    record["fps"] = piece["fps"]
    record["frames"] = piece["frames"]
    record["frame_count"] = len(piece["frames"])
    return record


def test_gremio_slices():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert [piece["first_frame"] for piece in fixture["slices"]] == [row[0] for row in EXPECTED]
    for piece, (first_frame, passes, turnovers) in zip(fixture["slices"], EXPECTED):
        detected = replay_event_record(_slice_record(fixture, piece))["passes"]
        found = [
            (event["from_track_id"], event["to_track_id"], event.get("rolled_past_track_id"))
            for event in detected["events"]
        ]
        lost = [(event["from_track_id"], event["to_track_id"]) for event in detected["turnover_events"]]
        assert found == passes, (first_frame, found)
        assert lost == turnovers, (first_frame, lost)
        assert detected["roll_past_turnovers"] == (1 if passes else 0), (first_frame, detected["roll_past_turnovers"])


def main():
    test_gremio_slices()
    print("gremio roll past tests passed")


if __name__ == "__main__":
    main()
