#!/usr/bin/env python3
"""Argentina v Spain (demonstration clip): two slices of the event log.

1. 0 s to 10 s. Spain's 8 plays it 150 px to a man coming in from the
   touchline, who turns back with it; possession only confirms after he has
   carried it back past the last ball seen. Then Spain's 10 (track 27) carries
   the ball behind his legs for two seconds and plays it to the man beside
   him (39). Read before: only 39 to 27 and 39 to 54.
2. 26 s to 33.3 s. Two Spanish players get a new track id while dribbling
   (118 to 119, 122 to 128), which read as two passes.

Expected: 8 to 39, 39 to 27, 27 to 39, 39 to 54; then 119 to 122 and
128 to 117, with the two id changes read as the same player keeping the ball.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from event_replay import replay_event_record  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "event-argentina-hidden-ball.json"

EXPECTED = [
    # first frame, passes, identity handoffs
    (1, [(8, 39), (39, 27), (27, 39), (39, 54)], 0),
    (780, [(119, 122), (128, 117)], 2),
]


def _slice_record(fixture, piece):
    record = {key: value for key, value in fixture.items() if key != "slices"}
    record["frames"] = piece["frames"]
    record["frame_count"] = len(piece["frames"])
    return record


def test_argentina_slices():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert [piece["first_frame"] for piece in fixture["slices"]] == [row[0] for row in EXPECTED]
    for piece, (first_frame, passes, handoffs) in zip(fixture["slices"], EXPECTED):
        detected = replay_event_record(_slice_record(fixture, piece))["passes"]
        pairs = [(event["from_track_id"], event["to_track_id"]) for event in detected["events"]]
        assert pairs == passes, (first_frame, pairs)
        assert detected["turnover_events"] == [], (first_frame, detected["turnover_events"])
        assert detected["identity_handoffs"] == handoffs, (first_frame, detected["identity_handoffs"])


def main():
    test_argentina_slices()
    print("argentina hidden ball tests passed")


if __name__ == "__main__":
    main()
