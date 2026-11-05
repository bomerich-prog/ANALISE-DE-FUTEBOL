#!/usr/bin/env python3
"""Identity facts that can be read from the event log, plus the record shape.

The log has no shirt label. Track 31 changes team after a gap, so frames 93
and 100 of that track still need a person to look at the shirt. This file
does not invent that label.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from player_identity_record import record_player  # noqa: E402


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "event-regression-81c0fcfb.json"
# The event log shows track 31 as team 0 at frame 93 and team 1 at frame 100,
# with the player missing in between. The pixels of the shirt are not stored.
TRACK_31_ANNOTATION_FRAMES = (93, 100)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def appearances(record):
    tracks = {}
    for frame in record["frames"]:
        seen = set()
        for row in frame["players"]:
            track_id = row[0]
            if track_id in seen:
                continue
            seen.add(track_id)
            tracks.setdefault(track_id, []).append(
                {"frame": frame["frame"], "team": row[1], "x": row[2], "y": row[3]}
            )
    return tracks


def flips(samples):
    changes = []
    previous = None
    for sample in samples:
        if previous is not None and sample["team"] != previous["team"]:
            changes.append((previous, sample))
        previous = sample
    return changes


def test_record_keeps_raw_id_and_vote_apart_from_the_published_id():
    row = record_player(
        19,
        {"track_id": 4, "team_id": 0, "box": [10, 20, 30, 80]},
        class_id=2,
        team_vote=None,
        team_locked=True,
        raw_class_id=2,
        hue=110,
        accepted=True,
    )
    assert_equal(row["raw_track_id"], 19, "ByteTrack id")
    assert_equal(row["track_id"], 4, "id the rest of the pipeline sees")
    assert_equal(row["team_vote"], None, "a cached frame has no new vote")
    assert_equal(row["team_locked"], True, "lock flag")
    assert_equal(row["hue"], 110.0, "torso hue stored for a later replay")


def _box(foot_x, foot_y=400, width=30, height=70):
    return [foot_x - width / 2.0, foot_y - height, foot_x + width / 2.0, foot_y]


def test_a_record_replays_a_gap_and_a_flickering_vote_without_yolo():
    """The sidecar is the input. No frame and no detector."""
    from player_track_stitch import PlayerTrackStitch
    from roboflow_team import TeamTrackLock

    frames = []
    for frame, foot_x in enumerate((100, 108, 116, 124, 132), start=1):
        frames.append(
            {
                "frame": frame,
                "players": [
                    {
                        "raw_track_id": 4,
                        "track_id": 4,
                        "raw_class_id": 2,
                        "box": _box(foot_x),
                        "hue": 110.0,
                        "team_vote": 0,
                        "accepted": True,
                    }
                ],
            }
        )
    frames.append(
        {
            "frame": 10,
            "players": [
                {
                    "raw_track_id": 19,
                    "track_id": 4,
                    "raw_class_id": 2,
                    "box": _box(172),
                    "hue": 110.0,
                    "team_vote": 1,
                    "accepted": True,
                },
                {
                    "raw_track_id": 49,
                    "track_id": 49,
                    "raw_class_id": 2,
                    "box": _box(900),
                    "hue": 20.0,
                    "team_vote": 0,
                    "accepted": True,
                },
            ],
        }
    )
    for offset, vote in enumerate((1, 0, 1, 0, 1)):
        frames.append(
            {
                "frame": 11 + offset,
                "players": [
                    {
                        "raw_track_id": 4,
                        "track_id": 4,
                        "raw_class_id": 2,
                        "box": _box(180 + offset * 8),
                        "hue": 110.0,
                        "team_vote": 0,
                        "accepted": True,
                    },
                    {
                        "raw_track_id": 49,
                        "track_id": 49,
                        "raw_class_id": 2,
                        "box": _box(900),
                        "hue": 20.0,
                        "team_vote": vote,
                        "accepted": True,
                    },
                ],
            }
        )

    stitch = PlayerTrackStitch()
    lock = TeamTrackLock()
    published_teams = []
    for frame in frames:
        players = [player for player in frame["players"] if player["accepted"]]
        rewritten = list(
            stitch.apply(
                frame["frame"],
                [player["raw_track_id"] for player in players],
                [player["box"] for player in players],
                hues=[player["hue"] for player in players],
                classes=[player["raw_class_id"] for player in players],
            )
        )
        assert_equal(
            rewritten,
            [player["track_id"] for player in players],
            f"stitched ids at frame {frame['frame']}",
        )
        teams = []
        for player, track_id in zip(players, rewritten):
            teams.append(
                lock.assign(
                    track_id,
                    player["team_vote"],
                    frame["frame"],
                    lock.is_locked(track_id),
                )
            )
        published_teams.append(teams)

    assert_equal(published_teams[5][0], 0, "the returning player keeps the first shirt")
    assert_equal(published_teams[-1][1], 0, "track 49 stays on the first clear vote")
    assert_equal(lock.metrics()["unlocks"], 0, "a locked shirt does not open")


def test_recorded_team_flips_match_the_geometry_in_the_log():
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tracks = appearances(record)
    edge = flips(tracks[49])
    assert_equal(len(edge), 6, "track 49 flips six times in this log")
    for previous, current in edge:
        assert_equal(current["frame"] - previous["frame"], 1, "track 49 flips on the next frame")
        step = math.hypot(current["x"] - previous["x"], current["y"] - previous["y"])
        assert_true(step < 5.0, f"track 49 barely moves during a flip ({step:.2f} px)")

    late = flips(tracks[2])
    assert_equal(len(late), 1, "track 2 flips once")
    previous, current = late[0]
    assert_equal(current["frame"], 347, "the late flip frame")
    assert_equal(previous["frame"], 346, "the frame before it")
    assert_true(len(tracks[2]) > 300, "the same id was already present for most of the clip")

    gap = flips(tracks[31])
    assert_equal(len(gap), 1, "track 31 flips once")
    previous, current = gap[0]
    assert_equal((previous["frame"], current["frame"]), TRACK_31_ANNOTATION_FRAMES, "frames that need a shirt label")
    missing = current["frame"] - previous["frame"] - 1
    assert_equal(missing, 6, "six frames without this id")
    assert_true(
        math.hypot(current["x"] - previous["x"], current["y"] - previous["y"]) > 20.0,
        "the foot moved, so the shirt is not decided by the log",
    )


def test_possessor_teams_do_not_flip_in_the_event_log():
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tracks = appearances(record)
    possessors = {frame["possessor"] for frame in record["frames"] if frame.get("possessor") is not None}
    for track_id in sorted(possessors):
        teams = {sample["team"] for sample in tracks[track_id] if sample["team"] is not None}
        assert_equal(len(teams), 1, f"possessor {track_id} team")


if __name__ == "__main__":
    test_record_keeps_raw_id_and_vote_apart_from_the_published_id()
    test_a_record_replays_a_gap_and_a_flickering_vote_without_yolo()
    test_recorded_team_flips_match_the_geometry_in_the_log()
    test_possessor_teams_do_not_flip_in_the_event_log()
    print("player identity tests passed")
    print(
        "manual annotation still required: track 31 frames "
        + ", ".join(str(frame) for frame in TRACK_31_ANNOTATION_FRAMES)
    )
