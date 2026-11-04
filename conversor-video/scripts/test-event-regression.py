#!/usr/bin/env python3
"""Replay possession and passes from a recorded clip, without YOLO.

The fixture is the input the analysis loop already gave to the possession
and pass engines. October's pass count is not the expected value.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server/services/analysis/yolo"))

from event_replay import (  # noqa: E402
    explain_ball_after,
    frame_by_number,
    load_event_record,
    possessor_identity_report,
    replay_event_record,
    team_changes,
)
from pass_detector import PASS_MAX_GAP_SECONDS  # noqa: E402


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "event-regression-81c0fcfb.json"
CLIP_NAME = "81c0fcfb-e357-44b5-9678-21bf85c90242.mp4"
FRAME_47_X = 730.8
FRAME_47_Y = 232.1
FRAME_47_CONFIDENCE = 0.603


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def player(track_id, team_id, x, y):
    return [track_id, team_id, x, y, None, None]


def ball(x, y, source="observed"):
    return {"source": source, "x": x, "y": y, "field_x": None, "field_y": None}


def test_replay_confirms_only_after_five_observed_frames():
    frames = []
    for frame in range(1, 5):
        frames.append(
            {
                "frame": frame,
                "ball_state": "observed",
                "published_ball": {"x": 100.0, "y": 200.0, "confidence": 0.8},
                "refused_centers": [],
                "players": [player(7, 0, 100.0, 210.0)],
                "ball": ball(100.0, 200.0),
                "possessor": None,
                "team": None,
            }
        )
    frames.append(
        {
            "frame": 5,
            "ball_state": "observed",
            "published_ball": {"x": 100.0, "y": 200.0, "confidence": 0.8},
            "refused_centers": [],
            "players": [player(7, 0, 100.0, 210.0)],
            "ball": ball(100.0, 200.0),
            "possessor": 7,
            "team": "Team 0",
        }
    )
    replay = replay_event_record({"fps": 30.0, "frames": frames})
    assert_equal(replay["timeline"][3]["possessor"], None, "four observed frames do not confirm")
    assert_equal(replay["timeline"][4]["possessor"], 7, "the fifth observed frame confirms")
    assert_equal(replay["timeline"][4]["team"], "Team 0", "team follows the player")


def test_missing_field_distance_keeps_the_one_second_window():
    frames = []
    for frame in range(1, 6):
        frames.append(
            {
                "frame": frame,
                "ball_state": "observed",
                "published_ball": {"x": 100.0, "y": 200.0, "confidence": 0.8},
                "refused_centers": [],
                "players": [player(15, 0, 100.0, 220.0), player(11, 0, 400.0, 280.0)],
                "ball": ball(100.0, 200.0),
                "possessor": None,
                "team": None,
            }
        )
    for frame in range(6, 70):
        frames.append(
            {
                "frame": frame,
                "ball_state": "lost",
                "published_ball": None,
                "refused_centers": [],
                "players": [player(15, 0, 100.0, 220.0), player(11, 0, 400.0, 280.0)],
                "ball": None,
                "possessor": None,
                "team": None,
            }
        )
    for frame in range(70, 75):
        frames.append(
            {
                "frame": frame,
                "ball_state": "recovered",
                "published_ball": {"x": 400.0, "y": 260.0, "confidence": 0.7},
                "refused_centers": [],
                "players": [player(15, 0, 100.0, 220.0), player(11, 0, 400.0, 280.0)],
                "ball": ball(400.0, 260.0),
                "possessor": None,
                "team": None,
            }
        )
    replay = replay_event_record({"fps": 30.0, "frames": frames})
    rejected = replay["passes"]["rejected_events"]
    assert_true(rejected, "a late receiver without field distance is rejected")
    assert_equal(rejected[0]["reason"], "timeout", "the rejection reason is the clock")
    assert_equal(rejected[0].get("distance_cm"), None, "no field distance was available")
    assert_equal(
        float(rejected[0]["max_timing_seconds"]),
        float(PASS_MAX_GAP_SECONDS),
        "the window stays at one second",
    )
    assert_equal(replay["passes"]["total_passes"], 0, "the late pass is not counted")


def _event_key(event):
    return {
        "type": event.get("type", "pass"),
        "from_track_id": event.get("from_track_id"),
        "to_track_id": event.get("to_track_id"),
        "team": event.get("team"),
        "frame": event.get("frame"),
        "reason": event.get("reason"),
        "release_frame": event.get("release_frame"),
        "arrival_frame": event.get("arrival_frame"),
        "max_timing_seconds": event.get("max_timing_seconds"),
    }


def test_fixture_replays_the_recorded_clip():
    record = load_event_record(FIXTURE)
    assert_equal(record.get("clip_name"), CLIP_NAME, "fixture clip")
    assert_equal(record.get("frame_count"), 401, "401 frames")
    assert_true(record.get("homography_feeds_decisions") is False, "homography stays out of decisions")
    replay = replay_event_record(record)
    mismatches = []
    for recorded, replayed in zip(record["frames"], replay["timeline"]):
        if recorded.get("possessor") != replayed["possessor"] or recorded.get("team") != replayed["team"]:
            mismatches.append(
                {
                    "frame": recorded["frame"],
                    "recorded": [recorded.get("possessor"), recorded.get("team")],
                    "replay": [replayed["possessor"], replayed["team"]],
                }
            )
    assert_true(not mismatches, f"possessor replay diverged on {mismatches[:5]}")
    recorded_passes = record["passes"]
    replay_passes = replay["passes"]
    for key in ("total_passes", "turnovers", "disputes", "discarded_timeout"):
        assert_equal(replay_passes[key], recorded_passes[key], key)
    assert_equal(
        [_event_key(event) for event in replay_passes["events"]],
        [_event_key(event) for event in recorded_passes["events"]],
        "confirmed passes",
    )
    assert_equal(
        [_event_key(event) for event in replay_passes["rejected_events"]],
        [_event_key(event) for event in recorded_passes["rejected_events"]],
        "rejected passes",
    )
    assert_equal(
        [_event_key(event) for event in replay_passes["turnover_events"]],
        [_event_key(event) for event in recorded_passes["turnover_events"]],
        "turnovers",
    )
    assert_equal(
        [_event_key(event) for event in replay_passes["dispute_events"]],
        [_event_key(event) for event in recorded_passes["dispute_events"]],
        "disputes",
    )
    assert_equal(
        replay["possession_by_team"]["percent_team_0"],
        record["possession_by_team"]["percent_team_0"],
        "team 0 possession percent",
    )
    assert_equal(
        replay["possession_by_team"]["percent_team_1"],
        record["possession_by_team"]["percent_team_1"],
        "team 1 possession percent",
    )
    assert_equal(
        replay["possession_by_team"]["percent_without_possessor"],
        record["possession_by_team"]["percent_without_possessor"],
        "open possession percent",
    )
    for event in replay_passes["rejected_events"]:
        if event.get("distance_cm") is None:
            assert_equal(
                float(event["max_timing_seconds"]),
                1.0,
                f"frame {event.get('frame')} keeps the one second window",
            )
    identity = possessor_identity_report(record)
    assert_true(
        not identity["team_mismatches"],
        f"possessor team mismatch: {identity['team_mismatches'][:5]}",
    )
    return record, replay, identity


def test_grass_interval_publishes_no_ball(record):
    for frame_number in range(32, 47):
        frame = frame_by_number(record, frame_number)
        assert_true(frame is not None, f"frame {frame_number} is in the log")
        assert_true(
            frame.get("published_ball") is None,
            f"frame {frame_number} published {frame.get('published_ball')}",
        )


def test_frame_47_published_ball(record):
    frame = frame_by_number(record, 47)
    assert_true(frame is not None, "frame 47 is in the log")
    published = frame.get("published_ball")
    assert_true(published is not None, "frame 47 has no published ball in this recording")
    assert_true(abs(float(published["x"]) - FRAME_47_X) <= 1.0, f"x was {published['x']}")
    assert_true(abs(float(published["y"]) - FRAME_47_Y) <= 1.0, f"y was {published['y']}")
    assert_true(
        abs(float(published["confidence"]) - FRAME_47_CONFIDENCE) <= 0.01,
        f"confidence was {published['confidence']}",
    )


def test_frame_103_is_reported_from_the_log(record):
    frame = frame_by_number(record, 103)
    assert_true(frame is not None, "frame 103 is in the log")
    return {
        "frame": 103,
        "ball_state": frame.get("ball_state"),
        "published_ball": frame.get("published_ball"),
        "possessor": frame.get("possessor"),
        "team": frame.get("team"),
        "source": (frame.get("ball") or {}).get("source"),
    }


def main():
    test_replay_confirms_only_after_five_observed_frames()
    test_missing_field_distance_keeps_the_one_second_window()
    if not FIXTURE.exists():
        raise SystemExit(
            f"missing fixture {FIXTURE}. Record it with run_yolo_analysis.py --event-record."
        )
    record, replay, identity = test_fixture_replays_the_recorded_clip()
    test_grass_interval_publishes_no_ball(record)
    test_frame_47_published_ball(record)
    frame_103 = test_frame_103_is_reported_from_the_log(record)
    gap = explain_ball_after(record, 47)
    changes = team_changes(record)
    summary = {
        "clip": record.get("clip_name"),
        "frames": record.get("frame_count"),
        "passes": replay["passes"]["total_passes"],
        "turnovers": replay["passes"]["turnovers"],
        "disputes": replay["passes"]["disputes"],
        "rejected": [
            {
                "from": event.get("from_track_id"),
                "to": event.get("to_track_id"),
                "reason": event.get("reason"),
                "timing_seconds": event.get("timing_seconds"),
                "max_timing_seconds": event.get("max_timing_seconds"),
                "release_frame": event.get("release_frame"),
                "arrival_frame": event.get("arrival_frame"),
                "distance_cm": event.get("distance_cm"),
            }
            for event in replay["passes"]["rejected_events"]
        ],
        "possession_percent": {
            "team_0": replay["possession_by_team"]["percent_team_0"],
            "team_1": replay["possession_by_team"]["percent_team_1"],
            "without": replay["possession_by_team"]["percent_without_possessor"],
        },
        "after_frame_47": {
            "published_run": gap["published_run"],
            "observed_run_including_frame": gap["observed_run_including_frame"],
            "first_missing_frame": gap["first_missing_frame"],
            "first_missing_state": gap["first_missing_state"],
            "first_missing_source": gap["first_missing_source"],
            "refused_on_first_missing": gap["refused_on_first_missing"],
            "next_published_frame": gap["next_published_frame"],
            "next_possessor_frame": gap["next_possessor_frame"],
            "next_possessor": gap["next_possessor"],
            "next_possessor_team": gap["next_possessor_team"],
        },
        "frame_103": frame_103,
        "team_id_changes": len(changes),
        "possessor_held_without_box": len(identity["held_without_box"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("event regression tests passed")


if __name__ == "__main__":
    main()
