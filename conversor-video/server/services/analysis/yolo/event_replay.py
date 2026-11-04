"""Replay possession and passes from a recorded frame log.

The log is the input the live loop already handed to PossessionEngine and
PassDetector: accepted players, the ball state, and whether a ball box was
published. Replay does not run YOLO and does not decide a new ball.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pass_detector import PassDetector
from possession import TEAM_NAME_BY_ID, PossessionEngine
from possession_stats import PossessionTeamAccumulator


RECORD_VERSION = 1


def _json_default(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return float(value) if hasattr(value, "item") else str(value)


def _number(value, digits=4):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def _optional_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def published_ball_record(detection):
    """Center of a ball box that was actually published. None keeps the frame empty."""
    if not detection:
        return None
    box = detection.get("box") or detection.get("bounding_box")
    if box and len(box) >= 4:
        center_x = (float(box[0]) + float(box[2])) / 2.0
        center_y = (float(box[1]) + float(box[3])) / 2.0
    else:
        center_x = detection.get("center_x")
        center_y = detection.get("center_y")
    if center_x is None or center_y is None:
        return None
    return {
        "x": _number(center_x),
        "y": _number(center_y),
        "confidence": _number(detection.get("confidence")),
    }


def record_event_frame(
    frame_number,
    ball_state,
    published_detection,
    players,
    ball,
    possessor_id,
    possessor_team,
    refused_centers=None,
):
    """Copy one frame of the live decision inputs. This does not change them."""
    slim_players = []
    for player in players or []:
        slim_players.append(
            [
                _optional_int(player.get("track_id")),
                _optional_int(player.get("team_id")),
                _number(player.get("x")),
                _number(player.get("y")),
                _number(player.get("field_x")),
                _number(player.get("field_y")),
            ]
        )
    slim_ball = None
    if ball:
        slim_ball = {
            "source": ball.get("source"),
            "x": _number(ball.get("x")),
            "y": _number(ball.get("y")),
            "field_x": _number(ball.get("field_x")),
            "field_y": _number(ball.get("field_y")),
        }
    refused = []
    for item in refused_centers or []:
        if len(item) < 3 or int(item[0]) != int(frame_number):
            continue
        refused.append([_number(item[1], 1), _number(item[2], 1)])
    return {
        "frame": int(frame_number),
        "ball_state": ball_state,
        "published_ball": published_ball_record(published_detection),
        "refused_centers": refused,
        "players": slim_players,
        "ball": slim_ball,
        "possessor": _optional_int(possessor_id),
        "team": possessor_team,
    }


def _player_from_row(row):
    return {
        "track_id": row[0],
        "team_id": row[1],
        "x": row[2],
        "y": row[3],
        "field_x": row[4] if len(row) > 4 else None,
        "field_y": row[5] if len(row) > 5 else None,
    }


def _ball_from_record(ball):
    if not ball:
        return None
    return {
        "source": ball.get("source"),
        "x": ball.get("x"),
        "y": ball.get("y"),
        "field_x": ball.get("field_x"),
        "field_y": ball.get("field_y"),
    }


def replay_event_record(record):
    """Run the possession and pass engines on a saved log."""
    fps = float(record.get("fps") or 30.0)
    possession = PossessionEngine()
    passes = PassDetector()
    teams = PossessionTeamAccumulator(fps)
    timeline = []
    for frame in record.get("frames") or []:
        players = [_player_from_row(row) for row in frame.get("players") or []]
        ball = _ball_from_record(frame.get("ball"))
        snapshot = possession.update(players, ball)
        metrics = passes.update(
            frame_number=int(frame["frame"]),
            fps=fps,
            snapshot=snapshot,
            players=players,
            ball=ball,
        )
        team_stats = teams.observe(snapshot.current_possession_team, metrics["events"])
        timeline.append(
            {
                "frame": int(frame["frame"]),
                "possessor": snapshot.current_possessor_track_id,
                "team": snapshot.current_possession_team,
            }
        )
    return {
        "timeline": timeline,
        "possession": possession.metrics(),
        "possession_by_team": team_stats,
        "passes": passes.finalize(),
    }


def write_event_record(path, clip_name, fps, frames, possession_by_team, passes, homography_feeds_decisions):
    payload = {
        "version": RECORD_VERSION,
        "clip_name": clip_name,
        "fps": float(fps),
        "frame_count": len(frames),
        "homography_feeds_decisions": bool(homography_feeds_decisions),
        "possession_by_team": possession_by_team,
        "passes": passes,
        "frames": frames,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default),
        encoding="utf-8",
    )
    return destination


def load_event_record(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame_by_number(record, frame_number):
    for frame in record.get("frames") or []:
        if int(frame["frame"]) == int(frame_number):
            return frame
    return None


def explain_ball_after(record, frame_number):
    """Describe how long the published ball lasts after one frame.

    Every number is read from the log. A run of observed sources shorter than
    the possession confirmation leaves the possessor empty.
    """
    frames = list(record.get("frames") or [])
    target = frame_by_number(record, frame_number)
    if target is None:
        return {"frame": int(frame_number), "present_in_log": False}
    later = [frame for frame in frames if int(frame["frame"]) > int(frame_number)]
    published_run = []
    first_missing = None
    for frame in [target] + later:
        published = frame.get("published_ball")
        if published:
            published_run.append(
                {
                    "frame": int(frame["frame"]),
                    "x": published.get("x"),
                    "y": published.get("y"),
                    "confidence": published.get("confidence"),
                    "ball_state": frame.get("ball_state"),
                }
            )
            continue
        first_missing = frame
        break
    observed_run = 0
    for frame in [target] + later:
        if (frame.get("ball") or {}).get("source") == "observed":
            observed_run += 1
            continue
        break
    next_published = None
    if first_missing is not None:
        next_published = next(
            (
                frame
                for frame in later
                if int(frame["frame"]) > int(first_missing["frame"]) and frame.get("published_ball")
            ),
            None,
        )
    next_possessor = next((frame for frame in later if frame.get("possessor") is not None), None)
    return {
        "frame": int(frame_number),
        "present_in_log": True,
        "published_ball": target.get("published_ball"),
        "ball_state": target.get("ball_state"),
        "possessor": target.get("possessor"),
        "published_run": published_run,
        "observed_run_including_frame": observed_run,
        "first_missing_frame": None if first_missing is None else int(first_missing["frame"]),
        "first_missing_state": None if first_missing is None else first_missing.get("ball_state"),
        "first_missing_source": None if first_missing is None else (first_missing.get("ball") or {}).get("source"),
        "refused_on_first_missing": [] if first_missing is None else list(first_missing.get("refused_centers") or []),
        "next_published_frame": None if next_published is None else int(next_published["frame"]),
        "next_published_ball": None if next_published is None else next_published.get("published_ball"),
        "next_possessor_frame": None if next_possessor is None else int(next_possessor["frame"]),
        "next_possessor": None if next_possessor is None else next_possessor.get("possessor"),
        "next_possessor_team": None if next_possessor is None else next_possessor.get("team"),
    }


def possessor_identity_report(record):
    """Compare the possessor with the players recorded on the same frame.

    A possessor can remain for a few frames after his box drops. That is the
    possession hold, not a second identity. A team_id that disagrees with the
    snapshot is a real inconsistency.
    """
    mismatches = []
    held_without_box = []
    for frame in record.get("frames") or []:
        possessor = frame.get("possessor")
        if possessor is None:
            continue
        match = None
        for row in frame.get("players") or []:
            if row and row[0] == possessor:
                match = row
                break
        if match is None:
            held_without_box.append({"frame": int(frame["frame"]), "possessor": possessor})
            continue
        team_id = match[1]
        expected = TEAM_NAME_BY_ID.get(team_id)
        if expected != frame.get("team"):
            mismatches.append(
                {
                    "frame": int(frame["frame"]),
                    "possessor": possessor,
                    "player_team_id": team_id,
                    "snapshot_team": frame.get("team"),
                }
            )
    return {"team_mismatches": mismatches, "held_without_box": held_without_box}


def team_changes(record):
    """Tracks whose team_id is not constant in this log. Empty means no flip was recorded."""
    seen = {}
    changes = []
    for frame in record.get("frames") or []:
        for row in frame.get("players") or []:
            if not row or row[1] is None:
                continue
            track_id = row[0]
            team_id = row[1]
            previous = seen.get(track_id)
            if previous is None:
                seen[track_id] = team_id
                continue
            if previous != team_id:
                changes.append(
                    {
                        "frame": int(frame["frame"]),
                        "track_id": track_id,
                        "from_team_id": previous,
                        "to_team_id": team_id,
                    }
                )
                seen[track_id] = team_id
    return changes
