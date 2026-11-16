"""2D radar: the pitch seen from above, with the players and the ball of this frame.

The map calibrations come in two field frames. A penalty matrix puts the goal
line at x = 0 or x = 12000 cm, and a centre matrix puts the halfway line at
x = 6000 cm. Both measure real centimetres from that mark, and both put the
middle of the pitch width at y = 3500 cm. The radar draws a FIFA standard
pitch, 105 by 68 m, and moves each frame's points into it from the mark the
calibration was read from. Distances from that mark stay true.

A point is published only when its pixel is on the grass and it lands on the
pitch. The radar never feeds possession or passes.
"""

import cv2
import numpy as np

RADAR_LENGTH_CM = 10500.0
RADAR_WIDTH_CM = 6800.0
_HALF_LENGTH = RADAR_LENGTH_CM / 2.0
_HALF_WIDTH = RADAR_WIDTH_CM / 2.0
_TEMPLATE_LENGTH_CM = 12000.0
_TEMPLATE_MIDDLE_Y_CM = 3500.0
_MARGIN_CM = 300.0
# Steps longer than this between two frames are a new place, not a run.
_SMOOTH_RESET_CM = 400.0
_SMOOTH_KEEP = 0.45

TEAM_BGR = {0: (235, 120, 30), 1: (40, 40, 220)}
NEUTRAL_BGR = (190, 190, 190)
REFEREE_BGR = (0, 230, 255)
BALL_BGR = (0, 140, 255)
_GRASS_BGR = (52, 120, 58)
_STRIPE_BGR = (58, 130, 64)
_LINE_BGR = (235, 235, 235)
_VIEW_BGR = (255, 255, 255)


def to_radar(field_x, field_y, region, positive_x):
    """Radar centimetres, origin at the low-x corner, from one calibration's field point."""
    if region == "centre":
        x = float(field_x) - _TEMPLATE_LENGTH_CM / 2.0 + _HALF_LENGTH
    elif positive_x:
        x = RADAR_LENGTH_CM - (_TEMPLATE_LENGTH_CM - float(field_x))
    else:
        x = float(field_x)
    y = float(field_y) - _TEMPLATE_MIDDLE_Y_CM + _HALF_WIDTH
    return x, y


def on_radar(x, y, margin=_MARGIN_CM):
    return -margin <= x <= RADAR_LENGTH_CM + margin and -margin <= y <= RADAR_WIDTH_CM + margin


class TeamColours:
    """Each team's shirt colour, read from the torso of its players in the video.

    The radar paints a team the colour it wears, so a red side on the video is
    red on the radar. Until a team has been seen, a fixed colour stands in.
    """

    def __init__(self, keep=0.9):
        self._keep = keep
        self._colour = {}

    def observe(self, frame, detections):
        if frame is None:
            return
        height, width = frame.shape[:2]
        samples = {}
        for detection in detections:
            team = detection.get("team_id")
            if team is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in detection["box"]]
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width < 6 or box_height < 14:
                continue
            left = int(max(0, x1 + 0.3 * box_width))
            right = int(min(width, x2 - 0.3 * box_width))
            top = int(max(0, y1 + 0.18 * box_height))
            bottom = int(min(height, y1 + 0.45 * box_height))
            if right - left < 2 or bottom - top < 2:
                continue
            crop = frame[top:bottom, left:right].reshape(-1, 3).astype(np.float64)
            hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV).reshape(-1, 3)
            # Leave the grass between the arms out.
            shirt = crop[~((hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 60))]
            if len(shirt) < 4:
                continue
            samples.setdefault(int(team), []).append(np.median(shirt, axis=0))
        for team, colours in samples.items():
            colour = np.median(np.array(colours), axis=0)
            previous = self._colour.get(team)
            self._colour[team] = colour if previous is None else self._keep * previous + (1 - self._keep) * colour

    def colours(self):
        return {team: tuple(int(round(v)) for v in colour) for team, colour in self._colour.items()}


class RadarSmoother:
    """Light smoothing per track, so a jittering foot box does not shake the dot."""

    def __init__(self):
        self._last = {}

    def smooth(self, key, frame_number, point):
        if key is None:
            return point
        previous = self._last.get(key)
        if previous is not None:
            last_frame, last_point = previous
            jump = float(np.hypot(point[0] - last_point[0], point[1] - last_point[1]))
            if frame_number - last_frame <= 2 and jump <= _SMOOTH_RESET_CM:
                point = (
                    _SMOOTH_KEEP * last_point[0] + (1.0 - _SMOOTH_KEEP) * point[0],
                    _SMOOTH_KEEP * last_point[1] + (1.0 - _SMOOTH_KEEP) * point[1],
                )
        self._last[key] = (frame_number, point)
        return point


# A lost player keeps his dot this long (0.6 s at 30 fps). Longer, a player who
# comes back under a new ID far from his dot is drawn twice.
RADAR_HOLD_SECONDS = 0.6
RADAR_HOLD_FRAMES = 18
# A held dot goes as soon as a player of the same team is seen this close to it:
# almost always the same player back under a new ID.
_SHADOW_CM = 500.0
# A new ID this close to a lost dot of the same team is the same player.
_RELINK_BASE_CM = 300.0
_RELINK_CM_PER_FRAME = 30.0
_RELINK_MAX_CM = 1500.0
_RELINK_CLEAR_RATIO = 1.3


def _role_group(role):
    return "Referee" if role == "Referee" else "Player"


class RadarMemory:
    """Players on the radar, kept on their last spot while the detector misses them.

    Each dot is a slot. A track ID owns its slot while it is seen. A missed
    player keeps his dot where he was last mapped, for up to hold_frames.
    When the tracker gives a lost player a new ID, the new ID takes over the
    lost dot if it appears close to it, on the same team, and no other lost
    dot is about as close. A held dot with a seen player of the same team
    within 5 m goes away: it is almost always him under another ID. Only the
    drawing uses this: possession, passes and the contest read the tracker as
    before.
    """

    def __init__(self, hold_frames=RADAR_HOLD_FRAMES):
        self.hold_frames = int(hold_frames)
        self._slots = {}
        self._slot_of_track = {}
        self._next_slot = 0
        self.held_dots = 0
        self.relinked_ids = 0
        self.dropped_shadows = 0

    def update(self, frame_number, players):
        """All dots of this frame: the players seen now and the lost ones still held.

        players: dicts with x, y, role, team_id, track_id (track_id may be None).
        """
        seen = []
        loose = []
        fresh = []
        for player in players:
            track = player.get("track_id")
            if track is None:
                loose.append(dict(player))
                continue
            slot = self._slot_of_track.get(track)
            if slot is not None and slot in self._slots and self._slots[slot]["track_id"] == track:
                seen.append((slot, player))
            else:
                fresh.append(player)
        taken = {slot for slot, _player in seen}
        for slot, player in seen:
            self._store(slot, player, frame_number)
        for player, slot in self._relink(fresh, taken, frame_number):
            if slot is None:
                slot = self._next_slot
                self._next_slot += 1
            else:
                self.relinked_ids += 1
            self._slot_of_track[player["track_id"]] = slot
            taken.add(slot)
            self._store(slot, player, frame_number)
        self._expire(frame_number)
        self._drop_shadows(taken, loose)
        dots = []
        for slot, entry in self._slots.items():
            dot = dict(entry["player"])
            dot["tracker_id"] = entry["track_id"]
            dot["track_id"] = entry["label"]
            dot["slot"] = slot
            dot["held"] = slot not in taken
            if dot["held"]:
                self.held_dots += 1
            dots.append(dot)
        return dots + loose

    def _store(self, slot, player, frame_number):
        # The dot keeps the number it was first drawn with, even after a new tracker ID.
        label = self._slots[slot]["label"] if slot in self._slots else player["track_id"]
        self._slots[slot] = {
            "player": dict(player),
            "track_id": player["track_id"],
            "label": label,
            "frame": frame_number,
        }

    def _relink(self, fresh, taken, frame_number):
        """Each new ID with the lost dot it continues, or None. Nearest pairs first."""
        lost = [
            slot
            for slot, entry in self._slots.items()
            if slot not in taken and entry["frame"] < frame_number
        ]
        options = []
        for index, player in enumerate(fresh):
            near = []
            for slot in lost:
                entry = self._slots[slot]
                old = entry["player"]
                if _role_group(old.get("role")) != _role_group(player.get("role")):
                    continue
                if old.get("team_id") != player.get("team_id"):
                    continue
                gap = frame_number - entry["frame"]
                reach = min(_RELINK_BASE_CM + _RELINK_CM_PER_FRAME * gap, _RELINK_MAX_CM)
                distance = float(np.hypot(player["x"] - old["x"], player["y"] - old["y"]))
                if distance <= reach:
                    near.append((distance, slot))
            near.sort()
            if not near:
                continue
            # Two lost dots about as close: no guess.
            if len(near) > 1 and near[1][0] <= near[0][0] * _RELINK_CLEAR_RATIO + 50.0:
                continue
            options.append((near[0][0], index, near[0][1]))
        chosen = {}
        used = set()
        for _distance, index, slot in sorted(options):
            if index in chosen or slot in used:
                continue
            chosen[index] = slot
            used.add(slot)
        return [(player, chosen.get(index)) for index, player in enumerate(fresh)]

    def _drop_shadows(self, taken, loose):
        """Remove held dots with a seen player of the same team and role close by."""
        seen = [self._slots[slot]["player"] for slot in taken if slot in self._slots] + list(loose)
        for slot in [s for s in self._slots if s not in taken]:
            old = self._slots[slot]["player"]
            for player in seen:
                if _role_group(old.get("role")) != _role_group(player.get("role")):
                    continue
                if old.get("team_id") != player.get("team_id"):
                    continue
                if float(np.hypot(player["x"] - old["x"], player["y"] - old["y"])) <= _SHADOW_CM:
                    track = self._slots[slot]["track_id"]
                    del self._slots[slot]
                    if self._slot_of_track.get(track) == slot:
                        del self._slot_of_track[track]
                    self.dropped_shadows += 1
                    break

    def _expire(self, frame_number):
        for slot in [s for s, entry in self._slots.items() if frame_number - entry["frame"] > self.hold_frames]:
            track = self._slots[slot]["track_id"]
            del self._slots[slot]
            if self._slot_of_track.get(track) == slot:
                del self._slot_of_track[track]


def _layout(frame_shape, width_share=0.30, margin_px=16):
    """Pixel box of the radar in the bottom-right corner, and centimetres per pixel."""
    height, width = frame_shape[:2]
    pad = max(6, int(width * 0.006))
    radar_width = int(width * width_share)
    scale = (radar_width - 2 * pad) / (RADAR_LENGTH_CM + 2 * _MARGIN_CM)
    radar_height = int((RADAR_WIDTH_CM + 2 * _MARGIN_CM) * scale) + 2 * pad
    left = width - radar_width - margin_px
    top = height - radar_height - margin_px
    return left, top, radar_width, radar_height, pad, scale


def _to_pixel(layout, x, y):
    left, top, _width, _height, pad, scale = layout
    return (
        int(round(left + pad + (x + _MARGIN_CM) * scale)),
        int(round(top + pad + (y + _MARGIN_CM) * scale)),
    )


def _pitch_lines(canvas, layout, thickness):
    def line(points):
        pixels = np.array([_to_pixel(layout, x, y) for x, y in points], dtype=np.int32)
        cv2.polylines(canvas, [pixels], False, _LINE_BGR, thickness, cv2.LINE_AA)

    length, width = RADAR_LENGTH_CM, RADAR_WIDTH_CM
    line([(0, 0), (length, 0), (length, width), (0, width), (0, 0)])
    line([(_HALF_LENGTH, 0), (_HALF_LENGTH, width)])
    angles = np.linspace(0.0, 2.0 * np.pi, 64)
    line([(_HALF_LENGTH + 915.0 * np.cos(a), _HALF_WIDTH + 915.0 * np.sin(a)) for a in angles])
    for goal, inward in ((0.0, 1.0), (length, -1.0)):
        front = goal + inward * 1650.0
        line([(goal, _HALF_WIDTH - 2016.0), (front, _HALF_WIDTH - 2016.0), (front, _HALF_WIDTH + 2016.0), (goal, _HALF_WIDTH + 2016.0)])
        small = goal + inward * 550.0
        line([(goal, _HALF_WIDTH - 916.0), (small, _HALF_WIDTH - 916.0), (small, _HALF_WIDTH + 916.0), (goal, _HALF_WIDTH + 916.0)])
        spot = goal + inward * 1100.0
        reach = np.arccos(550.0 / 915.0)
        line([(spot + inward * 915.0 * np.cos(a), _HALF_WIDTH + 915.0 * np.sin(a)) for a in np.linspace(-reach, reach, 24)])
        line([(goal, _HALF_WIDTH - 366.0), (goal - inward * 120.0, _HALF_WIDTH - 366.0), (goal - inward * 120.0, _HALF_WIDTH + 366.0), (goal, _HALF_WIDTH + 366.0)])


def draw_radar(frame, players, ball=None, view=None, team_colours=None):
    """Draw the radar in the bottom-right corner of the frame.

    players: dicts with radar x and y in cm, role ("Player", "Goalkeeper",
    "Referee"), team_id and track_id. ball: (x, y) in cm or None. view: the
    camera's footprint on the pitch, a list of (x, y) in cm, or None.
    team_colours: BGR per team_id, read from the shirts.
    """
    palette = dict(TEAM_BGR)
    if team_colours:
        palette.update(team_colours)
    if frame is None:
        return
    layout = _layout(frame.shape)
    left, top, radar_width, radar_height, pad, scale = layout
    height, width = frame.shape[:2]
    if left < 0 or top < 0:
        return
    region = frame[top : top + radar_height, left : left + radar_width]
    canvas = np.empty_like(region)
    canvas[:] = _GRASS_BGR
    stripes = 10
    for index in range(0, stripes, 2):
        x0, _y0 = _to_pixel(layout, RADAR_LENGTH_CM * index / stripes, 0.0)
        x1, _y1 = _to_pixel(layout, RADAR_LENGTH_CM * (index + 1) / stripes, 0.0)
        cv2.rectangle(canvas, (x0 - left, 0), (x1 - left, radar_height), _STRIPE_BGR, -1)
    shifted = (0, 0, radar_width, radar_height, pad, scale)
    line_px = max(1, int(round(width / 900)))
    _pitch_lines(canvas, shifted, line_px)
    if view is not None and len(view) >= 3:
        outline = np.array([_to_pixel(shifted, x, y) for x, y in view], dtype=np.int32)
        shade = canvas.copy()
        cv2.fillPoly(shade, [outline], (90, 160, 95))
        cv2.addWeighted(shade, 0.35, canvas, 0.65, 0, canvas)
        cv2.polylines(canvas, [outline], True, _VIEW_BGR, line_px, cv2.LINE_AA)
    radius = max(4, int(round(width / 230)))
    font_scale = max(0.3, width / 4200)
    for player in players:
        centre = _to_pixel(shifted, player["x"], player["y"])
        role = player.get("role")
        if role == "Referee":
            colour = REFEREE_BGR
        else:
            colour = palette.get(player.get("team_id"), NEUTRAL_BGR)
        cv2.circle(canvas, centre, radius, colour, -1, cv2.LINE_AA)
        ring = (0, 0, 0) if role != "Goalkeeper" else (255, 255, 255)
        cv2.circle(canvas, centre, radius, ring, max(1, line_px), cv2.LINE_AA)
        track = player.get("track_id")
        if track is not None and radar_width >= 360:
            label = str(track)
            size, _base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            origin = (centre[0] - size[0] // 2, centre[1] - radius - 3)
            cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    if ball is not None:
        centre = _to_pixel(shifted, ball[0], ball[1])
        cv2.circle(canvas, centre, max(3, radius - 1), BALL_BGR, -1, cv2.LINE_AA)
        cv2.circle(canvas, centre, max(3, radius - 1), (255, 255, 255), max(1, line_px), cv2.LINE_AA)
    cv2.addWeighted(canvas, 0.9, region, 0.1, 0, region)
    cv2.rectangle(frame, (left, top), (left + radar_width - 1, top + radar_height - 1), (255, 255, 255), line_px)
