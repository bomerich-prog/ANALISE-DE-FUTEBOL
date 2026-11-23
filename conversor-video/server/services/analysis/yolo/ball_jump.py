"""Hold back a ball that jumps to another object and then crawls.

A real ball that covers a big distance in one frame was kicked hard, and it is
still fast on the next frame. When the ball layer jumps to a lookalike (a white
boot, a sock, a head) the point leaps once and then moves at walking pace.

Argentina v Spain, 39.5 s: the ball rolled slowly at the referee's foot, the
detector switched to the boot of Spain's 16 for seven frames (a 75 px leap,
then 7 px steps), and that boot took possession from Argentina's 9. The real
ball came back afterwards where its old path pointed.

Frames of such a jump keep their position but lose the "observed" source, so
they neither give nor take possession, start a touch, or close a pass. The jump
clears when the next step is fast too (a kick), when the point stays put for
long enough to be trusted (a camera cut), or when the ball comes back after a
gap at a pace its old path allows.
"""

import math

JUMP_MIN_PX = 40.0
JUMP_SPEED_FACTOR = 4.0
JUMP_MIN_PREVIOUS_SPEED_PX = 2.0
KICK_FOLLOW_RATIO = 0.5
JUMP_TRUST_FRAMES = 15
DUPLICATE_STEP_PX = 1.0
JUMP_LANDING_PX = 30.0
SUSPECT_SOURCE = "jump"


def _point(ball):
    if not ball or ball.get("source") != "observed":
        return None
    try:
        x = float(ball.get("x"))
        y = float(ball.get("y"))
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isnan(y):
        return None
    return x, y


class BallJumpGuard:
    def __init__(self):
        self.frame_index = 0
        self.trusted = None  # (frame_index, x, y) of the last trusted observation
        self.speed = None  # px per frame of the last trusted step
        self.suspect = None  # {"start", "last", "count", "jump"}
        self.jumps = 0

    def metrics(self):
        return {"ball_jumps_held_back": self.jumps}

    def filter(self, ball, players=None, possessor_id=None):
        self.frame_index += 1
        point = _point(ball)
        if point is None:
            return ball
        index = self.frame_index
        suspect = self.suspect
        if suspect is not None:
            last_index, last_x, last_y = suspect["last"]
            step = math.hypot(point[0] - last_x, point[1] - last_y)
            if index - last_index == 1 and step < DUPLICATE_STEP_PX:
                # Screen recordings repeat frames: a repeated frame says nothing.
                suspect["last"] = (index, point[0], point[1])
                return self._held(ball)
            if index - last_index == 1 and suspect["count"] == 1 and step >= KICK_FOLLOW_RATIO * suspect["jump"]:
                # Fast again on the next frame: a kick, not a lookalike.
                self.suspect = None
                self._trust(index, point, step)
                return ball
            if index - last_index > 1 and self.trusted is not None:
                trusted_index, trusted_x, trusted_y = self.trusted
                pace = math.hypot(point[0] - trusted_x, point[1] - trusted_y) / max(1, index - trusted_index)
                if pace < JUMP_MIN_PX:
                    # Back after a hole at a pace the old path allows.
                    self.suspect = None
                    self._trust(index, point, None)
                    return ball
            suspect["last"] = (index, point[0], point[1])
            suspect["count"] += 1
            if suspect["count"] > JUMP_TRUST_FRAMES:
                self.suspect = None
                self._trust(index, point, None)
                return ball
            return self._held(ball)
        if self.trusted is not None and index - self.trusted[0] == 1:
            step = math.hypot(point[0] - self.trusted[1], point[1] - self.trusted[2])
            if step < DUPLICATE_STEP_PX:
                self.trusted = (index, point[0], point[1])
                return ball
            previous = self.speed
            if (
                step >= JUMP_MIN_PX
                and previous is not None
                and step >= JUMP_SPEED_FACTOR * max(previous, JUMP_MIN_PREVIOUS_SPEED_PX)
                and _lands_on_rival(point, players, possessor_id)
            ):
                self.jumps += 1
                self.suspect = {"last": (index, point[0], point[1]), "count": 1, "jump": step}
                return self._held(ball)
            self._trust(index, point, step)
            return ball
        self._trust(index, point, None)
        return ball

    def _trust(self, index, point, step):
        self.trusted = (index, point[0], point[1])
        self.speed = step

    @staticmethod
    def _held(ball):
        held = dict(ball)
        held["source"] = SUSPECT_SOURCE
        return held


def _lands_on_rival(point, players, possessor_id):
    """The jump lands at the feet of someone other than the player on the ball.

    That is the shape of a lookalike: a boot or a sock right at a player. A kick
    sends the ball into open grass, and a jump at the possessor's own feet takes
    nothing from anyone.
    """
    if possessor_id is None:
        return False
    best = None
    owner = None
    for player in players or []:
        try:
            x = float(player.get("x"))
            y = float(player.get("y"))
            track_id = int(player.get("track_id"))
        except (TypeError, ValueError):
            continue
        distance = math.hypot(x - point[0], y - point[1])
        if track_id == int(possessor_id):
            owner = distance
        elif best is None or distance < best:
            best = distance
    if best is None or best > JUMP_LANDING_PX:
        return False
    return owner is None or best < owner


def jump_guard_ball(guard, possession, ball, players=None):
    """The ball the possession and pass engines should see this frame.

    A held-back jump only keeps a confirmed possessor from losing the ball to
    the lookalike. With nobody on the ball, the point is used as observed: the
    jump may be the real ball coming back after a wrong detection, as in the
    Barcelona clip, where the receiver must still be able to take it.
    """
    possessor_id = getattr(possession, "possessor_id", None)
    guarded = guard.filter(ball, players, possessor_id)
    if guarded is ball or possessor_id is None:
        return ball
    return guarded
