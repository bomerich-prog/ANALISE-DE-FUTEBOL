from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

TEAM_LOCK_CONFIRM_STREAK = 3
TEAM_LOCK_UNLOCK_STREAK = 3
TEAM_BLUE_ID = 0
TEAM_RED_ID = 1
TEAM_DARK_ID = 0
TEAM_LIGHT_ID = 1
TORSO_MIN_HEIGHT = 32
TORSO_MIN_WIDTH = 12
TORSO_MIN_PIXELS = 24
TORSO_GRASS_MAX = 0.55
TORSO_HUE_SATURATION_MIN = 45
TORSO_HUE_MIN_PIXELS = 12
TORSO_YELLOW_HUE_MIN = 16
TORSO_YELLOW_HUE_MAX = 45
TORSO_YELLOW_SATURATION_MIN = 120
TORSO_YELLOW_VALUE_MIN = 120
TORSO_YELLOW_FRACTION = 0.40
TORSO_MIXED_FRACTION = 0.28
TORSO_SIDE_MIN_SAMPLES = 4
TORSO_DISTANCE_SCALE_VALUE = 45.0
TORSO_DISTANCE_SCALE_SATURATION = 55.0
TORSO_DISTANCE_MARGIN = 0.40
TORSO_DISTANCE_MAX = 2.4
TORSO_HUE_SCALE = 40.0


def torso_region(crop: np.ndarray) -> np.ndarray:
    """Upper-middle shirt area, excluding most of the head, legs and sidelines."""
    if crop is None or crop.size == 0:
        return crop
    height, width = crop.shape[:2]
    y1 = int(height * 0.22)
    y2 = max(y1 + 1, int(height * 0.62))
    x1 = int(width * 0.22)
    x2 = max(x1 + 1, int(width * 0.78))
    torso = crop[y1:y2, x1:x2]
    if torso.size == 0:
        return crop
    return torso


def jersey_color_profile(crop: np.ndarray) -> Dict[str, float]:
    """How much of the torso reads as blue vs red, ignoring grass."""
    empty = {"blue": 0.0, "red": 0.0, "chroma": 0.0, "pixels": 0.0}
    if crop is None or getattr(crop, "size", 0) == 0:
        return empty
    torso = torso_region(crop)
    if torso is None or torso.size == 0:
        return empty
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)
    grass = (hue >= 35) & (hue <= 85) & (saturation >= 40) & (value >= 30)
    valid = (saturation >= 35) & (value >= 40) & ~grass
    if int(valid.sum()) < 8:
        valid = value >= 25
    pixel_count = int(valid.sum())
    if pixel_count == 0:
        return empty
    hues = hue[valid]
    red = float(np.mean((hues <= 12) | (hues >= 168)))
    blue = float(np.mean((hues >= 95) & (hues <= 130)))
    bgr = torso.astype(np.float32)
    blue_channel = float(bgr[:, :, 0][valid].mean())
    red_channel = float(bgr[:, :, 2][valid].mean())
    return {
        "blue": blue,
        "red": red,
        "chroma": blue_channel - red_channel,
        "pixels": float(pixel_count),
    }


def jersey_blueness(profile: Dict[str, float]) -> float:
    return float(profile["blue"]) - float(profile["red"]) + 0.002 * float(profile["chroma"])


def _weak_torso(reason: str) -> Dict[str, object]:
    return {
        "abstain": True,
        "reason": reason,
        "value": 0.0,
        "saturation": 0.0,
        "hue": None,
        "pixels": 0,
    }


def _hue_distance(left: float, right: float) -> float:
    gap = abs(float(left) - float(right))
    return min(gap, 180.0 - gap)


def _circular_mean_hue(hues: List[float]) -> float:
    angles = np.deg2rad(np.array(hues, dtype=np.float64) * 2.0)
    degrees = float(np.rad2deg(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())))
    return (degrees % 360.0) / 2.0


def measure_torso(crop: np.ndarray) -> Dict[str, object]:
    """Brightness and saturation of the shirt. Hue is kept only when it is saturated.

    A weak crop abstains: too small, too much grass, too few shirt pixels, a neon
    yellow official, or both a light shirt and a dark shirt in the same torso.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return _weak_torso("empty")
    height, width = crop.shape[:2]
    if height < TORSO_MIN_HEIGHT or width < TORSO_MIN_WIDTH:
        return _weak_torso("small")
    torso = torso_region(crop)
    if torso is None or torso.size == 0:
        return _weak_torso("empty")
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)
    grass = (hue >= 35) & (hue <= 85) & (saturation >= 40) & (value >= 30)
    if float(grass.mean()) > TORSO_GRASS_MAX:
        return _weak_torso("grass")
    yellow = (
        (hue >= TORSO_YELLOW_HUE_MIN)
        & (hue <= TORSO_YELLOW_HUE_MAX)
        & (saturation >= TORSO_YELLOW_SATURATION_MIN)
        & (value >= TORSO_YELLOW_VALUE_MIN)
    )
    if float(yellow.mean()) >= TORSO_YELLOW_FRACTION:
        return _weak_torso("yellow")
    usable = ~grass
    pixel_count = int(usable.sum())
    if pixel_count < TORSO_MIN_PIXELS:
        return _weak_torso("few_pixels")
    bright = usable & (value >= 170) & (saturation <= 60)
    dark = usable & (value <= 130) & (saturation >= 50)
    if (
        float(bright.sum()) / float(pixel_count) > TORSO_MIXED_FRACTION
        and float(dark.sum()) / float(pixel_count) > TORSO_MIXED_FRACTION
    ):
        return _weak_torso("mixed")
    saturated = usable & (saturation >= TORSO_HUE_SATURATION_MIN)
    hue_value = None
    if int(saturated.sum()) >= TORSO_HUE_MIN_PIXELS:
        hue_value = float(np.median(hue[saturated]))
    return {
        "abstain": False,
        "reason": None,
        "value": float(np.median(value[usable])),
        "saturation": float(np.median(saturation[usable])),
        "hue": hue_value,
        "pixels": pixel_count,
    }


def anchor_cluster_labels(crops: List[np.ndarray], raw_labels: Iterable[int]):
    """Map raw KMeans ids onto team 0 (blue shirt) and team 1 (red shirt)."""
    labels = [int(label) for label in raw_labels]
    buckets: Dict[int, List[Dict[str, float]]] = {}
    for crop, label in zip(crops, labels):
        buckets.setdefault(label, []).append(jersey_color_profile(crop))

    def mean_blueness(label: int) -> float:
        profiles = buckets.get(label) or []
        if not profiles:
            return 0.0
        return float(sum(jersey_blueness(profile) for profile in profiles) / len(profiles))

    present = sorted(buckets)
    blueness = {label: mean_blueness(label) for label in present}
    mapping = {0: TEAM_BLUE_ID, 1: TEAM_RED_ID}
    if len(present) == 1:
        only = present[0]
        mapping = {only: TEAM_BLUE_ID if blueness[only] >= 0.0 else TEAM_RED_ID}
    elif len(present) >= 2:
        ranked = sorted(present, key=lambda label: (blueness[label], -label), reverse=True)
        mapping = {ranked[0]: TEAM_BLUE_ID, ranked[1]: TEAM_RED_ID}
        for extra in ranked[2:]:
            mapping[extra] = TEAM_RED_ID
    report = {
        "semantic": {"0": "Team 0", "1": "Team 1"},
        "cluster_blueness": {str(label): blueness[label] for label in present},
        "cluster_to_team": {str(label): int(mapping[label]) for label in mapping},
        "labels_were_remapped": any(int(label) != int(mapping[label]) for label in mapping),
    }
    return mapping, report


def reject_training_crop(crop: np.ndarray) -> Optional[str]:
    """Drop flags, posts, grass and low-chroma officials before the team fit."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return "empty"
    height, width = crop.shape[:2]
    if height < 40 or width < 18:
        return "small"
    aspect = width / float(height)
    if aspect < 0.18:
        return "thin_non_player"
    if aspect > 1.05:
        return "wide_non_player"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    grass = (hue >= 35) & (hue <= 90) & (saturation > 35) & (value > 30)
    if float(grass.mean()) > 0.72:
        return "mostly_grass"

    if aspect < 0.32:
        top = hsv[: max(1, height // 3)]
        flag_hue = (top[:, :, 0] <= 25) | ((top[:, :, 0] >= 15) & (top[:, :, 0] <= 45))
        flag = flag_hue & (top[:, :, 1] > 80) & (top[:, :, 2] > 60)
        if float(flag.mean()) > 0.22:
            return "corner_flag"

    torso = torso_region(crop)
    torso_hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    dark_neutral = (torso_hsv[:, :, 1] < 42) & (torso_hsv[:, :, 2] < 85)
    if float(dark_neutral.mean()) > 0.62:
        return "low_chroma_official"
    return None


class TeamTrackLock:
    """Keep one shirt color per track. Refresh can vote, not overwrite."""

    def __init__(
        self,
        confirm_streak: int = TEAM_LOCK_CONFIRM_STREAK,
        unlock_streak: int = TEAM_LOCK_UNLOCK_STREAK,
    ):
        self.confirm_streak = max(1, int(confirm_streak))
        self.unlock_streak = max(2, int(unlock_streak))
        self.tracks = {}
        self.prelock_team_switches = 0
        self.refresh_checks = 0
        self.refresh_agreements = 0
        self.refresh_contradictions = 0
        self.unlocks = 0
        self.team_changes_after_lock = []
        self.ignored_weak_votes = 0

    def needs_observation(self, track_id: int, frame_number: int, refresh_interval: int) -> bool:
        state = self.tracks.get(int(track_id))
        if state is None or not state["locked"]:
            return True
        return int(frame_number) - int(state["last_checked_frame"]) > int(refresh_interval)

    def is_locked(self, track_id: int) -> bool:
        state = self.tracks.get(int(track_id))
        return bool(state and state["locked"])

    def current_team(self, track_id: int):
        state = self.tracks.get(int(track_id))
        if state is None:
            return None
        return int(state["team_id"])

    def assign(self, track_id: int, predicted_team, frame_number: int, verifying: bool):
        """Record a clear vote. A missing vote leaves the track untouched."""
        track_id = int(track_id)
        frame_number = int(frame_number)
        if predicted_team is None:
            self.ignored_weak_votes += 1
            state = self.tracks.get(track_id)
            if state is None:
                return None
            return int(state["team_id"])
        predicted_team = int(predicted_team)
        state = self.tracks.get(track_id)
        if state is None:
            state = {
                "team_id": predicted_team,
                "locked": False,
                "confirm_streak": 1,
                "contradiction_streak": 0,
                "last_checked_frame": frame_number,
            }
            if state["confirm_streak"] >= self.confirm_streak:
                state["locked"] = True
            self.tracks[track_id] = state
            return int(state["team_id"])

        state["last_checked_frame"] = frame_number
        if not state["locked"]:
            if predicted_team == int(state["team_id"]):
                state["confirm_streak"] += 1
                state["contradiction_streak"] = 0
            else:
                # One opposite vote used to replace the team immediately, so a
                # flickering edge crop never reached the lock.
                state["contradiction_streak"] += 1
                if state["contradiction_streak"] >= self.confirm_streak:
                    self.prelock_team_switches += 1
                    state["team_id"] = predicted_team
                    state["confirm_streak"] = state["contradiction_streak"]
                    state["contradiction_streak"] = 0
            if state["confirm_streak"] >= self.confirm_streak:
                state["locked"] = True
                state["contradiction_streak"] = 0
            return int(state["team_id"])

        if not verifying:
            return int(state["team_id"])

        self.refresh_checks += 1
        if predicted_team == int(state["team_id"]):
            self.refresh_agreements += 1
            state["contradiction_streak"] = 0
            return int(state["team_id"])

        # A locked shirt stays. Later shadow or a partial crop must not rename the team.
        self.refresh_contradictions += 1
        state["contradiction_streak"] += 1
        return int(state["team_id"])

    def metrics(self) -> Dict[str, object]:
        final_team_by_track = {
            str(track_id): int(state["team_id"])
            for track_id, state in sorted(self.tracks.items(), key=lambda item: item[0])
        }
        return {
            "confirm_streak_required": self.confirm_streak,
            "unlock_streak_required": self.unlock_streak,
            "tracks_seen": len(self.tracks),
            "tracks_locked": sum(1 for state in self.tracks.values() if state["locked"]),
            "prelock_team_switches": self.prelock_team_switches,
            "team_changes_after_lock": len(self.team_changes_after_lock),
            "team_change_events": list(self.team_changes_after_lock),
            "refresh_checks": self.refresh_checks,
            "refresh_agreements": self.refresh_agreements,
            "refresh_contradictions": self.refresh_contradictions,
            "unlocks": self.unlocks,
            "ignored_weak_votes": self.ignored_weak_votes,
            "final_team_by_track": final_team_by_track,
        }


def _split_brightness(samples: List[Dict[str, object]]):
    """Two groups from torso brightness and saturation. The darker group is first."""
    points = np.array(
        [[float(item["value"]), float(item["saturation"])] for item in samples],
        dtype=np.float64,
    )
    center = points.mean(axis=0)
    scale = points.std(axis=0)
    scale[scale < 1.0] = 1.0
    normalized = (points - center) / scale
    dark_index = int(np.argmin(points[:, 0]))
    light_index = int(np.argmax(points[:, 0]))
    if dark_index == light_index:
        return samples, []
    centers = np.stack([normalized[dark_index], normalized[light_index]])
    labels = np.zeros(len(samples), dtype=np.int32)
    for _ in range(16):
        distances = np.linalg.norm(normalized[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        if len(set(labels.tolist())) < 2:
            break
        for group in (0, 1):
            chosen = normalized[labels == group]
            if len(chosen) == 0:
                continue
            centers[group] = chosen.mean(axis=0)
    darker = [samples[index] for index, label in enumerate(labels) if int(label) == 0]
    lighter = [samples[index] for index, label in enumerate(labels) if int(label) == 1]
    if not darker or not lighter:
        return samples, []
    darker_value = float(np.mean([float(item["value"]) for item in darker]))
    lighter_value = float(np.mean([float(item["value"]) for item in lighter]))
    if darker_value <= lighter_value:
        return darker, lighter
    return lighter, darker


class TeamClassifier:
    """Assign Team 0 to the dark shirt and Team 1 to the light shirt.

    The decision is the torso color measured once on the training crops.
    SigLIP, UMAP and KMeans do not vote. A weak torso returns -1 so the
    track lock can keep the last stable team.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.mode = "value"
        self.prototypes = {
            TEAM_DARK_ID: {"value": 90.0, "saturation": 120.0, "hue": None},
            TEAM_LIGHT_ID: {"value": 200.0, "saturation": 20.0, "hue": None},
        }
        self.color_anchor = {
            "semantic": {"0": "Team 0", "1": "Team 1"},
            "decision": "torso_color",
            "mode": self.mode,
            "team_0": "dark",
            "team_1": "light",
            "cluster_to_team": {"0": TEAM_DARK_ID, "1": TEAM_LIGHT_ID},
            "labels_were_remapped": False,
        }
        self.vote_count = 0
        self.abstain_count = 0

    def fit(self, crops: List[np.ndarray]) -> None:
        samples = []
        for crop in crops:
            measured = measure_torso(crop)
            if measured["abstain"]:
                continue
            samples.append(measured)
        if len(samples) < TORSO_SIDE_MIN_SAMPLES * 2:
            raise RuntimeError("TeamClassifier: poucos torsos com cor utilizável.")
        dark, light = _split_brightness(samples)
        if len(dark) >= TORSO_SIDE_MIN_SAMPLES and len(light) >= TORSO_SIDE_MIN_SAMPLES:
            dark_prototype = _prototype(dark)
            light_prototype = _prototype(light)
            value_gap = float(light_prototype["value"]) - float(dark_prototype["value"])
            saturation_gap = float(dark_prototype["saturation"]) - float(light_prototype["saturation"])
            if value_gap >= 20.0 or saturation_gap >= 25.0:
                self.mode = "value"
                self.prototypes = {TEAM_DARK_ID: dark_prototype, TEAM_LIGHT_ID: light_prototype}
                self.color_anchor = _anchor_report(self.mode, "dark", "light", self.prototypes, dark, light)
                return
        blue = [item for item in samples if _hue_is_blue(item.get("hue"))]
        red = [item for item in samples if _hue_is_red(item.get("hue"))]
        if len(blue) < TORSO_SIDE_MIN_SAMPLES or len(red) < TORSO_SIDE_MIN_SAMPLES:
            raise RuntimeError("TeamClassifier: os torsos não formam dois times pela cor.")
        self.mode = "hue"
        self.prototypes = {
            TEAM_DARK_ID: _prototype(blue),
            TEAM_LIGHT_ID: _prototype(red),
        }
        self.color_anchor = _anchor_report(self.mode, "blue", "red", self.prototypes, blue, red)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([], dtype=np.int32)
        teams = [self._predict_one(crop) for crop in crops]
        return np.array(teams, dtype=np.int32)

    def _predict_one(self, crop: np.ndarray) -> int:
        measured = measure_torso(crop)
        if measured["abstain"]:
            self.abstain_count += 1
            return -1
        if self.mode == "hue":
            if measured["hue"] is None:
                self.abstain_count += 1
                return -1
            distances = {
                team_id: _hue_distance(float(measured["hue"]), float(prototype["hue"])) / TORSO_HUE_SCALE
                for team_id, prototype in self.prototypes.items()
                if prototype.get("hue") is not None
            }
        else:
            distances = {
                team_id: _value_distance(measured, prototype)
                for team_id, prototype in self.prototypes.items()
            }
        if len(distances) < 2:
            self.abstain_count += 1
            return -1
        ranked = sorted(distances, key=lambda team_id: (distances[team_id], team_id))
        nearest = ranked[0]
        margin = distances[ranked[1]] - distances[nearest]
        if distances[nearest] > TORSO_DISTANCE_MAX or margin <= TORSO_DISTANCE_MARGIN:
            self.abstain_count += 1
            return -1
        self.vote_count += 1
        return int(nearest)


def _prototype(samples: List[Dict[str, object]]) -> Dict[str, object]:
    values = [float(item["value"]) for item in samples]
    saturations = [float(item["saturation"]) for item in samples]
    hues = [float(item["hue"]) for item in samples if item.get("hue") is not None]
    return {
        "value": float(np.median(values)),
        "saturation": float(np.median(saturations)),
        "hue": _circular_mean_hue(hues) if hues else None,
    }


def _value_distance(measured: Dict[str, object], prototype: Dict[str, object]) -> float:
    value_gap = abs(float(measured["value"]) - float(prototype["value"])) / TORSO_DISTANCE_SCALE_VALUE
    saturation_gap = (
        abs(float(measured["saturation"]) - float(prototype["saturation"]))
        / TORSO_DISTANCE_SCALE_SATURATION
    )
    return value_gap + saturation_gap


def _hue_is_blue(hue) -> bool:
    if hue is None:
        return False
    return 95.0 <= float(hue) <= 130.0


def _hue_is_red(hue) -> bool:
    if hue is None:
        return False
    hue = float(hue)
    return hue <= 15.0 or hue >= 165.0


def _anchor_report(mode, team_0_name, team_1_name, prototypes, group_0, group_1) -> Dict[str, object]:
    return {
        "semantic": {"0": "Team 0", "1": "Team 1"},
        "decision": "torso_color",
        "mode": mode,
        "team_0": team_0_name,
        "team_1": team_1_name,
        "prototype_value": {
            "0": round(float(prototypes[0]["value"]), 1),
            "1": round(float(prototypes[1]["value"]), 1),
        },
        "prototype_saturation": {
            "0": round(float(prototypes[0]["saturation"]), 1),
            "1": round(float(prototypes[1]["saturation"]), 1),
        },
        "samples": {"0": len(group_0), "1": len(group_1)},
        "cluster_to_team": {"0": TEAM_DARK_ID, "1": TEAM_LIGHT_ID},
        "labels_were_remapped": False,
    }
