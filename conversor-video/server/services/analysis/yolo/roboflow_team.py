from typing import Dict, Generator, Iterable, List, Optional, TypeVar

import cv2
import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"
TEAM_LOCK_CONFIRM_STREAK = 3
TEAM_LOCK_UNLOCK_STREAK = 3
TEAM_BLUE_ID = 0
TEAM_RED_ID = 1


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


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
        "semantic": {"0": "blue", "1": "red"},
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

    def assign(self, track_id: int, predicted_team: int, frame_number: int, verifying: bool) -> int:
        track_id = int(track_id)
        predicted_team = int(predicted_team)
        frame_number = int(frame_number)
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
            if predicted_team != int(state["team_id"]):
                self.prelock_team_switches += 1
                state["team_id"] = predicted_team
                state["confirm_streak"] = 1
            else:
                state["confirm_streak"] += 1
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

        self.refresh_contradictions += 1
        state["contradiction_streak"] += 1
        if state["contradiction_streak"] < self.unlock_streak:
            return int(state["team_id"])

        previous_team = int(state["team_id"])
        self.unlocks += 1
        self.team_changes_after_lock.append(
            {
                "track_id": track_id,
                "from_team": previous_team,
                "to_team": predicted_team,
                "frame": frame_number,
            }
        )
        state["locked"] = False
        state["team_id"] = predicted_team
        state["confirm_streak"] = 1
        state["contradiction_streak"] = 0
        if state["confirm_streak"] >= self.confirm_streak:
            state["locked"] = True
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
            "final_team_by_track": final_team_by_track,
        }


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.

    Cluster ids are then anchored to shirt color: 0 = blue, 1 = red.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        """
        Initialize the TeamClassifier with device and batch size.

        Args:
            device (str): The device to run the model on ('cpu' or 'cuda').
            batch_size (int): The batch size for processing images.
        """
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3, random_state=0)
        self.cluster_model = KMeans(n_clusters=2, random_state=0, n_init=10)
        self.cluster_to_team = {0: TEAM_BLUE_ID, 1: TEAM_RED_ID}
        self.color_anchor = {
            "semantic": {"0": "blue", "1": "red"},
            "cluster_blueness": {},
            "cluster_to_team": {"0": TEAM_BLUE_ID, "1": TEAM_RED_ID},
            "labels_were_remapped": False,
        }

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc="Embedding extraction"):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        mapping, report = anchor_cluster_labels(crops, self.cluster_model.labels_)
        self.cluster_to_team = {int(key): int(value) for key, value in mapping.items()}
        self.color_anchor = report

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict semantic team ids (0=blue, 1=red) for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted team ids anchored by shirt color.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        raw_labels = self.cluster_model.predict(projections)
        return np.array(
            [int(self.cluster_to_team.get(int(label), int(label))) for label in raw_labels],
            dtype=np.int32,
        )
