"""Experimental monocular collision-risk indicators; never a safety determination."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


def intersection_over_union(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def box_area_ratio(box: list[float], width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / (width * height)


@dataclass
class Track:
    track_id: str
    label: str
    observations: deque = field(default_factory=lambda: deque(maxlen=8))
    misses: int = 0


class CollisionRiskEngine:
    """Tracks image-plane boxes and emits cautious experimental approach indicators.

    Apparent size growth is not distance, speed, or time-to-collision. This engine
    intentionally never estimates TTC and can only emit NORMAL, MEDIUM, or
    UNAVAILABLE assessments. Its output is not a crash prediction or safety alert.
    """

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

    def __init__(self, max_observation_gap: float = 1.5, alert_cooldown: float = 3.0,
                 minimum_confidence: float = 0.35):
        self.max_observation_gap = max_observation_gap
        self.alert_cooldown = alert_cooldown
        self.minimum_confidence = minimum_confidence
        self.tracks: list[Track] = []
        self.next_track_id = 1
        self.last_timestamp: float | None = None
        self.last_alert_timestamp: float | None = None
        self.active_alert_track: str | None = None
        self.benign_streak = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.last_timestamp = None
        self.last_alert_timestamp = None
        self.active_alert_track = None
        self.benign_streak = 0

    def _unavailable(self, timestamp: float, reason: str) -> dict[str, Any]:
        return {
            "risk_level": "UNAVAILABLE", "timestamp": timestamp, "status": "UNAVAILABLE",
            "experimental": True, "objects": [], "evidence": [], "reason": reason,
            "uncertainty": "High. Monocular image-plane observations are not calibrated distance or collision measurements.",
            "estimated_time_to_collision_seconds": None, "alert": None,
        }

    def update(
        self, timestamp: float, frame_width: int, frame_height: int,
        detections: list[dict[str, Any]], inference_latency_ms: float,
    ) -> dict[str, Any]:
        if frame_width <= 0 or frame_height <= 0:
            return self._unavailable(timestamp, "Frame dimensions are unavailable.")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            self.reset()
            return self._unavailable(timestamp, "Out-of-order or repeated simulation timestamp; temporal tracks were reset.")
        if self.last_timestamp is not None and timestamp - self.last_timestamp > self.max_observation_gap:
            self.tracks.clear()
            self.active_alert_track = None
            self.last_timestamp = timestamp
            return self._unavailable(timestamp, "Inference observations have a gap; temporal risk is unavailable.")
        self.last_timestamp = timestamp
        if inference_latency_ms > 1500:
            self.tracks.clear()
            self.active_alert_track = None
            return self._unavailable(timestamp, "Inference latency exceeded the 1.5-second freshness limit.")

        vehicles = [d for d in detections if d.get("label") in self.VEHICLE_CLASSES
                    and len(d.get("bbox", [])) == 4
                    and float(d.get("confidence", 0.0)) >= self.minimum_confidence]
        for track in self.tracks:
            track.misses += 1

        matched: set[str] = set()
        for detection in vehicles:
            box = [float(value) for value in detection["bbox"]]
            candidates = [t for t in self.tracks if t.label == detection["label"] and t.track_id not in matched and t.misses <= 2]
            track = max(candidates, key=lambda t: intersection_over_union(t.observations[-1][1], box), default=None)
            if track is not None and intersection_over_union(track.observations[-1][1], box) < 0.12:
                track = None
            if track is None:
                track = Track(f"vehicle-{self.next_track_id}", detection["label"])
                self.next_track_id += 1
                self.tracks.append(track)
            track.misses = 0
            track.observations.append((timestamp, box, float(detection.get("confidence", 0.0))))
            matched.add(track.track_id)

        self.tracks = [track for track in self.tracks if track.misses <= 2]
        candidates: list[tuple[Track, float, float, float, float]] = []
        for track in self.tracks:
            if track.misses or len(track.observations) < 3:
                continue
            first_time, first_box, _ = track.observations[0]
            last_time, last_box, last_confidence = track.observations[-1]
            elapsed = last_time - first_time
            first_area = box_area_ratio(first_box, frame_width, frame_height)
            last_area = box_area_ratio(last_box, frame_width, frame_height)
            growth = last_area / first_area if first_area > 0 else 0.0
            center_x = (last_box[0] + last_box[2]) / (2 * frame_width)
            center_y = (last_box[1] + last_box[3]) / (2 * frame_height)
            center_x_first = (first_box[0] + first_box[2]) / (2 * frame_width)
            center_y_first = (first_box[1] + first_box[3]) / (2 * frame_height)
            relative_dx = center_x - center_x_first
            relative_dy = center_y - center_y_first
            if (elapsed >= 0.5 and last_confidence >= self.minimum_confidence
                    and 0.2 <= center_x <= 0.8 and center_y >= 0.3
                    and last_area >= 0.06 and growth >= 1.5):
                candidates.append((track, growth, last_area, relative_dx, relative_dy))

        if candidates:
            track, growth, area, relative_dx, relative_dy = max(candidates, key=lambda candidate: candidate[1])
            alert = None
            if self.active_alert_track != track.track_id and (
                self.last_alert_timestamp is None or timestamp - self.last_alert_timestamp >= self.alert_cooldown
            ):
                alert = {
                    "type": "POTENTIAL_FORWARD_OBSTACLE",
                    "timestamp": timestamp,
                    "risk_level": "MEDIUM",
                    "track_id": track.track_id,
                    "evidence": [f"Tracked {track.label} box grew {growth:.2f}x over {track.observations[-1][0] - track.observations[0][0]:.2f}s", f"Tracked center shifted ({relative_dx:+.3f}, {relative_dy:+.3f}) in normalized image coordinates", f"Current box covers {area:.1%} of the image at confidence {track.observations[-1][2]:.2f}"],
                    "status": "EXPERIMENTAL",
                }
                self.last_alert_timestamp = timestamp
            self.active_alert_track = track.track_id
            self.benign_streak = 0
            return {
                "risk_level": "MEDIUM", "timestamp": timestamp, "status": "EXPERIMENTAL",
                "experimental": True, "objects": [track.track_id],
                "evidence": [f"Tracked {track.label} box grew {growth:.2f}x over {track.observations[-1][0] - track.observations[0][0]:.2f}s", f"Tracked center shifted ({relative_dx:+.3f}, {relative_dy:+.3f}) in normalized image coordinates", "Object center remains in the broad forward-view region"],
                "reason": "Image-plane size growth crossed the configured experimental indicator threshold.",
                "uncertainty": "High. Apparent size can change with motion, camera vibration, or object shape. No calibrated distance or TTC is available.",
                "estimated_time_to_collision_seconds": None, "alert": alert,
            }

        self.benign_streak += 1
        if self.benign_streak >= 2:
            self.active_alert_track = None
        if not vehicles:
            return self._unavailable(timestamp, f"No supported vehicle detection met the {self.minimum_confidence:.2f} confidence threshold; this is not evidence that the road is clear.")
        return {
            "risk_level": "NORMAL", "timestamp": timestamp, "status": "EXPERIMENTAL",
            "experimental": True, "objects": [track.track_id for track in self.tracks if not track.misses],
            "evidence": ["No configured image-plane approach indicator was observed in current vehicle tracks"],
            "reason": "No experimental approach criterion was met; this does not mean the scene is safe.",
            "uncertainty": "High. Monocular detections do not provide calibrated distance, speed, or collision probability.",
            "estimated_time_to_collision_seconds": None, "alert": None,
        }
