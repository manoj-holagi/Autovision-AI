"""Experimental MediaPipe facial-landmark indicators for prerecorded video."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


def _distance(a: Any, b: Any) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def eye_aspect_ratio(landmarks: Any, indices: tuple[int, ...]) -> float:
    p = [landmarks[index] for index in indices]
    horizontal = _distance(p[0], p[3])
    return ((_distance(p[1], p[5]) + _distance(p[2], p[4])) / (2 * horizontal)) if horizontal else 0.0


def create_face_landmarker(model_path: str | Path):
    """Load the repository's local Face Landmarker model; never downloads weights."""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=4,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


class DriverMonitor:
    """Tracks whether one visible face has sustained low eye aspect ratio.

    This is an uncalibrated indicator. It does not diagnose impairment and is
    unavailable if the driver's face is not distinctly visible in the camera.
    """

    def __init__(self, landmarker: Any, ear_threshold: float = 0.21, closure_seconds: float = 1.5):
        self.landmarker = landmarker
        self.ear_threshold = ear_threshold
        self.closure_seconds = closure_seconds
        self.closed_since: float | None = None
        self.last_timestamp_ms = -1
        self.last_timestamp_seconds: float | None = None

    def process(self, frame: np.ndarray, timestamp_seconds: float) -> tuple[np.ndarray, dict[str, Any]]:
        import mediapipe as mp

        timestamp_ms = max(self.last_timestamp_ms + 1, round(timestamp_seconds * 1000))
        self.last_timestamp_ms = timestamp_ms
        if self.last_timestamp_seconds is not None and timestamp_seconds - self.last_timestamp_seconds > 1.5:
            self.closed_since = None
        self.last_timestamp_seconds = timestamp_seconds
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        faces = result.face_landmarks or []
        info: dict[str, Any] = {
            "module": "driver_monitoring", "status": "FACE_NOT_VISIBLE" if not faces else "AMBIGUOUS",
            "timestamp": timestamp_seconds, "face_count": len(faces), "ear": None,
            "indicator": None, "experimental": True,
            "limitation": "Uncalibrated eye-closure indicator; not a diagnosis. A road-facing camera may not see the driver.",
        }
        if len(faces) != 1:
            self.closed_since = None
            label = "FACE NOT VISIBLE" if not faces else "MULTIPLE FACES — AMBIGUOUS"
        else:
            landmarks = faces[0]
            ear = (eye_aspect_ratio(landmarks, LEFT_EYE) + eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2
            info["ear"] = round(ear, 4)
            if ear < self.ear_threshold:
                if self.closed_since is None:
                    self.closed_since = timestamp_seconds
                closed_duration = timestamp_seconds - self.closed_since
                indicator = "POSSIBLE_PROLONGED_EYE_CLOSURE" if closed_duration >= self.closure_seconds else "EYES_CLOSED"
                info.update(status="EXPERIMENTAL", indicator=indicator,
                            closure_duration_seconds=round(closed_duration, 3))
                label = indicator.replace("_", " ")
            else:
                self.closed_since = None
                info.update(status="EXPERIMENTAL", indicator="FACE_VISIBLE")
                label = "FACE VISIBLE · EAR %.2f" % ear
        cv2.putText(frame, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (30, 210, 240) if "POSSIBLE" in label else (180, 235, 210), 2, cv2.LINE_AA)
        return frame, info

    def close(self) -> None:
        self.landmarker.close()
