from types import SimpleNamespace

import cv2
import numpy as np

from backend.driver_monitor import DriverMonitor
from backend.lane_baseline import estimate_lanes


class FakeLandmarker:
    def __init__(self, results):
        self.results = iter(results)
        self.timestamps = []

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return next(self.results)

    def close(self):
        pass


def make_face(ear):
    landmarks = [SimpleNamespace(x=.5, y=.5) for _ in range(478)]
    for indices, offset in (((33,160,158,133,153,144), .35), ((362,385,387,263,373,380), .60)):
        points = ((offset,.4),(offset+.04,.4),(offset+.04,.4+ear*.1),(offset+.1,.4),
                  (offset+.04,.4-ear*.1),(offset+.04,.4))
        for index, (x,y) in zip(indices,points):
            landmarks[index]=SimpleNamespace(x=x,y=y)
    return landmarks


def test_driver_monitor_uses_timestamped_face_landmarks_and_temporal_window():
    face = make_face(.05)
    landmark = FakeLandmarker([
        SimpleNamespace(face_landmarks=[face]),
        SimpleNamespace(face_landmarks=[face]),
        SimpleNamespace(face_landmarks=[face]),
    ])
    monitor = DriverMonitor(landmark)
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    _, first = monitor.process(frame.copy(), 0.0)
    _, second = monitor.process(frame.copy(), .8)
    _, third = monitor.process(frame.copy(), 1.6)
    assert first["indicator"] == "EYES_CLOSED"
    assert second["indicator"] == "EYES_CLOSED"
    assert third["indicator"] == "POSSIBLE_PROLONGED_EYE_CLOSURE"
    assert landmark.timestamps == [0, 800, 1600]


def test_driver_monitor_reports_missing_or_ambiguous_faces():
    landmark = FakeLandmarker([
        SimpleNamespace(face_landmarks=[]),
        SimpleNamespace(face_landmarks=[make_face(.05), make_face(.05)]),
    ])
    monitor = DriverMonitor(landmark)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    assert monitor.process(frame.copy(), 0)[1]["status"] == "FACE_NOT_VISIBLE"
    assert monitor.process(frame.copy(), .1)[1]["status"] == "AMBIGUOUS"


def test_lane_baseline_returns_unavailable_on_blank_scene_and_estimates_synthetic_lines():
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    _, unavailable = estimate_lanes(blank)
    assert unavailable["status"] == "UNAVAILABLE"

    road = blank.copy()
    cv2.line(road, (40, 239), (135, 140), (255,255,255), 5)
    cv2.line(road, (280, 239), (185, 140), (255,255,255), 5)
    _, result = estimate_lanes(road)
    assert result["status"] == "ESTIMATED"
    assert {lane["side"] for lane in result["lanes"]} == {"left", "right"}

