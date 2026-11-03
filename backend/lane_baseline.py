"""Experimental OpenCV lane-line baseline for forward-facing road footage."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def estimate_lanes(frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate two image-space lane boundaries from edges and Hough segments.

    Assumes a roughly centered forward camera and visible painted boundaries.
    Returns UNAVAILABLE rather than extrapolating when evidence is weak.
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (int(width * .38), int(height * .57)),
                         (int(width * .62), int(height * .57)), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi_edges = cv2.bitwise_and(edges, mask)
    segments = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=35,
                               minLineLength=max(20, width // 18), maxLineGap=max(15, width // 30))
    groups: dict[str, list[tuple[float, float, float]]] = {"left": [], "right": []}
    if segments is not None:
        for segment in segments[:, 0, :]:
            x1, y1, x2, y2 = (int(x) for x in segment)
            dx = x2 - x1
            if abs(dx) < 2:
                continue
            slope = (y2 - y1) / dx
            midpoint = (x1 + x2) / 2
            side = "left" if slope < 0 and midpoint < width * .55 else "right" if slope > 0 and midpoint > width * .45 else None
            if side:
                length = float(np.hypot(dx, y2 - y1))
                groups[side].extend(((x1, y1, length), (x2, y2, length)))

    annotated = frame.copy()
    lanes: list[dict[str, Any]] = []
    y_bottom, y_top = height - 1, int(height * .60)
    for side, points in groups.items():
        if len(points) < 4:
            continue
        xs = np.asarray([p[0] for p in points], dtype=np.float64)
        ys = np.asarray([p[1] for p in points], dtype=np.float64)
        weights = np.asarray([p[2] for p in points], dtype=np.float64)
        try:
            slope, intercept = np.polyfit(ys, xs, 1, w=weights)
        except (ValueError, np.linalg.LinAlgError):
            continue
        x_bottom, x_top = int(slope * y_bottom + intercept), int(slope * y_top + intercept)
        if not (0 <= x_bottom < width and 0 <= x_top < width):
            continue
        confidence = min(0.95, float(np.sum(weights) / (height * 1.4)))
        if confidence < 0.12:
            continue
        lanes.append({"side": side, "points": [[x_bottom, y_bottom], [x_top, y_top]],
                      "confidence": round(confidence, 3)})
    left = next((lane for lane in lanes if lane["side"] == "left"), None)
    right = next((lane for lane in lanes if lane["side"] == "right"), None)
    had_pair = left is not None and right is not None
    geometry_valid = False
    if left and right:
        left_bottom, left_top = left["points"]
        right_bottom, right_top = right["points"]
        geometry_valid = (left_bottom[0] < right_bottom[0]
                          and left_top[0] + width * .05 < right_top[0]
                          and right_bottom[0] - left_bottom[0] > width * .18)
    if not geometry_valid and len(lanes) == 2:
        lanes = []
    for lane in lanes:
        (x1, y1), (x2, y2) = lane["points"]
        cv2.line(annotated, (x1, y1), (x2, y2), (40, 220, 235), 4, cv2.LINE_AA)
    status = "ESTIMATED" if geometry_valid else "PARTIAL" if lanes else "UNAVAILABLE"
    info = {
        "module": "lane_detection", "status": status, "timestamp": None,
        "lanes": lanes, "experimental": True,
        "limitation": "OpenCV edge/Hough baseline only; assumes a forward camera and visible painted lane boundaries. Not calibrated for steering.",
    }
    if had_pair and not geometry_valid:
        info["reason"] = "The left/right estimates cross or have implausible image-space width."
    if not lanes:
        cv2.putText(annotated, "LANE ESTIMATE UNAVAILABLE", (14, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (40, 180, 240), 2, cv2.LINE_AA)
    return annotated, info
