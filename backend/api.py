"""Local API and dashboard for the AutoVision AI portfolio demo."""
from __future__ import annotations

import json
import importlib.util
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.risk import CollisionRiskEngine
from backend.driver_monitor import DriverMonitor, create_face_landmarker
from backend.lane_baseline import estimate_lanes

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("AUTOVISION_DATA_DIR", str(ROOT / "data"))).resolve()
MEDIA_DIR = DATA_DIR / "media"
DB_PATH = DATA_DIR / "autovision.sqlite3"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "250")) * 1024 * 1024
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:8001,http://127.0.0.1:8001").split(",") if x.strip()]
VEHICLE_WEIGHTS = Path(os.getenv("VEHICLE_MODEL_PATH", str(ROOT / "yolo26n.pt"))).resolve()
POTHOLE_WEIGHTS = Path(os.getenv("POTHOLE_MODEL_PATH", str(ROOT / "runs/segment/runs/pothole/pothole_seg_test/weights/best.pt"))).resolve()
DRIVER_LANDMARKER_WEIGHTS = Path(os.getenv("DRIVER_LANDMARKER_MODEL_PATH", str(ROOT / "models/face_landmarker.task"))).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA_DIR / "ultralytics"))


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()


def initialize_database() -> None:
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, filename TEXT NOT NULL,
            media_type TEXT NOT NULL, detector TEXT NOT NULL, detections_json TEXT NOT NULL,
            annotated_path TEXT NOT NULL, frame_count INTEGER NOT NULL DEFAULT 1
        )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at DESC)")
        # Expand databases created by earlier demo versions without discarding history.
        existing_columns = {row[1] for row in db.execute("PRAGMA table_info(events)")}
        for column, definition in {
            "sampled_frame_count": "INTEGER",
            "source_frame_count": "INTEGER",
            "processing_time_seconds": "REAL",
            "average_inference_ms": "REAL",
            "model_used": "TEXT",
            "confidence_threshold": "REAL",
            "frame_interval": "INTEGER",
            "completion_status": "TEXT",
            "sampling_rate_hz": "REAL",
            "risk_assessments_json": "TEXT",
            "module_results_json": "TEXT",
            "module_results_json": "TEXT",
        }.items():
            if column not in existing_columns:
                db.execute(f"ALTER TABLE events ADD COLUMN {column} {definition}")


initialize_database()


class EventPage(BaseModel):
    items: list[dict[str, Any]]
    page: int
    page_size: int
    total: int


class SimulationStartRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    detector: Literal["vehicle", "pothole"] = "vehicle"
    profile: Literal["automatic", "road_facing", "driver_facing", "mixed", "general"] = "automatic"
    confidence: float = Field(default=0.35, ge=0.05, le=0.95)
    sampling_rate_hz: float = Field(default=4.0, ge=1.0, le=10.0)
    features: list[Literal["drowsiness", "lane", "collision_risk"]] = Field(default_factory=lambda: ["drowsiness", "lane", "collision_risk"])


class SimulationStartResponse(BaseModel):
    session_id: str
    model_used: str
    model_status: str
    detector: str
    confidence_threshold: float
    sampling_rate_hz: float
    simulation_status: Literal["ready"]
    capability_status: dict[str, str] = Field(default_factory=dict)
    profile: str
    assessment: dict[str, Any] = Field(default_factory=dict)


class SimulationDetection(BaseModel):
    label: str
    class_name: str
    confidence: float
    bbox: list[float]
    timestamp: float
    frame_id: int
    detector_id: str
    processing_latency_ms: float
    model_status: str
    uncertainty: str


class SimulationFrameResponse(BaseModel):
    session_id: str
    timestamp: float
    frame_id: int
    detections: list[SimulationDetection]
    risk_assessment: dict[str, Any]
    model_used: str
    model_status: str
    processing_latency_ms: float
    inference_status: Literal["complete", "delayed"]
    driver_monitoring: dict[str, Any]
    lane_estimate: dict[str, Any]
    pothole_status: dict[str, Any] = Field(default_factory=dict)
    module_status: dict[str, Any] = Field(default_factory=dict)
    profile_assessment: dict[str, Any] = Field(default_factory=dict)


class SimulationFinishRequest(BaseModel):
    frames_presented: int = Field(default=0, ge=0)
    status: Literal["complete", "stopped", "error"]


class SimulationFrameRequest(BaseModel):
    timestamp: float = Field(ge=0, le=180)
    frame_id: int = Field(ge=0)
    width: int = Field(ge=1, le=7680)
    height: int = Field(ge=1, le=4320)


@dataclass
class SimulationSession:
    session_id: str
    filename: str
    detector: str
    confidence: float
    sampling_rate_hz: float
    profile: str = "automatic"
    features: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    last_timestamp: float = -1.0
    last_frame_id: int = -1
    in_flight: bool = False
    frames_processed: int = 0
    sampled_frame_count: int = 0
    detections: list[dict[str, Any]] = field(default_factory=list)
    risk_assessments: list[dict[str, Any]] = field(default_factory=list)
    module_results: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    inference_seconds: float = 0.0
    preview: np.ndarray | None = None
    processing_started: float = field(default_factory=time.perf_counter)
    risk_engine: CollisionRiskEngine = field(default_factory=CollisionRiskEngine)
    driver_monitor: DriverMonitor | None = None
    driver_monitor_error: str | None = None
    pothole_error: str | None = None
    last_pothole_event: float = -10.0
    last_risk_event: float = -10.0
    road_evidence_frames: int = 0
    face_visible_frames: int = 0


app = FastAPI(title="AutoVision AI", version="0.1.0", description="Automotive perception portfolio demo. Not a certified safety system.")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")

_models: dict[str, Any] = {}
_model_lock = threading.RLock()
_model_health: dict[str, dict[str, str]] = {}
_feature_health: dict[str, dict[str, str]] = {}
_sessions: dict[str, SimulationSession] = {}
_sessions_lock = threading.RLock()


def detector_status() -> list[dict[str, Any]]:
    has_yolo = importlib.util.find_spec("ultralytics") is not None
    vehicle_state = _model_health.get("vehicle", {}).get("state")
    pothole_state = _model_health.get("pothole", {}).get("state")
    vehicle_status = ("missing_model" if not VEHICLE_WEIGHTS.is_file() else "unavailable" if not has_yolo or vehicle_state == "unavailable"
                      else "ready" if vehicle_state == "inference_verified" else "available")
    pothole_status = ("missing_model" if not POTHOLE_WEIGHTS.is_file() else "unavailable" if not has_yolo or pothole_state == "unavailable"
                      else "experimental")
    face_path = DRIVER_LANDMARKER_WEIGHTS
    has_mediapipe = importlib.util.find_spec("mediapipe") is not None
    driver_state = _feature_health.get("drowsiness", {}).get("state")
    driver_status = "missing_model" if not face_path.is_file() else "unavailable" if not has_mediapipe or driver_state == "unavailable" else "experimental"
    lane_state = _feature_health.get("lane", {}).get("state", "available")
    risk_state = _feature_health.get("collision_risk", {}).get("state", "available")
    return [
        {"id": "vehicle", "name": "Vehicle & object detection", "status": vehicle_status,
         "runtime_status": vehicle_state or ("dependency_missing" if not has_yolo else "not_inference_verified"),
         "weights": VEHICLE_WEIGHTS.name if VEHICLE_WEIGHTS.is_file() else None,
         "classes": ["car", "motorcycle", "bus", "truck", "person", "cell phone"],
         "limitation": "Runtime Ready is set only after model load and a successful inference. General COCO phone detections do not establish driver use."},
        {"id": "pothole", "name": "Pothole segmentation", "status": pothole_status,
         "runtime_status": pothole_state or ("dependency_missing" if not has_yolo else "configured_not_verified"),
         "weights": POTHOLE_WEIGHTS.name if POTHOLE_WEIGHTS.is_file() else None,
         "classes": ["Pothole"] if POTHOLE_WEIGHTS.is_file() else [],
         "limitation": "Experimental one-class local segmentation model. Inference works, but prior smoke testing found a false positive on a clear-car image."},
        {"id": "drowsiness", "name": "Drowsiness / driver monitoring", "status": driver_status,
         "runtime_status": driver_state or ("configured_not_initialized" if has_mediapipe and face_path.is_file() else "dependency_or_model_missing"),
         "weights": str(face_path.relative_to(ROOT)).replace("\\", "/") if face_path.is_file() else None,
         "classes": ["face visibility", "eye closure indicator"],
         "limitation": "Integrated MediaPipe face-landmark prototype for sampled video only. A road-facing clip often cannot see the driver; EAR thresholds are uncalibrated and not diagnostic."},
        {"id": "seatbelt", "name": "Seatbelt use", "status": "missing_model", "runtime_status": "no_compatible_weights",
         "weights": None, "classes": [], "limitation": "No seatbelt-specific weights or labeled supported camera-angle data are present. COCO object detection cannot infer belt use."},
        {"id": "alcohol", "name": "Alcohol impairment", "status": "not_implemented", "runtime_status": "sensor_integration_required",
         "weights": None, "classes": [], "limitation": "Requires a validated sensor or legitimate external evidence source. Facial appearance is not used to infer impairment or BAC."},
        {"id": "lane", "name": "Lane detection", "status": "unavailable" if lane_state == "unavailable" else "experimental", "runtime_status": lane_state,
         "weights": None, "classes": ["left image-space boundary", "right image-space boundary"],
         "limitation": "OpenCV Canny/Hough baseline, not a trained lane model. Assumes a forward camera and visible painted lines; no steering/calibration claim."},
        {"id": "crash_risk", "name": "Collision-risk indicator", "status": "unavailable" if risk_state == "unavailable" else "experimental", "runtime_status": risk_state,
         "weights": None, "classes": ["potential forward obstacle"],
         "limitation": "IoU-associated image-plane tracks, confidence, apparent size and center movement only. No TTC, crash probability, or safety determination."},
    ]


VIDEO_FEATURES = {"drowsiness", "lane", "collision_risk"}


def parse_video_features(value: str | list[str] | None) -> set[str]:
    if value is None or value == "":
        return set()
    items = value if isinstance(value, list) else value.split(",")
    features = {item.strip() for item in items if item.strip()}
    unknown = features - VIDEO_FEATURES
    if unknown:
        raise HTTPException(422, f"Unsupported video feature(s): {', '.join(sorted(unknown))}")
    return features


def _create_driver_monitor() -> DriverMonitor:
    model_path = DRIVER_LANDMARKER_WEIGHTS
    if not model_path.is_file():
        _feature_health["drowsiness"] = {"state": "missing_model", "detail": model_path.name}
        raise FileNotFoundError("models/face_landmarker.task is missing")
    if importlib.util.find_spec("mediapipe") is None:
        _feature_health["drowsiness"] = {"state": "unavailable", "detail": "mediapipe dependency is not installed"}
        raise RuntimeError("MediaPipe dependency is not installed")
    try:
        monitor = DriverMonitor(create_face_landmarker(model_path))
    except Exception as exc:
        _feature_health["drowsiness"] = {"state": "unavailable", "detail": f"initialization failed: {type(exc).__name__}"}
        raise
    _feature_health["drowsiness"] = {"state": "initialized", "detail": model_path.name}
    return monitor


def _module_unavailable(module: str, timestamp: float, reason: str) -> dict[str, Any]:
    return {"module": module, "status": "UNAVAILABLE", "timestamp": timestamp,
            "experimental": True, "reason": reason}


def get_model(detector: str):
    if detector not in {"vehicle", "pothole"}:
        raise HTTPException(422, f"Unsupported detector: {detector}")
    path = VEHICLE_WEIGHTS if detector == "vehicle" else POTHOLE_WEIGHTS
    if not path.is_file():
        _model_health[detector] = {"state": "missing_model", "detail": str(path.name)}
        raise HTTPException(503, f"Configured {detector} model weights are unavailable")
    if importlib.util.find_spec("ultralytics") is None:
        _model_health[detector] = {"state": "unavailable", "detail": "ultralytics dependency is not installed"}
        raise HTTPException(503, "Ultralytics inference dependency is unavailable")
    with _model_lock:
        if detector not in _models:
            try:
                from ultralytics import YOLO
                model = YOLO(str(path))
                names = model.names.values() if isinstance(model.names, dict) else model.names
                normalized_names = {str(name).lower() for name in names}
                if detector == "vehicle" and not {"car", "truck", "bus", "motorcycle"} <= normalized_names:
                    raise ValueError("configured model is missing expected COCO vehicle classes")
                if detector == "pothole" and (model.task != "segment" or "pothole" not in normalized_names):
                    raise ValueError("configured model must be a segmentation model with a Pothole class")
                _models[detector] = model
                _model_health[detector] = {"state": "loaded", "detail": path.name}
            except Exception as exc:
                _model_health[detector] = {"state": "unavailable", "detail": type(exc).__name__}
                raise HTTPException(503, f"Could not initialize {detector} inference: {type(exc).__name__}") from exc
        return _models[detector]


def run_frame(frame: np.ndarray, detector: str, confidence: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    model = get_model(detector)
    try:
        with _model_lock:
            result = model.predict(frame, conf=confidence, verbose=False)[0]
    except Exception as exc:
        _model_health[detector] = {"state": "unavailable", "detail": f"inference failed: {type(exc).__name__}"}
        raise HTTPException(503, f"{detector} inference failed: {type(exc).__name__}") from exc
    _model_health[detector] = {"state": "inference_verified", "detail": model_name(detector)}
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = str(result.names[cls_id])
            if detector == "vehicle" and label not in {"car", "motorcycle", "bus", "truck", "person", "cell phone"}:
                continue
            coords = [round(float(x), 1) for x in box.xyxy[0].tolist()]
            detections.append({"label": label, "confidence": round(float(box.conf[0]), 4), "bbox": coords})
    return result.plot(), detections


def model_name(detector: str) -> str:
    path = VEHICLE_WEIGHTS if detector == "vehicle" else POTHOLE_WEIGHTS
    return f"{detector} ({path.name})"


def summarize_detections(detections: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(item["label"] for item in detections).items()))


def video_completion_status(processed_frames: int, source_frames: int | None) -> str:
    if source_frames is None or source_frames <= 0:
        return "finished_unverified"
    return "complete" if processed_frames >= source_frames else "partial"


async def read_upload(file: UploadFile, allowed: set[str]) -> tuple[bytes, str]:
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in allowed:
        raise HTTPException(415, f"Unsupported file extension: {ext or '(none)'}")
    mime_by_ext = {
        ".jpg": {"image/jpeg"}, ".jpeg": {"image/jpeg"}, ".png": {"image/png"}, ".webp": {"image/webp"},
        ".mp4": {"video/mp4"}, ".avi": {"video/x-msvideo", "video/avi"}, ".mov": {"video/quicktime"},
        ".mkv": {"video/x-matroska", "application/x-matroska"}, ".webm": {"video/webm"},
    }
    content_type = (file.content_type or "").lower()
    if content_type and content_type != "application/octet-stream" and content_type not in mime_by_ext.get(ext, set()):
        raise HTTPException(415, "Content type does not match the file extension")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload exceeds configured {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    if not content:
        raise HTTPException(400, "Uploaded file is empty")
    return content, name


def save_event(
    filename: str, media_type: str, detector: str, detections: list[dict], annotated: np.ndarray,
    frame_count: int, sampled_frame_count: int | None = None, source_frame_count: int | None = None,
    processing_time_seconds: float | None = None, average_inference_ms: float | None = None,
    confidence_threshold: float | None = None, frame_interval: int | None = None,
    completion_status: str | None = None, sampling_rate_hz: float | None = None,
    risk_assessments: list[dict[str, Any]] | None = None,
    module_results: list[dict[str, Any]] | None = None,
    model_used_override: str | None = None,
) -> dict:
    event_id = str(uuid.uuid4())
    out = MEDIA_DIR / f"{event_id}.jpg"
    if not cv2.imwrite(str(out), annotated):
        raise HTTPException(500, "Could not write annotated preview")
    created_at = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        db.execute(
            """INSERT INTO events (
                id, created_at, filename, media_type, detector, detections_json,
                annotated_path, frame_count, sampled_frame_count, source_frame_count,
                processing_time_seconds, average_inference_ms, model_used,
                confidence_threshold, frame_interval, completion_status,
                sampling_rate_hz, risk_assessments_json, module_results_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, created_at, filename, media_type, detector, json.dumps(detections), out.name,
             frame_count, sampled_frame_count, source_frame_count, processing_time_seconds,
             average_inference_ms, model_used_override or model_name(detector), confidence_threshold, frame_interval, completion_status,
             sampling_rate_hz, json.dumps(risk_assessments or []), json.dumps(module_results or [])),
        )
    return {
        "id": event_id, "created_at": created_at, "filename": filename, "media_type": media_type,
        "detector": detector, "model_used": model_used_override or model_name(detector), "detections": detections,
        "total_detections": len(detections), "detections_by_class": summarize_detections(detections),
        "annotated_url": f"/media/{out.name}", "frame_count": frame_count,
        "frames_processed": frame_count, "frames_presented": frame_count if sampling_rate_hz is not None else None,
        "sampled_frame_count": sampled_frame_count,
        "frames_sampled": sampled_frame_count, "source_frame_count": source_frame_count,
        "processing_time_seconds": processing_time_seconds, "average_inference_ms": average_inference_ms,
        "confidence_threshold": confidence_threshold, "frame_interval": frame_interval,
        "sampling_rate_hz": sampling_rate_hz, "completion_status": completion_status,
        "risk_assessments": risk_assessments or [],
        "module_results": module_results or [],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "database": "sqlite", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/ready")
def ready():
    return {"status": "ready", "available_detectors": [x["id"] for x in detector_status() if x["status"] == "ready"],
            "experimental_capabilities": [x["id"] for x in detector_status() if x["status"] == "experimental"]}


@app.get("/api/detectors")
def detectors():
    return {"items": detector_status(), "notice": "Research/demo tooling only; outputs are not safety advice."}


@app.post("/api/infer/image")
async def infer_image(file: UploadFile = File(...), detector: str = Query("vehicle", pattern="^(vehicle|pothole)$"), confidence: float = Query(0.35, ge=0.05, le=0.95)):
    content, name = await read_upload(file, {".jpg", ".jpeg", ".png", ".webp"})
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "File is not a decodable image")
    started = time.perf_counter()
    annotated, detections = await run_in_threadpool(run_frame, image, detector, confidence)
    elapsed = time.perf_counter() - started
    return save_event(name, "image", detector, detections, annotated, 1, 1, 1, elapsed, elapsed * 1000, confidence, 1, "complete")


@app.post("/api/infer/video")
async def infer_video(file: UploadFile = File(...), detector: str = Query("vehicle", pattern="^(vehicle|pothole)$"), confidence: float = Query(0.35, ge=0.05, le=0.95), frame_interval: int = Query(15, ge=1, le=300), features: str = Query("", max_length=100)):
    selected_features = parse_video_features(features)
    content, name = await read_upload(file, {".mp4", ".avi", ".mov", ".mkv", ".webm"})
    return await run_in_threadpool(_process_video_upload, content, name, detector, confidence, frame_interval, selected_features)


def _process_video_upload(content: bytes, name: str, detector: str, confidence: float, frame_interval: int,
                          features: set[str] | None = None) -> dict[str, Any]:
    features = features or set()
    tmp_path = None
    driver_monitor: DriverMonitor | None = None
    driver_error: str | None = None
    try:
        if "drowsiness" in features:
            try:
                driver_monitor = _create_driver_monitor()
            except Exception as exc:
                driver_error = f"{type(exc).__name__}: {exc}"
        with tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False, dir=DATA_DIR) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            cap.release()
            raise HTTPException(400, "Video could not be opened; check codec and file integrity")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            if not np.isfinite(fps) or fps <= 0:
                fps = 0
            source_frame_value = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            source_frame_count = int(round(source_frame_value)) if source_frame_value > 0 else None
            duration = source_frame_value / fps if fps > 0 else 0
            if duration > 180:
                raise HTTPException(413, "Videos longer than 180 seconds are not supported")
            aggregate, preview, processed, sampled = [], None, 0, 0
            inference_seconds = 0.0
            module_results: list[dict[str, Any]] = []
            risk_assessments: list[dict[str, Any]] = []
            risk_engine = CollisionRiskEngine() if "collision_risk" in features else None
            processing_started = time.perf_counter()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if processed % frame_interval == 0:
                    inference_started = time.perf_counter()
                    preview, found = run_frame(frame, detector, confidence)
                    sampled += 1
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    ts = round(processed / fps, 3) if fps else (round(pos_ms / 1000, 3) if np.isfinite(pos_ms) and pos_ms > 0 else None)
                    if ts is not None:
                        aggregate.extend({**d, "timestamp": ts} for d in found)
                    frame_modules: dict[str, Any] = {"timestamp": ts}
                    if "lane" in features:
                        try:
                            preview, lane_result = estimate_lanes(preview)
                            _feature_health["lane"] = {"state": "inference_verified", "detail": "OpenCV Canny/Hough baseline"}
                            lane_result["timestamp"] = ts
                            frame_modules["lane"] = lane_result
                        except Exception as exc:
                            _feature_health["lane"] = {"state": "unavailable", "detail": type(exc).__name__}
                            frame_modules["lane"] = _module_unavailable("lane", ts or 0.0, f"Lane baseline failed: {type(exc).__name__}")
                    if "drowsiness" in features:
                        if driver_monitor and ts is not None:
                            try:
                                preview, driver_result = driver_monitor.process(preview, ts)
                                _feature_health["drowsiness"] = {"state": "inference_verified", "detail": "MediaPipe Face Landmarker"}
                                frame_modules["drowsiness"] = driver_result
                            except Exception as exc:
                                driver_error = f"{type(exc).__name__}: {exc}"
                                _feature_health["drowsiness"] = {"state": "unavailable", "detail": f"inference failed: {type(exc).__name__}"}
                                try:
                                    driver_monitor.close()
                                except Exception:
                                    pass
                                driver_monitor = None
                        else:
                            frame_modules["drowsiness"] = _module_unavailable("drowsiness", ts or 0.0,
                                driver_error or "Valid media timestamps are required for temporal face analysis.")
                    if risk_engine:
                        if ts is None:
                            risk_result = _module_unavailable("collision_risk", 0.0, "Video timestamps are unavailable.")
                        else:
                            try:
                                traffic_detections = found if detector == "vehicle" else run_frame(frame, "vehicle", confidence)[1]
                                risk_result = risk_engine.update(ts, frame.shape[1], frame.shape[0], traffic_detections,
                                                                 (time.perf_counter() - inference_started) * 1000)
                                _feature_health["collision_risk"] = {"state": "inference_verified", "detail": "Temporal image-plane indicator"}
                            except Exception as exc:
                                risk_result = _module_unavailable("collision_risk", ts, f"Vehicle tracks unavailable: {type(exc).__name__}")
                        risk_assessments.append(risk_result)
                        frame_modules["collision_risk"] = risk_result
                    if len(features):
                        module_results.append(frame_modules)
                        module_results = module_results[-1000:]
                    inference_seconds += time.perf_counter() - inference_started
                processed += 1
            processing_seconds = time.perf_counter() - processing_started
        finally:
            cap.release()
        if processed == 0 or preview is None:
            raise HTTPException(400, "Video contains no readable frames")
        completion = video_completion_status(processed, source_frame_count)
        return save_event(
            name, "video", detector, aggregate, preview, processed, sampled, source_frame_count,
            processing_seconds, inference_seconds * 1000 / sampled if sampled else None,
            confidence, frame_interval, completion, None, risk_assessments, module_results,
        )
    finally:
        if driver_monitor:
            driver_monitor.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _get_session(session_id: str) -> SimulationSession:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Simulation session not found or expired")
    session.last_activity = time.monotonic()
    return session


@app.post("/api/simulation/start", response_model=SimulationStartResponse)
async def simulation_start(request: SimulationStartRequest):
    profile_features = {
        "road_facing": {"lane", "collision_risk"},
        "driver_facing": {"drowsiness"},
        "mixed": {"drowsiness", "lane", "collision_risk"},
        "general": {"drowsiness", "lane", "collision_risk"},
        "automatic": {"drowsiness", "lane", "collision_risk"},
    }
    features = profile_features[request.profile] & set(request.features)
    needs_road = request.profile != "driver_facing"
    if needs_road:
        await run_in_threadpool(get_model, "vehicle")
    try:
        if needs_road:
            await run_in_threadpool(get_model, "pothole")
        pothole_error = None if needs_road else "Not applicable to driver-facing profile"
    except Exception as exc:
        pothole_error = f"{type(exc).__name__}: {getattr(exc, 'detail', str(exc))}"
    driver_monitor, driver_error = None, None
    if "drowsiness" in features:
        try:
            driver_monitor = await run_in_threadpool(_create_driver_monitor)
        except Exception as exc:
            driver_error = f"{type(exc).__name__}: {exc}"
    with _sessions_lock:
        now = time.monotonic()
        for key, value in list(_sessions.items()):
            if now - value.last_activity > 3600:
                del _sessions[key]
        if len(_sessions) >= 32:
            raise HTTPException(503, "Too many active simulation sessions")
        session = SimulationSession(str(uuid.uuid4()), Path(request.filename).name[:255], "vehicle",
                                    request.confidence, request.sampling_rate_hz, profile=request.profile,
                                    features=features, driver_monitor=driver_monitor,
                                    driver_monitor_error=driver_error, pothole_error=pothole_error)
        _sessions[session.session_id] = session
    vehicle_status = "not_applicable" if not needs_road else ("ready" if _model_health.get("vehicle", {}).get("state") == "inference_verified" else "available")
    return SimulationStartResponse(session_id=session.session_id, model_used=model_name("vehicle") if needs_road else "Not run for driver-facing profile",
        model_status="NOT_APPLICABLE" if not needs_road else ("READY" if _model_health.get("vehicle", {}).get("state") == "inference_verified" else "AVAILABLE"),
        detector="vehicle", confidence_threshold=request.confidence,
        sampling_rate_hz=request.sampling_rate_hz, simulation_status="ready",
        capability_status={"vehicle": vehicle_status,
                           "pothole": "not_applicable" if not needs_road else ("experimental" if pothole_error is None else "unavailable"),
                           "drowsiness": "experimental" if driver_monitor else ("unavailable" if "drowsiness" in features else "not_applicable"),
                           "lane": "experimental" if "lane" in features else "not_applicable",
                           "collision_risk": "experimental" if "collision_risk" in features else "not_applicable"},
        profile=request.profile,
        assessment={"status": "AWAITING_FRAMES", "profile": request.profile,
                    "message": "No scene classification model is used. Applicability will be based on observed detector and face-monitor evidence; choose a profile when camera view is known."})


@app.post("/api/simulation/{session_id}/frame", response_model=SimulationFrameResponse)
async def simulation_frame(session_id: str, timestamp: float = Query(ge=0, le=180),
                           frame_id: int = Query(ge=0), width: int = Query(ge=1, le=7680),
                           height: int = Query(ge=1, le=4320), file: UploadFile = File(...)):
    session = _get_session(session_id)
    meta = SimulationFrameRequest(timestamp=timestamp, frame_id=frame_id, width=width, height=height)
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}:
        raise HTTPException(415, "Simulation frames must be JPEG, PNG, or WebP")
    content = await file.read(4 * 1024 * 1024 + 1)
    if not content or len(content) > 4 * 1024 * 1024:
        raise HTTPException(413 if content else 400, "Frame is empty or exceeds 4 MB")
    frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[1] != meta.width or frame.shape[0] != meta.height:
        raise HTTPException(400, "Frame dimensions do not match the supplied metadata")
    with _sessions_lock:
        if session.in_flight:
            raise HTTPException(409, "A frame is already being processed for this session")
        if meta.timestamp <= session.last_timestamp or meta.frame_id <= session.last_frame_id:
            raise HTTPException(409, "Simulation frames must arrive in strictly increasing order")
        session.in_flight = True
    started = time.perf_counter()
    try:
        annotated = frame.copy()
        vehicle_detections: list[dict[str, Any]] = []
        if session.profile != "driver_facing":
            annotated, vehicle_detections = await run_in_threadpool(run_frame, frame, "vehicle", session.confidence)
        vehicle_latency_ms = (time.perf_counter() - started) * 1000
        found = list(vehicle_detections)
        pothole_result: dict[str, Any] = {"status": "UNAVAILABLE", "message": session.pothole_error or "Pothole model not initialized", "detections": []}
        if session.profile == "driver_facing":
            pothole_result = {"status": "NOT_APPLICABLE", "message": "Road-surface model is not run for the driver-facing profile.", "detections": []}
        elif session.pothole_error is None:
            try:
                pothole_overlay, potholes = await run_in_threadpool(run_frame, frame, "pothole", session.confidence)
                # Preserve both sets of model overlays over the same source frame.
                annotated = cv2.addWeighted(annotated, 0.72, pothole_overlay, 0.28, 0)
                for hazard in potholes:
                    hazard["detector_id"] = "pothole"
                    hazard["class_name"] = hazard["label"]
                    hazard["timestamp"] = round(meta.timestamp, 3)
                    hazard["frame_id"] = meta.frame_id
                    hazard["uncertainty"] = "Experimental pothole segmentation; review for false positives."
                    found.append(hazard)
                pothole_result = {"status": "INFERENCE_COMPLETE", "message": "Experimental segmentation output; review visually.", "detections": potholes}
                _model_health["pothole"] = {"state": "inference_verified", "detail": model_name("pothole")}
            except Exception as exc:
                session.pothole_error = f"{type(exc).__name__}: {getattr(exc, 'detail', str(exc))}"
                pothole_result = {"status": "UNAVAILABLE", "message": session.pothole_error, "detections": []}
        lane_result: dict[str, Any] = {"module": "lane_detection", "status": "NOT_APPLICABLE" if "lane" not in session.features else "NOT_REQUESTED", "timestamp": meta.timestamp}
        driver_result: dict[str, Any] = {"module": "driver_monitoring", "status": "NOT_APPLICABLE" if "drowsiness" not in session.features else "NOT_REQUESTED", "timestamp": meta.timestamp}
        module_result: dict[str, Any] = {"timestamp": meta.timestamp}
        if "lane" in session.features:
            try:
                lane_overlay, lane_result = await run_in_threadpool(estimate_lanes, frame.copy())
                if lane_result.get("status") == "ESTIMATED":
                    annotated = cv2.addWeighted(annotated, 0.8, lane_overlay, 0.2, 0)
                _feature_health["lane"] = {"state": "inference_verified", "detail": "OpenCV Canny/Hough baseline"}
                lane_result["timestamp"] = meta.timestamp
                module_result["lane"] = lane_result
            except Exception as exc:
                _feature_health["lane"] = {"state": "unavailable", "detail": type(exc).__name__}
                lane_result = _module_unavailable("lane", meta.timestamp, f"Lane baseline failed: {type(exc).__name__}")
                module_result["lane"] = lane_result
        if "drowsiness" in session.features:
            if session.driver_monitor:
                try:
                    driver_overlay, driver_result = await run_in_threadpool(session.driver_monitor.process, frame.copy(), meta.timestamp)
                    if driver_overlay is not None:
                        annotated = cv2.addWeighted(annotated, 0.8, driver_overlay, 0.2, 0)
                    _feature_health["drowsiness"] = {"state": "inference_verified", "detail": "MediaPipe Face Landmarker"}
                except Exception as exc:
                    session.driver_monitor_error = f"{type(exc).__name__}: {exc}"
                    _feature_health["drowsiness"] = {"state": "unavailable", "detail": f"inference failed: {type(exc).__name__}"}
                    try:
                        session.driver_monitor.close()
                    except Exception:
                        pass
                    session.driver_monitor = None
                    driver_result = _module_unavailable("driver_monitoring", meta.timestamp, session.driver_monitor_error)
            else:
                driver_result = _module_unavailable("driver_monitoring", meta.timestamp,
                    session.driver_monitor_error or "MediaPipe Face Landmarker is unavailable.")
            module_result["drowsiness"] = driver_result
        total_latency_ms = (time.perf_counter() - started) * 1000
        total_latency_ms = (time.perf_counter() - started) * 1000
        module_result["pothole"] = {"status": pothole_result["status"],
                                    "detection_count": len(pothole_result["detections"]),
                                    "timestamp": meta.timestamp,
                                    "message": pothole_result["message"]}
        for detection in found:
            detection.update(timestamp=round(meta.timestamp, 3), frame_id=meta.frame_id,
                             detector_id=detection.get("detector_id", "vehicle"), processing_latency_ms=round(total_latency_ms, 2),
                             model_status="READY" if detection.get("detector_id", "vehicle") == "vehicle" else "EXPERIMENTAL",
                             uncertainty=detection.get("uncertainty", "Object detection only; not confirmation of driver behavior or safety state."))
            detection["class_name"] = detection["label"]
        if "collision_risk" in session.features:
            risk_detections = vehicle_detections
            assessment = None
            if session.detector != "vehicle":
                try:
                    vehicle_started = time.perf_counter()
                    _, risk_detections = await run_in_threadpool(run_frame, frame, "vehicle", session.confidence)
                    vehicle_latency_ms += (time.perf_counter() - vehicle_started) * 1000
                except HTTPException as exc:
                    assessment = _module_unavailable("collision_risk", meta.timestamp, f"Vehicle model unavailable: {exc.detail}")
                    risk_detections = []
            if assessment is None:
                assessment = session.risk_engine.update(meta.timestamp, meta.width, meta.height, risk_detections, vehicle_latency_ms)
                _feature_health["collision_risk"] = {"state": "inference_verified", "detail": "Temporal image-plane indicator"}
            module_result["collision_risk"] = assessment
        else:
            assessment = {"status": "NOT_REQUESTED", "timestamp": meta.timestamp, "alert": None,
                          "reason": "Enable the collision-risk indicator to run this experimental analysis."}
        with _sessions_lock:
            session.last_timestamp, session.last_frame_id = meta.timestamp, meta.frame_id
            session.frames_processed = max(session.frames_processed, meta.frame_id + 1)
            session.sampled_frame_count += 1
            session.inference_seconds += total_latency_ms / 1000
            session.detections.extend(found)
            session.detections = session.detections[-10000:]
            if assessment.get("status") != "NOT_REQUESTED":
                session.risk_assessments.append(assessment)
            session.risk_assessments = session.risk_assessments[-1000:]
            if session.features:
                session.module_results.append(module_result)
                session.module_results = session.module_results[-1000:]
            if assessment.get("alert"):
                if meta.timestamp - session.last_risk_event >= 5.0:
                    session.alerts.append(assessment)
                    session.last_risk_event = meta.timestamp
                    session.alerts = session.alerts[-200:]
            if pothole_result["detections"] and meta.timestamp - session.last_pothole_event >= 3.0:
                module_result["pothole_event"] = {"timestamp": meta.timestamp, "count": len(pothole_result["detections"]), "status": "EXPERIMENTAL"}
                session.last_pothole_event = meta.timestamp
            session.preview = annotated
            road_vehicle_classes = {"car", "motorcycle", "bus", "truck", "bicycle"}
            if any(str(item.get("label", "")).lower() in road_vehicle_classes for item in vehicle_detections) or lane_result.get("status") == "ESTIMATED":
                session.road_evidence_frames += 1
            if driver_result.get("face_count", 0) == 1:
                session.face_visible_frames += 1
        profile_assessment = {
            "profile": session.profile,
            "status": "OBSERVATIONAL",
            "road_scene": "cues_observed" if session.road_evidence_frames else "not_established",
            "road_evidence_frames": session.road_evidence_frames,
            "driver_face": "observed" if session.face_visible_frames else ("not_visible_in_samples" if "drowsiness" in session.features else "not_assessed"),
            "driver_face_visible_samples": session.face_visible_frames,
            "sampled_frames": session.sampled_frame_count,
            "message": "These are detector/landmark observations, not a camera-view classifier. Lack of a cue does not prove the scene or person is absent."
        }
        module_result["profile"] = profile_assessment
        return SimulationFrameResponse(session_id=session_id, timestamp=meta.timestamp, frame_id=meta.frame_id,
            detections=found, risk_assessment=assessment, model_used=model_name(session.detector) if session.profile != "driver_facing" else "No road-scene model run", model_status="READY" if session.profile != "driver_facing" else "NOT_APPLICABLE",
            processing_latency_ms=round(total_latency_ms, 2),
            inference_status="delayed" if total_latency_ms > 1000 / session.sampling_rate_hz else "complete",
            driver_monitoring=driver_result, lane_estimate=lane_result, pothole_status=pothole_result,
            module_status={"vehicle": "not_applicable" if session.profile == "driver_facing" else "ready", "pothole": pothole_result["status"],
                           "drowsiness": driver_result.get("status", "NOT_REQUESTED"),
                           "lane": lane_result.get("status", "NOT_REQUESTED"),
                           "collision_risk": assessment.get("status", "NOT_REQUESTED")},
            profile_assessment=profile_assessment)
    finally:
        with _sessions_lock:
            session.in_flight = False


@app.post("/api/simulation/{session_id}/finish")
async def simulation_finish(session_id: str, request: SimulationFinishRequest):
    session = _get_session(session_id)
    with _sessions_lock:
        if session.in_flight:
            raise HTTPException(409, "Wait for current frame inference to finish")
        _sessions.pop(session_id, None)
    if session.driver_monitor:
        await run_in_threadpool(session.driver_monitor.close)
    if not session.sampled_frame_count or session.preview is None:
        return {"persisted": False, "status": request.status, "frames_processed": request.frames_presented,
                "frames_sampled": 0, "total_detections": 0, "detections_by_class": {},
                "message": "No sampled inference frame was available to persist."}
    elapsed = max(0, time.perf_counter() - session.processing_started)
    if session.profile == "driver_facing":
        session_models = "MediaPipe Face Landmarker" if session.driver_monitor_error is None else "Driver monitor unavailable"
    else:
        session_models = f"{model_name('vehicle')} + {model_name('pothole')}" if session.pothole_error is None else f"{model_name('vehicle')} + pothole unavailable"
    result = await run_in_threadpool(save_event, session.filename, "video", session.detector,
        session.detections, session.preview, request.frames_presented, session.sampled_frame_count, None,
        elapsed, session.inference_seconds * 1000 / session.sampled_frame_count,
        session.confidence, None, request.status, session.sampling_rate_hz, session.risk_assessments,
        session.module_results, session_models)
    result.update({"persisted": True, "alerts": session.alerts, "frames_presented": request.frames_presented,
                   "risk_assessments": session.risk_assessments})
    return result


@app.post("/api/simulation/{session_id}/discard")
def simulation_discard(session_id: str):
    with _sessions_lock:
        if session_id not in _sessions:
            return {"discarded": False}
        if _sessions[session_id].in_flight:
            raise HTTPException(409, "Wait for current frame inference to finish")
        session = _sessions.pop(session_id)
    if session.driver_monitor:
        session.driver_monitor.close()
    return {"discarded": True}


@app.get("/api/events", response_model=EventPage)
def events(page: int = Query(1, ge=1), page_size: int = Query(12, ge=1, le=100), search: str = Query("", max_length=100), detector: str | None = None):
    where, args = [], []
    if search:
        where.append("filename LIKE ?")
        args.append(f"%{search}%")
    if detector:
        where.append("detector = ?")
        args.append(detector)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    with connect() as db:
        total = db.execute("SELECT COUNT(*) FROM events" + clause, args).fetchone()[0]
        rows = db.execute("SELECT * FROM events" + clause + " ORDER BY created_at DESC LIMIT ? OFFSET ?", [*args, page_size, (page - 1) * page_size]).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        detections = json.loads(item["detections_json"])
        risk_assessments = json.loads(item.get("risk_assessments_json") or "[]")
        module_results = json.loads(item.get("module_results_json") or "[]")
        item.update({
            "detections": detections,
            "risk_assessments": risk_assessments,
            "module_results": module_results,
            "total_detections": len(detections),
            "detections_by_class": summarize_detections(detections),
            "frames_processed": item["frame_count"],
            "frames_presented": item["frame_count"] if item["sampling_rate_hz"] is not None else None,
            "frames_sampled": item["sampled_frame_count"],
            "model_used": item["model_used"],
            "completion_status": item["completion_status"] or ("finished_unverified" if item["media_type"] == "video" else "complete"),
            "detection_timestamps": len({d["timestamp"] for d in detections if "timestamp" in d}),
            "annotated_url": f"/media/{item['annotated_path']}",
        })
        # Existing rows predate summary metadata and must stay explicitly unknown.
        if item["media_type"] == "video" and item["sampled_frame_count"] is None:
            item["frames_sampled_note"] = "Not recorded for this run"
        items.append(item)
    for item in items:
        item.pop("detections_json")
        item.pop("annotated_path")
    return {"items": items, "page": page, "page_size": page_size, "total": total}


@app.get("/api/analytics")
def analytics():
    with connect() as db:
        total = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        kinds = db.execute("SELECT detector, COUNT(*) n FROM events GROUP BY detector").fetchall()
        detected = db.execute("SELECT COUNT(*) FROM events WHERE detections_json != '[]'").fetchone()[0]
    return {"total_analyses": total, "events_with_detections": detected, "by_detector": {row["detector"]: row["n"] for row in kinds}}


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "frontend" / "index.html")
