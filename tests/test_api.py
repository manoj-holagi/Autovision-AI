from fastapi.testclient import TestClient
import sqlite3

from backend.api import app

client = TestClient(app)


def test_health_and_detector_capabilities():
    assert client.get("/api/health").json()["status"] == "ok"
    items = client.get("/api/detectors").json()["items"]
    assert next(x for x in items if x["id"] == "alcohol")["status"] == "not_implemented"
    assert next(x for x in items if x["id"] == "seatbelt")["status"] == "missing_model"
    assert next(x for x in items if x["id"] == "lane")["status"] == "experimental"
    assert next(x for x in items if x["id"] == "crash_risk")["status"] == "experimental"


def test_upload_rejects_unsupported_extension():
    response = client.post("/api/infer/image", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert response.status_code == 415


def test_upload_rejects_mismatched_mime():
    response = client.post("/api/infer/image", files={"file": ("bad.jpg", b"not an image", "text/plain")})
    assert response.status_code == 415


def test_upload_enforces_configured_byte_limit(monkeypatch):
    import backend.api as api
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    response = client.post("/api/infer/image", files={"file": ("large.png", b"12345", "image/png")})
    assert response.status_code == 413


def test_upload_rejects_corrupt_image():
    response = client.post("/api/infer/image", files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400


def test_upload_rejects_missing_model(tmp_path, monkeypatch):
    import backend.api as api
    import cv2
    import numpy as np
    monkeypatch.setattr(api, "VEHICLE_WEIGHTS", tmp_path / "missing.pt")
    ok, buf = cv2.imencode(".png", np.zeros((12, 12, 3), dtype=np.uint8))
    assert ok
    response = client.post("/api/infer/image", files={"file": ("valid.png", buf.tobytes(), "image/png")})
    assert response.status_code == 503


def test_events_and_analytics_shape():
    assert client.get("/api/events").status_code == 200
    assert "total_analyses" in client.get("/api/analytics").json()


def test_dashboard_and_openapi_are_served():
    page = client.get("/").text
    assert "AI Video Simulation" in page
    assert 'id="sim-profile"' in page
    assert "Run analysis" not in page
    assert 'id="sim-start"' in page
    assert client.get("/openapi.json").status_code == 200


def test_completion_status_distinguishes_partial_from_unverified():
    from backend.api import video_completion_status
    assert video_completion_status(714, 771) == "partial"
    assert video_completion_status(771, 771) == "complete"
    assert video_completion_status(714, None) == "finished_unverified"


def test_legacy_event_database_migrates_without_losing_rows(tmp_path, monkeypatch):
    import backend.api as api
    legacy_db = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(legacy_db) as db:
        db.execute("""CREATE TABLE events (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, filename TEXT NOT NULL,
            media_type TEXT NOT NULL, detector TEXT NOT NULL, detections_json TEXT NOT NULL,
            annotated_path TEXT NOT NULL, frame_count INTEGER NOT NULL DEFAULT 1
        )""")
        db.execute("INSERT INTO events VALUES ('legacy', 'now', 'clip.mp4', 'video', 'vehicle', '[]', 'x.jpg', 714)")
    monkeypatch.setattr(api, "DB_PATH", legacy_db)
    api.initialize_database()
    with sqlite3.connect(legacy_db) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(events)")}
        assert {"sampled_frame_count", "source_frame_count", "processing_time_seconds", "average_inference_ms", "model_used", "confidence_threshold", "frame_interval", "completion_status", "sampling_rate_hz", "risk_assessments_json", "module_results_json"} <= columns
        assert db.execute("SELECT frame_count FROM events WHERE id='legacy'").fetchone()[0] == 714
        assert db.execute("SELECT sampled_frame_count FROM events WHERE id='legacy'").fetchone()[0] is None


def test_video_summary_fields_round_trip_through_events_api(tmp_path, monkeypatch):
    import backend.api as api
    import cv2
    import numpy as np
    db_path = tmp_path / "events.sqlite3"
    media_path = tmp_path / "media"
    media_path.mkdir()
    with sqlite3.connect(db_path) as db:
        db.execute("""CREATE TABLE events (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, filename TEXT NOT NULL,
            media_type TEXT NOT NULL, detector TEXT NOT NULL, detections_json TEXT NOT NULL,
            annotated_path TEXT NOT NULL, frame_count INTEGER NOT NULL DEFAULT 1,
            sampled_frame_count INTEGER, source_frame_count INTEGER, processing_time_seconds REAL,
            average_inference_ms REAL, model_used TEXT, confidence_threshold REAL,
            frame_interval INTEGER, completion_status TEXT
        )""")
    monkeypatch.setattr(api, "DB_PATH", db_path)
    monkeypatch.setattr(api, "MEDIA_DIR", media_path)
    monkeypatch.setattr(api, "VEHICLE_WEIGHTS", tmp_path / "yolo26n.pt")
    api.initialize_database()
    detections = [
        {"label": "car", "confidence": 0.8, "bbox": [0, 0, 1, 1], "timestamp": 0.0},
        {"label": "car", "confidence": 0.7, "bbox": [0, 0, 1, 1], "timestamp": 0.6},
        {"label": "truck", "confidence": 0.6, "bbox": [0, 0, 1, 1], "timestamp": 0.6},
    ]
    api.save_event("clip.mp4", "video", "vehicle", detections, np.zeros((8, 8, 3), dtype=np.uint8), 771, 52, 771, 2.5, 30.0, 0.35, 15, "complete")
    item = client.get("/api/events").json()["items"][0]
    assert item["frames_processed"] == 771
    assert item["frames_sampled"] == 52
    assert item["total_detections"] == 3
    assert item["detections_by_class"] == {"car": 2, "truck": 1}
    assert item["processing_time_seconds"] == 2.5
    assert item["average_inference_ms"] == 30.0
    assert item["model_used"] == "vehicle (yolo26n.pt)"
    assert item["completion_status"] == "complete"


def test_simulation_frame_is_real_inference_and_persists_summary(tmp_path, monkeypatch):
    import backend.api as api
    import cv2
    import numpy as np
    db_path, media_path = tmp_path / "simulation.sqlite3", tmp_path / "media"
    media_path.mkdir()
    monkeypatch.setattr(api, "DB_PATH", db_path)
    monkeypatch.setattr(api, "MEDIA_DIR", media_path)
    monkeypatch.setattr(api, "VEHICLE_WEIGHTS", tmp_path / "yolo26n.pt")
    api.initialize_database()
    monkeypatch.setattr(api, "get_model", lambda detector: object())
    def fake_inference(frame, detector, confidence):
        detections = [{"label": "car", "confidence": .8, "bbox": [2, 2, 9, 9]}] if detector == "vehicle" else []
        return frame.copy(), detections
    monkeypatch.setattr(api, "run_frame", fake_inference)
    start = client.post("/api/simulation/start", json={"filename": "dash.mp4", "detector": "vehicle", "confidence": .35, "sampling_rate_hz": 4, "features":["lane"]})
    assert start.status_code == 200, start.text
    session_id = start.json()["session_id"]
    ok, encoded = cv2.imencode(".jpg", np.zeros((12, 16, 3), dtype=np.uint8))
    assert ok
    frame = client.post(f"/api/simulation/{session_id}/frame?timestamp=0.25&frame_id=1&width=16&height=12",
                        files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")})
    assert frame.status_code == 200, frame.text
    assert frame.json()["detections"][0]["timestamp"] == .25
    assert frame.json()["model_status"] == "READY"
    assert frame.json()["lane_estimate"]["status"] == "UNAVAILABLE"
    assert frame.json()["risk_assessment"]["status"] == "NOT_REQUESTED"
    assert frame.json()["pothole_status"]["status"] == "INFERENCE_COMPLETE"
    assert frame.json()["processing_latency_ms"] >= 0
    finish = client.post(f"/api/simulation/{session_id}/finish", json={"frames_presented": 4, "status": "stopped"})
    assert finish.status_code == 200, finish.text
    assert finish.json()["persisted"] is True
    assert finish.json()["completion_status"] == "stopped"
    assert finish.json()["frames_sampled"] == 1
    assert finish.json()["frames_presented"] == 4
    assert finish.json()["total_detections"] == 1
    assert "vehicle (yolo26n.pt) + pothole (best.pt)" == finish.json()["model_used"]
    persisted = client.get("/api/events").json()["items"][0]
    assert persisted["filename"] == "dash.mp4"
    assert persisted["model_used"] == finish.json()["model_used"]
    assert persisted["frames_presented"] == 4
    assert persisted["module_results"][0]["lane"]["status"] == "UNAVAILABLE"


def test_simulation_requires_increasing_frames(tmp_path, monkeypatch):
    import backend.api as api
    import cv2
    import numpy as np
    monkeypatch.setattr(api, "get_model", lambda detector: object())
    monkeypatch.setattr(api, "run_frame", lambda frame, detector, confidence: (frame.copy(), []))
    start = client.post("/api/simulation/start", json={"filename": "dash.mp4"}).json()
    ok, encoded = cv2.imencode(".jpg", np.zeros((12, 16, 3), dtype=np.uint8))
    assert ok
    files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")}
    first=client.post(f"/api/simulation/{start['session_id']}/frame?timestamp=0.25&frame_id=1&width=16&height=12", files=files)
    repeated=client.post(f"/api/simulation/{start['session_id']}/frame?timestamp=0.25&frame_id=2&width=16&height=12", files=files)
    assert first.status_code == 200
    assert repeated.status_code == 409


def test_driver_facing_profile_skips_road_models_and_marks_road_features_not_applicable(monkeypatch):
    import cv2
    import numpy as np
    import backend.api as api

    class Monitor:
        def process(self, frame, timestamp):
            return frame, {"module": "driver_monitoring", "status": "FACE_NOT_VISIBLE", "face_count": 0,
                           "timestamp": timestamp, "indicator": None}
        def close(self):
            pass

    def road_model_must_not_load(*args, **kwargs):
        raise AssertionError("driver-facing profile should not initialize a road model")

    monkeypatch.setattr(api, "get_model", road_model_must_not_load)
    monkeypatch.setattr(api, "_create_driver_monitor", lambda: Monitor())
    monkeypatch.setattr(api, "run_frame", road_model_must_not_load)
    started = client.post("/api/simulation/start", json={"filename": "cabin.mp4", "profile": "driver_facing"})
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["capability_status"]["vehicle"] == "not_applicable"
    assert body["capability_status"]["pothole"] == "not_applicable"
    assert body["capability_status"]["lane"] == "not_applicable"
    assert body["model_status"] == "NOT_APPLICABLE"

    ok, encoded = cv2.imencode(".jpg", np.zeros((12, 16, 3), dtype=np.uint8))
    assert ok
    frame = client.post(f"/api/simulation/{body['session_id']}/frame?timestamp=0.25&frame_id=1&width=16&height=12",
                        files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")})
    assert frame.status_code == 200, frame.text
    result = frame.json()
    assert result["detections"] == []
    assert result["pothole_status"]["status"] == "NOT_APPLICABLE"
    assert result["module_status"]["vehicle"] == "not_applicable"
    assert result["profile_assessment"]["driver_face"] == "not_visible_in_samples"
    client.post(f"/api/simulation/{body['session_id']}/discard")


def test_vehicle_registry_does_not_mark_unverified_file_ready(tmp_path, monkeypatch):
    import backend.api as api
    monkeypatch.setattr(api, "VEHICLE_WEIGHTS", tmp_path / "vehicle.pt")
    api.VEHICLE_WEIGHTS.touch()
    monkeypatch.delitem(api._models, "vehicle", raising=False)
    monkeypatch.delitem(api._model_health, "vehicle", raising=False)
    entry = next(item for item in api.detector_status() if item["id"] == "vehicle")
    assert entry["status"] == "available"
    assert entry["runtime_status"] != "inference_verified"


def test_simulation_keeps_vehicle_results_when_pothole_inference_fails(monkeypatch):
    import cv2
    import numpy as np
    import backend.api as api
    monkeypatch.setattr(api, "get_model", lambda detector: object())
    def fake_inference(frame, detector, confidence):
        if detector == "pothole":
            raise api.HTTPException(503, "simulated missing/incompatible pothole runtime")
        return frame.copy(), [{"label": "car", "confidence": .8, "bbox": [1, 1, 8, 8]}]
    monkeypatch.setattr(api, "run_frame", fake_inference)
    started=client.post("/api/simulation/start",json={"filename":"dash.mp4","features":["lane"]})
    assert started.status_code == 200
    session_id=started.json()["session_id"]
    ok, encoded=cv2.imencode(".jpg",np.zeros((12,16,3),dtype=np.uint8))
    assert ok
    response=client.post(f"/api/simulation/{session_id}/frame?timestamp=0.2&frame_id=1&width=16&height=12",files={"file":("frame.jpg",encoded.tobytes(),"image/jpeg")})
    assert response.status_code == 200, response.text
    body=response.json()
    assert [item["label"] for item in body["detections"]] == ["car"]
    assert body["pothole_status"]["status"] == "UNAVAILABLE"
    assert body["module_status"]["vehicle"] == "ready"
    assert client.post(f"/api/simulation/{session_id}/discard").json()["discarded"]


def test_model_loading_checks_task_and_class_names(tmp_path, monkeypatch):
    import sys
    import types
    import backend.api as api
    weights = tmp_path / "bad.pt"
    weights.touch()
    class WrongModel:
        task = "classify"
        names = {0: "road"}
    module = types.ModuleType("ultralytics")
    module.YOLO = lambda path: WrongModel()
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(api.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(api, "VEHICLE_WEIGHTS", weights)
    monkeypatch.delitem(api._models, "vehicle", raising=False)
    monkeypatch.delitem(api._model_health, "vehicle", raising=False)
    try:
        api.get_model("vehicle")
        assert False, "wrong model classes should be rejected"
    except api.HTTPException as exc:
        assert exc.status_code == 503
    assert api._model_health["vehicle"]["state"] == "unavailable"


def test_model_loading_accepts_compatible_classes_but_ready_follows_successful_inference(tmp_path, monkeypatch):
    import sys
    import types
    import backend.api as api
    import numpy as np
    class EmptyBoxes:
        def __iter__(self): return iter(())
    class Result:
        boxes = EmptyBoxes()
        names = {0:"car",1:"truck",2:"bus",3:"motorcycle"}
        def plot(self): return np.zeros((8,8,3),dtype=np.uint8)
    class ValidModel:
        task = "detect"
        names = {0:"car",1:"truck",2:"bus",3:"motorcycle"}
        def predict(self,*args,**kwargs): return [Result()]
    path=tmp_path/"vehicle.pt";path.touch()
    module=types.ModuleType("ultralytics");module.YOLO=lambda model_path:ValidModel()
    monkeypatch.setitem(sys.modules,"ultralytics",module)
    monkeypatch.setattr(api.importlib.util,"find_spec",lambda name: object())
    monkeypatch.setattr(api,"VEHICLE_WEIGHTS",path)
    monkeypatch.delitem(api._models,"vehicle",raising=False)
    monkeypatch.delitem(api._model_health,"vehicle",raising=False)
    model=api.get_model("vehicle")
    assert model.task=="detect"
    assert api._model_health["vehicle"]["state"]=="loaded"
    api.run_frame(np.zeros((8,8,3),dtype=np.uint8),"vehicle",.35)
    assert api._model_health["vehicle"]["state"]=="inference_verified"


def test_inference_error_updates_registry_runtime_state(monkeypatch):
    import backend.api as api
    import numpy as np
    monkeypatch.delitem(api._model_health, "vehicle", raising=False)
    class BrokenModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("test inference failure")
    monkeypatch.setattr(api, "get_model", lambda detector: BrokenModel())
    try:
        api.run_frame(np.zeros((8, 8, 3), dtype=np.uint8), "vehicle", .35)
        assert False, "model exception should propagate as HTTP 503"
    except api.HTTPException as exc:
        assert exc.status_code == 503
    assert api._model_health["vehicle"]["state"] == "unavailable"


def test_driver_monitor_status_reports_missing_dependency(monkeypatch):
    import backend.api as api
    original=api.importlib.util.find_spec
    monkeypatch.delitem(api._feature_health,"drowsiness",raising=False)
    monkeypatch.setattr(api.importlib.util,"find_spec",lambda name: None if name=="mediapipe" else original(name))
    try:
        api._create_driver_monitor()
        assert False, "missing dependency should prevent initialization"
    except RuntimeError:
        pass
    assert api._feature_health["drowsiness"]["state"]=="unavailable"


def test_vehicle_model_missing_inference_dependency_is_unavailable(tmp_path, monkeypatch):
    import backend.api as api
    path=tmp_path/"vehicle.pt";path.touch()
    monkeypatch.setattr(api,"VEHICLE_WEIGHTS",path)
    original=api.importlib.util.find_spec
    monkeypatch.setattr(api.importlib.util,"find_spec",lambda name: None if name=="ultralytics" else original(name))
    monkeypatch.delitem(api._model_health,"vehicle",raising=False)
    try:
        api.get_model("vehicle")
        assert False,"missing YOLO dependency should fail initialization"
    except api.HTTPException as exc:
        assert exc.status_code==503
    assert api._model_health["vehicle"]["state"]=="unavailable"


def test_uploaded_video_pipeline_invokes_pothole_model_and_temporal_modules(tmp_path, monkeypatch):
    import backend.api as api
    import cv2
    import numpy as np
    db_path, media_path=tmp_path/"video.sqlite3",tmp_path/"media"
    media_path.mkdir()
    monkeypatch.setattr(api,"DB_PATH",db_path);monkeypatch.setattr(api,"MEDIA_DIR",media_path)
    api.initialize_database()
    class FakeCapture:
        def __init__(self,path): self.index=0
        def isOpened(self): return True
        def get(self,prop):
            if prop==cv2.CAP_PROP_FPS:return 10
            if prop==cv2.CAP_PROP_FRAME_COUNT:return 7
            if prop==cv2.CAP_PROP_POS_MSEC:return self.index*100
            return 0
        def read(self):
            if self.index>=7:return False,None
            frame=np.full((240,320,3),self.index,dtype=np.uint8);self.index+=1
            return True,frame
        def release(self): pass
    monkeypatch.setattr(api.cv2,"VideoCapture",FakeCapture)
    calls=[]
    sizes=[40,55,70,85,100,120,140]
    def fake_inference(frame,detector,confidence):
        calls.append(detector)
        size=sizes[int(frame[0,0,0])] if detector=="vehicle" else 45
        box=[160-size//2,190-size//2,160+size//2,190+size//2]
        return frame.copy(),[{"label":"car" if detector=="vehicle" else "Pothole","confidence":.9,"bbox":box}]
    monkeypatch.setattr(api,"run_frame",fake_inference)
    result=api._process_video_upload(b"test video","dash.avi","pothole",.35,1,{"lane","collision_risk"})
    assert result["frames_processed"]==7 and result["frames_sampled"]==7
    assert result["total_detections"]==7
    assert calls.count("pothole")==7 and calls.count("vehicle")==7
    assert result["module_results"][-1]["lane"]["status"] in {"ESTIMATED","UNAVAILABLE"}
    assert any(item.get("alert") for item in result["risk_assessments"])
