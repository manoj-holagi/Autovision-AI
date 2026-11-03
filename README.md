# AutoVision AI

Automotive perception portfolio demo built around the original OpenCV/YOLO/MediaPipe learning scripts. Its primary workflow is one uploaded-video simulation: select a clip from a dashcam, phone, cabin camera, or other source, choose a processing profile, and inspect frame-synchronized results. It is research/demo tooling, not a certified driver-assistance or safety system.

## Current capabilities

- Progressive simulated-live playback runs vehicle detection and attempts pothole segmentation on every sampled frame. It also evaluates driver visibility/eye closure, lane estimates, and experimental temporal collision-risk evidence.
- General vehicle/object detection uses the configured local YOLO model (COCO classes include car, motorcycle, bus, truck, person, and cell phone).
- Experimental pothole segmentation uses the local `best.pt`; known false positives are displayed as candidates and recorded with a review warning.
- Annotated previews and SQLite-backed event history/summary metrics.
- Registry status distinguishes loadable/unverified weights, inference-verified vehicle detection, experimental modules, missing models, and unsupported integrations.
- Existing standalone OpenCV and MediaPipe experiments remain in `backend/`; the webcam script is still not invoked by the API.

There is no seatbelt-specific detector or supporting model, no alcohol sensor/BAC integration, and no validated crash-prediction model. The collision-risk module only emits an experimental image-plane indicator; it is not a crash prediction. Lane detection uses a fragile OpenCV baseline, not a trained model. Driver monitoring is an uncalibrated face/eye-closure prototype; many road-facing clips will not show the driver. A cell phone detection does not prove driver use.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/API.md](docs/API.md), [docs/MODELS.md](docs/MODELS.md), and [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). The initial repository findings and prioritised plan are in [PROJECT_AUDIT.md](PROJECT_AUDIT.md).

## Requirements and local start

Use Python 3.12 (or another version supported by the selected MediaPipe release). The pre-existing `.venv` in this checkout is Python 3.14 with newer NumPy/OpenCV versions than MediaPipe supports; create a compatible environment instead of downgrading that existing environment in place.

### Windows PowerShell

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn backend.api:app --reload --env-file .env --port 8001
```

Open <http://127.0.0.1:8001>. Health check: <http://127.0.0.1:8001/api/health>. OpenAPI: <http://127.0.0.1:8001/docs>.

The default request size limit is 250 MiB per file (`MAX_UPLOAD_MB`); offline videos are limited to 180 seconds. The service looks for the general model at `./yolo26n.pt` and pothole weights at `./runs/segment/runs/pothole/pothole_seg_test/weights/best.pt`. Configure alternative paths using the environment variables in `.env.example`. If weights are absent, the model reports unavailable and inference returns HTTP 503. No model weights are downloaded automatically.

For production configuration, copy `.env.example` to `.env`, set a persistent `AUTOVISION_DATA_DIR`, restrict `CORS_ORIGINS` to the deployed frontend origin, and supply vetted model paths. The current application has no authentication and should not be exposed as a multi-user service.

## Docker

Docker Compose mounts the expected local model files read-only and persists SQLite/media in a named volume:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The Compose model bind mounts require the files to exist. For an environment without either model, remove the corresponding `./...pt:/models/...` mount and leave that detector unavailable, or change the model path/mount. Review licensing before building/distributing any model-containing image. Docker/Compose execution has not been verified in this environment.

## Testing

```powershell
python -m pytest -q
```

Tests cover health/capabilities, upload validation, missing weights, database migration and summary persistence, simulated inference session ordering, and experimental risk-indicator behavior. Simulation API tests mock only the model boundary. A separate real-weight sampled-frame check is documented in implementation status; it is not a browser video replay or accuracy evaluation. Run `python -m compileall -q backend tests` and `node --check frontend/app.js` for syntax checks.

## Data and privacy

Uploaded media is processed locally. Events store a filename, detector, detections, timestamp, and annotated preview under the configured data directory. Original video bytes are held in a temporary file during processing and removed afterward. There is no authentication, deletion API, retention policy, user separation, or production-grade privacy control yet. Do not upload sensitive media to a shared or public deployment.

## Screenshots

Run the application and capture the dashboard at `http://127.0.0.1:8001`; no screenshot is included yet.

## Simulated-live video replay

The homepage is the video simulation. Upload a clip, select a view profile, choose confidence and sampling rate, and start. Road profiles run configured vehicle and pothole models; driver-facing runs driver monitoring without initializing road models; mixed and automatic profiles run both groups. Playback does not wait for inference; if a frame is still processing at the next sample time, that sample is skipped and the skipped count is shown. Detection overlays appear only when their result still matches a paused source frame; otherwise timestamped results remain in the timeline without being drawn over a later frame. The selected file remains local to the browser; only sampled frames are sent to the local API. No camera or microphone is accessed. Browser codec support varies by format.

At stop or completion, the dashboard persists the session summary in event history. It shows browser-reported frames presented, frame samples, actual detection instances and class totals, average inference latency, model, confidence and status. Pause time is included in total replay duration. Historical uploads will still show metrics as unrecorded where the earlier backend did not capture them.

The collision-risk indicator is experimental and based on image-plane bounding-box association/growth. It does not estimate time-to-collision, crash likelihood, or a safe driving state. It can report unavailable when observations are insufficient or stale. This is not a safety product.

## Universal video input and profiles

The homepage is an uploaded-video simulation, not dashcam-only. Choose a clip recorded by a phone, cabin camera, dashcam, or another source, subject to browser codec support, 250 MB and 180-second limits. The browser validates successful decode/duration before start, preserves the decoded portrait/landscape ratio, and sends only sampled frames (not the original video) to the API. No camera permission is requested.

Profiles are General / automatic assessment (default), Road-facing, Driver-facing, and Mixed road and cabin. Automatic is observational; it is not a scene classifier. Road-oriented modules can report no usable estimate when the sampled frames do not support them. A driver-facing session does not load/run road detectors. Choose a known camera profile when automatic applicability is uncertain. Per-frame assessment summarizes actual lane/vehicle/face cues and explicitly does not prove a scene/person is absent.

Repository clips inspected: `datasets/samplevid.mp4` (848×478, ~24.96 FPS, 1,000 frames) and the pothole project's sample (1280×720, 25 FPS, 375 frames), both landscape MP4. These do not verify portrait, cabin-facing, mixed view, variable-frame-rate, metadata-rotated, or damaged footage. Browser decoding support and orientation behavior vary. No live camera capture is implemented.
