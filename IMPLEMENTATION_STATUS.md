# Implementation status

Updated 2026-10-28. This status reflects repository inspection and checks actually run.

## Implemented in this pass

- Added a FastAPI application with health/readiness, detector capabilities, image/video inference, paginated events, analytics, static dashboard, restricted configurable CORS, upload size/extension/content checks, video duration/frame sampling, UUID preview storage, local SQLite event persistence, and temporary video cleanup.
- Added a responsive dashboard with uploads, model selector, confidence threshold, annotated result display, actual detection labels/scores/timestamps, event search/filter/pagination, and API status.
- Configured available local general-object and experimental pothole weights by path. API lazily initializes models and returns 503 if the selected model is absent/unavailable.
- Added audit, architecture/API/model docs, setup instructions, dependency declarations, sample environment config, Docker files, and API validation tests.
- Added backward-compatible SQLite columns for decoded/sampled/source frames, completion status, timing, threshold, interval, and model filename. New video history/result views show these values and class counts; legacy rows display missing values as unrecorded.
- Clarified dashboard metrics: analyses with detections count event records, while a per-video detection total counts frame-level detection instances. The capability count no longer treats model-file presence alone as Ready.
- Preserved existing OpenCV/MediaPipe scripts and all untracked datasets, model weights, and experiment outputs.

## Genuine capabilities and limitations

- Genuine inference code is wired for YOLO general-object/vehicle classes and the local pothole fine-tune when configured. Both processed media through the API in this pass.
- The pothole model reported a 0.4993 pothole box on the repository's clear car photo (`datasets/sample.jpg`), a false positive. Treat the custom model as experimental and review detections; no accuracy claim is justified.
- The original drowsiness script remains a standalone webcam prototype. Its MediaPipe task is now integrated as an optional uploaded-video/replay eye-closure indicator; no live webcam endpoint exists. Seatbelt model and alcohol sensor are absent. Lane and collision-risk modules are experimental, not validated capabilities or steering/crash prediction.
- SQLite local single-user persistence is implemented; authentication, authorization, user isolation, deletion/retention policy, Alembic migrations, PostgreSQL, job queue, and production media storage are not implemented.
- Event history and analytics are connected. Date-range charts, profile/authentication screens, browser webcam, and live camera streaming are not implemented.
- The user's two repeated video records each contain 147 detection instances across 714 successfully decoded frames (126 car, 20 truck, 1 bus); detections occur at 47 distinct timestamps. The reported 771-frame value is not present in those persisted rows. The old schema did not preserve source-frame metadata or sampling count, so complete-file status and exact old-run sample count cannot be reconstructed; migrated records are labeled finished/unverified.
- Existing rows also do not preserve the model filename, threshold, or timings; the current code path uses the configured `vehicle` weights and confidence sent by the UI, but old-run specifics remain unrecorded. Their stored confidences range from 0.3518 to 0.9142, consistent with—but not proof of—the default 0.35 threshold.
- The API now records decode counts separately from container-reported frames and reports partial completion on a mismatch. Sampling uses frame indices 0, 15, 30, … when called from the dashboard. Detections are concatenated per sampled frame and are not deduplicated across frames.
- Updated-pipeline real video smoke run on `datasets/samplevid.mp4`: HTTP 200, 1,000/1,000 source frames decoded, 67 sampled, 4 detection instances (2 car, 2 truck), 24.29 seconds processing time, 320.0 ms average model-call time per sample, model `vehicle (yolo26n.pt)`, threshold 0.35, interval 15, completion `complete`.
- Compose file exists; YAML syntax was checked, but Docker validation/run has not been performed because Docker is unavailable.

## Checks

- Python 3.12 dependency installation succeeded in isolated `.venv312`; existing Python 3.14 `.venv` was left untouched.
- `.venv312\Scripts\python.exe -m pytest -q`: **11 passed** (one Starlette/httpx deprecation warning).
- `python -m py_compile backend\api.py tests\test_api.py`: passed.
- `node --check frontend/app.js`: passed.
- Real image inference through the API with the local `yolo26n.pt`: **HTTP 200**, detected one `car` at confidence 0.8932 in `datasets/sample.jpg`, returned annotated preview and persisted event.
- Real video inference through the API with the local `yolo26n.pt`: **HTTP 200**, processed 1,000 frames from `datasets/samplevid.mp4` at every 30th frame, returned three timestamped vehicle detections and persisted event.
- Uvicorn startup completed; live `/api/health`, `/`, and `/docs` requests returned successfully.
- Docker Compose YAML parsed successfully; Docker CLI is not installed, so Compose runtime/build validation was unavailable.

## Highest-priority follow-up

1. Verify model provenance and evaluate each configured detector on held-out, representative footage; establish thresholds and limitations.
2. Add authentication/user isolation, delete/retention controls, migrations, and production database/storage.
3. Add queued video jobs with progress/cancellation and frontend webcam support only after privacy and performance design.

## Simulated-live replay implementation

- Added browser-local video playback with sequential, timestamp-ordered frame sampling and actual configured YOLO inference; inference latency is measured per call.
- Added start, pause, resume, restart, stop, selectable detector/confidence, 2/4/8 Hz sampling, 0.5×–2× playback speed, synchronized bounding-box overlay, class totals, frame/sample counters, experimental alert timeline, and persistence to SQLite event history.
- Added an experimental IoU-associated temporal vehicle-box growth indicator with explicit uncertainty and `UNAVAILABLE` handling for missing, stale, out-of-order, or delayed evidence. No TTC or crash prediction is generated.
- Offline video decode and image/video inference work is now offloaded to a thread pool.
- The replay retains the original source file only in browser memory; it posts sampled frames (maximum 4 MiB each). Replay session state is in process memory, expires after one hour, and is not crash-resilient.
- Risk: no end-to-end browser run against an actual video has yet been executed in this turn. API session integration is exercised with mocked inference in tests; prior real model smoke-test results are documented above. Simulated-live currently needs a compatible browser video decoder and the chosen model weights.

## Checks (2026-10-28)

- `.venv312\\Scripts\\python.exe -m pytest -q`: **16 passed**, with one Starlette/httpx deprecation warning.
- Python `compileall` on backend/tests: passed.
- `node --check frontend/app.js`: passed.
- The simulation API test mocks the model boundary and verifies request/response, monotonic timestamps, persistence, and honest stopped status; it does not represent real detector accuracy.
- This turn did not run a browser video replay or Docker build. Prior real vehicle-model upload smoke tests and their measured results are recorded above; they are not simulated-live runs.
- Simulated-live actual model check (temporary database): two sampled frames from `datasets/sample.jpg` were passed through the configured `vehicle (yolo26n.pt)` model; each returned one car at confidence 0.8928. The cold first prediction took 8,654.6 ms (risk `UNAVAILABLE` due to stale inference); the next took 150.9 ms and risk status `EXPERIMENTAL`. Average over those two calls was 4,402.7 ms due to warm-up. The stopped session persisted two samples/two frame-level detection instances to the temporary database, which was removed. This validates the real inference API path, not video decoding/browser playback or detector accuracy.

## Upload limit update

- Default API upload limit is now 250 MiB (`MAX_UPLOAD_MB=250`); the environment variable can still configure a different cap. Dashboard validation and the local replay file picker reject files over the same 250 MiB limit. The offline video duration limit remains 180 seconds. Neither Dockerfile nor Compose adds an ingress proxy with a smaller body-size cap.

## Model registry and video modules update (2026-10-28)

| Capability | Registry status | Actual implementation and remaining limits |
|---|---|---|
| Vehicle/object detection | `available` until successful inference; then `ready` | Real `yolo26n.pt` detector, validated task/classes, successful inference updates runtime status. General object classes; phone object does not establish driver distraction. |
| Pothole segmentation | `experimental` | Local `best.pt` is an actual one-class YOLO segmentation model and is wired for image and sampled-video inference. A real two-frame video request on the clear-car sample returned two false-positive Pothole detections; no quality claim. |
| Drowsiness/driver monitoring | `experimental` | MediaPipe Face Landmarker task integrated into uploaded video and simulated replay. Real clip smoke test returned `FACE_NOT_VISIBLE`; threshold is uncalibrated and a forward road camera may not see the driver. |
| Seatbelt use | `missing_model` | No belt-specific weights or validated supported camera-angle dataset. |
| Alcohol impairment | `not_implemented` | Requires a legitimate, validated sensor/evidence integration. No facial-appearance inference. |
| Lane detection | `experimental` | Canny/Hough image-space baseline, geometry consistency check, and partial/unavailable responses. No trained lane model/calibration. Clear-car sample lane hypotheses were rejected as crossing/implausible. |
| Collision-risk indicator | `experimental` | IoU-associated vehicle boxes with timestamps, confidence threshold, apparent size growth and normalized center movement. Evidence/uncertainty only; no TTC, probability, or crash prediction. |

Uploaded video and simulated replay accept optional `drowsiness`, `lane`, and `collision_risk` features. `module_results_json` stores per-sample module output separately from detection-instance counts. When pothole segmentation is the selected video detector and collision risk is enabled, the vehicle model is also run for risk inputs. Model and dependency failures report `unavailable` without discarding primary detections when possible. Registry details live in [docs/MODELS.md](docs/MODELS.md); the API feature contract is in [docs/API.md](docs/API.md).

### Verification for this update

- API/unit suite: **27 passed** (one Starlette/httpx deprecation warning).
- Python compileall, Node JavaScript syntax check, generated OpenAPI route check, and dashboard/API health check passed.
- Real local pothole weights through `/api/infer/video`: HTTP 200, 2/2 frames processed and sampled, `pothole (best.pt)`, 2 `Pothole` detections on two clear-car frames (false-positive example).
- Real MediaPipe-enabled video path: HTTP 200; drowsiness reported `FACE_NOT_VISIBLE`, lane baseline reported estimates before the new crossing-geometry rejection (the post-fix baseline now returns `UNAVAILABLE` for the clear-car sample), and risk returned unavailable on the first slow/stale sample then experimental on the next. No driver face or drowsiness event was fabricated.
- Risk tests verify low-confidence rejection, stale timestamp handling, and an evidence-bearing experimental warning. API pipeline tests use explicit test doubles for predictable video-frame sequences; these are tests, not model-performance evidence.
- Browser rendering is verified on the updated dashboard, but a browser-selected video was not driven end to end in this pass. Docker runtime was not available.
- On the final live server, the registry reported vehicle `available / not_inference_verified` before inference; one actual `yolo26n.pt` sampled frame returned a car at 0.8928, after which `/api/detectors` reported vehicle `ready / inference_verified`. The verification session was discarded, so it added no event-history row.

## Unified dashcam homepage update (2026-10-28)

- Removed the user-facing batch `Run analysis` panel, duplicate image/video upload, and “Analyze media” navigation. The homepage's primary and only media workflow is now **Upload Video → Start Dashcam Simulation**. Event history and Model Registry remain secondary navigation/diagnostics.
- The replay request now always runs the configured vehicle detector and attempts the configured pothole segmentation model on each sampled frame. Pothole failure is surfaced as unavailable and does not suppress vehicle results. Both detectors' frame-level instances and model IDs are included in responses and persisted summaries.
- Driver monitoring, lane baseline, and collision-risk assessment run in the same sequential frame request. Lane/face processing receives the original source frame to avoid using detector annotation pixels as evidence. Segmentation, lane, and face overlays are composited. Bounding-box overlays are cleared once playback advances beyond the sampled timestamp.
- Added live capability observations and explicit driver-not-visible messaging, distinct per-module metrics, pipeline latency/rate, and deduplicated timestamped pothole/risk event entries. Vehicle readiness before the first successful inference is shown as `available`, then `ready`.
- Local startup and Compose host mapping now use port 8001 (container Uvicorn remains on 8000). The still-supported batch API routes remain for compatibility but are not exposed as a second dashboard workflow.
- Automated suite after changes: **28 passed**, one Starlette/httpx deprecation warning. `node --check frontend/app.js`, Python compileall, and `git diff --check` passed.
- Live browser-served page check: title is `AutoVision AI — Dashcam Live Simulation`, the old `Run analysis` action is absent, the unified start action is present, and `/api/health` returned `ok`. Registry returned vehicle `ready`, pothole/drowsiness/lane/collision-risk `experimental`, seatbelt `missing_model`, alcohol `not_implemented`.
- Real unified simulation API smoke check using `datasets/sample.jpg`: HTTP 200; one actual vehicle detection and one actual pothole-model candidate (known clear-car false-positive risk); drowsiness `FACE_NOT_VISIBLE`; lane and collision risk `UNAVAILABLE` for that single still-image observation. Full simulation session was discarded without adding history. This confirms both model calls and honest insufficient-evidence statuses, not browser playback, video performance, or model quality.
- Browser-driven complete replay: a 40.08-second local clip decoded and played through the dashboard. The API processed 403 browser-presented frames and 53 sequential sampled frames; the run completed with 219 detection instances: 187 car, 27 person, 1 motorcycle, and 4 pothole candidates. Average combined pipeline latency was 515.9 ms/sample. Driver monitoring reported `FACE_NOT_VISIBLE` on all 53 samples; lane estimates were available on 0/53 samples; three pothole timeline events were deduplicated; one risk sample carried an experimental warning. This is a smoke run, not detector quality evidence.
- Browser controls: pause and resume toggled paused/play states and button availability; restart discarded the old session and initialized a new one. A separate explicit Stop-button run on `samplevid.mp4` stopped and persisted a 3-sample `stopped` summary (29 frames presented, 0 detection instances, 3,278.6 ms average including cold-start inference). Its generated QA row and preview, and the other browser QA record, were removed afterward; pre-existing history rows were preserved.
- Remaining verification: Docker is unavailable. No cabin-facing sample exists, so driver monitoring on visible-driver footage remains unverified. Pothole false positives remain a known limitation. Browser replay confirms workflow and synchronization behavior on local clips, not model accuracy or safety performance.

## Universal uploaded-video update (2026-10-28)

- Replaced dashcam-only primary copy with AI Video Simulation and added General/automatic, Road-facing, Driver-facing, and Mixed profiles.
- Added profile request/response state and per-frame observational road/face evidence. Automatic is explicitly not a trained scene classifier.
- Driver-facing requests skip vehicle and pothole model loading/inference, mark road functions not applicable, and run driver monitoring if its local MediaPipe task dependency exists. Road profile selection runs the shared road modules. The frontend video remains local and sends sequential timestamped decoded frames; no direct camera APIs or permission prompts were introduced.
- The browser reports decoded frame size/orientation and keeps source aspect ratio/overlay alignment through contain scaling. Supported codecs are determined by actual browser decode.
- Inspected two actual repository videos with OpenCV: landscape MP4 848×478 at ~24.96 FPS/1,000 reported frames, and landscape MP4 1280×720 at 25 FPS/375 reported frames. They do not constitute portrait/cabin/rotated/VFR validation. Browser E2E with those additional sources was not performed in this pass.
- Added an API test that rejects accidental road-model use for a driver-facing session and checks `NOT_APPLICABLE` responses.

## Playback stutter fix (2026-11-03)

- The previous replay paused the HTML video at every sample and waited for the full inference response before resuming. Observed historical per-frame inference latency was roughly 330–450 ms, longer than the 250 ms interval at the default 4 samples/second, so each sample caused a visible freeze.
- Playback now continues while a single sampled frame is processed asynchronously. While that request is busy, due sample slots are dropped and counted instead of queued. The browser reports actual processed frames and skipped slots separately.
- Results retain their source timestamp. Bounding boxes are drawn only if inference completed while playback is paused on the same nearby source time; otherwise the overlay is hidden and the result stays timestamped, avoiding boxes on a later frame.
