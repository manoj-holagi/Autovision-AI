# API

Interactive schema: `/docs`; JSON OpenAPI: `/openapi.json`.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Process/database health |
| GET | `/api/ready` | Service readiness, inference-ready detectors, and experimental capabilities |
| GET | `/api/detectors` | Capability/readiness and limitations |
| POST | `/api/infer/image?detector=vehicle&confidence=0.35` | Multipart image inference; detector is `vehicle` or `pothole` |
| POST | `/api/infer/video?detector=vehicle&confidence=0.35&frame_interval=15&features=drowsiness,lane,collision_risk` | Multipart video inference with optional temporal modules |
| POST | `/api/simulation/start` | Start replay with an automatic/general, road-facing, driver-facing, or mixed profile; driver-facing skips road models |
| POST | `/api/simulation/{session_id}/frame?timestamp=...&frame_id=...&width=...&height=...` | Multipart frame; selected profile modules run sequentially for that timestamp |
| POST | `/api/simulation/{session_id}/finish` | Persist complete, stopped, or error replay summary |
| POST | `/api/simulation/{session_id}/discard` | Discard an empty/restarted in-memory replay session |
| GET | `/api/events?page=1&page_size=12&search=&detector=` | Paginated event history |
| GET | `/api/analytics` | Persisted analysis/event summary |

Image formats: JPG/JPEG, PNG, WebP. Video formats: MP4, AVI, MOV, MKV, WebM. Default upload cap is 250 MB per file; set `MAX_UPLOAD_MB` to change it. Video duration is capped at 180 seconds. Detection boxes are pixel `[x1,y1,x2,y2]` coordinates. Video detections include seconds timestamps. Inference returns 503 when selected weights are missing or cannot initialize; invalid media returns 400; unsupported extension returns 415; excessive size/duration returns 413. No authentication endpoints exist.

Video responses and event rows also include `frames_processed` (frames successfully decoded), `frames_sampled` (model calls), `source_frame_count` (container-reported frame total when available), `total_detections` (detection instances aggregated over sampled frames), `detections_by_class`, `processing_time_seconds` (decode/inference loop wall time), `average_inference_ms` (average time per sampled model call), `model_used`, `confidence_threshold`, `frame_interval`, and `completion_status`. A record is `complete` only when decoded frames meet the container-reported count; a mismatch is `partial`. If the container has no usable count, status is `finished_unverified`. Detection instances are not unique tracked vehicles. Older records predate these fields; unknown values remain null and are displayed as not recorded. Average model-call time includes lazy model initialization on the first sample after a cold start.

The current dashboard sends `frame_interval=15` and default confidence `0.35`. The API validates confidence in `[0.05, 0.95]` and the interval in `[1, 300]`. The inference loop checks frame indices `0, interval, 2*interval, ...`; a sampled frame may have zero detections, so the number of distinct detection timestamps is not necessarily the number of sampled frames.

`features` is a comma-separated optional subset of `drowsiness,lane,collision_risk`. `drowsiness` requires the local MediaPipe task model and computes an uncalibrated eye-aspect-ratio closure indicator over timestamps; a missing face/multiple faces are reported without claiming drowsiness. `lane` uses the experimental OpenCV Canny/Hough image-space baseline and reports partial/unavailable results for weak or geometrically inconsistent boundaries. `collision_risk` uses the vehicle detector even when the primary video detector is pothole segmentation; it reports temporal evidence/uncertainty only. These results are returned in `module_results` and persisted with the event. Feature failures are reported as unavailable without discarding the primary model detections.

## Registry status semantics

`status` expresses implementation/readiness and `runtime_status` expresses dependency/model/inference health. Vehicle weights are `available` until compatible classes load and one real prediction succeeds; only then are they `ready`. Pothole remains `experimental` even after verified inference because its local weights have not been validated. Driver monitoring, lane baseline, and image-plane collision-risk indicators are `experimental`; runtime health is reported separately. Seatbelt is `missing_model`; alcohol is `not_implemented` pending a validated sensor integration. Missing dependencies or a runtime inference failure report `unavailable`. A weight file's mere existence is never sufficient for `ready`.

## Simulated-live video replay

The homepage is the only video workflow. The browser opens the selected prerecorded video using a local object URL; it does not upload the whole source. It sends at most one sampled frame at a time while playback continues; sample slots that occur during inference are skipped. Late results remain timestamped and are not overlaid on a different current frame. The same sequential frame request runs the modules selected for its profile (road models, driver monitoring, lane baseline, and collision-risk indicator). Responses carry video timestamps and presented-frame IDs; detections identify their model. Pause/resume/restart/stop, playback speeds 0.5×/1×/2×, and 2/4/8 samples per second are supported. Seeking is disabled while active to preserve strict ordering. Browser codec support varies.

At finish, SQLite stores browser-reported presented frames, sampled calls, detection instances across both models, per-class counts, timing, model/configuration, completion state, risk assessments, module observations, and the last annotated frame. Presented-frame count is a browser counter, not a container-verified source total. Frames are sent individually (maximum 4 MiB); the original file stays in browser memory. Session state is process-local and expires after one hour; process restart ends unpersisted sessions. Metrics count detection instances per sampled frame, not unique tracked objects.

The experimental risk engine tracks same-class vehicle boxes by IoU and considers sustained, central, sufficiently large apparent box growth. It can report a cautious potential-forward-obstacle indicator with explicit uncertainty. It does not estimate distance, speed, time-to-collision, crash probability, or safety. No alert is not evidence of a clear or safe road. This temporal indicator is not evidence of model validation.

For simulated replay, average latency includes model warm-up on the first prediction after loading weights. The dashboard presents each measured call and marks a sample delayed when it exceeds the selected sampling interval. No latency or event-rate target is guaranteed.


## Video profiles and frame API

`POST /api/simulation/start` accepts `profile`: `automatic` (default), `general`, `road_facing`, `driver_facing`, or `mixed`. Road-facing selects vehicle/pothole plus requested lane/risk; driver-facing selects driver monitoring only and skips road detector initialization; mixed and automatic/general select the requested driver/lane/risk modules plus road detectors. The frontend selects the profile and requests all optional modules; the backend intersects those requests with the profile. Start and frame responses expose applicability/readiness and an observational `profile_assessment`. Automatic is not a scene-classification model. A frame response's `profile_assessment` reports cues in sampled frames, not proof of scene absence.

The source video stays local to the browser. Browser decoding supplies upright frame pixels and dimensions to the common frame endpoint, which verifies bytes and dimension match. The UI reports decoded resolution/orientation and uses contain scaling for overlays. Supported formats depend on the browser's actual decoder. Direct camera capture is not enabled.
