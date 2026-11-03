# Model registry, provenance, and limits

The registry endpoint is `GET /api/detectors`. It separates implementation status from runtime status. `ready` requires a model to load with expected task/classes and complete at least one real inference successfully. `available` means a configured compatible dependency/asset exists but successful inference has not yet been observed in this process. Experimental methods retain the `experimental` status even when their runtime path works; model quality is not validated by a successful call. Runtime errors and missing dependencies are shown separately.

## Vehicle/object detection — Ready after inference verification

- Asset: repository-local `yolo26n.pt`, configured with `VEHICLE_MODEL_PATH`.
- Verified model metadata: Ultralytics task `detect`; COCO classes include car, motorcycle, bus, truck, person, and cell phone.
- Actual API path: `get_model` validates the task and expected vehicle classes; `run_frame` performs YOLO prediction and marks registry runtime inference-verified only on success.
- Provenance/license is not recorded in this repository. Confirm the weight distribution terms before redistributing it. No weight is downloaded automatically.
- Phone detections are object observations; they do not establish driver use or distraction.

## Pothole segmentation — Experimental

- Asset: `runs/segment/runs/pothole/pothole_seg_test/weights/best.pt`, configured with `POTHOLE_MODEL_PATH`.
- Verified metadata: Ultralytics task `segment`, one `Pothole` class. Dataset metadata identifies Roboflow Pothole Segmentation YOLOv8 v1, CC BY 4.0. Confirm the checkpoint's provenance and license separately.
- Image inference and uploaded-video inference use the actual segmentation model. The unified simulated-live frame pipeline attempts it on sampled frames for road-oriented profiles alongside vehicle detection. Driver-facing profiles mark road models not applicable and do not initialize them. It returns plotted segmentation previews and timestamped boxes; a missing/unavailable pothole model does not block vehicle inference.
- Actual video smoke test on two frames of the repository's clear-car sample returned two `Pothole` instances. This is a false-positive example, not proof of useful pothole recognition. Do not treat the model as road-hazard ready; evaluate on representative footage.

## Driver monitoring / drowsiness — Experimental prototype

- Asset: `models/face_landmarker.task`, configurable through `DRIVER_LANDMARKER_MODEL_PATH`. The MediaPipe package is declared in `requirements.txt`.
- `backend/driver_monitor.py` integrates MediaPipe Face Landmarker VIDEO mode into sampled uploaded video and simulated replay. It handles no face, multiple faces, timestamp gaps, eye-aspect-ratio values, and prolonged closure over a 1.5-second window. EAR threshold `0.21` comes from the original experiment and is not calibrated or validated.
- The actual uploaded-video smoke test initialized this model successfully and reported `FACE_NOT_VISIBLE` on a road-facing sample. This is expected when the camera cannot see the driver; it is not an inference of drowsiness.
- The original `backend/drowsiness.py` remains a standalone webcam program and is not imported by the API. No live camera permission is used. Face model provenance/license is not recorded; verify before distribution.

## Lane detection — Experimental OpenCV baseline

- No trained lane model is present. `backend/lane_baseline.py` uses a lower-image ROI, Canny edges, Hough line segments, slope/side grouping, and left/right geometry checks.
- It returns image-space boundary points and a confidence heuristic, or `PARTIAL`/`UNAVAILABLE`. A basic geometry check was added after the clear-car sample produced crossing lane hypotheses. This remains highly sensitive to camera angle, shadows, road markings, and scene edges. No calibration or steering output is provided.

## Collision-risk indicator — Experimental, not crash prediction

- No trained predictive model or representative crash dataset is configured.
- `backend/risk.py` tracks same-class vehicle boxes by IoU across timestamped sampled frames. It checks a minimum detection confidence (default `0.35`), temporal continuity, apparent box area/growth, central position, and image-plane center movement. Warnings include observed growth/area/time/confidence evidence and uncertainty.
- Monocular image-plane change is not calibrated distance, speed, TTC, or collision probability. `UNAVAILABLE` and no-alert results are not evidence that the road is safe.

## Seatbelt use — Missing model

There are no seatbelt-specific weights or labeled, representative cabin-camera data in the repository. General COCO weights cannot infer belt use. A compatible model and validated supported viewpoints are required before an implementation can be called a detector; status remains `missing_model`.

## Alcohol impairment — Not implemented

There is no validated alcohol sensor or legitimate external evidence integration. Facial appearance is not used to infer impairment or blood-alcohol concentration. Status remains `not_implemented` until an appropriate sensor integration is specified and validated.

No model downloads occur at runtime. Local weight paths can be overridden in `.env`; place the files yourself and verify provenance/licensing.
