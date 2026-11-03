# AutoVision AI repository audit

Audit date: 2026-10-28

## Current state

The tracked repository is a Python/OpenCV learning project. It has a set of image manipulation scripts, a standalone webcam drowsiness experiment, a small sample dataset, and a MediaPipe Face Landmarker asset. The README describes intended technologies but the tracked project does not contain a frontend, API, database, authentication, migrations, Docker configuration, or application setup instructions.

The working tree already contained untracked user work at audit time: YOLO26 model files, pothole segmentation dataset and experiment outputs, plus experiments for phone/person detection, driver region, head pose, and potholes. These assets are preserved. A pothole dataset metadata file identifies Roboflow Pothole Segmentation YOLOv8, version 1, CC BY 4.0, with one class (`Pothole`). Existing model provenance for the supplied YOLO26 files and MediaPipe task asset is not recorded in the repository.

## Architecture and findings

- Python scripts call OpenCV/YOLO/MediaPipe directly. Several scripts immediately access webcams, open GUI windows, or train models when executed; they are experiments, not import-safe services.
- The original requirements file pins only NumPy 2.5.1 and OpenCV 5.0.0.93 and omits all API/frontend dependencies.
- The `.venv` has OpenCV, Ultralytics, and MediaPipe, but not FastAPI, Uvicorn, Pydantic, SQLAlchemy, multipart upload support, or pytest at audit time.
- Existing weights include general YOLO detection/pose/segmentation assets in the working tree, but no model card or verifiable provenance for those weights. COCO class detection can support object/vehicle detection; it does not establish seatbelt use, phone distraction, alcohol impairment, or crash prediction.
- A task file exists for MediaPipe face landmarks. The drowsiness experiment measures eye aspect ratio from landmarks but only runs as a webcam GUI loop; it has no service boundary, temporal test coverage, or calibrated operating thresholds.
- There is no persistent event/job store, upload validation, user isolation, CORS policy, auth, deployment setup, or API documentation.
- Existing tracked scripts and sample assets remain untouched to preserve the original learning exercises.

## Risks

1. Model asset licensing/provenance is not verified; do not distribute weights until their source and terms are confirmed.
2. Drowsiness, phone-use, and driver monitoring are sensitive, camera-angle-dependent estimates and require representative validation before safety use.
3. No sensor or validated model supports alcohol/BAC inference. Facial appearance must not be used to claim impairment.
4. An experimental risk score is not a crash prediction and must not be represented as one.
5. Local demonstration storage is not a production-grade multi-user media or identity system.

## Prioritised implementation plan

1. Add a typed FastAPI service with health/readiness, detector capabilities, bounded image/video uploads, real configured YOLO inference, and local event persistence.
2. Add a responsive dashboard served by the API that uploads media, renders annotated output and events, and displays explicit detector readiness/limitations.
3. Add input/security configuration, tests, API/architecture/model docs, and reproducible local run instructions.
4. Keep absent/unsupported models unavailable; document production gaps including verified model provenance, authentication, migrations, background job queue, and deployment credentials.
