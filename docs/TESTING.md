# Testing

Run the suite with Python 3.12:

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe -m compileall -q backend tests
node --check frontend/app.js
```

Tests cover health/registry state semantics, compatible/mismatched model task and classes, inference exceptions, missing Ultralytics/MediaPipe, upload validation, database migrations, optional uploaded-video module processing, simulated-frame ordering and persistence, drowsiness temporal indicators, lane geometry, and risk confidence/timestamp/warning conditions. Test doubles are isolated to tests; no sample inference is represented as real capability verification.

Separately recorded real-weight checks in `IMPLEMENTATION_STATUS.md` exercised vehicle inference, pothole inference through uploaded video, and MediaPipe face landmarks through uploaded video. They demonstrate functioning code paths only; they do not establish detector accuracy. Browser-selected-video E2E and Docker runtime checks remain unconfigured/unverified.
