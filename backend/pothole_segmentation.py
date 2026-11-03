import cv2
import time
from ultralytics import YOLO


# ============================================================
# AutoVision AI - Fast YOLO Segmentation
# ============================================================

print("=" * 60)
print("AutoVision AI - Fast YOLO Segmentation")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO("yolo26n-seg.pt")

print("YOLO Segmentation model loaded successfully.")


# ============================================================
# OPEN EXTERNAL CAMERA
# ============================================================

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ERROR: Could not open external camera.")
    exit()


# ============================================================
# CAMERA SETTINGS
# ============================================================

# Lower resolution = faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Reduce camera buffer
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.40

# Process smaller image
IMAGE_SIZE = 416

# Process every second frame
FRAME_SKIP = 1


# ============================================================
# FPS VARIABLES
# ============================================================

previous_time = time.time()

fps = 0

frame_counter = 0

last_result = None


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    frame_counter += 1


    # --------------------------------------------------------
    # YOLO PROCESSING
    # --------------------------------------------------------

    # Run YOLO only on selected frames
    if frame_counter % (FRAME_SKIP + 1) == 0:

        results = model(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False
        )

        last_result = results[0]


    # --------------------------------------------------------
    # DRAW RESULT
    # --------------------------------------------------------

    if last_result is not None:

        display_frame = last_result.plot()

    else:

        display_frame = frame


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    current_time = time.time()

    elapsed = current_time - previous_time

    if elapsed > 0:

        fps = 1.0 / elapsed

    previous_time = current_time


    # --------------------------------------------------------
    # FPS DISPLAY
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"FPS: {fps:.1f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        "Fast Segmentation Test",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # SHOW
    # --------------------------------------------------------

    cv2.imshow(
        "AutoVision AI - Fast Segmentation",
        display_frame
    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Segmentation stopped.")