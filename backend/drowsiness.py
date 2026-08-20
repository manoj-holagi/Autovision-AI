import cv2
import time
import math
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# AutoVision AI - Drowsiness Detection
# MediaPipe 1.x Face Landmarker
# ============================================================

print("=" * 60)
print("AutoVision AI - Drowsiness Detection")
print("=" * 60)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "models/face_landmarker.task"


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    options
)

print("MediaPipe Face Landmarker loaded successfully.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ERROR: Could not open external camera.")
    exit()


# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Reduce camera buffer to reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ============================================================
# EYE LANDMARKS
# ============================================================

# Left eye
LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

# Right eye
RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ============================================================
# DROWSINESS SETTINGS
# ============================================================

# Initial EAR threshold.
# We will calibrate this later based on your eyes.
EAR_THRESHOLD = 0.21

# Eyes must remain closed for this duration
# to trigger drowsiness.
DROWSY_TIME = 1.5


# ============================================================
# STATE VARIABLES
# ============================================================

blink_count = 0

eyes_closed = False

eyes_closed_start = None

timestamp_ms = 0


# ============================================================
# FPS
# ============================================================

previous_time = time.time()


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# EAR CALCULATION
# ============================================================

def calculate_ear(
    landmarks,
    eye_indices,
    width,
    height
):

    points = []

    for index in eye_indices:

        landmark = landmarks[index]

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )


    # Vertical eye distances
    vertical_1 = distance(
        points[1],
        points[5]
    )

    vertical_2 = distance(
        points[2],
        points[4]
    )


    # Horizontal eye distance
    horizontal = distance(
        points[0],
        points[3]
    )


    if horizontal == 0:
        return 0.0


    # Eye Aspect Ratio
    ear = (
        vertical_1 +
        vertical_2
    ) / (
        2.0 *
        horizontal
    )


    return ear


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera.")
        break


    height, width, _ = frame.shape


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MEDIAPIPE IMAGE
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp_ms += 33


    # --------------------------------------------------------
    # FACE LANDMARK DETECTION
    # --------------------------------------------------------

    result = face_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    status = "FACE NOT DETECTED"

    status_color = (
        255,
        255,
        255
    )

    average_ear = None


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # ----------------------------------------------------
        # CALCULATE LEFT EAR
        # ----------------------------------------------------

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE,
            width,
            height
        )


        # ----------------------------------------------------
        # CALCULATE RIGHT EAR
        # ----------------------------------------------------

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE,
            width,
            height
        )


        # ----------------------------------------------------
        # AVERAGE EAR
        # ----------------------------------------------------

        average_ear = (
            left_ear +
            right_ear
        ) / 2.0


        # ====================================================
        # DRAW LEFT EYE LANDMARKS
        # ====================================================

        for index in LEFT_EYE:

            landmark = landmarks[index]

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


        # ====================================================
        # DRAW RIGHT EYE LANDMARKS
        # ====================================================

        for index in RIGHT_EYE:

            landmark = landmarks[index]

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


        # ====================================================
        # EYE STATE
        # ====================================================

        if average_ear < EAR_THRESHOLD:

            # ------------------------------------------------
            # EYES CLOSED
            # ------------------------------------------------

            if not eyes_closed:

                eyes_closed = True

                eyes_closed_start = time.time()


            closed_duration = (
                time.time()
                -
                eyes_closed_start
            )


            # ------------------------------------------------
            # DROWSY
            # ------------------------------------------------

            if closed_duration >= DROWSY_TIME:

                status = "DROWSY"

                status_color = (
                    0,
                    0,
                    255
                )

            else:

                status = "EYES CLOSED"

                status_color = (
                    0,
                    165,
                    255
                )


        else:

            # ------------------------------------------------
            # EYES OPEN
            # ------------------------------------------------

            if eyes_closed:

                if eyes_closed_start is not None:

                    closure_duration = (
                        time.time()
                        -
                        eyes_closed_start
                    )


                    # Short eye closure = blink
                    if closure_duration < DROWSY_TIME:

                        blink_count += 1


            eyes_closed = False

            eyes_closed_start = None


            status = "AWAKE"

            status_color = (
                0,
                255,
                0
            )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    fps = 1.0 / max(
        current_time -
        previous_time,
        0.001
    )

    previous_time = current_time


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "AutoVision AI - Drowsiness Detection",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # EAR
    # ========================================================

    if average_ear is not None:

        cv2.putText(
            frame,
            f"EAR: {average_ear:.3f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


    # ========================================================
    # BLINK COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"BLINKS: {blink_count}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # ========================================================
    # THRESHOLD
    # ========================================================

    cv2.putText(
        frame,
        f"EAR THRESHOLD: {EAR_THRESHOLD:.2f}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1
    )


    # ========================================================
    # STATUS
    # ========================================================

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (20, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        3
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "AutoVision AI - Drowsiness Detection",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

face_landmarker.close()

cv2.destroyAllWindows()

print("Drowsiness detection stopped.")