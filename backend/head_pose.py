import cv2
import time
import math
import numpy as np
from ultralytics import YOLO


# ============================================================
# AutoVision AI - Driver Attention Detection
# ============================================================

print("=" * 60)
print("AutoVision AI - Driver Attention")
print("=" * 60)


# ============================================================
# LOAD YOLO POSE MODEL
# ============================================================

model = YOLO("yolo26n-pose.pt")

print("Pose model loaded successfully.")


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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ============================================================
# GENERAL SETTINGS
# ============================================================

POSE_CONFIDENCE = 0.25
KEYPOINT_CONFIDENCE = 0.15

CALIBRATION_FRAMES = 60

SMOOTHING_FRAMES = 5


# ============================================================
# CALIBRATION STORAGE
# ============================================================

calibration_h = []
calibration_v = []
calibration_pitch = []

calibrated = False

baseline_h = 0.0
baseline_v = 0.0
baseline_pitch = 0.0


# ============================================================
# SMOOTHING STORAGE
# ============================================================

history_h = []
history_v = []
history_pitch = []


# ============================================================
# FPS
# ============================================================

previous_time = time.time()


# ============================================================
# HELPER: SMOOTH VALUE
# ============================================================

def smooth_value(history, value, max_length=5):

    history.append(value)

    if len(history) > max_length:
        history.pop(0)

    return sum(history) / len(history)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # READ CAMERA
    # ========================================================

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera frame.")
        break


    height, width, _ = frame.shape


    # ========================================================
    # DEFAULT
    # ========================================================

    direction = "UNKNOWN"

    current_h = None
    current_v = None
    current_pitch = None

    delta_h = 0.0
    delta_v = 0.0
    delta_pitch = 0.0


    # ========================================================
    # YOLO POSE
    # ========================================================

    results = model(
        frame,
        imgsz=416,
        conf=POSE_CONFIDENCE,
        verbose=False
    )

    result = results[0]


    # ========================================================
    # CHECK PERSON
    # ========================================================

    if (
        result.keypoints is not None
        and len(result.keypoints.xy) > 0
    ):

        keypoints = result.keypoints.xy[0]
        confidences = result.keypoints.conf[0]


        # ====================================================
        # COCO KEYPOINTS
        # ====================================================
        #
        # 0 = Nose
        # 1 = Left Eye
        # 2 = Right Eye
        # 3 = Left Ear
        # 4 = Right Ear
        #
        # ====================================================

        nose_x = float(keypoints[0][0])
        nose_y = float(keypoints[0][1])

        left_eye_x = float(keypoints[1][0])
        left_eye_y = float(keypoints[1][1])

        right_eye_x = float(keypoints[2][0])
        right_eye_y = float(keypoints[2][1])

        left_ear_x = float(keypoints[3][0])
        left_ear_y = float(keypoints[3][1])

        right_ear_x = float(keypoints[4][0])
        right_ear_y = float(keypoints[4][1])


        nose_conf = float(confidences[0])
        left_eye_conf = float(confidences[1])
        right_eye_conf = float(confidences[2])
        left_ear_conf = float(confidences[3])
        right_ear_conf = float(confidences[4])


        # ====================================================
        # DRAW KEYPOINTS
        # ====================================================

        points = [

            (
                nose_x,
                nose_y,
                "NOSE",
                nose_conf
            ),

            (
                left_eye_x,
                left_eye_y,
                "L-EYE",
                left_eye_conf
            ),

            (
                right_eye_x,
                right_eye_y,
                "R-EYE",
                right_eye_conf
            ),

            (
                left_ear_x,
                left_ear_y,
                "L-EAR",
                left_ear_conf
            ),

            (
                right_ear_x,
                right_ear_y,
                "R-EAR",
                right_ear_conf
            )

        ]


        for x, y, label, confidence in points:

            if confidence > KEYPOINT_CONFIDENCE:

                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    5,
                    (255, 0, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        int(x) + 6,
                        int(y)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (255, 0, 0),
                    1
                )


        # ====================================================
        # BOTH EYES AVAILABLE
        # ====================================================

        if (
            nose_conf > KEYPOINT_CONFIDENCE
            and
            left_eye_conf > KEYPOINT_CONFIDENCE
            and
            right_eye_conf > KEYPOINT_CONFIDENCE
        ):

            # ------------------------------------------------
            # EYE CENTER
            # ------------------------------------------------

            eye_center_x = (
                left_eye_x +
                right_eye_x
            ) / 2.0

            eye_center_y = (
                left_eye_y +
                right_eye_y
            ) / 2.0


            # ------------------------------------------------
            # EYE DISTANCE
            # ------------------------------------------------

            eye_distance = math.sqrt(

                (
                    right_eye_x -
                    left_eye_x
                ) ** 2

                +

                (
                    right_eye_y -
                    left_eye_y
                ) ** 2

            )


            if eye_distance > 5:

                # =================================================
                # ORIGINAL HORIZONTAL / VERTICAL METRICS
                # =================================================

                raw_h = (
                    nose_x -
                    eye_center_x
                ) / eye_distance


                raw_v = (
                    nose_y -
                    eye_center_y
                ) / eye_distance


                current_h = smooth_value(
                    history_h,
                    raw_h,
                    SMOOTHING_FRAMES
                )


                current_v = smooth_value(
                    history_v,
                    raw_v,
                    SMOOTHING_FRAMES
                )


                # =================================================
                # NEW PITCH METRIC
                # =================================================
                #
                # We use the nose position relative to the ears.
                #
                # Looking DOWN:
                # nose moves further below the ears
                #
                # Looking UP:
                # nose moves closer to / above the ears
                #
                # =================================================

                ears_available = (
                    left_ear_conf > KEYPOINT_CONFIDENCE
                    and
                    right_ear_conf > KEYPOINT_CONFIDENCE
                )


                if ears_available:

                    ear_center_y = (
                        left_ear_y +
                        right_ear_y
                    ) / 2.0


                    ear_eye_distance = abs(
                        eye_center_y -
                        ear_center_y
                    )


                    if ear_eye_distance > 5:

                        raw_pitch = (
                            nose_y -
                            ear_center_y
                        ) / eye_distance


                        current_pitch = smooth_value(
                            history_pitch,
                            raw_pitch,
                            SMOOTHING_FRAMES
                        )


                # =================================================
                # CALIBRATION
                # =================================================

                if not calibrated:

                    calibration_h.append(
                        current_h
                    )

                    calibration_v.append(
                        current_v
                    )

                    if current_pitch is not None:

                        calibration_pitch.append(
                            current_pitch
                        )


                    remaining = (
                        CALIBRATION_FRAMES -
                        len(calibration_h)
                    )


                    # ---------------------------------------------
                    # CALIBRATION MESSAGE
                    # ---------------------------------------------

                    cv2.putText(
                        frame,
                        "CALIBRATING...",
                        (
                            20,
                            90
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2
                    )


                    cv2.putText(
                        frame,
                        "Look STRAIGHT at the camera",
                        (
                            20,
                            120
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )


                    cv2.putText(
                        frame,
                        f"Frames: {len(calibration_h)}/{CALIBRATION_FRAMES}",
                        (
                            20,
                            150
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )


                    # =================================================
                    # FINISH CALIBRATION
                    # =================================================

                    if len(calibration_h) >= CALIBRATION_FRAMES:

                        baseline_h = float(
                            np.median(
                                calibration_h
                            )
                        )


                        baseline_v = float(
                            np.median(
                                calibration_v
                            )
                        )


                        if len(calibration_pitch) > 0:

                            baseline_pitch = float(
                                np.median(
                                    calibration_pitch
                                )
                            )

                        else:

                            baseline_pitch = 0.0


                        calibrated = True


                        print()
                        print("=" * 60)
                        print("CALIBRATION COMPLETE")
                        print(
                            f"Baseline H     : {baseline_h:.3f}"
                        )
                        print(
                            f"Baseline V     : {baseline_v:.3f}"
                        )
                        print(
                            f"Baseline Pitch : {baseline_pitch:.3f}"
                        )
                        print("=" * 60)
                        print()


                # =================================================
                # ATTENTION DETECTION
                # =================================================

                else:

                    # ------------------------------------------------
                    # DELTA FROM FORWARD POSITION
                    # ------------------------------------------------

                    delta_h = (
                        current_h -
                        baseline_h
                    )


                    delta_v = (
                        current_v -
                        baseline_v
                    )


                    if current_pitch is not None:

                        delta_pitch = (
                            current_pitch -
                            baseline_pitch
                        )


                    # ------------------------------------------------
                    # ABSOLUTE VALUES
                    # ------------------------------------------------

                    abs_h = abs(delta_h)
                    abs_v = abs(delta_v)
                    abs_pitch = abs(delta_pitch)


                    # =================================================
                    # THRESHOLDS
                    # =================================================

                    HORIZONTAL_THRESHOLD = 0.28

                    UP_THRESHOLD = 0.20

                    DOWN_V_THRESHOLD = 0.15

                    DOWN_PITCH_THRESHOLD = 0.12


                    # =================================================
                    # DIRECTION PRIORITY
                    # =================================================

                    # ------------------------------------------------
                    # DOWN
                    # ------------------------------------------------
                    #
                    # Either:
                    #
                    # 1. Vertical movement is clearly down
                    #
                    # OR
                    #
                    # 2. New pitch metric indicates down
                    #
                    # ------------------------------------------------

                    down_detected = False


                    if (
                        delta_v >
                        DOWN_V_THRESHOLD
                        and
                        abs_v >= abs_h * 0.8
                    ):

                        down_detected = True


                    if (
                        delta_pitch >
                        DOWN_PITCH_THRESHOLD
                    ):

                        down_detected = True


                    if down_detected:

                        direction = "LOOKING_DOWN"


                    # ------------------------------------------------
                    # UP
                    # ------------------------------------------------

                    elif (
                        delta_v <
                        -UP_THRESHOLD
                        and
                        abs_v >
                        abs_h
                    ):

                        direction = "LOOKING_UP"


                    # ------------------------------------------------
                    # RIGHT
                    # ------------------------------------------------

                    elif (
                        delta_h <
                        -HORIZONTAL_THRESHOLD
                    ):

                        direction = "LOOKING_RIGHT"


                    # ------------------------------------------------
                    # LEFT
                    # ------------------------------------------------

                    elif (
                        delta_h >
                        HORIZONTAL_THRESHOLD
                    ):

                        direction = "LOOKING_LEFT"


                    # ------------------------------------------------
                    # FORWARD
                    # ------------------------------------------------

                    else:

                        direction = "LOOKING_FORWARD"


                    # =================================================
                    # NOSE-EYE LINE
                    # =================================================

                    cv2.line(
                        frame,

                        (
                            int(eye_center_x),
                            int(eye_center_y)
                        ),

                        (
                            int(nose_x),
                            int(nose_y)
                        ),

                        (0, 255, 255),

                        2
                    )


    # ========================================================
    # STATUS TEXT
    # ========================================================

    if direction == "LOOKING_FORWARD":

        status_text = "ATTENTION: FORWARD"

        status_color = (
            0,
            255,
            0
        )


    elif direction == "LOOKING_LEFT":

        status_text = "ATTENTION: LEFT"

        status_color = (
            0,
            165,
            255
        )


    elif direction == "LOOKING_RIGHT":

        status_text = "ATTENTION: RIGHT"

        status_color = (
            0,
            165,
            255
        )


    elif direction == "LOOKING_UP":

        status_text = "ATTENTION: UP"

        status_color = (
            255,
            255,
            0
        )


    elif direction == "LOOKING_DOWN":

        status_text = "ATTENTION: DOWN"

        status_color = (
            0,
            0,
            255
        )


    else:

        status_text = "ATTENTION: UNKNOWN"

        status_color = (
            255,
            255,
            255
        )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    fps = 1 / max(
        current_time -
        previous_time,
        0.001
    )

    previous_time = current_time


    # ========================================================
    # TOP INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (
            20,
            40
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "AutoVision AI - Driver Attention",
        (
            20,
            75
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DEBUG VALUES
    # ========================================================

    if current_h is not None:

        cv2.putText(
            frame,
            f"dH: {delta_h:.2f}",
            (
                20,
                185
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


    if current_v is not None:

        cv2.putText(
            frame,
            f"dV: {delta_v:.2f}",
            (
                20,
                215
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


    if current_pitch is not None:

        cv2.putText(
            frame,
            f"Pitch: {delta_pitch:.2f}",
            (
                20,
                245
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


    # ========================================================
    # RAW VALUES
    # ========================================================

    if current_h is not None:

        cv2.putText(
            frame,
            f"H: {current_h:.2f}",
            (
                20,
                275
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


    if current_v is not None:

        cv2.putText(
            frame,
            f"V: {current_v:.2f}",
            (
                20,
                305
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


    # ========================================================
    # CALIBRATION STATUS
    # ========================================================

    if calibrated:

        cv2.putText(
            frame,
            "CALIBRATED",
            (
                20,
                335
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2
        )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    cv2.putText(
        frame,
        status_text,
        (
            20,
            height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        status_color,
        3
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AutoVision AI - Driver Attention",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Q = QUIT
    if key == ord("q"):

        break


    # R = RECALIBRATE
    if key == ord("r"):

        print("Recalibrating...")

        calibration_h = []
        calibration_v = []
        calibration_pitch = []

        calibrated = False

        baseline_h = 0.0
        baseline_v = 0.0
        baseline_pitch = 0.0

        history_h = []
        history_v = []
        history_pitch = []


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("AutoVision AI stopped.")