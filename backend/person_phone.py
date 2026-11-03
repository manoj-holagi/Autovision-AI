import cv2
import time
import math
from ultralytics import YOLO


# ============================================================
# LOAD MODELS
# ============================================================

# YOLO Pose model
pose_model = YOLO("yolo26n-pose.pt")

# YOLO object detection model
object_model = YOLO("yolo26n.pt")


# ============================================================
# FIND CLASS IDs
# ============================================================

person_class_id = None
phone_class_id = None

for class_id, class_name in object_model.names.items():

    if class_name == "person":
        person_class_id = class_id

    elif class_name == "cell phone":
        phone_class_id = class_id


print("Person class ID:", person_class_id)
print("Phone class ID:", phone_class_id)


# ============================================================
# CHECK CLASS IDs
# ============================================================

if person_class_id is None:
    print("ERROR: Person class not found.")
    exit()

if phone_class_id is None:
    print("ERROR: Cell phone class not found.")
    exit()


# ============================================================
# OPEN EXTERNAL CAMERA
# ============================================================

cap = cv2.VideoCapture(1)

if not cap.isOpened():

    print("Error: Could not open external camera.")
    exit()


# ============================================================
# CAMERA SETTINGS
# ============================================================

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ============================================================
# FPS
# ============================================================

prev_time = time.time()


# ============================================================
# PHONE CONFIDENCE
# ============================================================

PHONE_CONFIDENCE = 0.45


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Read frame
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("Error: Could not read frame.")
        break


    height, width, _ = frame.shape


    # --------------------------------------------------------
    # Lists
    # --------------------------------------------------------

    persons = []
    phones = []
    ears = []


    # ========================================================
    # STEP 1 — PERSON DETECTION
    # ========================================================

    person_results = object_model(
        frame,
        imgsz=416,
        classes=[person_class_id],
        conf=0.30,
        verbose=False
    )

    person_result = person_results[0]


    for box in person_result.boxes:

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        # Make sure coordinates stay inside image

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)


        if x2 <= x1 or y2 <= y1:
            continue


        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2


        persons.append({
            "box": (x1, y1, x2, y2),
            "center": (center_x, center_y),
            "confidence": confidence
        })


    # ========================================================
    # STEP 2 — POSE DETECTION
    # ========================================================

    pose_results = pose_model(
        frame,
        imgsz=416,
        conf=0.30,
        verbose=False
    )

    pose_result = pose_results[0]


    if pose_result.keypoints is not None:

        for person_index in range(
            len(pose_result.keypoints.xy)
        ):

            keypoints = pose_result.keypoints.xy[
                person_index
            ]

            keypoint_conf = pose_result.keypoints.conf[
                person_index
            ]


            # ------------------------------------------------
            # COCO KEYPOINTS
            #
            # 0 = Nose
            # 1 = Left Eye
            # 2 = Right Eye
            # 3 = Left Ear
            # 4 = Right Ear
            # ------------------------------------------------


            # =================================================
            # LEFT EAR
            # =================================================

            left_ear_x = float(
                keypoints[3][0]
            )

            left_ear_y = float(
                keypoints[3][1]
            )

            left_ear_conf = float(
                keypoint_conf[3]
            )


            if left_ear_conf > 0.30:

                ears.append({
                    "x": int(left_ear_x),
                    "y": int(left_ear_y),
                    "side": "LEFT EAR",
                    "confidence": left_ear_conf
                })


            # =================================================
            # RIGHT EAR
            # =================================================

            right_ear_x = float(
                keypoints[4][0]
            )

            right_ear_y = float(
                keypoints[4][1]
            )

            right_ear_conf = float(
                keypoint_conf[4]
            )


            if right_ear_conf > 0.30:

                ears.append({
                    "x": int(right_ear_x),
                    "y": int(right_ear_y),
                    "side": "RIGHT EAR",
                    "confidence": right_ear_conf
                })


    # ========================================================
    # STEP 3 — PHONE DETECTION
    # ========================================================

    for person in persons:

        px1, py1, px2, py2 = person["box"]


        person_width = px2 - px1
        person_height = py2 - py1


        if person_width <= 0 or person_height <= 0:
            continue


        # ----------------------------------------------------
        # Take upper portion of person
        # ----------------------------------------------------

        upper_y2 = py1 + int(
            person_height * 0.70
        )


        upper_y2 = min(
            height,
            upper_y2
        )


        crop = frame[
            py1:upper_y2,
            px1:px2
        ]


        if crop.size == 0:
            continue


        # ----------------------------------------------------
        # Enlarge crop
        # ----------------------------------------------------

        enlarged_crop = cv2.resize(
            crop,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_LINEAR
        )


        # ====================================================
        # PHONE DETECTION
        # ====================================================

        phone_results = object_model(
            enlarged_crop,
            imgsz=640,
            classes=[phone_class_id],

            # IMPORTANT:
            # Increased from 0.20 to 0.45
            conf=PHONE_CONFIDENCE,

            verbose=False
        )


        phone_result = phone_results[0]


        for box in phone_result.boxes:

            confidence = float(
                box.conf[0]
            )


            # ------------------------------------------------
            # Extra confidence protection
            # ------------------------------------------------

            if confidence < PHONE_CONFIDENCE:
                continue


            cx1, cy1, cx2, cy2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # Convert from enlarged crop
            # ------------------------------------------------

            cx1 = int(cx1 / 2)
            cy1 = int(cy1 / 2)
            cx2 = int(cx2 / 2)
            cy2 = int(cy2 / 2)


            # ------------------------------------------------
            # Convert to original frame
            # ------------------------------------------------

            phone_x1 = px1 + cx1
            phone_y1 = py1 + cy1
            phone_x2 = px1 + cx2
            phone_y2 = py1 + cy2


            # Keep inside frame

            phone_x1 = max(
                0,
                phone_x1
            )

            phone_y1 = max(
                0,
                phone_y1
            )

            phone_x2 = min(
                width,
                phone_x2
            )

            phone_y2 = min(
                height,
                phone_y2
            )


            phone_center_x = (
                phone_x1 + phone_x2
            ) // 2

            phone_center_y = (
                phone_y1 + phone_y2
            ) // 2


            phones.append({
                "box": (
                    phone_x1,
                    phone_y1,
                    phone_x2,
                    phone_y2
                ),
                "center": (
                    phone_center_x,
                    phone_center_y
                ),
                "confidence": confidence
            })


    # ========================================================
    # STEP 4 — DRAW PERSON
    # ========================================================

    for person in persons:

        x1, y1, x2, y2 = person["box"]

        confidence = person["confidence"]


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Person {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ========================================================
    # STEP 5 — DRAW EAR KEYPOINTS
    # ========================================================

    for ear in ears:

        x = ear["x"]
        y = ear["y"]

        side = ear["side"]


        # Draw ear point

        cv2.circle(
            frame,
            (x, y),
            8,
            (255, 0, 0),
            -1
        )


        # Draw label

        cv2.putText(
            frame,
            side,
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )


    # ========================================================
    # STEP 6 — CHECK PHONE NEAR EAR
    # ========================================================

    phone_near_ear = False

    nearest_distance = None


    for phone in phones:

        x1, y1, x2, y2 = phone["box"]

        phone_x, phone_y = phone["center"]

        confidence = phone["confidence"]


        # ----------------------------------------------------
        # Draw phone bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            3
        )


        # ----------------------------------------------------
        # Phone center
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (phone_x, phone_y),
            6,
            (0, 0, 255),
            -1
        )


        # ----------------------------------------------------
        # Phone label
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"PHONE {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )


        # ====================================================
        # COMPARE PHONE WITH EARS
        # ====================================================

        for ear in ears:

            ear_x = ear["x"]
            ear_y = ear["y"]


            # Calculate distance

            distance = math.sqrt(
                (phone_x - ear_x) ** 2
                +
                (phone_y - ear_y) ** 2
            )


            if (
                nearest_distance is None
                or distance < nearest_distance
            ):

                nearest_distance = distance


            # ------------------------------------------------
            # Dynamic distance threshold
            # ------------------------------------------------

            threshold = width * 0.20


            # ------------------------------------------------
            # Phone near ear
            # ------------------------------------------------

            if distance < threshold:

                phone_near_ear = True


                # Draw connection line

                cv2.line(
                    frame,
                    (phone_x, phone_y),
                    (ear_x, ear_y),
                    (0, 0, 255),
                    3
                )


                # Show distance

                midpoint_x = (
                    phone_x + ear_x
                ) // 2

                midpoint_y = (
                    phone_y + ear_y
                ) // 2


                cv2.putText(
                    frame,
                    f"{distance:.0f}px",
                    (
                        midpoint_x,
                        midpoint_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )


    # ========================================================
    # STEP 7 — STATUS
    # ========================================================

    if phone_near_ear:

        cv2.putText(
            frame,
            "WARNING: PHONE NEAR EAR!",
            (20, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3
        )


    elif phones:

        cv2.putText(
            frame,
            "PHONE DETECTED",
            (20, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 165, 255),
            3
        )


    else:

        cv2.putText(
            frame,
            "STATUS: SAFE",
            (20, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            3
        )


    # ========================================================
    # STEP 8 — FPS
    # ========================================================

    current_time = time.time()


    fps = 1 / max(
        current_time - prev_time,
        0.001
    )


    prev_time = current_time


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # ========================================================
    # STEP 9 — COUNTS
    # ========================================================

    cv2.putText(
        frame,
        f"Persons: {len(persons)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Phones: {len(phones)}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AutoVision AI - Driver Phone Detection",
        frame
    )


    # ========================================================
    # PRESS Q TO EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()