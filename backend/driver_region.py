import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo26n.pt")

# Classes AutoVision AI is interested in
allowed_classes = {
    "person",
    "cell phone"
}

# Open external camera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open external camera.")
    exit()

prev_time = time.time()

while True:

    # Read camera frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Get frame dimensions
    height, width, channels = frame.shape

    # Define driver region
    region_x1 = width // 4
    region_y1 = height // 6

    region_x2 = (width * 3) // 4
    region_y2 = (height * 5) // 6

    # Draw driver region
    cv2.rectangle(
        frame,
        (region_x1, region_y1),
        (region_x2, region_y2),
        (255, 0, 0),
        2
    )

    # Label driver region
    cv2.putText(
        frame,
        "DRIVER REGION",
        (region_x1, region_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    # Run YOLO
    results = model(
        frame,
        imgsz=320,
        verbose=False
    )

    result = results[0]

    object_count = 0

    # Process detections
    for box in result.boxes:

        confidence = float(box.conf[0])

        if confidence < 0.50:
            continue

        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        if class_name not in allowed_classes:
            continue

        object_count += 1

        # Bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Object center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Check whether center is inside driver region
        inside_region = (
            region_x1 <= center_x <= region_x2
            and
            region_y1 <= center_y <= region_y2
        )

        # Choose label
        if inside_region:
            location = "INSIDE"
        else:
            location = "OUTSIDE"

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw center
        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        # Object label
        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 30, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Location label
        cv2.putText(
            frame,
            location,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Display object count
    cv2.putText(
        frame,
        f"Objects: {object_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Display webcam
    cv2.imshow(
        "AutoVision AI - Driver Region",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()