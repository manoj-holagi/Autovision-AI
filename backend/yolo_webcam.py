import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo26n.pt")

# Classes we want AutoVision AI to focus on
allowed_classes = {
    "person",
    "cell phone"
}

# Open external camera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open external camera.")
    exit()

# FPS tracking
prev_time = time.time()

while True:

    # Read camera frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection
    results = model(
        frame,
        imgsz=320,
        verbose=False
    )

    # Get first result
    result = results[0]

    # Count detected objects
    object_count = 0

    # Process every detected object
    for box in result.boxes:

        # Get confidence
        confidence = float(box.conf[0])

        # Ignore weak detections
        if confidence < 0.50:
            continue

        # Get class ID
        class_id = int(box.cls[0])

        # Get object name
        class_name = result.names[class_id]

        # Only process selected classes
        if class_name not in allowed_classes:
            continue

        object_count += 1

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Calculate object center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw center point
        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        # Create object label
        label = f"{class_name} {confidence:.2f}"

        # Draw object label
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Display center coordinates
        coordinates = f"({center_x}, {center_y})"

        cv2.putText(
            frame,
            coordinates,
            (center_x + 10, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
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
        "AutoVision AI - YOLO",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()