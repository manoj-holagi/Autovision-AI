import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo26n.pt")

# Classes we care about
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

    cv2.putText(
        frame,
        "DRIVER REGION",
        (region_x1, max(region_y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    # Run YOLO detection
    # 416 gives the model more visual detail than 320
    results = model(
        frame,
        imgsz=416,
        verbose=False
    )

    result = results[0]

    # Count objects
    object_count = 0

    # Phone warning status
    phone_warning = False

    # Process every detection
    for box in result.boxes:

        # Get confidence
        confidence = float(box.conf[0])

        # Lower threshold temporarily for testing
        if confidence < 0.30:
            continue

        # Get class ID
        class_id = int(box.cls[0])

        # Get class name
        class_name = result.names[class_id]

        # Ignore objects we don't care about
        if class_name not in allowed_classes:
            continue

        object_count += 1

        # Get bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Calculate center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Check if center is inside driver region
        inside_region = (
            region_x1 <= center_x <= region_x2
            and
            region_y1 <= center_y <= region_y2
        )

        # Phone inside driver region
        if class_name == "cell phone" and inside_region:
            phone_warning = True

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

        # Location
        location = "INSIDE" if inside_region else "OUTSIDE"

        cv2.putText(
            frame,
            location,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
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

    # Display warning/status
    if phone_warning:

        cv2.putText(
            frame,
            "WARNING: PHONE DETECTED!",
            (20, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            frame,
            "STATUS: SAFE",
            (20, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3
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
        "AutoVision AI - Phone Warning",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()

# Close windows
cv2.destroyAllWindows()