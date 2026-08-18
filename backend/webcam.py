import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = cap.read()

while ret:

    # Get frame dimensions
    height, width, channels = frame.shape

    # Calculate center of the frame
    center_x = width // 2
    center_y = height // 2

    # Rectangle size
    rect_width = 200
    rect_height = 150

    # Calculate rectangle coordinates
    x1 = center_x - rect_width // 2
    y1 = center_y - rect_height // 2
    x2 = center_x + rect_width // 2
    y2 = center_y + rect_height // 2

    # Draw text
    cv2.putText(
        frame,
        "OpenCV Learning",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Draw center rectangle
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

    # Display webcam
    cv2.imshow("AutoVision AI - Webcam", frame)

    # Press q to exit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    # Read next frame
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()