import cv2

start_x = 0
start_y = 0
end_x = 0
end_y = 0
drawing = False
roi = None


def mouse_callback(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, drawing, roi

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x = x
        start_y = y
        end_x = x
        end_y = y
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_x = x
        end_y = y

    elif event == cv2.EVENT_LBUTTONUP:
        end_x = x
        end_y = y
        drawing = False

        # Make sure coordinates are in the correct order
        x1 = min(start_x, end_x)
        x2 = max(start_x, end_x)
        y1 = min(start_y, end_y)
        y2 = max(start_y, end_y)

        # Crop selected region
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2].copy()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

window_name = "AutoVision AI - ROI Crop"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

ret, frame = cap.read()

while ret:

    display = frame.copy()

    # Draw selection rectangle while dragging
    if drawing:
        cv2.rectangle(
            display,
            (start_x, start_y),
            (end_x, end_y),
            (0, 255, 0),
            2
        )

    # Show selected ROI
    if roi is not None:
        cv2.imshow("Selected ROI", roi)

    cv2.putText(
        display,
        "Click and drag to crop | Q = quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(window_name, display)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()