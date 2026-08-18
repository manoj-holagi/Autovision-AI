import cv2

# Variables for mouse selection
start_x = 0
start_y = 0
end_x = 0
end_y = 0
drawing = False


# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, drawing

    # Mouse button pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x = x
        start_y = y
        end_x = x
        end_y = y
        drawing = True

    # Mouse is moving while button is pressed
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_x = x
        end_y = y

    # Mouse button released
    elif event == cv2.EVENT_LBUTTONUP:
        end_x = x
        end_y = y
        drawing = False


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

window_name = "AutoVision AI - ROI Selection"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

ret, frame = cap.read()

while ret:

    # Create a copy so we don't permanently draw
    # the rectangle on the original frame
    display = frame.copy()

    # Draw rectangle while selecting
    if drawing:
        cv2.rectangle(
            display,
            (start_x, start_y),
            (end_x, end_y),
            (0, 255, 0),
            2
        )

    # Draw final rectangle after mouse release
    elif start_x != end_x and start_y != end_y:
        cv2.rectangle(
            display,
            (start_x, start_y),
            (end_x, end_y),
            (0, 255, 0),
            2
        )

    # Display instructions
    cv2.putText(
        display,
        "Click and drag to select ROI | Press Q to quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(window_name, display)

    # Press Q to exit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    # Read next frame
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()