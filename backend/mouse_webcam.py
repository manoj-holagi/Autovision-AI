import cv2

# Store the latest mouse click coordinates
click_x = 0
click_y = 0
clicked = False


# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global click_x, click_y, clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        click_x = x
        click_y = y
        clicked = True
        print("Clicked:", x, y)


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Window name
window_name = "AutoVision AI - Mouse"

# Connect mouse callback to the window
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

ret, frame = cap.read()

while ret:

    # Get frame dimensions
    height, width, channels = frame.shape

    # Draw text
    cv2.putText(
        frame,
        "Click anywhere",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Draw circle where mouse was clicked
    if clicked:
        cv2.circle(
            frame,
            (click_x, click_y),
            10,
            (0, 255, 0),
            -1
        )

        # Display coordinates
        cv2.putText(
            frame,
            f"({click_x}, {click_y})",
            (click_x + 15, click_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display webcam
    cv2.imshow(window_name, frame)

    # Press q to exit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    # Read next frame
    ret, frame = cap.read()

# Cleanup
cap.release()
cv2.destroyAllWindows()