import cv2

cap = cv2.VideoCapture("datasets/samplevid.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, frame = cap.read()

while ret:
    cv2.imshow("Video", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()