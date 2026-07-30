import cv2

# Load the image
image = cv2.imread("datasets/sample.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not load the image.")
    exit()

# Draw a green rectangle
# (top-left), (bottom-right), (B, G, R), thickness
cv2.rectangle(image, (50, 50), (250, 200), (0, 255, 0), 3)

# Draw a blue circle
# (center), radius, (B, G, R), thickness
cv2.circle(image, (400, 150), 60, (255, 0, 0), 3)

# Draw a red line
# start point, end point, (B, G, R), thickness
cv2.line(image, (50, 300), (500, 300), (0, 0, 255), 3)

# Write text
# image, text, position, font, font scale, color, thickness
cv2.putText(
    image,
    "OpenCV Demo",
    (50, 350),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

# Display the image
cv2.imshow("Modified Image", image)

# Save the output image
cv2.imwrite("datasets/output.jpg", image)

# Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()