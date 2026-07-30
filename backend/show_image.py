import cv2

# Read the image
image = cv2.imread("datasets/sample.jpg")

# Display the image
cv2.imshow("Sample Image", image)

# Wait until a key is pressed
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()