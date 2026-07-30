import cv2

# Load the image
image = cv2.imread("datasets/sample.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not load the image.")
    exit()
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     

# Display the image
cv2.imshow("grayimage", grayscale)


# Save the output image
cv2.imwrite("datasets/grayimage.jpg", grayscale)

cv2.waitKey(0)
cv2.destroyAllWindows()