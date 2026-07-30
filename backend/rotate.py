import cv2

image = cv2.imread("datasets/sample.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not load the image.")
    exit()
rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("rot", rotated_90)


# Save the output image
cv2.imwrite("datasets/rotated_90.jpg", rotated_90)

cv2.waitKey(0)
cv2.destroyAllWindows()