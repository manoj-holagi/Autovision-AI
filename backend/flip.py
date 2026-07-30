import cv2

image = cv2.imread("datasets/sample.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not load the image.")
    exit()

flip = cv2.flip(image, 1)

cv2.imshow("flp", flip)


# Save the output image
cv2.imwrite("datasets/flip.jpg", flip)

cv2.waitKey(0)
cv2.destroyAllWindows()