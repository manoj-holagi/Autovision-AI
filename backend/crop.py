import cv2

image = cv2.imread("datasets/sample.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not load the image.")
    exit()

cropped = image[100:400,200:700]

cv2.imshow("crp", cropped)


# Save the output image
cv2.imwrite("datasets/cropped.jpg", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()