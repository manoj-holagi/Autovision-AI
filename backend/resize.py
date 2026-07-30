import cv2
image = cv2.imread("datasets/sample.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not load the image.")
    exit()

resized = cv2.resize(image,(800,600)) 

cv2.imshow("res", resized)


# Save the output image
cv2.imwrite("datasets/resized.jpg", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
