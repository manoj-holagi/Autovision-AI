import cv2

image = cv2.imread("datasets/sample.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not load the image.")
    exit()

resized = cv2.resize(image,(800,600))     
grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
flip = cv2.flip(grayscale, 1)

cv2.putText(
    flip,
    "AutoVision AI",
    (50, 350),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

cv2.imshow("flp", flip)


# Save the output image
cv2.imwrite("datasets/pipeline.jpg", flip)

cv2.waitKey(0)
cv2.destroyAllWindows()