import cv2

# Read an image
image = cv2.imread("datasets/sample.jpg")

# Check if the image was loaded
if image is None:
    print("Image not found!")
else:
    print("Image loaded successfully!")
    print("Shape    :", image.shape)
    print("Height   :", image.shape[0])
    print("Width    :", image.shape[1])
    print("Channels :", image.shape[2])