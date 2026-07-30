import cv2

image = cv2.imread("datasets/sample.jpg")

# Read the pixel at row 100, column 100
pixel = image[100, 100]

print("Pixel at (100,100):", pixel)
print("Blue :", pixel[0])
print("Green:", pixel[1])
print("Red  :", pixel[2])