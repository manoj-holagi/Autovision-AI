from ultralytics import YOLO
import cv2
import os


print("=" * 60)
print("AutoVision AI - Pothole Model Test")
print("=" * 60)


# ============================================================
# MODEL
# ============================================================

model_path = (
    "runs/segment/runs/pothole/"
    "pothole_seg_test/weights/best.pt"
)

model = YOLO(model_path)

print("Pothole model loaded successfully.")


# ============================================================
# TEST IMAGE
# ============================================================

image_path = "datasets/sample.jpg"

if not os.path.exists(image_path):
    print(f"ERROR: Image not found: {image_path}")
    exit()


# ============================================================
# RUN PREDICTION
# ============================================================

results = model(
    image_path,
    conf=0.35,
    imgsz=640,
    verbose=False
)


# ============================================================
# GET RESULT
# ============================================================

result = results[0]

print()
print("Detection Results:")
print("-" * 40)


if result.boxes is not None and len(result.boxes) > 0:

    for i, box in enumerate(result.boxes):

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        print(
            f"{i + 1}. "
            f"{class_name} "
            f"- Confidence: {confidence:.2f}"
        )

else:

    print("No pothole detected.")


# ============================================================
# DRAW RESULT
# ============================================================

annotated = result.plot()


# ============================================================
# SAVE RESULT
# ============================================================

output_path = "runs/pothole_test.jpg"

cv2.imwrite(output_path, annotated)

print()
print(f"Result saved to:")
print(output_path)


# ============================================================
# DISPLAY
# ============================================================

cv2.imshow(
    "AutoVision AI - Pothole Detection",
    annotated
)

print()
print("Press Q to close.")

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cv2.destroyAllWindows()

print("=" * 60)
print("Test completed.")
print("=" * 60)