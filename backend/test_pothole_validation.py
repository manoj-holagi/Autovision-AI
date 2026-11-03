from ultralytics import YOLO
import os
import glob


print("=" * 60)
print("AutoVision AI - Pothole Validation Test")
print("=" * 60)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model_path = (
    "runs/segment/runs/pothole/"
    "pothole_seg_test/weights/best.pt"
)

model = YOLO(model_path)

print("Pothole model loaded successfully.")


# ============================================================
# VALIDATION IMAGES
# ============================================================

image_folder = (
    "datasets/pothole/"
    "Pothole_Segmentation_YOLOv8/"
    "valid/images"
)

images = glob.glob(
    os.path.join(image_folder, "*.jpg")
)

# Test only first 5 images
images = images[:5]

print()
print(f"Testing {len(images)} validation images...")
print("-" * 60)


# ============================================================
# RUN PREDICTION
# ============================================================

for index, image_path in enumerate(images):

    print()
    print(f"Image {index + 1}: {os.path.basename(image_path)}")

    results = model(
        image_path,
        conf=0.30,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:

        for box in result.boxes:

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            print(
                f"  {class_name}: "
                f"{confidence:.2f}"
            )

    else:

        print("  No pothole detected.")


# ============================================================
# SAVE ALL VALIDATION RESULTS
# ============================================================

output_folder = "runs/pothole_validation"

os.makedirs(output_folder, exist_ok=True)


results = model(
    images,
    conf=0.30,
    imgsz=640,
    save=True,
    project="runs",
    name="pothole_validation",
    exist_ok=True,
    verbose=False
)


print()
print("=" * 60)
print("Validation test completed.")
print("=" * 60)

print()
print("Annotated images saved in:")

print(
    "runs/pothole_validation/"
)