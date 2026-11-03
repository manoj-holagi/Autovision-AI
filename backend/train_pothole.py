from ultralytics import YOLO


print("=" * 60)
print("AutoVision AI - Pothole Segmentation Training")
print("=" * 60)


# ------------------------------------------------------------
# Load YOLO26 segmentation model
# ------------------------------------------------------------

model = YOLO("yolo26n-seg.pt")


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

data_yaml = (
    "datasets/pothole/"
    "Pothole_Segmentation_YOLOv8/"
    "data.yaml"
)


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

results = model.train(
    data=data_yaml,
    epochs=5,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    project="runs/pothole",
    name="pothole_seg_test",
    exist_ok=True
)


print("=" * 60)
print("Training test completed.")
print("=" * 60)