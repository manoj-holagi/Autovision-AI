from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo26n.pt")

# Run detection
results = model("datasets/sample.jpg")

# Get first image result
result = results[0]

# Print detected objects
for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    coordinates = box.xyxy[0].tolist()

    class_name = result.names[class_id]

    print("Object:", class_name)
    print("Confidence:", confidence)
    print("Coordinates:", coordinates)
    print("--------------------")

# Display results
result.show()