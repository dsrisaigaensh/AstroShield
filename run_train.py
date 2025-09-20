import torch
from ultralytics import YOLO
import ultralytics.nn.tasks

# Allow YOLO DetectionModel for safe torch.load
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  

# Train the model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="SafetyDetectionYOLOv8_exp50"
)
