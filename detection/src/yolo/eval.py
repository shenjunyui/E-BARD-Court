from ultralytics import YOLO

# Load a YOLOv8 model, either from a checkpoint or a pre-trained version
custom_model_path = '../../model/BODD_yolov8n_0001.pt'

# Load the custom YOLOv8 model
model = YOLO(custom_model_path)
# Path to the dataset YAML file

dataset_path = "../../data/yolo/data.yaml"


from pathlib import Path

# Assuming you already have:
results = model.val(data=dataset_path, imgsz=704, batch=16, save=True, save_txt=True,save_conf=True, iou=0.5, conf=0.25)
