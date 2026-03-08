from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model_path = "./Ultralytics/YOLOv8/yolov8n" #your ultralytics path
model = YOLO(model_path)

yaml_path = "../../data/yolo/data.yaml"


model.train(data=yaml_path, epochs=50, batch=32, imgsz=704,amp=True, mosaic=1.0,copy_paste=0.5, auto_augment='randaugment')

model.val()

model_name = "../../model/BODD_yolov8n_0001.pt"

model.save(model_name)
