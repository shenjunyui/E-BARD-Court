import os
import sys
import torch
import numpy as np  # <-- 1. ADD THIS IMPORT
from tqdm import tqdm
from ultralytics import YOLO
from rfdetr import RFDETRNano  # Assuming rfdetr is installed
from PIL import Image
import supervision as sv
from supervision.metrics.mean_average_precision import MeanAveragePrecision
from supervision.metrics import Precision, Recall, F1Score

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = ".."
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Model checkpoints
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "BODD_yolov8n_0001.pt")
RFDETR_MODEL_PATH = os.path.join(MODEL_DIR, "BODD_rf-detr-nano_0000/checkpoint_best_total.pth")

# Test dataset
YOLO_TEST_IMAGES = os.path.join(DATA_DIR, "yolo","test", "images")
YOLO_TEST_LABELS = os.path.join(DATA_DIR, "yolo","test", "labels")

COCO_TEST_ANN = os.path.join(DATA_DIR, "coco", "test", "_annotations.coco.json")
COCO_TEST_IMAGES = os.path.join(DATA_DIR, "coco", "test", "images")

YOLO_DATA_YAML = os.path.join(DATA_DIR, "yolo", "data.yaml")

# Inference thresholds (identical for fairness)
# NOTE: For mAP calculation, CONF_THRESHOLD should be very low
# to get all detections for the Precision-Recall curve.
CONF_THRESHOLD = 0.25
IOU_NMS = 0.5

# ============================================================
# CHECK FILES AND FOLDERS
# ============================================================

required_paths = [
    YOLO_MODEL_PATH,
    RFDETR_MODEL_PATH,
    YOLO_TEST_IMAGES,
    YOLO_TEST_LABELS,
    COCO_TEST_ANN,
    YOLO_DATA_YAML,
]

print("\nChecking required files...")
for p in required_paths:
    if not os.path.exists(p):
        print(f"Missing: {p}")
        sys.exit(1)
print("All required files found.\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# ============================================================
# 0. DATASET CONSISTENCY CHECK
# ============================================================

print("Checking YOLO and COCO annotation consistency...")

# Load YOLO ground truth dataset
yolo_gt_dataset = sv.DetectionDataset.from_yolo(
    data_yaml_path=YOLO_DATA_YAML,
    images_directory_path=YOLO_TEST_IMAGES,
    annotations_directory_path=YOLO_TEST_LABELS,
)

# Count YOLO instances
yolo_instances = sum(len(ann.xyxy) for _, _, ann in yolo_gt_dataset)

# Load COCO ground truth dataset
coco_gt_dataset = sv.DetectionDataset.from_coco(
    images_directory_path=COCO_TEST_IMAGES,
    annotations_path=COCO_TEST_ANN,
)

# Count COCO instances
coco_instances = sum(len(ann.xyxy) for _, _, ann in coco_gt_dataset)

print(f"  YOLO instances: {yolo_instances}")
print(f"  COCO instances: {coco_instances}")

if yolo_instances != coco_instances:
    raise ValueError(
        f"Mismatch in number of annotations! "
        f"YOLO={yolo_instances} vs COCO={coco_instances}"
    )

print("Annotation counts match.\n")

# ============================================================
# 1. YOLOv8 EVALUATION (with Path Alignment)
# ============================================================

print("Running YOLOv8 inference and evaluation...")

yolo_model = YOLO(YOLO_MODEL_PATH)

# ---
# CORRECTION: Added conf=CONF_THRESHOLD and iou=IOU_NMS
# ---
yolo_results = yolo_model.predict(
    source=YOLO_TEST_IMAGES,
    imgsz=704,
    device=device,
    conf=CONF_THRESHOLD,  # Use the low confidence threshold
    iou=IOU_NMS           # Use the specified NMS threshold
)

# ---
# FIX: Create a lookup dictionary for Ground Truth annotations.
# This solves the path mismatch error caused by zip()
# when datasets are loaded in different orders.
# ---
print("\nCreating YOLO ground truth lookup dictionary...")
gt_lookup_yolo = {}
for (image_path, _, annotations) in tqdm(yolo_gt_dataset, desc="Loading YOLO GT annotations"):
    normalized_path = os.path.normpath(image_path)
    gt_lookup_yolo[normalized_path] = annotations
print(f"YOLO ground truth lookup created with {len(gt_lookup_yolo)} entries.")


pred_detections_yolo = []
gt_detections_yolo = []

# ---
# FIX: Loop over prediction results and use the lookup, not zip().
# ---
for pred_result in (yolo_results):
    normalized_pred_path = os.path.normpath(pred_result.path)
    
    if normalized_pred_path in gt_lookup_yolo:
        annotations = gt_lookup_yolo[normalized_pred_path]
        det = sv.Detections.from_ultralytics(pred_result)
        pred_detections_yolo.append(det)
        gt_detections_yolo.append(annotations)
    else:
        print(f"\nWarning: Could not find matching YOLO GT for prediction path:")
        print(f"  {normalized_pred_path}. Skipping this sample.")


# ---
# BLOCK REMOVED: The all-class metric computation is no longer needed.
# We will compute metrics inside a loop instead.
# ---
# yolo_map_metric = MeanAveragePrecision()
# ... (rest of the block removed) ...
# print(yolo_map_results)
# ... (per-class printing block also removed) ...


# ============================================================
# 1. (Continued) YOLOv8 PER-CLASS EVALUATION LOOP
# ============================================================

print("\nRunning YOLOv8 Per-Class Evaluation Loop...")
class_names_yolo = yolo_gt_dataset.classes
# Store results for a final summary
yolo_per_class_results = {}

for class_id, class_name in enumerate(class_names_yolo):
    print(f"\n--- Evaluating YOLOv8 for Class: {class_name} (ID: {class_id}) ---")
    
    pred_detections_class = []
    gt_detections_class = []

    # Filter our full, aligned lists for *only* this class
    for pred_det, gt_ann in zip(pred_detections_yolo, gt_detections_yolo):
        pred_filtered = pred_det[pred_det.class_id == class_id]
        gt_filtered = gt_ann[gt_ann.class_id == class_id]
        
        # We must append *something* for every image, even if empty,
        # to keep the list lengths identical for the evaluator.
        pred_detections_class.append(pred_filtered)
        gt_detections_class.append(gt_filtered)

    # We must instantiate a new metric object *inside* the loop
    # to reset the state for each class.
    map_metric_class = MeanAveragePrecision()
    precision_metric_class = Precision() 
    recall_metric_class = Recall() 
    f1_metric_class = F1Score()    

    map_metric_class.update(pred_detections_class, gt_detections_class)
    precision_metric_class.update(pred_detections_class, gt_detections_class) 
    recall_metric_class.update(pred_detections_class, gt_detections_class)    
    f1_metric_class.update(pred_detections_class, gt_detections_class) 
    
    map_results_class = map_metric_class.compute()
    precision_results_class = precision_metric_class.compute() 
    recall_results_class = recall_metric_class.compute()   
    f1_results_class = f1_metric_class.compute()      

    print("Precision")
    print(precision_results_class)
    print("Recall")
    print(recall_results_class)
    print("F1")
    print(f1_results_class)
    # Print all metrics for this class
    print(map_results_class)
    
    yolo_per_class_results[class_name] = {
        'map_50': getattr(map_results_class, 'map_50', 0.0),
        'map_50_95': getattr(map_results_class, 'map_50_95', 0.0)        
    }
    
    
# ============================================================
# 2. RF-DETR EVALUATION (with Path Alignment)
# ============================================================

print("\nRunning RF-DETR inference and evaluation...")

rfdetr_model = RFDETRNano(pretrain_weights=RFDETR_MODEL_PATH)

# ---
# FIX: Create a lookup dictionary for COCO Ground Truth annotations.
# ---
print("\nCreating COCO ground truth lookup dictionary...")
gt_lookup_coco = {}
# We use the coco_gt_dataset to get normalized paths and annotations
for (image_path, _, annotations) in (coco_gt_dataset):
    normalized_path = os.path.normpath(image_path)
    # Store the original path for Image.open() and the annotations
    gt_lookup_coco[normalized_path] = (image_path, annotations) 
print(f"COCO ground truth lookup created with {len(gt_lookup_coco)} entries.")

pred_detections_rfdetr = []
gt_detections_rfdetr = []

# ---
# FIX: Loop over the *ground truth* paths to run inference.
# This ensures we evaluate the same set of images as YOLO.
# ---
for normalized_gt_path, (original_path, annotations) in tqdm(gt_lookup_coco.items(), desc="Evaluating RF-DETR"):
    img = Image.open(original_path).convert("RGB")
    
    # ---
    # CORRECTION: Pass thresholds to RF-DETR's predict method.
    # ---
    try:
        detections = rfdetr_model.predict(
            img,
            resolution =704,
            conf_threshold=CONF_THRESHOLD,
            #nms_threshold=IOU_NMS
        )
    except TypeError as e:
        print(f"\nWarning: Could not pass thresholds to RFDETRNano.predict(): {e}")
        print("Falling back to default prediction. Metrics may be inaccurate.")
        detections = rfdetr_model.predict(img)

    pred_detections_rfdetr.append(detections)
    gt_detections_rfdetr.append(annotations)


# ============================================================
# 2. (Continued) RF-DETR PER-CLASS EVALUATION LOOP
# ============================================================

print("\nRunning RF-DETR Per-Class Evaluation Loop...")
class_names_coco = coco_gt_dataset.classes
rf_per_class_results = {}

for class_id, class_name in enumerate(class_names_coco):
    print(f"\n--- Evaluating RF-DETR for Class: {class_name} (ID: {class_id}) ---")
    
    pred_detections_class = []
    gt_detections_class = []

    # Filter our full, aligned lists for *only* this class
    for pred_det, gt_ann in zip(pred_detections_rfdetr, gt_detections_rfdetr):
        pred_filtered = pred_det[pred_det.class_id == class_id]
        gt_filtered = gt_ann[gt_ann.class_id == class_id]
        
        pred_detections_class.append(pred_filtered)
        gt_detections_class.append(gt_filtered)

    # Instantiate new metric object inside the loop
    map_metric_class = MeanAveragePrecision()
    precision_metric_class = Precision() 
    recall_metric_class = Recall() 
    f1_metric_class = F1Score()
    
    map_metric_class.update(pred_detections_class, gt_detections_class)    
    precision_metric_class.update(pred_detections_class, gt_detections_class)     
    recall_metric_class.update(pred_detections_class, gt_detections_class)    
    f1_metric_class.update(pred_detections_class, gt_detections_class)      

    map_results_class = map_metric_class.compute()
    precision_results_class = precision_metric_class.compute() 
    recall_results_class = recall_metric_class.compute()   
    f1_results_class = f1_metric_class.compute()      
    # Print all metrics for this class
    print(map_results_class)
    print("Precision")
    print(precision_results_class)
    print("Recall")
    print(recall_results_class)
    print("F1")
    print(f1_results_class)
    
    rf_per_class_results[class_name] = {
        'map_50': getattr(map_results_class, 'map_50', 0.0),
        'map_50_95': getattr(map_results_class, 'map_50_95', 0.0)       
    }
  





