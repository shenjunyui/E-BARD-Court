#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def plot_detections(image_path, detections, class_names=None,save_path="0.png"):
    """
    image_path : str
        Path to the source image
    detections : sv.Detections
        Detections object containing xyxy, class_id, confidence
    class_names : list of class names (optional)
    """

    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    
    xyxy = detections.xyxy
    class_ids = detections.class_id
    conf = detections.confidence

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box
        w, h = (x2 - x1), (y2 - y1)

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

        label = ""
        if class_names:
            label += class_names[class_ids[i]]
        if conf is not None:
            label += f" {conf[i]:.2f}"

        ax.text(
            x1, y1 - 5, label,
            color="yellow",
            fontsize=10,
            weight="bold",
            bbox=dict(facecolor="black", alpha=0.5)
        )

    ax.axis("off")
    plt.show()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

import json
import os
import re
import sys
from collections import defaultdict
import numpy as np

# --- 1. IMPORTS ---
# Using imports from both scripts
import supervision as sv
from supervision.metrics import Precision, Recall, F1Score

# --- 2. CONFIGURATION ---
# Using the IOU_THRESHOLD from your last script
IOU_THRESHOLD = 0.5

# --- 3. PARSING FUNCTION (From your JSON script) ---
def parse_bboxes(bbox_str, is_prediction=False):
    """Parses bbox JSON strings from the input file."""
    # Find the JSON list, ignoring markdown fences
    match = re.search(r'(\[.*\]|\{.*\})', bbox_str, re.DOTALL)
    if not match:
        return defaultdict(list)
    clean_str = match.group(1)

    try:
        bboxes = json.loads(clean_str)
        grouped_boxes = defaultdict(list)
        for bbox_info in bboxes:
            label = bbox_info.get("label")
            bbox_2d = bbox_info.get("bbox_2d")
            score = bbox_info.get("score", 1.0 if not is_prediction else 0.5) 
            
            if not (label and bbox_2d and len(bbox_2d) == 4):
                continue

            if is_prediction:
                # Prediction format is [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox_2d
                if x2 > x1 and y2 > y1:
                    w = x2 - x1
                    h = y2 - y1
                    grouped_boxes[label].append({"bbox": [x1, y1, w, h], "score": score})
            else:
                # Ground truth format is [x, y, w, h]
                grouped_boxes[label].append([bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3]])
        return grouped_boxes
    except json.JSONDecodeError:
        return defaultdict(list)
    except Exception:
        return defaultdict(list)

# --- 4. MAIN EVALUATION ---
def main():
    if len(sys.argv) < 2:
        print("? Error: Please provide the path to the predictions JSON file.")
        print(f"Usage: python {sys.argv[0]} <path_to_predictions.json>")
        return

    predictions_file = sys.argv[1]
    if not os.path.exists(predictions_file):
        print(f"? Error: Predictions file not found at {predictions_file}")
        return

    print(f"Loading predictions from: {predictions_file}\n")
    with open(predictions_file, 'r') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError as e:
            print(f"? Error: Failed to decode JSON file. {e}")
            return
    
    if not isinstance(results, list):
        print(f"? Error: Expected JSON file to contain a list of items.")
        return

    # --- Part 1: Data Collection (From JSON script) ---
    all_labels = set()
    all_detections_by_label = defaultdict(list)
    all_gt_boxes_by_label = defaultdict(list)
    num_images = len(results)

    print(f"Processing {num_images} images...")

    for item_id, item in enumerate(results):
        if 'ground_truth' not in item or 'prediction' not in item:
            print(f"Warning: Skipping item {item_id}, missing 'ground_truth' or 'prediction'.")
            continue
            
        gt_boxes_by_label = parse_bboxes(item['ground_truth'], is_prediction=False)
        pred_boxes_by_label = parse_bboxes(item['prediction'], is_prediction=True)

        current_labels = set(gt_boxes_by_label.keys()) | set(pred_boxes_by_label.keys())
        all_labels.update(current_labels)

        for label, gt_boxes in gt_boxes_by_label.items():
            for gt_box in gt_boxes:
                all_gt_boxes_by_label[label].append({
                    "bbox": gt_box, "image_id": item_id
                })
        
        for label, preds in pred_boxes_by_label.items():
            for p in preds:
                all_detections_by_label[label].append({
                    "bbox": p["bbox"], "score": p["score"], "image_id": item_id
                })

    # --- Part 2: Data Conversion (From JSON script) ---
    # Convert data into the format needed by the evaluation loop:
    # two lists (pred_sv_list, gt_sv_list) of sv.Detections, one per image.
    
    class_names = sorted(list(all_labels))
    if not class_names:
        print("No labels found. Cannot calculate metrics.")
        return
        
    class_names_to_id = {name: i for i, name in enumerate(class_names)}

    gt_sv_list = []
    pred_sv_list = []

    for i in range(num_images):
        gt_xyxy_list, gt_class_id_list = [], []
        pred_xyxy_list, pred_class_id_list, pred_confidence_list = [], [], []

        for label, gt_list in all_gt_boxes_by_label.items():
            class_id = class_names_to_id[label]
            for gt_item in gt_list:
                if gt_item["image_id"] == i:
                    x1, y1, w, h = gt_item["bbox"]
                    gt_xyxy_list.append([x1, y1, x1 + w, y1 + h])
                    gt_class_id_list.append(class_id)
        
        for label, det_list in all_detections_by_label.items():
            class_id = class_names_to_id[label]
            for det_item in det_list:
                if det_item["image_id"] == i:
                    x1, y1, w, h = det_item["bbox"]
                    pred_xyxy_list.append([x1, y1, x1 + w, y1 + h])
                    pred_class_id_list.append(class_id)
                    pred_confidence_list.append(det_item["score"])

        gt_sv_list.append(sv.Detections(
            xyxy=np.array(gt_xyxy_list),
            class_id=np.array(gt_class_id_list)
        ) if gt_xyxy_list else sv.Detections.empty())
            
        pred_sv_list.append(sv.Detections(
            xyxy=np.array(pred_xyxy_list),
            class_id=np.array(pred_class_id_list),
            confidence=np.array(pred_confidence_list)
        ) if pred_xyxy_list else sv.Detections.empty())

    
    for i in range(5):
        first_image_path = results[i]["image"]
        plot_detections(first_image_path, pred_sv_list[i], class_names,str(i)+"pred.png")
        plot_detections(first_image_path, gt_sv_list[i], class_names,str(i)+"gt.png")
    
    
    
if __name__ == "__main__":
    main()