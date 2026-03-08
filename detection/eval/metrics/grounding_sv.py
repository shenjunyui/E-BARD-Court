#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    # --- Part 3: Per-Class Evaluation Loop (Syntax from YOLO/RF-DETR script) ---
    
    print("\nRunning Per-Class Evaluation Loop...")
    # Store results for a final summary
    all_prec = []
    all_rec = []
    all_f1 = []

    for class_id, class_name in enumerate(class_names):
        print(f"\n--- Evaluating for Class: {class_name} (ID: {class_id}) ---")
        
        pred_detections_class = []
        gt_detections_class = []

        # Filter our full, aligned lists for *only* this class
        for pred_det, gt_ann in zip(pred_sv_list, gt_sv_list):
            pred_filtered = pred_det[pred_det.class_id == class_id]
            gt_filtered = gt_ann[gt_ann.class_id == class_id]
            
            # We must append *something* for every image, even if empty,
            # to keep the list lengths identical for the evaluator.
            pred_detections_class.append(pred_filtered)
            gt_detections_class.append(gt_filtered)

        # We must instantiate a new metric object *inside* the loop
        # to reset the state for each class.
        # We pass the IOU_THRESHOLD here.
        precision_metric_class = Precision()
        recall_metric_class = Recall()
        f1_metric_class = F1Score()

        # Update metrics with only this class's data
        precision_metric_class.update(pred_detections_class, gt_detections_class)
        recall_metric_class.update(pred_detections_class, gt_detections_class)
        f1_metric_class.update(pred_detections_class, gt_detections_class)
        
        # Compute results
        precision_results_class = precision_metric_class.compute()
        recall_results_class = recall_metric_class.compute()
        f1_results_class = f1_metric_class.compute()

        # Print metrics individually, like the template script
        print("Precision")
        print(precision_results_class.precision_at_50)
        print("Recall")
        print(recall_results_class.recall_at_50)
        print("F1")
        print(f1_results_class.f1_50) # Using f-string to match template
        
        # Store for summary
        all_prec.append(precision_results_class.precision_at_50)
        all_rec.append(recall_results_class.recall_at_50)
        all_f1.append(f1_results_class.f1_50)

    # --- Part 4: Final Summary Table (Adapted from JSON script) ---
    print("\n" + "="*70)
    print(f"?? Overall Summary (Macro Avg @ IoU={IOU_THRESHOLD})")
    print(f"   Source File: {os.path.basename(predictions_file)}")
    print("="*70)
    print(f"{'METRIC':<20} | {'SCORE':>10}")
    print("-"*70)
    
    avg_p = np.mean(all_prec) if all_prec else 0.0
    avg_r = np.mean(all_rec) if all_rec else 0.0
    avg_f1 = np.mean(all_f1) if all_f1 else 0.0
    
    print(f"{'Macro-Precision':<20} | {avg_p:10.4f}")
    print(f"{'Macro-Recall':<20} | {avg_r:10.4f}")
    print(f"{'Macro-F1-Score':<20} | {avg_f1:10.4f}")
    print("="*70)

if __name__ == "__main__":
    main()