import json
import os
import re
import sys
from collections import defaultdict
import numpy as np

# --- 1. Configuration ---
IOU_THRESHOLD = 0.5

# --- 2. IoU Calculation ---
def calculate_iou(boxA, boxB):
    xA_min, yA_min, wA, hA = boxA
    xB_min, yB_min, wB, hB = boxB
    xA_max, yA_max = xA_min + wA, yA_min + hA
    xB_max, yB_max = xB_min + wB, yB_min + hB

    x_inter_min = max(xA_min, xB_min)
    y_inter_min = max(yA_min, yB_min)
    x_inter_max = min(xA_max, xB_max)
    y_inter_max = min(yA_max, yB_max)

    inter_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)
    boxA_area = wA * hA
    boxB_area = wB * hB
    union_area = float(boxA_area + boxB_area - inter_area)
    return inter_area / union_area if union_area > 0 else 0

# --- 3. Parse BBoxes ---
def parse_bboxes(bbox_str, is_prediction=False):
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
            if label and bbox_2d and len(bbox_2d) == 4:
                if is_prediction:
                    x1, y1, x2, y2 = bbox_2d
                    if x2 > x1 and y2 > y1:
                        w = x2 - x1
                        h = y2 - y1
                        grouped_boxes[label].append({"bbox": [x1, y1, w, h], "score": score})
                else:
                    grouped_boxes[label].append([bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3]])
        return grouped_boxes
    except:
        return defaultdict(list)

# --- 4. Compute Average Precision ---
def compute_ap(rec, prec):
    """Compute AP using the 11-point interpolation (Pascal VOC 2007 style)."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec_at_t = prec[rec >= t].max() if np.any(rec >= t) else 0
        ap += prec_at_t / 11.0
    return ap

# --- 5. Main Evaluation ---
def main():
    if len(sys.argv) < 2:
        print("? Error: Please provide the path to the predictions JSON file.")
        print(f"Usage: python {sys.argv[0]} <path_to_predictions.json>")
        return

    predictions_file = sys.argv[1]
    if not os.path.exists(predictions_file):
        print(f"? Error: Predictions file not found at {predictions_file}")
        return

    with open(predictions_file, 'r') as f:
        results = json.load(f)

    stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    all_labels = set()

    # For mAP calculation
    all_detections = defaultdict(list)
    all_gt_boxes = defaultdict(list)

    for item_id, item in enumerate(results):
        gt_boxes_by_label = parse_bboxes(item['ground_truth'], is_prediction=False)
        pred_boxes_by_label = parse_bboxes(item['prediction'], is_prediction=True)

        all_labels.update(gt_boxes_by_label.keys())
        all_labels.update(pred_boxes_by_label.keys())

        # Collect GTs
        for label, gt_boxes in gt_boxes_by_label.items():
            for gt_box in gt_boxes:
                all_gt_boxes[label].append({"bbox": gt_box, "used": False, "image_id": item_id})

        # Evaluate grounding (F1)
        for label in set(gt_boxes_by_label.keys()) | set(pred_boxes_by_label.keys()):
            gt_boxes = gt_boxes_by_label[label]
            pred_boxes = [p["bbox"] for p in pred_boxes_by_label[label]]
            if not pred_boxes and not gt_boxes:
                continue
            gt_matched = [False] * len(gt_boxes)
            for pred_box in pred_boxes:
                best_iou, best_gt_idx = 0, -1
                for i, gt_box in enumerate(gt_boxes):
                    if not gt_matched[i]:
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou, best_gt_idx = iou, i
                if best_iou >= IOU_THRESHOLD:
                    if best_gt_idx != -1 and not gt_matched[best_gt_idx]:
                        stats[label]['tp'] += 1
                        gt_matched[best_gt_idx] = True
                    else:
                        stats[label]['fp'] += 1
                else:
                    stats[label]['fp'] += 1
            stats[label]['fn'] += gt_matched.count(False)

        # Collect detections for mAP
        for label, preds in pred_boxes_by_label.items():
            for p in preds:
                all_detections[label].append({
                    "bbox": p["bbox"],
                    "score": p["score"],
                    "image_id": item_id
                })

    # --- Report F1 ---
    print("\n" + "="*70)
    print(f"?? Grounding Performance Report (IoU Threshold = {IOU_THRESHOLD})")
    print(f"   Source File: {os.path.basename(predictions_file)}")
    print("="*70)
    print(f"{'CLASS':<20} | {'PRECISION':>10} | {'RECALL':>10} | {'F1-SCORE':>10} | {'SUPPORT (GT)':>12}")
    print("-"*70)

    total_tp, total_fp, total_fn, total_gt = 0, 0, 0, 0
    for label in sorted(list(all_labels)):
        tp, fp, fn = stats[label]['tp'], stats[label]['fp'], stats[label]['fn']
        support = tp + fn
        total_tp += tp; total_fp += fp; total_fn += fn; total_gt += support
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{label:<20} | {precision:10.4f} | {recall:10.4f} | {f1_score:10.4f} | {support:>12}")

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / total_gt if total_gt > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    print("-"*70)
    print(f"{'OVERALL (Micro Avg)':<20} | {micro_precision:10.4f} | {micro_recall:10.4f} | {micro_f1:10.4f} | {total_gt:>12}")
    print("="*70)

    # --- Report mAP ---
    print("\n" + "="*70)
    print("?? Detection Performance Report (mAP, IoU=0.5)")
    print("="*70)
    aps = []
    for label in sorted(list(all_labels)):
        detections = sorted(all_detections[label], key=lambda x: -x["score"])
        npos = len(all_gt_boxes[label])
        tp, fp = [], []

        for det in detections:
            ovmax, jmax = -np.inf, -1
            for j, gt in enumerate(all_gt_boxes[label]):
                if det["image_id"] == gt["image_id"]:
                    iou = calculate_iou(det["bbox"], gt["bbox"])
                    if iou > ovmax:
                        ovmax, jmax = iou, j
            if ovmax >= IOU_THRESHOLD and jmax != -1 and not all_gt_boxes[label][jmax]["used"]:
                tp.append(1)
                fp.append(0)
                all_gt_boxes[label][jmax]["used"] = True
            else:
                tp.append(0)
                fp.append(1)

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / npos if npos > 0 else np.zeros(len(tp))
        prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

        ap = compute_ap(rec, prec) if npos > 0 else 0
        aps.append(ap)
        print(f"{label:<20} | AP={ap:.4f} (GT={npos})")

    mAP = np.mean(aps) if aps else 0
    print("-"*70)
    print(f"{'mAP':<20} | {mAP:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()