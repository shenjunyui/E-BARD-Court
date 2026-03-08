import json
import os
import sys
from collections import Counter, defaultdict
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculates precision, recall, F1-score, and support for each class,
    overall accuracy, and average metrics.
    """
    all_labels = sorted(list(set(y_true + y_pred)))
    stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    # Calculate TP, FP, FN for each class
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            stats[true_label]['tp'] += 1
        else:
            stats[true_label]['fn'] += 1
            stats[pred_label]['fp'] += 1
            
    # --- Calculate metrics per class ---
    report = {}
    total_samples = len(y_true)

    for label in all_labels:
        tp = stats[label]['tp']
        fp = stats[label]['fp']
        fn = stats[label]['fn']
        tn = total_samples - (tp + fp + fn) # Calculate True Negatives for per-class accuracy
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
        
        support_gt = tp + fn 
        support_pred = tp + fp
        
        report[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
            'accuracy': accuracy,
            'support_gt': support_gt,
            'support_pred': support_pred
        }
    
    # --- Calculate Averages ---
    averages = {
        'macro_avg': defaultdict(float),
        'weighted_avg': defaultdict(float)
    }
    
    # Exclude 'nan' from average calculations if it exists for more meaningful averages
    labels_for_avg = [l for l in all_labels if l != 'nan']
    num_classes_for_avg = len(labels_for_avg)

    for label in labels_for_avg:
        metrics = report[label]
        support = metrics['support_gt']
        
        # Macro sums (unweighted)
        averages['macro_avg']['precision'] += metrics['precision']
        averages['macro_avg']['recall'] += metrics['recall']
        averages['macro_avg']['f1-score'] += metrics['f1-score']
        
        # Weighted sums (weighted by support)
        averages['weighted_avg']['precision'] += metrics['precision'] * support
        averages['weighted_avg']['recall'] += metrics['recall'] * support
        averages['weighted_avg']['f1-score'] += metrics['f1-score'] * support
        averages['weighted_avg']['accuracy'] += metrics['accuracy'] * support

    # Finalize averages by dividing by the total or number of classes
    if num_classes_for_avg > 0:
        for metric in ['precision', 'recall', 'f1-score']:
            averages['macro_avg'][metric] /= num_classes_for_avg
    
    if total_samples > 0:
        for metric in ['precision', 'recall', 'f1-score', 'accuracy']:
            averages['weighted_avg'][metric] /= total_samples
            
    # Overall Accuracy
    num_correct = sum(stats[label]['tp'] for label in all_labels)
    overall_accuracy = num_correct / total_samples if total_samples > 0 else 0
    
    return report, overall_accuracy, all_labels, num_correct, averages

def print_confusion_matrix(y_true, y_pred, labels):
    """Prints a formatted confusion matrix."""
    matrix_labels = [l for l in labels if l != 'nan']
    num_labels = len(matrix_labels)
    matrix = np.zeros((num_labels, num_labels), dtype=int)
    label_to_index = {label: i for i, label in enumerate(matrix_labels)}

    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_index.get(true)
        pred_idx = label_to_index.get(pred)
        if true_idx is not None and pred_idx is not None:
            matrix[true_idx, pred_idx] += 1

    print("\n" + "="*70)
    print("📊 Confusion Matrix")
    print("="*70)
    
    col_width = max([len(l) for l in matrix_labels] + [10]) + 2
    header = 'TRUE \\ PRED'.ljust(col_width) + " | ".join([f"{label:>{col_width-2}}" for label in matrix_labels])
    print(header)
    print("-"*len(header))

    for i, label in enumerate(matrix_labels):
        row_str = f"{label:<{col_width}}"
        for j in range(num_labels):
            row_str += f" | {matrix[i, j]:>{col_width-2}}"
        print(row_str)
    print("="*70)

def main():
    """
    Main function to load data and evaluate classification performance.
    """
    if len(sys.argv) < 2:
        print("❌ Error: Please provide the path to the predictions JSON file.")
        print(f"Usage: python {sys.argv[0]} <path_to_predictions.json>")
        return

    predictions_file = sys.argv[1]
    if not os.path.exists(predictions_file):
        print(f"❌ Error: Predictions file not found at {predictions_file}")
        return

    with open(predictions_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    y_true = [str(item['ground_truth']).lower() for item in results]
    y_pred = [str(item['prediction']).lower() for item in results]
    
    if not y_true:
        print("❌ Error: No data found in the JSON file.")
        return

    # --- Calculate Metrics ---
    report, overall_accuracy, all_labels, num_correct_total, averages = calculate_metrics(y_true, y_pred)

    # --- Calculate accuracy without NaN if 'nan' predictions exist ---
    accuracy_without_nan = None
    num_correct_no_nan = 0
    num_valid_samples = 0
    
    if 'nan' in all_labels:
        y_true_no_nan, y_pred_no_nan = [], []
        for true, pred in zip(y_true, y_pred):
            if pred != 'nan':
                y_true_no_nan.append(true)
                y_pred_no_nan.append(pred)
        
        num_correct_no_nan = sum(1 for t, p in zip(y_true_no_nan, y_pred_no_nan) if t == p)
        num_valid_samples = len(y_true_no_nan)
        accuracy_without_nan = num_correct_no_nan / num_valid_samples if num_valid_samples > 0 else 0

    # --- Report Results ---
    print("\n" + "="*100)
    print(f"📈 Classification Performance Report")
    print(f"   Source File: {os.path.basename(predictions_file)}")
    print("="*100)
    print(f"{'CLASS':<20} | {'PRECISION':>10} | {'RECALL':>10} | {'F1-SCORE':>10} | {'SUPPORT (GT)':>12} | {'SUPPORT (PRED)':>14}")
    print("-"*100)

    total_support_gt = len(y_true)
    total_support_pred = len(y_pred)
    
    for label in all_labels:
        metrics = report[label]
        print(f"{label:<20} | {metrics['precision']:10.4f} | {metrics['recall']:10.4f} | {metrics['f1-score']:10.4f} | {metrics['support_gt']:>12} | {metrics['support_pred']:>14}")

    print("-"*100)

    # --- Print Averages and Overall Accuracy ---
    macro_avg = averages['macro_avg']
    weighted_avg = averages['weighted_avg']
    
    print(f"{'MACRO AVG':<20} | {macro_avg['precision']:10.4f} | {macro_avg['recall']:10.4f} | {macro_avg['f1-score']:10.4f} |")
    print(f"{'WEIGHTED AVG':<20} | {weighted_avg['precision']:10.4f} | {weighted_avg['recall']:10.4f} | {weighted_avg['f1-score']:10.4f} | {total_support_gt:>12} | {total_support_pred:>14}")

    print("-"*100)

    total_samples = len(y_true)
    if accuracy_without_nan is not None:
        label_with_nan = "OVERALL ACC (incl. NaN as errors)"
        label_without_nan = "OVERALL ACC (excl. NaN predictions)"
        print(f"{label_with_nan:<45} | {overall_accuracy:10.4f} ({num_correct_total}/{total_samples})")
        print(f"{label_without_nan:<45} | {accuracy_without_nan:10.4f} ({num_correct_no_nan}/{num_valid_samples})")
    else:
        print(f"{'OVERALL ACCURACY':<45} | {overall_accuracy:10.4f} ({num_correct_total}/{total_samples})")

    print(f"{'WEIGHTED ACCURACY':<45} | {weighted_avg['accuracy']:10.4f}")
    print("="*100)

    # --- Confusion Matrix ---
    #print_confusion_matrix(y_true, y_pred, all_labels)

if __name__ == "__main__":
    main()