import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

# List of CSV files to compare
csv_files = [
    "../src/full/full_mi_qwen3b.csv",
    "../src/full/full_mi_qwen7b.csv",
    "../src/full/full_si_qwen7b.csv",
    "../src/full/full_si_olmo.csv",
]

shorts = ["Q3B_mi","Q7B_mi","Q7B_si","OLM_si"]


# Output file
output_path = "full.txt"

all_results = {}
all_classes = set()

for k,csv_path in enumerate(csv_files):
    df = pd.read_csv(csv_path, dtype=str)
    y_true = df["ground_truth"].astype(str)
    y_pred = df["prediction"].astype(str)
    classes = sorted(set(y_true))
    all_classes.update(classes)

    results = {}

    # Per-class metrics (only samples where GT == cls)
    for cls in classes:
        y_true_bin = [1 if y == cls else 0 for y in y_true]
        y_pred_bin = [1 if y == cls else 0 for y in y_pred]
        
        recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        
        # Restrict accuracy to just samples where GT == cls
        mask = (y_true == cls)
        gt_cls = y_true[mask]
        pred_cls = y_pred[mask]
        accuracy = accuracy_score(gt_cls, pred_cls)
       
        support = len(gt_cls)
        if support == 0:
            continue

        results[cls] = {
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "Support": support,
        }

    # Overall metrics (same as before, over all samples)
    overall_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_support = len(y_true)

    results["Overall"] = {
        "Recall": overall_recall,
        "Precision": overall_precision,
        "F1": overall_f1,
        "Accuracy": overall_accuracy,
        "Support": overall_support,
    }
    
    mask = (y_true != 'nan')
    gt_cls = y_true[mask]
    pred_cls = y_pred[mask]
    
    overall_recall = recall_score(gt_cls, pred_cls, average='weighted', zero_division=0)
    overall_precision = precision_score(gt_cls, pred_cls, average='weighted', zero_division=0)
    overall_f1 = f1_score(gt_cls, pred_cls, average='weighted', zero_division=0)
    overall_accuracy = accuracy_score(gt_cls, pred_cls)
    overall_support = len(gt_cls)
    
    results["Overall (no nan)"] = {
        "Recall": overall_recall,
        "Precision": overall_precision,
        "F1": overall_f1,
        "Accuracy": overall_accuracy,
        "Support": overall_support,
    }

    all_results[shorts[k]] = results

# Sort classes and include Overall
all_classes = sorted(all_classes) + ["Overall"] + ["Overall (no nan)"]

# Build header
header = ["Class", "Support"]
for csv_name in all_results.keys():
    header += [
        f"{csv_name}.R",
        f"{csv_name}.P",
        f"{csv_name}.F1",
    ]

lines = ["\t".join(header)]

# Build rows
for cls in all_classes:
    first_model = next(iter(all_results))
    support = all_results[first_model].get(cls, {}).get("Support", 0)
    row = [cls, str(support)]

    for csv_name, results in all_results.items():
        metrics = results.get(cls, {"Recall": 0, "Precision": 0, "F1": 0})
        row.extend([
            f"{metrics['Recall']:.4f}",
            f"{metrics['Precision']:.4f}",
            f"{metrics['F1']:.4f}",
        ])

    lines.append("\t".join(row))

# Save to file
with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"Metrics saved to {output_path}")
