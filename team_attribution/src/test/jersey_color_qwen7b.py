import os
import json
import pandas as pd # Still useful if you want to structure the results as a DataFrame later, but not strictly necessary for just loading the JSON and calculating metrics.
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import defaultdict # Useful for gathering all unique labels

# --- Configuration ---
# Path to the pre-computed prediction JSON file
prediction_json_path = "./color_Qwen2.5-VL-7B-Instruct.json"

# Directory for saving the figure
fig_dir = "figure"
# --- End Configuration ---

true_labels = []
predicted_labels = []
all_possible_labels = set()

# --- Load Predictions from JSON ---
try:
    with open(prediction_json_path, 'r') as f:
        prediction_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Prediction file not found at {prediction_json_path}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {prediction_json_path}")
    exit()

print(f"Loaded {len(prediction_data)} predictions from the file.")

# Extract true and predicted labels
for item in prediction_data:
    # Normalize labels to lowercase for consistent comparison and metrics calculation
    # This step is crucial because "purple" (ground truth) and "Purple" (prediction)
    # would be treated as different classes otherwise.
    true_label = item.get("ground_truth", "").lower()
    pred_label = item.get("prediction", "").lower()
    
    if true_label and pred_label:
        true_labels.append(true_label)
        predicted_labels.append(pred_label)
        all_possible_labels.add(true_label)
        all_possible_labels.add(pred_label) # Both true and predicted labels contribute to the set of all classes

# Check if we have any data to process
if not true_labels:
    print("No valid labels found in the JSON data. Exiting.")
    exit()

# Convert label set to sorted list for consistent confusion matrix
class_names = sorted(list(all_possible_labels))

print(f"Detected {len(class_names)} unique classes: {class_names}")

# --- Metrics Calculation and Reporting ---
print("\nClassification Report:")
# Use the collected true and predicted labels to generate the report
# We include all possible labels in 'class_names' to ensure the report covers all seen classes.
print(classification_report(true_labels, predicted_labels, labels=class_names, target_names=class_names, zero_division=0))

acc = accuracy_score(true_labels, predicted_labels)
print(f"Overall Accuracy: {acc:.4f}")

# --- Confusion Matrix Plotting ---
cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)

# Save confusion matrix plot
os.makedirs(fig_dir, exist_ok=True)

plt.figure(figsize=(max(8, len(class_names)*0.8), max(6, len(class_names)*0.8)))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Qwen2.5-VL-3B-Instruct Color Classification Confusion Matrix (Total Samples: {len(true_labels)})")
plt.tight_layout()

# Change the output filename to reflect the new model/source
output_filename = "Qwen2_color_confusion_matrix.png"
plt.savefig(os.path.join(fig_dir, output_filename))
plt.close()

print(f"\nConfusion matrix plot saved to {os.path.join(fig_dir, output_filename)}")