import os
import json
import pandas as pd
import torch
from PIL import Image
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set device
device = 0 if torch.cuda.is_available() else -1  # For pipeline

# --- Paths ---
image_csv_path = "../../data/test_labels.csv"
color_json_folder = "../../data/colors"
siglip_ckpt = "/leonardo_work/uBS25_EcoGiu/google/siglip2-so400m-patch14-384"  # or local path if needed

# --- Load zero-shot image classification pipeline ---
image_classifier = pipeline(model=siglip_ckpt, task="zero-shot-image-classification", device=device)

# --- Load dataset ---
df = pd.read_csv(image_csv_path)

true_labels = []
predicted_labels = []
all_possible_labels = set()

for idx, row in df.iterrows():
    image_path = row['image_path']
    true_label = row['label']

    basename = os.path.basename(image_path)
    match_prefix = basename.split('_')[0]
    color_json_path = os.path.join(color_json_folder, f"{match_prefix}_color.json")

    if not os.path.exists(color_json_path):
        print(f"Missing color file for: {image_path}")
        continue

    with open(color_json_path, 'r') as f:
        color_data = json.load(f)

    match_colors = sorted(set(color_data.values()))
    all_possible_labels.update(match_colors)

    candidate_labels = [f"A photo of a {color} jersey" for color in match_colors]

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        continue

    # Run inference using pipeline
    try:
        outputs = image_classifier(image, candidate_labels = candidate_labels)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        continue

    # Choose the label with the highest score
    best_result = max(outputs, key=lambda x: x["score"])
    pred_label = best_result["label"].replace("A photo of a ", "").replace(" jersey", "")

    true_labels.append(true_label)
    predicted_labels.append(pred_label)

    print(f"{basename} | True: {true_label} | Pred: {pred_label} | Top Score: {best_result['score']:.4f}")

# --- Evaluation ---

class_names = sorted(list(all_possible_labels))

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, labels=class_names, target_names=class_names, zero_division=0))

acc = accuracy_score(true_labels, predicted_labels)
print(f"Overall Accuracy: {acc:.4f}")

cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)

# Save confusion matrix plot
fig_dir = "figure"
os.makedirs(fig_dir, exist_ok=True)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("SigLIP (Pipeline) Color Classification Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "SigLIP_color_confusion_matrix_pipeline.png"))
plt.close()

print("\nScript finished. Confusion matrix saved.")
