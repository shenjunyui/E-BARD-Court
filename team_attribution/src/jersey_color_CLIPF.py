import os
import json
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
image_csv_path = "../data/image_labels.csv"
color_json_folder = "../data/colors"
clip_path = "/leonardo_scratch/large/userexternal/ggiudici/patrickjohncyh/fashion-clip"

# Load model
model = CLIPModel.from_pretrained(clip_path).to(device)
processor = CLIPProcessor.from_pretrained(clip_path)

# Load dataset
df = pd.read_csv(image_csv_path)

true_labels = []
predicted_labels = []
all_possible_labels = set()

for idx, row in df.iterrows():
    image_path = row['image_path']
    true_label = row['label']  # Ground truth color label
    
    # Extract match prefix
    basename = os.path.basename(image_path)
    match_prefix = basename.split('_')[0]  # e.g., was-vs-mil-0022400382
    color_json_path = os.path.join(color_json_folder, f"{match_prefix}_color.json")
    
    if not os.path.exists(color_json_path):
        print(f"Missing color file for: {image_path}")
        continue
    
    with open(color_json_path, 'r') as f:
        color_data = json.load(f)
    
    # Extract unique colors from JSON
    match_colors = sorted(set(color_data.values()))  # Only unique colors
    all_possible_labels.update(match_colors)

    # Generate text prompts for only available colors
    text_prompts = [f"A photo of a {color} jersey" for color in match_colors]
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    
    # Encode text
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Process image
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
        # Compute similarities
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        pred_label = match_colors[pred_idx]
    #outputs = model(**text_inputs)
    #logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    #probs = logits_per_image.softmax(dim=1)  
    #pred_idx = probs.argmax(dim=-1).item()
    #pred_label = match_colors[pred_idx]
    true_labels.append(true_label)
    predicted_labels.append(pred_label)

    print(f"{basename} | True: {true_label} | Pred: {pred_label} | Probs: {probs.squeeze().tolist()}")

# Convert label set to sorted list for consistent confusion matrix
class_names = sorted(list(all_possible_labels))

# Metrics
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, labels=class_names, target_names=class_names))

acc = accuracy_score(true_labels, predicted_labels)
print(f"Overall Accuracy: {acc:.4f}")

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)

# Save confusion matrix plot
fig_dir = "figure"
os.makedirs(fig_dir, exist_ok=True)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("CLIP Color Classification Confusion Matrix")
plt.tight_layout()

plt.savefig(os.path.join(fig_dir, "CLIP_color_confusion_matrix.png"))
plt.close()
