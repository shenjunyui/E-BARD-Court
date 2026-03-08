import os
import sys
import json
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Add perception_models to path
sys.path.append("/leonardo_scratch/large/userexternal/ggiudici/perception_models")

# Import PE-Core
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
image_csv_path = "../data/image_labels.csv"
color_json_folder = "../data/colors"

# Load model and preprocessing
print("Available configs:", pe.CLIP.available_configs())

# The name of the model configuration
model_name = 'PE-Core-L14-336'

# The path to the SPECIFIC weights file, not just the directory.

local_checkpoint_file = "/leonardo_scratch/large/userexternal/ggiudici/facebook/PE-Core-L14-336/PE-Core-L14-336.pt"

# Load the model using the correct arguments
print(f"Loading model '{model_name}' from local checkpoint: {local_checkpoint_file}")

model = pe.CLIP.from_config(
    name=model_name,
    pretrained=True,
    checkpoint_path=local_checkpoint_file
)

model = model.to(device)

preprocess = transforms.get_image_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)

# Load dataset
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
    
    text_prompts = [f"A photo of a {color} jersey" for color in match_colors]
    
    # Preprocess
    image_tensor = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text_tensor = tokenizer(text_prompts).to(device)

    # Forward pass
    with torch.no_grad(), torch.autocast(device_type='cuda'):
        image_features, text_features, logit_scale = model(image_tensor, text_tensor)
        probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

    pred_idx = probs.argmax(dim=-1).item()
    pred_label = match_colors[pred_idx]
    
    true_labels.append(true_label)
    predicted_labels.append(pred_label)

    print(f"{basename} | True: {true_label} | Pred: {pred_label} | Probs: {probs.squeeze().tolist()}")

# Sorted class names for metrics
class_names = sorted(list(all_possible_labels))

# Metrics
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, labels=class_names, target_names=class_names))

acc = accuracy_score(true_labels, predicted_labels)
print(f"Overall Accuracy: {acc:.4f}")

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)

# Plot
fig_dir = "figure"
os.makedirs(fig_dir, exist_ok=True)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("PE-Core Color Classification Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "PECore_color_confusion_matrix.png"))
plt.close()
