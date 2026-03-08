import torch
import sys
import os
import re
from PIL import Image
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor

# 1. Define the Custom Video Classifier Model
# This architecture MUST EXACTLY MATCH the one used for training.
# ---------------------------------------------------------------------------------------
class VideoClassifier(nn.Module):
    """
    A custom video classification model using a pre-trained VideoMAE backbone.
    """
    def __init__(self, num_classes: int, model_name: str = "OpenGVLab/VideoMAEv2-Base"):
        super().__init__()
        self.num_classes = num_classes
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.videomae = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        
        hidden_size = self.videomae.config.model_config['embed_dim']
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        pooled_features = self.videomae(pixel_values=pixel_values)
        logits = self.classifier(pooled_features)
        return logits

# 2. Define the Custom Dataset
# This class must process data IDENTICALLY to how it was done during training.
# -------------------------------------------------------------------------
class VideoFrameDataset(Dataset):
    """
    A PyTorch Dataset for loading video frames for inference.
    """
    def __init__(self, samples, root_dir, image_processor, num_frames=16):
        self.samples = samples
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.image_processor = image_processor

    def __len__(self):
        return len(self.samples)

    def _sort_frames(self, frame_list):
        """Helper function to sort frame filenames numerically."""
        return sorted(frame_list, key=lambda x: int(re.search(r'\d+', x).group()))

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video_folder_path = os.path.join(self.root_dir, video_path)
        
        # Robustness check for missing directories or frames
        if not os.path.isdir(video_folder_path):
            print(f"\nWarning: Directory not found: {video_folder_path}", file=sys.stderr)
            return None # Will be filtered by collate_fn

        frame_files = [f for f in os.listdir(video_folder_path) if f.endswith('.png')]
        if not frame_files:
            print(f"\nWarning: No frames found in: {video_folder_path}", file=sys.stderr)
            return None # Will be filtered by collate_fn
            
        sorted_frame_files = self._sort_frames(frame_files)
        
        # Sample or repeat frames to get the desired number
        if len(sorted_frame_files) >= self.num_frames:
            sampled_frames_files = sorted_frame_files[:self.num_frames]
        else:
            sampled_frames_files = (sorted_frame_files * (self.num_frames // len(sorted_frame_files) + 1))[:self.num_frames]

        frames = [Image.open(os.path.join(video_folder_path, f)).convert("RGB") for f in sampled_frames_files]
        
        processed_video = self.image_processor(frames, return_tensors="pt")
        video_tensor = processed_video['pixel_values'].squeeze(0)
        
        return video_tensor, label

# 3. Inference Script Main Execution
# ----------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    MODEL_NAME = "/leonardo_work/uBS25_EcoGiu/OpenGVLab/VideoMAEv2-Base"
    SAVED_MODEL_PATH = 'best_model.pth' # Path to your trained weights
    DATA_ROOT = "../data"
    VAL_ANNOTATION_FILE = os.path.join(DATA_ROOT, "val.txt")
    
    NUM_FRAMES = 16
    BATCH_SIZE = 128 # Adjust based on your GPU memory

    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Create Label Mapping (must be identical to training) ---
    print("\nBuilding label map...")
    unique_labels = set(["nan", "00"] + [str(i) for i in range(100)])
    label_to_int = {label: i for i, label in enumerate(sorted(list(unique_labels)))}
    NUM_CLASSES = len(label_to_int)
    print(f"Loaded {NUM_CLASSES} unique labels.")

    # --- Initialize Model, Processor, and Load Weights ---
    print("\nInitializing model and image processor...")
    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    model = VideoClassifier(num_classes=NUM_CLASSES, model_name=MODEL_NAME)

    print(f"Loading trained weights from '{SAVED_MODEL_PATH}'...")
    try:
        model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{SAVED_MODEL_PATH}'. Ensure the path is correct.", file=sys.stderr)
        sys.exit(1)
    
    model.to(device)
    model.eval() # CRITICAL: Set model to evaluation mode
    print("Model loaded successfully.")

    # --- Prepare Validation Data ---
    def load_samples(annotation_file, label_map):
        try:
            with open(annotation_file, 'r') as f:
                return [(line.strip().split()[0], label_map[line.strip().split()[2]]) for line in f]
        except FileNotFoundError:
            print(f"Error: Annotation file not found at '{annotation_file}'", file=sys.stderr)
            sys.exit(1)

    print(f"\nLoading validation data from '{VAL_ANNOTATION_FILE}'...")
    val_samples = load_samples(VAL_ANNOTATION_FILE, label_to_int)
    val_dataset = VideoFrameDataset(val_samples, DATA_ROOT, image_processor=image_processor, num_frames=NUM_FRAMES)
    
    # A collate function that filters out None items from failed file loads
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        return {
            "pixel_values": torch.stack([item[0] for item in batch]),
            "labels": torch.tensor([item[1] for item in batch], dtype=torch.long)
        }
        
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    print(f"Validation dataset loaded with {len(val_dataset)} samples.")

    # --- Run Inference ---
    print("\nStarting inference on the validation set...")
    all_preds, all_labels = [], []
    
    with torch.no_grad(): # Disable gradient calculation for efficiency
        for i, batch in enumerate(val_loader):
            if batch is None: continue # Skip batches that failed to load

            pixel_values = batch["pixel_values"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4)
            labels = batch["labels"].to(device, non_blocking=True)
            
            outputs = model(pixel_values)
            _, predicted = torch.max(outputs.data, 1)
            
            print(predicted.cpu())
            print(labels.cpu())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            print(f"\rProcessed batch {i+1}/{len(val_loader)}", end="", flush=True)

    print("\n\nInference complete.")

    # --- Calculate and Report Final Results ---
    if not all_labels:
        print("No valid samples were processed. Cannot calculate accuracy.", file=sys.stderr)
        sys.exit(1)

    correct = (np.array(all_preds) == np.array(all_labels)).sum()
    total = len(all_labels)
    accuracy = 100 * correct / total
    
    print("\n--- Validation Results ---")
    print(f"Total Samples Evaluated: {total}")
    print(f"Correct Predictions:     {correct}")
    print(f"Accuracy:                {accuracy:.2f}%")
    print("--------------------------")