import torch
print("import torch",flush=True)
torch.cuda.is_available()  # Force CUDA init early
from torch import nn

import copy # For saving initial model state

print("import nn",flush=True)

from transformers import AutoConfig,AutoModel,VideoMAEImageProcessor

print("import transformers ",flush=True)

import numpy as np
import os
from PIL import Image

print("import ut ",flush=True)

from torch.utils.data import DataLoader, Dataset

print("import dataset ")

import re # For sorting frame numbers correctly

print("loaded libraries",flush=True)


# 1. Define the Custom Video Classifier Model
# --------------------------------------------
class VideoClassifier(nn.Module):
    """
    A custom video classification model using a pre-trained VideoMAE backbone.
    """
    def __init__(self, num_classes: int, model_name: str = "OpenGVLab/VideoMAEv2-Base"):
        super().__init__()
        self.num_classes = num_classes
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.videomae = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        
        # Freeze the backbone parameters
        counter = 0
        for param in self.videomae.parameters():
            if counter < 150:
                param.requires_grad = False 
                counter = counter+1
            else:
                print('skipped layers')
        # Get the hidden size from the model's configuration
        hidden_size = self.videomae.config.model_config['embed_dim']
        # Define the new classifier head
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        # The output from the backbone is ALREADY the pooled features.
        pooled_features = self.videomae(pixel_values=pixel_values)

        # Pass the features directly to the classifier.
        logits = self.classifier(pooled_features)
        return logits

# 2. Define the Custom Dataset for Your Video Frames
# ------------------------------------------------------------
class VideoFrameDataset(Dataset):
    """
    A PyTorch Dataset for loading video frames and labels using a VideoMAEImageProcessor.
    """
    def __init__(self, samples, root_dir, image_processor, num_frames=16):
        """
        Args:
            samples (list): A list of tuples, where each tuple is (path, label_index).
            root_dir (str): The directory where the video folders are stored.
            image_processor: An instance of VideoMAEImageProcessor.
            num_frames (int): The number of frames to sample from each video.
        """
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
        
        frame_files = [f for f in os.listdir(video_folder_path) if f.endswith('.png')]
        sorted_frame_files = self._sort_frames(frame_files)
        
        # Sample or repeat frames to get the desired number
        if len(sorted_frame_files) >= self.num_frames:
            sampled_frames_files = sorted_frame_files[:self.num_frames]
        else:
            # If not enough frames, repeat the existing ones
            sampled_frames_files = (sorted_frame_files * (self.num_frames // len(sorted_frame_files) + 1))[:self.num_frames]

        # Load all sampled frames as a list of PIL Images
        frames = []
        for frame_file in sampled_frames_files:
            img_path = os.path.join(video_folder_path, frame_file)
            # Load image and ensure it's in RGB format for the processor
            image = Image.open(img_path).convert("RGB") 
            frames.append(image)
            
        # Use the image processor to handle resizing, normalization, and tensor conversion
        processed_video = self.image_processor(frames, return_tensors="pt")
        video_tensor = processed_video['pixel_values']
        video_tensor = video_tensor.squeeze(0)
        
        return video_tensor, label, video_folder_path

# 3. Full Training and Evaluation Procedure
# ------------------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    MODEL_NAME = "/leonardo_work/uBS25_EcoGiu/OpenGVLab/VideoMAEv2-Base"
    DATA_ROOT = "../data"
    TRAIN_ANNOTATION_FILE = os.path.join(DATA_ROOT, "train.txt")
    VAL_ANNOTATION_FILE = os.path.join(DATA_ROOT, "val.txt")
    TEST_ANNOTATION_FILE = os.path.join(DATA_ROOT, "test.txt")
    
    NUM_FRAMES = 16
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    # This is now a fallback value. If AUTO_FIND_LR is True, this will be overridden.
    LEARNING_RATE = 4e-05
    EARLY_STOPPING_PATIENCE = 5
    
    # --- Set this to True to automatically find the LR and start training ---
    AUTO_FIND_LR = False

    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}",flush=True)
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

    # --- Create Label Mapping ---
    print("\nBuilding label map...",flush=True)
    unique_labels = set(["nan", "00"] + [str(i) for i in range(100)])
    label_to_int = {label: i for i, label in enumerate(sorted(list(unique_labels)))}
    NUM_CLASSES = len(label_to_int)
    print(f"Found {NUM_CLASSES} unique labels.",flush=True)

    # --- Model and Processor Initialization ---
    print("\nInitializing model and image processor...", flush=True)
    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    model = VideoClassifier(num_classes=NUM_CLASSES, model_name=MODEL_NAME)
    model.to(device)
    
    # --- Data Loading ---
    def load_samples(annotation_file, label_map):
        with open(annotation_file, 'r') as f:
            return [(line.strip().split()[0], label_map[line.strip().split()[2]]) for line in f]

    train_samples = load_samples(TRAIN_ANNOTATION_FILE, label_to_int)
    val_samples = load_samples(VAL_ANNOTATION_FILE, label_to_int)
    test_samples = load_samples(TEST_ANNOTATION_FILE, label_to_int)

    train_dataset = VideoFrameDataset(train_samples, DATA_ROOT, image_processor=image_processor, num_frames=NUM_FRAMES)
    val_dataset = VideoFrameDataset(val_samples, DATA_ROOT, image_processor=image_processor, num_frames=NUM_FRAMES)
    test_dataset = VideoFrameDataset(test_samples, DATA_ROOT, image_processor=image_processor, num_frames=NUM_FRAMES)
    
    print(f"\nData loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([item[0] for item in batch]),
            "labels": torch.tensor([item[1] for item in batch], dtype=torch.long),
            "paths": [item[2] for item in batch]
        }
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # --- Training Setup ---
    # Calculate weights inversely proportional to class frequency
    #class_counts = pd.Series(labels).value_counts().sort_index()
    #weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float32)
    #weights = weights / weights.sum() # Normalize
    #weights = weights.to(device)

    # Use the weights in your criterion
    #criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.CrossEntropyLoss()
    
    # (NEW) 4. Automated Learning Rate Finder
    # --------------------------------------------------------------------------
    if AUTO_FIND_LR:
        print("\nStarting Automated LR Finder...", flush=True)
        try:
            from torch_lr_finder import LRFinder
        except ImportError:
            print("Please run 'pip install torch-lr_finder' to use the LR Finder.")
            exit()

        # Define a special collate function for the LR finder.
        # This will permute the tensor and return a simple (inputs, labels) tuple.
        def lr_finder_collate_fn(batch):
            # The dataset returns (video_tensor, label, path)
            video_tensors = torch.stack([item[0] for item in batch])
            labels_tensor = torch.tensor([item[1] for item in batch], dtype=torch.long)
            
            # Permute dimensions from (B, T, C, H, W) to (B, C, T, H, W)
            permuted_tensors = video_tensors.permute(0, 2, 1, 3, 4)
            
            return permuted_tensors, labels_tensor

        # Create a temporary DataLoader specifically for the LR finder
        # using our custom collate function.
        lr_finder_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            collate_fn=lr_finder_collate_fn, 
            num_workers=4
        )
        
        # Save initial model state because the finder will change the weights
        initial_state = copy.deepcopy(model.state_dict())

        temp_optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-7)
        
        # The LRFinder will now accept our loader because it's a real DataLoader instance
        lr_finder = LRFinder(model, temp_optimizer, criterion, device=device)
        lr_finder.range_test(lr_finder_loader, end_lr=1, num_iter=100, step_mode="exp")
        
        # --- Automatic analysis logic ---
        losses = np.array(lr_finder.history["loss"])[10:-5]
        lrs = np.array(lr_finder.history["lr"])[10:-5]
        gradients = np.gradient(losses)
        steepest_descent_idx = np.argmin(gradients)
        suggested_lr = lrs[steepest_descent_idx]
        
        LEARNING_RATE = suggested_lr
        print(f"Automated LR Finder finished. Suggested LR: {LEARNING_RATE:.2e}", flush=True)
        
        # Reset the model to its initial state before training
        model.load_state_dict(initial_state)
        
        # Clean up the temporary loader to free memory
        del lr_finder_loader
        del lr_finder
        # Optional: plot for verification
        # lr_finder.plot()
        # import matplotlib.pyplot as plt
        # plt.savefig('lr_finder_plot_auto.png')

    # --------------------------------------------------------------------------
    
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.AdamW([
    {'params': model.videomae.parameters(), 'lr': 1e-5},  # backbone
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # classifier head
    ])
    best_val_accuracy = 0.0
    epochs_no_improve = 0

    print("\nStarting training...",flush=True)
    # --- Training & Validation Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4)
            labels = batch["labels"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4)
                labels = batch["labels"].to(device, non_blocking=True)
                
                outputs = model(pixel_values)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"--- Epoch [{epoch+1}/{NUM_EPOCHS}] ---",flush=True)
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%",flush=True)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'video_mae_v2_ft.pth')
            print(f"New best model saved with accuracy: {val_accuracy:.2f}%",flush=True)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.",flush=True)
            break

    print("\nFinished Training!",flush=True)
    
    # --- Final Test Evaluation ---
    print("\nEvaluating on the test set with the best model...",flush=True)
    try:
        model.load_state_dict(torch.load('video_mae_v2_ft.pth'))
    except FileNotFoundError:
        print("Could not find 'best_model.pth'. Testing with the last model state.")
        
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(pixel_values)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
    test_accuracy = 100 * test_correct / test_total
    print(f"Final Test Accuracy: {test_accuracy:.2f}%",flush=True)