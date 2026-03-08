# 1. Import and initialize model
import torch
from rfdetr import RFDETRNano

model = RFDETRNano(pretrain_weights = "../../model/rf-detr-nano.pth") #your rfdetr path


# 2. Define your dataset path
dataset_path = "../../data/coco"

# 3. Start training
history = []

def on_epoch_end(data):
    history.append(data)

model.callbacks["on_fit_epoch_end"].append(on_epoch_end)

model.train(
    dataset_dir=dataset_path,
    train_ann_file=f"{dataset_path}/train/_annotations.coco.json",
    val_ann_file=f"{dataset_path}/valid/_annotations.coco.json",
    epochs=50,
    batch_size=16,
    grad_accum_steps=1,
    lr=1e-4,
    resolution=704,
    device="cuda",  # use GPU if available
    amp=True,       # enable mixed precision (faster on modern GPUs)
    early_stopping=True
)

