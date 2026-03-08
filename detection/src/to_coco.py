import os
import json
from PIL import Image
import shutil

# Paths
base_input_dir = '.'
base_output_dir = '.'
if base_input_dir ==".":
    raise(Exception("Fix your paths"))

splits = ['train', 'val', 'test']

# Define your class list (based on YOLO class ids)
class_names = ['basketball', 'hoop', 'player','referee']  # Replace with actual class names

def convert_split(split):
    input_img_dir = os.path.join(base_input_dir, split, 'images')
    input_lbl_dir = os.path.join(base_input_dir, split, 'labels')

    output_img_dir = os.path.join(base_output_dir, split, 'images')
    os.makedirs(output_img_dir, exist_ok=True)

    annotations = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for i, class_name in enumerate(class_names):
        annotations["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": "none"
        })

    annotation_id = 1
    image_id = 1

    for file in os.listdir(input_lbl_dir):
        if not file.endswith('.txt'):
            continue

        base_name = os.path.splitext(file)[0]
        image_path = os.path.join(input_img_dir, base_name + '.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(input_img_dir, base_name + '.png')
            if not os.path.exists(image_path):
                print(f"Image for {file} not found.")
                continue

        # Copy image to output folder
        shutil.copy(image_path, os.path.join(output_img_dir, os.path.basename(image_path)))

        with Image.open(image_path) as img:
            width, height = img.size

        annotations["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        })

        with open(os.path.join(input_lbl_dir, file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, w, h = map(float, parts)
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                x = x_center - w / 2
                y = y_center - h / 2

                annotations["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    # Save COCO JSON
    output_anno_dir = os.path.join(base_output_dir, 'annotations')
    os.makedirs(output_anno_dir, exist_ok=True)
    with open(os.path.join(output_anno_dir, f'instances_{split}.json'), 'w') as f:
        json.dump(annotations, f, indent=4)

    print(f"Converted {split} set to COCO format.")

# Run for all splits
for split in splits:
    convert_split(split)
