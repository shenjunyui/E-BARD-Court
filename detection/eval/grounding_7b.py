# /leonardo_work/uBS25_EcoGiu/pipeline_cv/yolobb/src/QWEN_FINE_TUNE/evaluation/run_grounding_inference.py

import os
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import warnings
from PIL import Image # <-- 1. IMPORT THE PILLOW (PIL) LIBRARY
from qwen_vl_utils import process_vision_info

# Suppress a known transformers warning
#warnings.filterwarnings("ignore", message="The models vision encoder did not receive absolute position embeddings")

# --- 1. Configuration ---
BASE_DIR = "/../model"
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
MODEL_PATH = os.path.join(BASE_DIR, "Qwen/Qwen2.5-VL-7B-Instruct")
# Reverted to original path for processor as recommended for stability
ORIGINAL_MODEL_PATH = MODEL_PATH
DATA_FILE = "../../EBQwen_dataset/test_grounding_dataset.json"
OUTPUT_DIR = "./output"
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "grounding_"+ MODEL_NAME+ ".json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_processor():
    """Loads the Qwen model and processor."""
    print(f"?? Loading fine-tuned model from: {MODEL_PATH}")
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}.")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    print(f"?? Loading processor from: {ORIGINAL_MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
    
    print("? Model and processor loaded successfully.")
    return model, processor

def main():
    """Main function to run inference and save predictions."""
    if not os.path.exists(MODEL_PATH):
        print(f"? Error: Model path not found: {MODEL_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model, processor = load_model_and_processor()
    
    print(f"?? Loading data from {DATA_FILE}...")
    with open(DATA_FILE, 'r') as f:
        dataset = json.load(f)
        
    results = []
    print(f"\n?? Starting inference on {len(dataset)} images...")

    for item in tqdm(dataset, desc="Processing Images"):
        image_path = item['image']
        
        human_prompt = ""
        for conv in item['conversations']:
            if conv['from'] == 'human':
                human_prompt = conv['value'].replace('<image>\n', '').strip()
                break
        
        if not os.path.exists(image_path):
            print(f"?? Warning: Image not found at {image_path}. Skipping.")
            continue
            
        if not human_prompt:
            print(f"?? Warning: No human prompt found for {image_path}. Skipping.")
            continue

        # <-- 2. OPEN THE IMAGE FILE AND CONVERT TO RGB -->
        # This converts the file path into actual image data the model can use.
        # .convert("RGB") is important to ensure all images have 3 color channels.
        image = Image.open(image_path).convert("RGB")

        # Prepare model inputs
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": human_prompt}]}]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=800, do_sample=False)
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        results.append({
            "image": image_path,
            "ground_truth": item['conversations'][1]['value'],
            "prediction": response
        })

    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n? Inference complete. Predictions saved to {PREDICTIONS_FILE}")

if __name__ == "__main__":
    main()