import os
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import warnings
from qwen_vl_utils import process_vision_info

# Suppress a known transformers warning
#warnings.filterwarnings("ignore", message="The models vision encoder did not receive absolute position embeddings")

# --- 1. Configuration ---
BASE_DIR = "."  #set you model folder
MODEL_NAME = "EBQwen2.5-VL-3B"
MODEL_PATH = os.path.join(BASE_DIR, "EBQwen2.5-VL-3B")
# Reverted to original path for processor as recommended for stability
ORIGINAL_MODEL_PATH = os.path.join(BASE_DIR, "Qwen/Qwen2.5-VL-3B-Instruct")
DATA_FILE = ["./data2024/model_input_dataset_2024.json","./data2025/model_input_dataset_2025.json"] #set your folder to BARD one
OUTPUT_DIR = "./output"
PREDICTIONS_FILE = [os.path.join(OUTPUT_DIR, "action_caption_"+ MODEL_NAME+ "2024.json"),os.path.join(OUTPUT_DIR, "action_caption_"+ MODEL_NAME+ "2025.json")]

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
    
    min_pixels = 256 * 28 * 28
    max_pixels = 448  * 28 * 28#1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(ORIGINAL_MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels
    )

    print("? Model and processor loaded successfully.")
    return model, processor

def main():
    """Main function to run inference and save predictions."""
    if not os.path.exists(MODEL_PATH):
        print(f"? Error: Model path not found: {MODEL_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model, processor = load_model_and_processor()
    for set_i in range(len(DATA_FILE)):
        print(f"?? Loading data from {DATA_FILE[set_i]}...")
        with open(DATA_FILE[set_i], 'r') as f:
            dataset = json.load(f)
            
        results = []
        print(f"\n?? Starting inference on {len(dataset)} videos...")
        
        fps = 3.0
        for item in tqdm(dataset, desc="Processing videos"):
            video_path = item['video']
            
            human_prompt = ""
            for conv in item['conversations']:
                if conv['from'] == 'human':
                    human_prompt = conv['value'].replace('<video>\n', '').strip()
                    break
            
            if not os.path.exists(video_path):
                print(f"?? Warning: video not found at {video_path}. Skipping.")
                continue
                
            if not human_prompt:
                print(f"?? Warning: No human prompt found for {video_path}. Skipping.")
                continue


            # Prepare model inputs
            messages = [{"role": "user", "content": [{"type": "video", "video": video_path,"resized_height": 420,
                    "resized_width": 784,"fps": fps,}, {"type": "text", "text": human_prompt}]}]
            
            # Preparation for inference
            print("chat temp")
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print("----process")
            
            # Process the vision info
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            # Before processor
            print("Before processor:")
            if image_inputs is None:
                print("image_inputs: None")
            else:
                print(f"image_inputs: type={type(image_inputs)}, shape={getattr(image_inputs, 'shape', 'N/A')}")

            if video_inputs is None:
                print("video_inputs: None")
            else:
                print(f"video_inputs: type={type(video_inputs)}, length={len(video_inputs)}")
                for i, v in enumerate(video_inputs):
                    print(f"  video[{i}]: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")

            print("video_kwargs:")
            for k, v in video_kwargs.items():
                print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}, value={v if not hasattr(v, 'shape') else 'tensor/array'}")

            # Pass through processor
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )

            # Print after processor
            print("\nAfter processor:")
            for k, v in inputs.items():
                print(f"{k}: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")

            # Move to CUDA
            inputs = inputs.to("cuda")
            print("\nAfter moving to CUDA:")
            for k, v in inputs.items():
                print(f"{k}: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")


            # Generate response
            print("-----generate")
            generated_ids = model.generate(**inputs, max_new_tokens=400, do_sample=False)
            generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            results.append({
                "image": video_path,
                "ground_truth": item['conversations'][1]['value'],
                "prediction": response
            })
            
            print(results[-1])

        with open(PREDICTIONS_FILE[set_i], 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"\n? Inference complete. Predictions saved to {PREDICTIONS_FILE[set_i]}")

if __name__ == "__main__":
    main()