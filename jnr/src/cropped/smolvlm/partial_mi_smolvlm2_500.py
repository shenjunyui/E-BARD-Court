import os
import sys
import glob
import re
from collections import Counter
import warnings
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Suppress a known transformers warning that is not critical for this task
warnings.filterwarnings("ignore", message="The models vision encoder did not receive absolute position embeddings")

# --- 1. Configuration: Set up all the paths and parameters ---

BASE_DIR = "/leonardo_scratch/large/userexternal/ggiudici"
MODEL_PATH = os.path.join(BASE_DIR, "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
DATA_DIR = "/leonardo_work/uBS25_EcoGiu/pipeline_cv/yolobb/src/JNR/data"
ANNOTATED_CROP_DIR = os.path.join(DATA_DIR, "annotated_crop")
VAL_FILE_PATH = os.path.join(DATA_DIR, "all.txt")
OUTPUT_FILE = "partial_mi_smolvlm2_500.csv"  # File to save results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "What is the number on the basketball player's jersey? Provide only the number. Consider all the provided images before giving a final answer."
NUM_FRAMES_TO_PROCESS = 8


def load_model_and_processor():
    """
    Loads the SmolVLM2 model and processor using the recommended classes.
    """
    print(f"?? Loading model and processor from: {MODEL_PATH}")
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}. Please check the path.")
    
    # Load the specific model class for SmolVLM2
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        _attn_implementation="flash_attention_2"
    ).eval()
    
    # Load the dedicated processor which handles both text and images
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    print("? Model and processor loaded successfully.")
    return model, processor


def parse_model_output(text: str):
    """
    Parses the model's text output to extract a jersey number.
    Returns a string of the number if found, otherwise 'nan'.
    """
    clean_text = text.lower().strip()
    if 'nan' in clean_text:
        return 'nan'
    
    match = re.search(r'\d+', clean_text)
    if match:
        try:
            return match.group(0)
        except (ValueError, IndexError):
            return 'nan'
    
    return 'nan'


def get_prediction_for_track(model, processor, image_paths: list):
    """
    Runs inference on a list of images using a single multi-image request.
    """
    if not image_paths:
        return 'nan'

    try:
        # Step 1: Create the message payload with multiple images and a single text query
        image_inputs = [Image.open(path).convert('RGB') for path in image_paths]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                ] + [{"type": "image", "path": img} for img in image_inputs],
            }
        ]
        
        # Step 2: Prepare inputs for the model using the processor
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        # Step 3: Generate the response
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=25)
        
        # Step 4: Decode the output, trimming the input prompt to get only the response
        response_list = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        
        response = response_list[0] if response_list else response_list
        
        print("response_list")

        print(response_list)
        return parse_model_output(response)

    except Exception as e:
        print(f"  - ?? Error processing track with multiple images: {e}")
        return 'nan'


def generate_report(filepath: str):
    """
    Reads the output file and prints a classification report and confusion matrix.
    """
    print("\n" + "="*50)
    print("?? FINAL PERFORMANCE REPORT ??")
    print("="*50)

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find the results file at {filepath}")
        return

    ground_truth = df['ground_truth'].astype(str)
    predictions = df['prediction'].astype(str)
    
    # Get all unique labels from both ground truth and predictions
    labels = sorted(list(set(ground_truth) | set(predictions)))

    # --- Classification Report ---
    print("\nClassification Report:")
    # Setting zero_division=0 handles cases where a class has no predictions
    report = classification_report(ground_truth, predictions, labels=labels, zero_division=0)
    print(report)

    # --- Confusion Matrix ---
    print("\nConfusion Matrix:")
    print("Rows: True Labels, Columns: Predicted Labels\n")
    conf_matrix = confusion_matrix(ground_truth, predictions, labels=labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print(conf_matrix_df)


def main():
    """
    Main function to orchestrate the loading, processing, and evaluation pipeline.
    """
    model, processor = load_model_and_processor()
    
    if not os.path.exists(VAL_FILE_PATH):
        print(f"? Error: Validation file not found at {VAL_FILE_PATH}")
        return

    # Lists to store results for final analysis
    all_ground_truths = []
    all_predictions = []
    
    with open(VAL_FILE_PATH, 'r') as f:
        lines = f.readlines()

    print(f"\n?? Starting evaluation on {len(lines)} tracks from {VAL_FILE_PATH}...")
    
    for i, line in enumerate(lines):
        try:
            old_path, _, class_label_str = line.strip().split()
            track_folder_name = old_path.replace('annotated/', '')
            track_folder_path = os.path.join(ANNOTATED_CROP_DIR, track_folder_name)
            true_jersey_number = class_label_str
        except (ValueError, IndexError):
            print(f"Skipping malformed line {i+1}: '{line.strip()}'")
            continue

        print(f"\n[{i+1}/{len(lines)}] Processing: {track_folder_name} | True Jersey: {true_jersey_number}")
        
        if not os.path.isdir(track_folder_path):
            print(f"  - Directory not found. Skipping.")
            continue
        
        image_files = sorted(glob.glob(os.path.join(track_folder_path, '*.png')))
        if len(image_files) < NUM_FRAMES_TO_PROCESS:
            print(f"  - Found {len(image_files)} frames, but {NUM_FRAMES_TO_PROCESS} are required. Skipping.")
            continue
        
        # --- NEW: Process all frames in a single request ---
        selected_images = image_files[:NUM_FRAMES_TO_PROCESS]
        final_prediction = get_prediction_for_track(model, processor, selected_images)
        
        # Store the results
        all_ground_truths.append(true_jersey_number)
        all_predictions.append(final_prediction)
        
        print(f"  -> Final Prediction (Multi-Image): {final_prediction}")
        
        if str(final_prediction) == str(true_jersey_number):
            print(f"  -> Result: ? CORRECT")
        else:
            print(f"  -> Result: ? INCORRECT (Ground Truth: {true_jersey_number})")
    
    # --- Save results to a file ---
    results_df = pd.DataFrame({
        'ground_truth': all_ground_truths,
        'prediction': all_predictions
    })
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n?? Results saved to {OUTPUT_FILE}")
    
    # --- Generate and print the final report from the saved file ---
    if os.path.exists(OUTPUT_FILE):
        generate_report(OUTPUT_FILE)

if __name__ == "__main__":
    main()