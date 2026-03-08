import os
import sys
import glob
import re
from collections import Counter
import warnings
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import base64
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import get_image_token_id

# Suppress warnings if needed
warnings.filterwarnings("ignore")

# --- 1. Configuration: Set up all the paths and parameters ---
BASE_DIR = "../../model"
MODEL_PATH = os.path.join(BASE_DIR, "RolmOCR")
DATA_DIR = os.path.join(BASE_DIR, "../data/")
ANNOTATED_CROP_DIR = os.path.join(DATA_DIR, "annotated_crop")
VAL_FILE_PATH = os.path.join(DATA_DIR, "all.txt")
OUTPUT_FILE = "cropped_si_rolmo.csv" # File to save results

print("Starting RolmOCR inference with vLLM (Single-Image)...")

PROMPT = "What is the player jersey number that you see in the images?The possible numbers are: 00 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 up to 99. If you cannot identify the number, answer 'nan'. If you can recognize it, answer with the jersey number only"
NUM_FRAMES_TO_PROCESS = 16

def load_model():
    """
    Loads the RolmOCR model using vLLM's LLM class.
    """
    print(f"?? Loading model from: {MODEL_PATH}")
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}. Please check the path.")
    
    # Load the model using the LLM class from vLLM
    model = LLM(MODEL_PATH)
    
    print("? RolmOCR model loaded successfully.")
    return model

def encode_image(image_path):
    """
    Encodes an image to a base64 string.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

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
        return match.group(0)
    
    return 'nan'

def get_prediction_for_image(model, image_path: str):
    """
    Runs inference on a single image using vLLM's generate method.
    """
    try:
        # Step 1: Create the prompt string with the image placeholder
        image_token_id = get_image_token_id()
        prompt_with_image_token = f"<image>{PROMPT}"
        
        # Step 2: Define sampling parameters
        sampling_params = SamplingParams(temperature=0.0, max_tokens=25, top_p=0.95, top_k=40)
        
        # Step 3: Run the generation
        # The generate method takes a list of prompts and a list of image paths
        output = model.generate(
            prompt_with_image_token, 
            sampling_params=sampling_params, 
            images=[image_path]
        )
        
        # Step 4: Extract and parse the generated text
        response = output[0].outputs[0].text if output else ""
        
        return parse_model_output(response)

    except Exception as e:
        print(f"  - ?? Error processing image {os.path.basename(image_path)}: {e}")
        return 'nan'

def calculate_mode(predictions: list):
    """
    Calculates the mode (most frequent item) from a list of predictions.
    """    
    if not predictions:
        return 'nan'
    counts = Counter(predictions)
    mode, _ = counts.most_common(1)[0]
    return mode

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
    
    labels = sorted(list(set(ground_truth) | set(predictions)))
    
    print("\nClassification Report:")
    report = classification_report(ground_truth, predictions, labels=labels, zero_division=0)
    print(report)
    
    print("\nConfusion Matrix:")
    print("Rows: True Labels, Columns: Predicted Labels\n")
    conf_matrix = confusion_matrix(ground_truth, predictions, labels=labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print(conf_matrix_df)

def main():
    """
    Main function to orchestrate the loading, processing, and evaluation pipeline.
    """
    model = load_model()
    
    if not os.path.exists(VAL_FILE_PATH):
        print(f"? Error: Validation file not found at {VAL_FILE_PATH}")
        return

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
        
        frame_predictions = []
        for img_path in image_files[:NUM_FRAMES_TO_PROCESS]:
            prediction = get_prediction_for_image(model, img_path)
            frame_predictions.append(prediction)
            print(f"  - Frame: {os.path.basename(img_path):<12} -> Predicted: {prediction}")

        final_prediction = calculate_mode(frame_predictions)
        
        all_ground_truths.append(true_jersey_number)
        all_predictions.append(final_prediction)
        
        print(f"  -> Raw Predictions: {frame_predictions}")
        print(f"  -> Final Prediction (Mode): {final_prediction}")
        
        if str(final_prediction) == str(true_jersey_number):
            print(f"  -> Result: ? CORRECT")
        else:
            print(f"  -> Result: ? INCORRECT (Ground Truth: {true_jersey_number})")
    
    results_df = pd.DataFrame({
        'ground_truth': all_ground_truths,
        'prediction': all_predictions
    })
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n?? Results saved to {OUTPUT_FILE}")
    
    if os.path.exists(OUTPUT_FILE):
        generate_report(OUTPUT_FILE)

if __name__ == "__main__":
    main()