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
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Suppress a known transformers warning that is not critical for this task
warnings.filterwarnings("ignore", message="The models vision encoder did not receive absolute position embeddings")

# --- 1. Configuration: Set up all the paths and parameters ---
# Updated BASE_DIR to your current project location
BASE_DIR = "../../model"

# Model and data paths are now relative to the updated BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "allenai/olmOCR-7B-0225-preview")
PROCESSOR_PATH = os.path.join(BASE_DIR, "Qwen/Qwen2-VL-7B-Instruct")


# Assuming the rest of your data structure is inside the new BASE_DIR
# If not, you may need to adjust this DATA_DIR path as well
DATA_DIR = os.path.join(BASE_DIR, "../data")
ANNOTATED_CROP_DIR = os.path.join(DATA_DIR, "annotated")
VAL_FILE_PATH = os.path.join(DATA_DIR, "all.txt")
OUTPUT_FILE = "full_si_olmo.csv"  # File to save results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "What is the player jersey number that you see in the images?The possible numbers are: 00 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 up to 99. If you cannot identify the number, answer 'nan'. If you can recognize it, answer with the jersey number only"
NUM_FRAMES_TO_PROCESS = 16

def load_model_and_processor():
    """ Loads the olmOCR model and its corresponding processor. """
    print(f"🕵️ Loading model from: {MODEL_PATH}")
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}. Please check the path.")

    # Load the specific model class for olmOCR
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto", # or torch.bfloat16
        device_map="auto"
    ).eval()

    # Per the olmOCR example, it uses the Qwen2-VL-7B-Instruct processor
    print(f"🕵️ Loading processor from: {PROCESSOR_PATH}")
    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)

    print("✅ Model and processor loaded successfully.")
    return model, processor

def parse_model_output(text: str):
    """ Parses the model's text output to extract a jersey number. """
    clean_text = text.lower().strip()
    if 'nan' in clean_text:
        return 'nan'
    # Search for one or more digits
    match = re.search(r'\d+', clean_text)
    if match:
        return match.group(0)
    return 'nan'

def get_prediction_for_image(model, processor, image_path: str):
    """ Runs inference on a single image using the olmOCR model. """
    try:
        # Step 1: Open the image using PIL
        image = Image.open(image_path).convert("RGB")

        # Step 2: Create the message payload in the format required by the processor
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image"}, # Placeholder for the image
            ]
        }]
        
        # Step 3: Prepare inputs for the model using the processor
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        # Step 4: Generate the response
        # Using do_sample=False for deterministic OCR, which is generally better
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False
        )

        # Step 5: Decode the output, trimming the input prompt to get only the response
        input_token_len = inputs.input_ids.shape[1]
        response_ids = generated_ids[:, input_token_len:]
        
        response = processor.batch_decode(
            response_ids,
            skip_special_tokens=True
        )[0]
        
        return parse_model_output(response)

    except Exception as e:
        print(f"  - ❌ Error processing image {os.path.basename(image_path)}: {e}")
        return 'nan'

def calculate_mode(predictions: list):
    """ Calculates the mode (most frequent item) from a list of predictions. """
    if not predictions:
        return 'nan'
    # Filter out 'nan' before calculating mode if you want to prioritize numbers
    valid_predictions = [p for p in predictions if p != 'nan']
    if not valid_predictions:
        return 'nan'
        
    counts = Counter(valid_predictions)
    mode, _ = counts.most_common(1)[0]
    return mode

def generate_report(filepath: str):
    """ Reads the output file and prints a classification report and confusion matrix. """
    print("\n" + "="*50)
    print("📊 FINAL PERFORMANCE REPORT 📊")
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
    report = classification_report(ground_truth, predictions, labels=labels, zero_division=0)
    print(report)

    # --- Confusion Matrix ---
    print("\nConfusion Matrix:")
    print("Rows: True Labels, Columns: Predicted Labels\n")
    conf_matrix = confusion_matrix(ground_truth, predictions, labels=labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print(conf_matrix_df)

def main():
    """ Main function to orchestrate the loading, processing, and evaluation pipeline. """
    model, processor = load_model_and_processor()

    if not os.path.exists(VAL_FILE_PATH):
        print(f"❌ Error: Validation file not found at {VAL_FILE_PATH}")
        return

    all_ground_truths = []
    all_predictions = []

    with open(VAL_FILE_PATH, 'r') as f:
        lines = f.readlines()
    
    print(f"\n🚀 Starting evaluation on {len(lines)} tracks from {VAL_FILE_PATH}...")
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
            prediction = get_prediction_for_image(model, processor, img_path)
            frame_predictions.append(prediction)
            print(f"  - Frame: {os.path.basename(img_path):<12} -> Predicted: {prediction}")

        final_prediction = calculate_mode(frame_predictions)
        
        all_ground_truths.append(true_jersey_number)
        all_predictions.append(final_prediction)

        print(f"  -> Raw Predictions: {frame_predictions}")
        print(f"  -> Final Prediction (Mode): {final_prediction}")
        if str(final_prediction) == str(true_jersey_number):
            print(f"  -> Result: ✅ CORRECT")
        else:
            print(f"  -> Result: ❌ INCORRECT (Ground Truth: {true_jersey_number})")

    # --- Save results to a file ---
    results_df = pd.DataFrame({
        'ground_truth': all_ground_truths,
        'prediction': all_predictions
    })
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Results saved to {OUTPUT_FILE}")

    # --- Generate and print the final report from the saved file ---
    if os.path.exists(OUTPUT_FILE):
        generate_report(OUTPUT_FILE)

if __name__ == "__main__":
    main()