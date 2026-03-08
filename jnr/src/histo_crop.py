import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_folder_distribution(base_dir: str, cap_value: int = 16):
    """
    Analyzes file distribution and plots a histogram with a capped maximum value.

    Args:
        base_dir (str): The path to the main directory to analyze.
        cap_value (int): The value at which to cap the histogram. 
                         All higher values will be grouped into this bin.
    """
    print(f"Analyzing directory: {base_dir}\n")

    # --- 1. Data Collection ---
    if not os.path.isdir(base_dir):
        print(f"❌ Error: Directory not found at '{base_dir}'")
        return

    image_counts = []
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                num_images = len([
                    f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                image_counts.append(num_images)
            except OSError as e:
                print(f"⚠️ Warning: Could not access files in {folder_path}. Error: {e}")

    if not image_counts:
        print("❌ No subdirectories with images found to analyze.")
        return

    # --- 2. Calculate Statistics (on original, full data) ---
    counts_array = np.array(image_counts)
    
    print("--- 📊 FULL STATS (uncut data) ---")
    print(f"Total 'track' folders analyzed: {len(counts_array)}")
    print(f"Total cropped images found:     {np.sum(counts_array)}")
    print("-" * 20)
    print(f"Average images per folder:      {np.mean(counts_array):.2f}")
    print(f"Median images per folder:       {np.median(counts_array)}")
    print(f"Min images in a single folder:  {np.min(counts_array)}")
    print(f"Max images in a single folder:  {np.max(counts_array)}")
    print("-" * 20)
    
    # --- 3. Generate Histogram (Capped at 16) ---
    print(f"\nGenerating histogram, capping display at {cap_value}...")
    
    # Create a new array where all values > cap_value are set to cap_value
    capped_counts = np.clip(counts_array, a_min=None, a_max=cap_value)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # Define bins to be centered on integers from 0 to 16
    bins = np.arange(0, cap_value + 2) - 0.5

    plt.hist(capped_counts, bins=bins, edgecolor='black', alpha=0.75)

    plt.title(f'Distribution of Frames per Stub', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Images Remaining in Folder', fontsize=12)
    plt.ylabel('Number of Folders (Frequency)', fontsize=12)
    
    # Set x-axis ticks to be integers and label the last one as "16+"
    x_ticks = np.arange(0, cap_value + 1)
    x_labels = [str(tick) for tick in x_ticks]
    x_labels[-1] = f'{cap_value}+' # Label the last tick
    plt.xticks(ticks=x_ticks, labels=x_labels)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    output_filename = '../figure/image_distribution_histogram_capped.png'
    plt.savefig(output_filename)
    print(f"✅ Histogram saved as '{output_filename}'")
    
    plt.show()


if __name__ == '__main__':
    folder_to_analyze = "../data/annotated_crop"
    
    # Run the analysis, capping the histogram at 16
    analyze_folder_distribution(folder_to_analyze, cap_value=16)