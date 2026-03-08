import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pathlib
import numpy as np
from scipy.stats import chi2_contingency

# Set Seaborn style for beautiful graphs
sns.set_theme(style="whitegrid", context="talk")

# --- Part 1: Load and process players.csv (Ground Truth) ---

players_path = "../data/players.csv"

# Read CSV with Number as string
df = pd.read_csv(players_path, dtype={"Number": str},sep=";")

# Filter out numbers containing commas
df = df[~df["Number"].str.contains(",", na=False)]

# Count occurrences of each jersey number
label_counts_players = df["Number"].value_counts()

# Total valid entries for players
total_players = label_counts_players.sum()

# Convert to relative frequencies
relative_freq_players = label_counts_players / total_players

# Custom sort function (Extended to handle nan)
def sort_key(x):
    s = str(x).strip().lower()
    if s == 'nan':
        return (4, 0) # Place NaN at the end
    if s == "0":
        return (0, 0)
    elif s == "00":
        return (1, 0)
    else:
        try:
            return (2, int(s))
        except ValueError:
            return (3, s)

sorted_labels_players = sorted(relative_freq_players.index, key=sort_key)


# --- Part 2: Load and process output_all_annotated_validated_str_zero.txt ---

txt_path = pathlib.Path(r"../data/all.txt")

# We will create two lists: 
# 1. labels_clean (No NaN) - For your original relative comparison
# 2. labels_all (With NaN) - For the new absolute count graph
labels_clean = []
labels_all = []

try:
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            val = parts[-1].strip()
            
            # Add to the "All" list (handling NaN string normalization)
            if val.lower() == "nan":
                labels_all.append("NaN")
            else:
                labels_all.append(val)
                # Add to "Clean" list only if not nan (Original Logic)
                labels_clean.append(val)

except FileNotFoundError:
    print(f"CRITICAL ERROR: Could not find {txt_path}")

print(f"Total valid examples (excluding NaN): {len(labels_clean)}")
print(f"Total examples (including NaN): {len(labels_all)}")

# --- Process Data for Plot 1 (Relative Frequency Comparison - No NaN) ---

label_counts_txt_clean = Counter(labels_clean)
total_txt_clean = sum(label_counts_txt_clean.values())
relative_freq_txt = {k: v / total_txt_clean for k, v in label_counts_txt_clean.items()}

# Prepare DataFrame for Seaborn (Plot 1)
data_rel = []
all_clean_labels = sorted(set(relative_freq_players.index) | set(relative_freq_txt.keys()), key=sort_key)

for label in all_clean_labels:
    # Ground Truth
    data_rel.append({
        "Jersey Number": label,
        "Value": relative_freq_players.get(label, 0),
        "Type": "Relative Frequency",
        "Dataset": "Ground Truth"
    })
    # Dataset (Clean)
    data_rel.append({
        "Jersey Number": label,
        "Value": relative_freq_txt.get(label, 0),
        "Type": "Relative Frequency",
        "Dataset": "Dataset"
    })

df_rel = pd.DataFrame(data_rel)

# --- Process Data for Plot 2 (Absolute Count - With NaN) ---

label_counts_all = Counter(labels_all)

# Prepare DataFrame for Seaborn (Plot 2)
data_abs = []
all_labels_inc_nan = sorted(label_counts_all.keys(), key=sort_key)

for label in all_labels_inc_nan:
    data_abs.append({
        "Jersey Number": label,
        "Value": label_counts_all[label],
        "Type": "Absolute Count",
        "Dataset": "Dataset"
    })

df_abs = pd.DataFrame(data_abs)


# --- Part 3: Generate Graphs ---

# GRAPH 1: Relative Distribution Comparison (Original Request)
plt.figure(figsize=(14, 7))
ax1 = sns.barplot(
    data=df_rel,
    x="Jersey Number",
    y="Value",
    hue="Dataset",
    palette={"Ground Truth": "#3498db", "Dataset": "#e74c3c"},
    edgecolor="black",
    linewidth=1,
    alpha=0.85
)
plt.title("Dataset and Ground Truth Relative Distribution Comparison", fontsize=18, weight='bold')
plt.xlabel("Jersey Number", fontsize=14)
plt.ylabel("Relative Frequency", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.legend()
sns.despine()
plt.tight_layout()
plt.show() # Display first graph

output_filename = '../figure/dataset_rel_number.png'
plt.savefig(output_filename)


# GRAPH 2: Dataset Absolute Distribution with NaN (New Request)
plt.figure(figsize=(14, 7))
# Use a specific color for NaN to highlight it
colors = ['#95a5a6' if x == 'NaN' else '#2ecc71' for x in df_abs['Jersey Number']]

ax2 = sns.barplot(
    data=df_abs,
    x="Jersey Number",
    y="Value",
    palette=colors, # Custom palette to highlight NaN
    edgecolor="black",
    linewidth=1,
    alpha=0.9
)

# Annotate bars with exact counts
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height())}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha = 'center', va = 'center', 
                 xytext = (0, 9), 
                 textcoords = 'offset points',
                 fontsize=12, fontweight='bold')

plt.title("Dataset Absolute Distribution", fontsize=18, weight='bold')
plt.xlabel("Jersey Number", fontsize=14)
plt.ylabel("Absolute Frequency", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
sns.despine()
plt.tight_layout()
plt.show() # Display second graph

output_filename = '../figure/dataset_abs_number.png'
plt.savefig(output_filename)


# --- Part 4: Statistical Test (Original Logic) ---

# Create a sorted list of all labels that appear in either distribution (Clean version)
all_labels_stat = sorted(set(sorted_labels_players) | set(labels_clean), key=sort_key)

# Convert relative frequencies to counts (using total counts)
obs_counts = []
exp_counts = []

for label in all_labels_stat:
    # Players data
    obs = label_counts_players.get(label, 0)
    # TXT data (Clean)
    exp = label_counts_txt_clean.get(label, 0)
    
    obs_counts.append(obs)
    exp_counts.append(exp)

# Scale expected counts to match observed sum
if sum(exp_counts) > 0:
    exp_counts_scaled = [c * sum(obs_counts) / sum(exp_counts) for c in exp_counts]
else:
    exp_counts_scaled = [0] * len(exp_counts)

exp_counts_scaled = np.array(exp_counts_scaled)
obs_counts = np.array(obs_counts)

# Build contingency table
contingency_table = np.vstack([obs_counts, exp_counts_scaled])

# Run chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print("\n=== Chi-square test of independence ===")
print(f"Chi-square statistic: {chi2_stat:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Reject null hypothesis: Distributions differ.")
else:
    print("Fail to reject null hypothesis: No significant difference.")