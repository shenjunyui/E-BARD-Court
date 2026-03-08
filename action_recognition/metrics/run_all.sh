#!/bin/bash

# Output file
LOGFILE="run_output.txt"

# Clear the log file at start
> "$LOGFILE"
# Run caption



python caption_nograph.py ../output/action_caption_Qwen2.5-VL-7B-Instruct_20242025.json       >> "$LOGFILE" 2>&1
python caption_nograph.py ../output/action_caption_Qwen2.5-VL-3B-Instruct_20242025.json       >> "$LOGFILE" 2>&1
python caption_nograph.py ../output/action_caption_EBQwen2.5-VL-3B_20242025.json       >> "$LOGFILE" 2>&1
python caption_nograph.py ../output/action_caption_gemini-2.5-pro-preview-06-05_20242025.json       >> "$LOGFILE" 2>&1
python caption_nograph.py ../output/action_caption_BQwen2.5-VL-3B_20242025.json       >> "$LOGFILE" 2>&1
