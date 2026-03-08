#!/bin/bash

# Output file
LOGFILE="run_output.txt"

# Clear the log file at start
> "$LOGFILE"


# Run classification - classification
python3 classification.py ../output/classification_Qwen2.5-VL-3B-Instruct.json >> "$LOGFILE" 2>&1
python3 classification.py ../output/classification_Qwen2.5-VL-7B-Instruct.json >> "$LOGFILE" 2>&1
python3 classification.py ../output/classification_EBQwen2.5-VL-3B.json       >> "$LOGFILE" 2>&1



