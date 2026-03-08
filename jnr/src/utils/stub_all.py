import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
import os
import cv2
import numpy as np

# --- Configuration ---
# Base path for source video files
SOURCE_VIDEO_DIR = "../../data/all"
# Path to your pre-trained YOLO model
YOLO_MODEL_PATH = "../../model/BODD_yolov8n_0001.pt"

# Base path where all annotated videos will be saved
ANNOTATED_VIDEO_BASE_DIR = "../../data/annotated_videos"
# Base path to save the extracted stubs
STUB_OUTPUT_BASE_DIR = "../../data/stub"

# Model confidence and IOU thresholds
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.6
# Class ID for players (assuming 'player' is class_id 2 in your model)
PLAYER_CLASS_ID = 2

# --- Setup Directories ---
# Create base output directories if they don't exist
os.makedirs(ANNOTATED_VIDEO_BASE_DIR, exist_ok=True)
os.makedirs(STUB_OUTPUT_BASE_DIR, exist_ok=True)

# --- Main Processing Logic ---

# Load the local YOLO model once
model = YOLO(YOLO_MODEL_PATH)

# Initialize tracker and annotators once
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Get a list of all video files in the source directory
video_files = [f for f in os.listdir(SOURCE_VIDEO_DIR) if f.endswith('.mp4')]

if not video_files:
    print(f"No .mp4 video files found in {SOURCE_VIDEO_DIR}. Please check the path.")
else:
    print(f"Found {len(video_files)} video(s) to process in {SOURCE_VIDEO_DIR}.")

    # Loop through each video file
    for video_filename in video_files:
        source_video_path = os.path.join(SOURCE_VIDEO_DIR, video_filename)
        video_name_without_ext = os.path.splitext(video_filename)[0]

        # Define paths for the current video's outputs
        target_video_path = os.path.join(ANNOTATED_VIDEO_BASE_DIR, f"{video_name_without_ext}_annotated.mp4")
        current_video_stub_base_folder = os.path.join(STUB_OUTPUT_BASE_DIR, video_name_without_ext)

        print(f"\n--- Processing video: {video_filename} ---")
        print(f"  Saving annotated video to: {target_video_path}")
        print(f"  Saving player stubs to: {current_video_stub_base_folder}/track_ID/frame_IDX.png")

        # Create the video-specific stub output directory
        os.makedirs(current_video_stub_base_folder, exist_ok=True)

        # Get video information and set up frame generator
        video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

        # Open a video sink to save the annotated video
        with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
            # Loop through each frame of the video with an index
            for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc=f"Processing {video_filename}")):
                # Perform inference on the frame
                results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]

                # Convert ultralytics results to supervision detections
                detections = sv.Detections.from_ultralytics(results)

                # Update the tracker with new detections
                detections = tracker.update_with_detections(detections)

                # --- Logic to extract and save player stubs ---
                # Iterate over each detected object
                for i in range(len(detections)):
                    # Check if the detected object is a player and has a tracker ID
                    if detections.class_id[i] == PLAYER_CLASS_ID and detections.tracker_id[i] is not None:
                        # Get the bounding box and tracker ID
                        bbox = detections.xyxy[i]
                        tracker_id = detections.tracker_id[i]

                        # Crop the player stub from the original frame
                        # Convert coordinates to integers
                        x1, y1, x2, y2 = map(int, bbox)
                        # Ensure coordinates are within frame bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)

                        player_stub = frame[y1:y2, x1:x2]

                        # Construct the tracker-specific subfolder path
                        current_tracker_stub_folder = os.path.join(current_video_stub_base_folder, f"track_{tracker_id}")
                        os.makedirs(current_tracker_stub_folder, exist_ok=True)

                        # Construct the filename for the stub
                        stub_filename = f"frame_{frame_idx}.png"
                        save_path = os.path.join(current_tracker_stub_folder, stub_filename)

                        # Save the stub, but only if the crop is not empty
                        if player_stub.size > 0 and player_stub.shape[0] > 0 and player_stub.shape[1] > 0:
                            cv2.imwrite(save_path, player_stub)
                        # else:
                        #     print(f"Skipping empty stub for {video_filename}, frame {frame_idx}, track {tracker_id}")

                # Create labels for annotations
                labels = [
                    f"{model.model.names[class_id]} {confidence:0.2f} ID:{tracker_id}"
                    for confidence, class_id, tracker_id
                    in zip(detections.confidence, detections.class_id, detections.tracker_id)
                ]
                # Annotate the frame with bounding boxes
                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(), detections=detections
                )

                # Annotate the frame with labels
                annotated_labeled_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )

                # Write the annotated frame to the output video
                sink.write_frame(frame=annotated_labeled_frame)

        print(f"? Finished processing {video_filename}.")
        print(f"  Annotated video saved to: {target_video_path}")
        print(f"  Player stubs saved in: {current_video_stub_base_folder}")

    print("\nAll videos processed successfully!")
