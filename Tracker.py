import cv2
import numpy as np
from collections import defaultdict
import os
import csv
from datetime import datetime
from ultralytics import YOLO
import torch
import torch.nn as nn

# Load YOLOv8s model (you can use yolov5s, yolov5m, etc. depending on your model)
model = YOLO("yolov5m.pt")  # You can also try yolov5m.pt or yolov5l.pt

# Input video path
input_video_path = "/home/gaurav/Desktop/Projects/company/obj-tracking-video.mp4"

# Output CSV path
output_csv_path = "object_tracking_metrics.csv"

# Object tracking storage
object_tracks = defaultdict(list)  # {id: [(x, y), ...]}
object_entry_frame = {}  # {id: first_seen_frame}
object_last_seen_frame = {}  # {id: last_seen_frame}
frame_count = 0

# Initialize Kalman filter dictionary for each object
kalman_filters = {}

# RNN Model for Temporal Prediction (This part is for temporal modeling but not visualized)
class RNNModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Output the predicted position (x, y)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the last output of the sequence
        return out

# Initialize RNN model
rnn_model = RNNModel()
rnn_model.eval()  # Set to evaluation mode

# Function to create Kalman filter
def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    return kf

# Function to add text with a black outline
def put_text_with_outline(frame, text, position, font, font_scale, color, thickness):
    x, y = position
    # Outline (black)
    cv2.putText(frame, text, (x - 1, y - 1), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(frame, text, (x + 1, y - 1), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(frame, text, (x - 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(frame, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1)
    # Main text (white)
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

# Load video
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))

# Object tracker (YOLOv8)
tracker = model.track(source=input_video_path, stream=True, persist=True, conf=0.4, iou=0.5)

# Create a blank canvas for the top view (same size as the original video)
top_view_canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Process video frame-by-frame
for result in tracker:
    frame = result.orig_img.copy()
    frame_count += 1

    # Display stats on screen
    current_ids = set()

    if result.boxes.id is None:
        continue

    for box in result.boxes:
        obj_id = int(box.id.item())
        cls_id = int(box.cls.item())
        if cls_id != 0:  # Only track 'person' class
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw bounding box and centroid
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Add ID text with black outline
        put_text_with_outline(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Initialize Kalman Filter if not already initialized
        if obj_id not in kalman_filters:
            kalman_filters[obj_id] = create_kalman_filter()
            kalman_filters[obj_id].correct(np.array([cx, cy], np.float32))

        # Kalman prediction
        predicted = kalman_filters[obj_id].predict()
        pred_x, pred_y = int(predicted[0]), int(predicted[1])

        # Update Kalman filter with new measurements
        kalman_filters[obj_id].correct(np.array([cx, cy], np.float32))

        # Store the positions for the object
        if obj_id not in object_tracks:
            object_tracks[obj_id] = []

        object_tracks[obj_id].append((cx, cy))

        # Speed Calculation - Euclidean distance between consecutive positions
        if len(object_tracks[obj_id]) > 1:
            prev_x, prev_y = object_tracks[obj_id][-2]
            distance = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
            speed = distance * fps  # speed in pixels per second
            cv2.putText(frame, f"Speed: {round(speed, 2)} px/s", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Use RNN to predict future position (visualize this as magenta dot)
        if len(object_tracks[obj_id]) > 10:  # Use last 10 positions for prediction
            input_seq = np.array(object_tracks[obj_id][-10:])
            input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            predicted_pos = rnn_model(input_seq).detach().numpy()

            # Visualize RNN prediction (magenta dot)
            rnn_pred_x, rnn_pred_y = int(predicted_pos[0][0]), int(predicted_pos[0][1])
            cv2.circle(frame, (rnn_pred_x, rnn_pred_y), 4, (255, 0, 255), -1)  # Magenta prediction dot

        # Draw Kalman prediction (yellow dot)
        cv2.circle(frame, (pred_x, pred_y), 4, (0, 255, 255), -1)  # Yellow Kalman predicted dot

        # Update top view canvas (normalize coordinates for 3D path projection)
        normalized_x = int((cx / width) * width)  # Normalize x to [0, width]
        normalized_y = int((cy / height) * height)  # Normalize y to [0, height]
        normalized_pred_x = int((pred_x / width) * width)
        normalized_pred_y = int((pred_y / height) * height)

        # Plot on top view canvas with the assumption of a flat 2D projection for 3D movement
        cv2.circle(top_view_canvas, (normalized_x, normalized_y), 4, (0, 255, 0), -1)  # Actual Position (Green)
        cv2.circle(top_view_canvas, (normalized_pred_x, normalized_pred_y), 4, (0, 255, 255), -1)  # Kalman Prediction (Yellow)
        if len(object_tracks[obj_id]) > 10:
            cv2.circle(top_view_canvas, (int(rnn_pred_x), int(rnn_pred_y)), 4, (255, 0, 255), -1)  # RNN Prediction (Magenta)

        # Connect all previous points (object path) in the frame to visualize its trajectory
        if len(object_tracks[obj_id]) > 1:
            for i in range(1, len(object_tracks[obj_id])):
                prev_point = object_tracks[obj_id][i - 1]
                curr_point = object_tracks[obj_id][i]
                cv2.line(frame, prev_point, curr_point, (0, 0, 255), 2)  # Red path line

        current_ids.add(obj_id)

    # Show count overlay
    total_unique = len(object_tracks)
    current_count = len(current_ids)
    cv2.putText(frame, f"Current: {current_count} | Total Detected: {total_unique}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Show the main video with the updated visualization
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save metrics to CSV
print("\nSaving metrics to CSV...")
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "Frames Seen", "Time in Video (s)", "Speed (px/s)"])
    for obj_id in object_tracks:
        start = object_entry_frame.get(obj_id, 0)
        end = object_last_seen_frame.get(obj_id, start)
        frames_seen = end - start + 1
        time_seconds = round(frames_seen / fps, 2)
        if time_seconds >= 1.0:  # Filter noise IDs
            # Calculate speed
            if len(object_tracks[obj_id]) > 1:
                prev_x, prev_y = object_tracks[obj_id][-2]
                cx, cy = object_tracks[obj_id][-1]
                speed = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2) * fps
            else:
                speed = 0

            writer.writerow([obj_id, frames_seen, time_seconds, round(speed, 2)])

print("CSV saved to", output_csv_path)

