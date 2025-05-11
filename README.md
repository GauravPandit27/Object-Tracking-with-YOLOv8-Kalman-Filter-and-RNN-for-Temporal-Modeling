
---

# Object Tracking with YOLOv8, Kalman Filter, and RNN for Temporal Modeling

This project provides an end-to-end solution for tracking objects in a video using a combination of the YOLOv8 model for object detection, Kalman Filters for state prediction, and an RNN for temporal prediction of object trajectories. It provides real-time object tracking, visualizing predicted positions using Kalman and RNN models, and logs key metrics like object speed and total frames detected in a CSV file.

## Table of Contents

* [Features](#features)
* [Requirements](#requirements)
* [Setup Instructions](#setup-instructions)
* [Usage](#usage)
* [Code Explanation](#code-explanation)

  * [YOLOv8 Object Detection](#yolov8-object-detection)
  * [Kalman Filter for Tracking](#kalman-filter-for-tracking)
  * [RNN Temporal Prediction](#rnn-temporal-prediction)
  * [Top-Down View Canvas](#top-down-view-canvas)
* [CSV Metrics](#csv-metrics)
* [Acknowledgments](#acknowledgments)

---

## Features

* **Real-time Object Detection**: Using YOLOv8 model for detecting objects in the video.
* **Object Tracking**: Combines Kalman Filters and RNN predictions for accurate tracking and future position predictions.
* **Speed Calculation**: Measures object speed in pixels per second.
* **Top-down View Visualization**: Displays a 2D top-down canvas showing object trajectories, predictions, and movement.
* **CSV Logging**: Saves object detection and movement metrics to a CSV file, including object ID, frames seen, time in video, and speed.

---

## Requirements

The following Python packages are required to run the project:

1. **OpenCV**: For video processing and drawing on frames.
2. **NumPy**: For numerical operations.
3. **Ultralytics YOLO**: For YOLOv8-based object detection.
4. **Torch**: For the RNN model (LSTM).
5. **CSV**: For saving tracking metrics.
6. **Datetime**: For logging the time of object detection.

You can install these dependencies via pip:

```bash
pip install opencv-python numpy ultralytics torch
```

---

## Setup Instructions

1. **Clone the Repository**: Download or clone this repository to your local machine.

   ```bash
   git clone https://github.com/your-username/object-tracking-yolo-kalman-rnn.git
   cd object-tracking-yolo-kalman-rnn
   ```

2. **Download YOLOv8 Model**: The project uses the YOLOv8 pre-trained model. Download the appropriate model (`yolov5m.pt` or `yolov5s.pt`) from [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8) and place it in the project directory.

3. **Prepare Your Input Video**: Ensure you have a video file (`.mp4`) that you want to process. Place it in the project folder or specify the correct path in the `input_video_path` variable.

4. **Install Dependencies**: Install the necessary Python libraries by running:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Edit the Input Video Path**: In the `main.py` script, set the `input_video_path` variable to the path of your input video file:

   ```python
   input_video_path = "/path/to/your/video.mp4"
   ```

2. **Run the Script**: Once everything is set up, run the Python script:

   ```bash
   python main.py
   ```

3. **Viewing the Results**: The script will display a window with the real-time video feed showing the bounding boxes, tracking ID, speed, and Kalman and RNN predictions.

   * **Green Circles**: Represent the actual positions of tracked objects.
   * **Yellow Circles**: Represent the predicted positions using Kalman Filter.
   * **Magenta Circles**: Represent the predicted positions using the RNN model.

4. **Exit the Video Feed**: To stop the video feed, press `Q` while the video is running.

5. **CSV Logging**: The script will save a CSV file named `object_tracking_metrics.csv` with the following columns:

   * **Object ID**: The unique identifier for the object.
   * **Frames Seen**: The number of frames the object has been detected in.
   * **Time in Video (s)**: The time the object was tracked in the video.
   * **Speed (px/s)**: The speed of the object in pixels per second.

---

## Code Explanation

### YOLOv8 Object Detection

* The `YOLO` model is loaded using the `Ultralytics YOLO` library.
* It processes each frame of the input video and detects objects with a confidence threshold of 0.4 (`conf=0.4`) and an IoU threshold of 0.5 (`iou=0.5`).

```python
model = YOLO("yolov5m.pt")
tracker = model.track(source=input_video_path, stream=True, persist=True, conf=0.4, iou=0.5)
```

Each detected object is assigned a unique ID and class. The script tracks only 'person' class objects.

### Kalman Filter for Tracking

* **Kalman Filter** is used to predict the state (position and velocity) of objects over time.
* The `cv2.KalmanFilter` is initialized for each object and used to predict its next position based on the previous position.
* The prediction helps stabilize tracking, even in the presence of noisy or missing data.

```python
kf = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
```

* **Predicted position**: The Kalman Filter is used to predict the object's next position and draw a yellow dot to represent the predicted trajectory.

### RNN Temporal Prediction

* **RNN (LSTM)** is used to predict the future position of an object based on the last 10 detected positions.
* This helps estimate where the object might be in the next few frames.

```python
rnn_model = RNNModel()
rnn_model.eval()
```

* The RNN model predicts the future position of the object (in magenta) when enough historical data (10 positions) is available.

### Top-Down View Canvas

* A top-down canvas is created to visualize object trajectories in 2D.
* The actual position, Kalman prediction, and RNN prediction are projected onto this canvas to show the object's movement path in a simplified form.

```python
top_view_canvas = np.zeros((height, width, 3), dtype=np.uint8)
```

### CSV Metrics

* Object metrics, including object ID, frames detected, time tracked, and speed, are saved in a CSV file for further analysis.
* The CSV file allows you to assess the tracking performance and analyze the movement of detected objects.

```python
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "Frames Seen", "Time in Video (s)", "Speed (px/s)"])
```

---

## CSV Metrics

The CSV file `object_tracking_metrics.csv` includes the following columns:

* **Object ID**: The unique identifier of the object.
* **Frames Seen**: The number of frames in which the object has been detected.
* **Time in Video (s)**: The total time (in seconds) the object was tracked.
* **Speed (px/s)**: The calculated speed of the object in pixels per second based on its movement.

This CSV file is useful for performance analysis and can be used for further statistical modeling or optimization.

---

## Acknowledgments

* **YOLOv8**: Ultralytics for the YOLOv8 pre-trained model and implementation.
* **OpenCV**: For providing the tools to process and display video frames.
* **PyTorch**: For implementing the RNN model used for temporal predictions.

---

Feel free to modify or extend this project according to your needs, and share any improvements with the community!
