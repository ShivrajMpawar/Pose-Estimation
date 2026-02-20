# Pose Estimation Video Pipeline (MediaPipe + MoveNet ONNX)

This repository contains two pose estimation pipelines for video processing:

* **MediaPipe Pose** implementation (real-time pose tracking)
* **MoveNet ONNX** implementation using ONNX Runtime

Both scripts run pose estimation on a video file, draw keypoints and skeleton, and display FPS and inference time.

---

## Features

* Real-time pose detection on video files
* 17 keypoint skeleton visualization
* FPS calculation
* Inference time measurement
* Modular class-based architecture
* Works with MediaPipe and MoveNet ONNX

---

## Project Structure

```
pose-project/
│── mediapipe_pose_video.py
│── movenet_onnx_video.py
│── Movenet.onnx        # optional (model file)
│── requirements.txt
│── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/pose-project.git
cd pose-project
```

Install dependencies:

```
pip install -r requirements.txt
```

Or install manually:

```
pip install opencv-python mediapipe numpy onnxruntime
```

---

## Usage

### MediaPipe Pose

Update the video path inside the script and run:

```
python mediapipe_pose_video.py
```

---

### MoveNet ONNX Pose

Place the `Movenet.onnx` model in the project folder and run:

```
python movenet_onnx_video.py
```

---

## Output

The scripts display:

* Pose keypoints
* Skeleton connections
* FPS (frames per second)
* Model inference time
* Optional saved sample frame (MediaPipe version)

---

## Notes

* ONNX model files may exceed GitHub size limits.
  If the model is large, host it externally (Google Drive / HuggingFace) and provide a download link.

* Ensure video paths are updated before running.

---

## Requirements

```
opencv-python
mediapipe
numpy
onnxruntime
```

---

## Future Improvements

* Exercise repetition counter
* Angle calculation utilities
* Multi-person tracking
* Webcam support
* TensorRT / GPU acceleration
* Pose analytics pipeline

---

## License

This project is for learning and experimentation purposes.
