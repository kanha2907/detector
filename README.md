# detector
Real-time object detection using a custom-trained YOLOv3 model based on the COCO dataset, integrated with OpenCV for video and image inference.
# 🚀 YOLOv3 Object Detection – COCO Trained AI Model

This project implements real-time object detection using a YOLOv3 model trained on the COCO dataset. It utilizes OpenCV's DNN module for fast inference and supports image, video, and webcam detection.

---

## 📦 Contents

```bash
ai_model/
├── yolov3.cfg         # YOLOv3 configuration file
├── yolov3.weights     # Trained weights (COCO)
├── coco.names         # COCO class labels (80 classes)
├── detect.py          # Python script for running inference

##🧠 Model Info
Model Type: YOLOv3

Dataset: COCO (Common Objects in Context)

Classes: 80 (person, bicycle, car, dog, etc.)

Framework: OpenCV with DNN backend

Input Size: 416×416

##🔧 Setup
Install dependencies:

bash
Copy
Edit
pip install opencv-python numpy
Download YOLOv3 weights if not present:
https://pjreddie.com/media/files/yolov3.weights

Run the detection script:

bash
Copy
Edit
python detect.py --image input.jpg
Or for webcam:

bash
Copy
Edit
python detect.py --webcam


