# 🎥 obj_count – Real-Time Object Detection & Counting Web App

**obj_count** is a Flask-powered web application that uses a **webcam feed** to perform **real-time object detection** using YOLO models. It displays the **name and count of each detected object**, and supports multiple YOLO versions including YOLOv3, YOLOv5, and YOLOv8.

---

## 🚀 Features

- 📷 Real-time webcam-based object detection
- 🔢 Displays object names and their live count
- 🧠 Supports YOLOv3, YOLOv5 (n, s, su), and YOLOv8
- 🌐 Interactive web interface using Flask

---

## 🗂 Folder Structure

 obj_count/
├── obj_count.py # Main Flask app

├── requirements.txt # Python dependencies

│

├── static/ # Static assets (CSS, JS)

├── templates/ # HTML templates (web UI)

├── uploads/ # (optional) Uploads folder

├── yolo_model/ # YOLO scripts/helpers 

│

├── coco.names # Class labels

├── yolov3.weights # YOLOv3 model weights (not in GitHub)

├── yolo5n.pt

├── yolo5s.pt

├── yolo5su.pt

├── yolov8m.onnx

├── yolov8m.py

## 🧩 Tech Stack

- **Backend**: Python, Flask
- **Models**: YOLOv3, YOLOv5, YOLOv8
- **Vision**: OpenCV
- **Speech** (optional): pyttsx3
- **Frontend**: HTML, JS (via templates)

---

## ✅ Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/obj_count.git
   cd obj_count
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   python obj_count.py
   ```

📸 How It Works

1.The app accesses your webcam using OpenCV.

2.YOLO model detects objects in each frame.

3.Detected objects are:

Displayed with bounding boxes

Counted and shown in real-time
