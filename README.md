# 🌿 Smart Weed Detection Web App

A responsive real-time weed detection web application built using Flask and YOLOv10 Nano. The system is designed to assist farmers and agricultural researchers in identifying weeds in crop fields through image, video, or live camera input.

---

## 📌 Project Overview
This project integrates a deep learning object detection model with a web-based interface to enable easy and accessible weed detection in real agricultural scenarios. The system uses a lightweight YOLOv10 Nano model trained on a public dataset and wraps it with a Flask backend and a mobile-friendly frontend.

---

## 🧠 Model Training Workflow

### 📂 Dataset
- **Source**: Kaggle
- **Dataset**: [Crop and Weed Detection Data with Bounding Boxes](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes)
- **Content**: Labeled bounding boxes of crops and weeds from drone-captured field images.
- **Format**: YOLO-compatible `.yaml` and image/label folder structure.

### ⚙️ Training Configuration
- **Model**: YOLOv10 Nano (`yolov10n.pt` from Ultralytics)
- **Input Image Size**: 416x416
- **Epochs**: 50
- **Batch Size**: 8 (adjustable based on system capacity)
- **Patience**: 10 (early stopping)
- **Confidence Threshold**: 0.25
- **Hardware**: CUDA-enabled GPU (or CPU fallback)

### 🧪 Validation Metrics (Example)
| Metric       | Value    |
|--------------|----------|
| Precision    | 93.1%    |
| Recall       | 95.0%    |
| mAP@0.5      | 94.2%    |
| FPS (CPU)    | ~12-15   |

> Model output: Bounding boxes for detected weeds with class confidence.

---

## 🔧 Technical Architecture

### 🧱 Backend (Flask)
- **Framework**: Flask (Python)
- **Model Inference**: Uses `ultralytics` YOLO Python API
- **Routes**:
  - `/` — UI home
  - `/predict` — Image/Video upload handling
  - `/predict_camera` — Base64 camera image decoding + detection
- **Output**: Saves result with detection overlays (OpenCV visualization)

### 🎨 Frontend (HTML/CSS + JS)
- **Templating**: Jinja2 (Flask default)
- **Camera Access**: JavaScript `getUserMedia()` API
- **Responsiveness**: CSS media queries and mobile-first design
- **Layout**:
  - Upload section
  - Camera section
  - Result section (side-by-side on desktop, stacked on mobile)

### 📁 Project Structure
```
├── app.py                # Flask backend logic
├── runs/
│   └── detect/
|         └── weed_detection8/
|                 └── best.pt/              # Trained YOLOv10 model
├── static/
│   ├── style.css         # Responsive UI styling
│   ├── uploads/          # Original media
│   ├── results/          # Detection results
│   └── images/logo.png   # App logo
├── templates/
│   └── index.html        # Main UI template
├── requirements.txt
```

---

## 🖼️ Visualization Results on test data

### 🧠 Detection Results
![confusion_matrix_normalized](https://github.com/user-attachments/assets/b5d4f88d-52fc-44be-bf4a-1729ffe4c4ac)
![PR_curve](https://github.com/user-attachments/assets/30d31afc-f96e-4386-a342-a07ad5d8a1e6)
![P_curve](https://github.com/user-attachments/assets/5fa2e488-d0b3-4539-83bd-cf9cf2981f82)
![F1_curve](https://github.com/user-attachments/assets/a1825e56-1197-4442-a329-ab57e7ebffae)
![R_curve](https://github.com/user-attachments/assets/3a7dba54-354e-4f18-a7c5-529f91342c72)
![val_batch0_labels](https://github.com/user-attachments/assets/440a15ec-56d1-4962-adf9-eb90b41c7aa0)
![val_batch2_pred](https://github.com/user-attachments/assets/5925a5ea-412f-4aa0-9b39-b5722b719662)

---

## 🚀 Running the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Trained Model
```bash
/model/best.pt
```

### 3. Run App
```bash
python app.py
```
Go to `http://127.0.0.1:5000/` in your browser.

---

## 📊 Results Summary
- Inference works on **CPU and GPU**.
- Optimized for mobile access.
- Works with **.jpg, .png, .mp4** and camera feed.
- All detections are visualized using `OpenCV.plot()` and saved.

---
## Additional Information
Below is the weed.yaml file structure that you can add in your processed_dataset folder after preprocessing the data:
```bash
path: 
C:/Users/91843/Desktop/Projects/Weed_Detection_Project/Weed_Detection_Project/Weed_Detection_Project/processed_dataset  #Path to your processed_dataset file
train: images/train
val: images/val

names:
  0: crop
  1: weed
```
---
## 🤝 Credits

- **Ultralytics YOLOv10**: For providing a lightweight yet powerful object detector.
- **Kaggle Dataset by @ravirajsinh45**: [Weed and Crop Detection](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes)



