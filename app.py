# File: app.py
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
import base64
import cv2
import numpy as np
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Load trained YOLO model
model = YOLO('runs/detect/weed-detection8/weights/best.pt')

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['media']
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Inference
    results = model(filepath, conf=0.25)
    result_image = results[0].plot()

    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    cv2.imwrite(result_path, result_image)

    return render_template('index.html', uploaded_media=filename, result_media=filename)

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    image_data = request.form['image_data']
    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    filename = str(uuid.uuid4()) + ".jpg"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)

    cv2.imwrite(upload_path, image)

    # Inference
    results = model(upload_path, conf=0.25)
    result_img = results[0].plot()
    cv2.imwrite(result_path, result_img)

    return render_template('index.html', uploaded_media=filename, result_media=filename)

if __name__ == '__main__':
    app.run(debug=True)
