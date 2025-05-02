import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import base64
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = 'weed_detection_secret_key'

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = None

def load_model():
    global model
    try:
        model = YOLO('best.pt')
        model.to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image with YOLO model
        try:
            # Run inference
            results = model(filepath, conf=0.15)
            
            # Get the annotated image
            for result in results:
                im_array = result.plot()
                
                # Save the result image
                result_filename = f"result_{filename}"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                cv2.imwrite(result_path, im_array)
                
                # Convert detections to a list for display
                detections = []
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = result.names[cls]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': round(conf * 100, 2),
                        'box': [round(x, 2) for x in [x1, y1, x2, y2]]
                    })
                
                # Return the path to the result image and detection details
                return render_template('result.html', 
                                      original_image=f"/{UPLOAD_FOLDER}/{filename}",
                                      result_image=f"/{RESULT_FOLDER}/{result_filename}",
                                      detections=detections)
        
        except Exception as e:
            flash(f'Error during detection: {str(e)}')
            return redirect('/')
    
    flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
    return redirect('/')

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for programmatic access"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image with YOLO model
        try:
            # Run inference
            results = model(filepath, conf=0.25)
            
            # Get the annotated image
            for result in results:
                im_array = result.plot()
                
                # Save the result image
                result_filename = f"result_{filename}"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                cv2.imwrite(result_path, im_array)
                
                # Convert to base64 for API response
                _, buffer = cv2.imencode('.jpg', im_array)
                img_str = base64.b64encode(buffer).decode('utf-8')
                
                # Extract detections
                detections = []
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = result.names[cls]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': round(conf * 100, 2),
                        'box': [round(x, 2) for x in [x1, y1, x2, y2]]
                    })
                
                return jsonify({
                    'success': True,
                    'image_base64': img_str,
                    'detections': detections,
                    'original_url': f"/{UPLOAD_FOLDER}/{filename}",
                    'result_url': f"/{RESULT_FOLDER}/{result_filename}"
                })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
