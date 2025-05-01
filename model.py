from ultralytics import YOLO
import torch

# Print if GPU is available
print("GPU Available:", torch.cuda.is_available())

# Load a lightweight YOLOv10 Nano model
model = YOLO('yolov10n.pt')

# Train the model with optimized settings
model.train(
    data='processed_dataset/weed.yaml',
    epochs=50,
    imgsz=416,           # Reduced from 640 to speed up training
    batch=8,             # Try 4 if this still lags
    name='weed-detection',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    workers=2,           # Lower workers to reduce CPU load
    patience=10,         # Early stopping
    conf=0.25
)

# Evaluate the model
metrics = model.val()
print(metrics)

# Inference + Visualization
import cv2
import matplotlib.pyplot as plt

# Load trained model
model = YOLO('runs/detect/weed-detection/weights/best.pt')

# Inference
results = model('path_to_image.jpg', conf=0.25)

# Visualize
for result in results:
    im_array = result.plot()
    plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show() 