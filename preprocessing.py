import os
import random
import shutil
from pathlib import Path

# Paths
base_dir = Path('dataset/agri_data/data/')
output_dir = Path('processed_dataset')
images_dir = output_dir / 'images'
labels_dir = output_dir / 'labels'

# Create directories
for split in ['train', 'val']:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

# Gather image files
image_files = list(base_dir.glob("*.jpeg"))
random.shuffle(image_files)

# 80/20 split
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def move_files(files, split):
    for img_path in files:
        label_path = img_path.with_suffix('.txt')
        shutil.copy(img_path, images_dir / split / img_path.name)
        shutil.copy(label_path, labels_dir / split / label_path.name)

move_files(train_files, 'train')
move_files(val_files, 'val')
