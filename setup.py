import os
import shutil
import random
from pathlib import Path
import numpy as np

# Define Paths
MAIN_DIR = r"C:\Coding\Earendel"  # Update this to your actual main directory
DATASET_SRC = os.path.join(MAIN_DIR, "dessert_dataset")
DATASET_DEST = os.path.join(MAIN_DIR, "dessert_dataset_yolo")
DATASET_CNN = os.path.join(MAIN_DIR, "dessert_dataset_cnn")

CLASSES = [
    "kek_lapis", "Kuih_Bahulu", "kuih_kaswi_pandan", "Kuih_Ketayap",
    "Kuih_Lapis", "Kuih_Seri_Muka", "Kuih_Talam", "Kuih_Ubi_Kayu", "Onde_Onde"
]
CATEGORIES = ["train", "val", "test"]

# Create directories if they don't exist
os.makedirs(DATASET_DEST, exist_ok=True)
os.makedirs(DATASET_CNN, exist_ok=True)

# Create Train, Val, Test directories
for category in CATEGORIES:
    os.makedirs(os.path.join(DATASET_DEST, category), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DEST, category, "images"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DEST, category, "labels"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_CNN, category), exist_ok=True)

# Create class directories for CNN dataset
for split in CATEGORIES:
    for cls in CLASSES:
        os.makedirs(os.path.join(DATASET_CNN, split, cls), exist_ok=True)

def prepare_yolo_dataset():
    print("Preparing YOLO Dataset")
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

    # Create dataset.yaml for detection
    with open(os.path.join(DATASET_DEST, "dataset.yaml"), "w") as f:
        f.write(f"path: {DATASET_DEST}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

    # Process each class
    for cls in CLASSES:
        src_dir = os.path.join(DATASET_SRC, cls)
        if not os.path.exists(src_dir):
            print(f"Warning: {src_dir} does not exist, skipping...")
            continue

        # Get all image files
        img_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            img_files.extend(list(Path(src_dir).glob(f"*{ext}")))

        if not img_files:
            print(f"Warning: No images found in {src_dir}")
            continue

        # Shuffle and split
        random.shuffle(img_files)
        
        train_len = int(len(img_files) * 0.7)
        val_len = int(len(img_files) * 0.2)
        
        train_files = img_files[:train_len]
        val_files = img_files[train_len:train_len + val_len]
        test_files = img_files[train_len + val_len:]

        # Move images to train, val, and test directories
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            counter = 1
            for img_path in files:
                unique_name = f"{cls}_{counter:03d}{img_path.suffix}"
                dest_img_path = os.path.join(DATASET_DEST, split_name, "images", unique_name)
                shutil.copy(img_path, dest_img_path)
                counter += 1

    print("Image processing completed")
    print("Please check dataset.yaml")
    print("Label the images using LabelImg")

def prepare_cnn_dataset():
    print("Preparing CNN Dataset")
    
    for cls in CLASSES:
        src_dir = os.path.join(DATASET_SRC, cls)
        if not os.path.exists(src_dir):
            print(f"Warning: {src_dir} does not exist, skipping...")
            continue

        # Get all image files
        img_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            img_files.extend(list(Path(src_dir).glob(f"*{ext}")))

        if not img_files:
            print(f"Warning: No images found in {src_dir}")
            continue

        # Shuffle and split
        random.shuffle(img_files)
        
        train_len = int(len(img_files) * 0.7)
        val_len = int(len(img_files) * 0.2)
        
        train_files = img_files[:train_len]
        val_files = img_files[train_len:train_len + val_len]
        test_files = img_files[train_len + val_len:]

        # Copy files
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dest_dir = os.path.join(DATASET_CNN, split_name, cls)
            for img_path in files:
                shutil.copy(img_path, os.path.join(dest_dir, img_path.name))

    # Create dataset.yaml for classification
    with open(os.path.join(DATASET_CNN, "dataset.yaml"), "w") as f:
        f.write(f"path: {DATASET_CNN}\n")
        f.write("train: train\n")
        f.write("val: val\n")
        f.write("test: test\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

    print(f"CNN dataset prepared at {DATASET_CNN}")
    print("Classification dataset.yaml created")

if __name__ == "__main__":
    prepare_yolo_dataset()
    prepare_cnn_dataset()
    print("All processes completed")