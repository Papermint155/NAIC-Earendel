import os
import shutil
import random
import subprocess
from pathlib import Path

# Define paths
MAIN_DIR = r"C:\Coding\Earendel"
DATASET_SRC = r"C:\Users\menth\Downloads\dessert_dataset"
DATASET_DEST = os.path.join(MAIN_DIR, "dessert_dataset_yolo")
CLASSES = [
    "kek_lapis", "Kuih_Bahulu", "kuih_kaswi_pandan", "Kuih_Ketayap", 
    "Kuih_Lapis", "Kuih_Seri_Muka", "Kuih_Talam", "Kuih_Ubi_Kayu", "Onde_Onde"
]

# Create main directory if it doesn't exist
os.makedirs(MAIN_DIR, exist_ok=True)

# Install ultralytics YOLOv8 package
print("Installing Ultralytics YOLOv8...")
subprocess.run(["pip", "install", "ultralytics"], check=True)

# Create dataset directories
os.makedirs(DATASET_DEST, exist_ok=True)
for split in ["train", "val", "test"]:
    for subdir in ["images", "labels"]:
        os.makedirs(os.path.join(DATASET_DEST, split, subdir), exist_ok=True)

# Create class mapping file
with open(os.path.join(DATASET_DEST, "classes.txt"), "w") as f:
    for cls in CLASSES:
        f.write(f"{cls}\n")

# Prepare dataset with train/val/test split (70/20/10)
def prepare_dataset():
    print("Preparing YOLOv8 dataset...")
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
    
    # Create dataset.yaml
    with open(os.path.join(DATASET_DEST, "dataset.yaml"), "w") as f:
        f.write(f"path: {DATASET_DEST}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"test: test/images\n")
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
        
        # Copy files to respective directories with unique names
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            # Counter for unique naming
            counter = 1
            for img_path in files:
                # Create unique filename using class name and counter
                unique_name = f"{cls}_{counter:03d}{img_path.suffix}"
                dest_img_path = os.path.join(DATASET_DEST, split_name, "images", unique_name)
                shutil.copy(img_path, dest_img_path)
                
                # Create corresponding label file
                label_name = f"{Path(dest_img_path).stem}.txt"
                label_path = os.path.join(DATASET_DEST, split_name, "labels", label_name)
                
                # Create a default label file with a box covering 80% of the image
                with open(label_path, "w") as f:
                    f.write(f"{class_to_idx[cls]} 0.5 0.5 0.8 0.8\n")  # class_id center_x center_y width height
                
                counter += 1
    
    print(f"Dataset prepared at {DATASET_DEST}")
    print("Note: You'll need to refine the labels using labelImg tool with YOLO format")
# Run preparation
prepare_dataset()

print("\nSetup complete!")
print(f"Dataset directory: {DATASET_DEST}")
print("\nNext steps:")
print("1. Use labelImg to refine the annotations in the dataset_yolo directory")
print("   - Make sure to select 'YOLO' format in labelImg")
print("   - Open the 'images' directory in each split (train/val/test)")
print("2. Run the training script for YOLOv8")
