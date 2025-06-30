import os
import subprocess
import sys
import time

def create_directory_structure():
    """Create the necessary directory structure"""
    print("Creating directory structure...")
    
    MAIN_DIR = r"C:\Coding\Earendel"
    os.makedirs(MAIN_DIR, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(MAIN_DIR, "dessert_dataset_yolo"), exist_ok=True)
    os.makedirs(os.path.join(MAIN_DIR, "dessert_dataset_cnn"), exist_ok=True)
    os.makedirs(os.path.join(MAIN_DIR, "yolo_model"), exist_ok=True)
    os.makedirs(os.path.join(MAIN_DIR, "cnn_model"), exist_ok=True)
    
    print(f"Directory structure created at {MAIN_DIR}")
    return MAIN_DIR

def check_dependencies():
    """Check if necessary dependencies are installed"""
    print("Checking dependencies...")
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print("CUDA is available! GPU acceleration will be used.")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA is not available. Models will run on CPU, which may be slow.")
    except ImportError:
        print("PyTorch is not installed. Installing required packages...")
        subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio"], check=True)
    
    # Install other dependencies
    dependencies = [
        "numpy", "matplotlib", "pandas", "seaborn", 
        "scikit-learn", "pillow", "tqdm"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.run(["pip", "install", dep], check=True)
    
    print("All dependencies are installed.")

def run_all():
    """Run the entire pipeline: setup, training, and evaluation"""
    main_dir = create_directory_structure()
    check_dependencies()
    
    # Run YOLOv5 setup and training
    print("\n===== STEP 1: YOLOv5 Setup and Training =====")
    print("This step will:")
    print("1. Clone the YOLOv5 repository")
    print("2. Setup the dataset structure for YOLOv5")
    print("3. Generate placeholder annotations (to be refined with labelImg)")
    
    proceed = input("Do you want to proceed with YOLOv5 setup? (y/n): ")
    if proceed.lower() == 'y':
        # Execute the YOLOv5 setup script
        print("Running YOLOv5 setup...")
        yolo_setup_path = os.path.join(main_dir, "yolov5_setup.py")
        
        # Copy the script content to file
        with open(yolo_setup_path, 'w') as f:
            # Here you would paste the content of yolov5-setup script
            f.write("# YOLOv5 setup script content goes here\n")
            f.write("# This would be populated by the actual content from the yolov5-setup artifact\n")
        
        subprocess.run([sys.executable, yolo_setup_path], check=True)
        
        print("\nYOLOv5 setup complete!")
        print("Before proceeding to training, you should use labelImg to refine the annotations.")
        print("Instructions:")
        print("1. Install labelImg: pip install labelImg")
        print("2. Run labelImg: labelImg")
        print("3. Open the image directory and create accurate bounding boxes for each dessert")
        
        input("Press Enter when you have completed the annotation process...")
        
        # Run YOLOv5 training
        print("\nRunning YOLOv5 training...")
        yolo_train_path = os.path.join(main_dir, "yolov5_train.py")
        
        # Copy the script content to file
        with open(yolo_train_path, 'w') as f:
            # Here you would paste the content of yolov5-train script
            f.write("# YOLOv5 training script content goes here\n")
            f.write("# This would be populated by the actual content from the yolov5-train artifact\n")
        
        subprocess.run([sys.executable, yolo_train_path], check=True)
    
    # Run CNN training
    print("\n===== STEP 2: CNN Model Training =====")
    proceed = input("Do you want to proceed with CNN training? (y/n): ")
    if proceed.lower() == 'y':
        # Execute the CNN training script
        print("Running CNN training...")
        cnn_train_path = os.path.join(main_dir, "cnn_train.py")
        
        # Copy the script content to file
        with open(cnn_train_path, 'w') as f:
            # Here you would paste the content of cnn-train script
            f.write("# CNN training script content goes here\n")
            f.write("# This would be populated by the actual content from the cnn-train artifact\n")
        
        subprocess.run([sys.executable, cnn_train_path], check=True)
    
    # Run prediction script
    print("\n===== STEP 3: Setup Prediction Script =====")
    prediction_script_path = os.path.join(main_dir, "predict_dessert.py")
    
    # Copy the script content to file
    with open(prediction_script_path, 'w') as f:
        # Here you would paste the content of prediction-script
        f.write("# Prediction script content goes here\n")
        f.write("# This would be populated by the actual content from the prediction-script artifact\n")
    
    print(f"Prediction script has been created at: {prediction_script_path}")
    print("You can now use this script to classify new dessert images.")
    print(f"Run it with: python {prediction_script_path}")

if __name__ == "__main__":
    print("==================================================")
    print("       Dessert Recognition System Setup           ")
    print("==================================================")
    print("This script will guide you through the process of:")
    print("1. Setting up YOLOv5 for object detection")
    print("2. Training a CNN classifier with TensorFlow/PyTorch")
    print("3. Creating a prediction script that combines both models")
    print("==================================================")
    
    proceed = input("Do you want to proceed? (y/n): ")
    if proceed.lower() == 'y':
        run_all()
        print("\nSetup complete! Your dessert recognition system is ready.")
    else:
        print("Setup cancelled.")
