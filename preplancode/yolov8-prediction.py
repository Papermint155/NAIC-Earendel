import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
from pathlib import Path
from ultralytics import YOLO

# Define paths and constants
MAIN_DIR = r"C:\Coding\Earendel"#change this to your main directory
CNN_MODEL_PATH = os.path.join(MAIN_DIR, "cnn_model", "best_cnn_model.pth")
YOLO_MODEL_PATH = os.path.join(MAIN_DIR, "yolo_model", "dessert_detector", "weights", "best.pt")
CLASSES = [
    "kek_lapis", "Kuih_Bahulu", "kuih_kaswi_pandan", "Kuih_Ketayap", 
    "Kuih_Lapis", "Kuih_Seri_Muka", "Kuih_Talam", "Kuih_Ubi_Kayu", "Onde_Onde"
]

# Function to load CNN model
def load_cnn_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model architecture
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    # Load trained weights
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

# Function to load YOLOv8 model
def load_yolo_model():
    model = YOLO(YOLO_MODEL_PATH)
    return model

# Function to predict with CNN model
def predict_with_cnn(model, device, img_path):
    # Data transforms for inference
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    pred_class = CLASSES[pred_idx.item()]
    confidence = conf.item()
    
    return pred_class, confidence, probs.squeeze().cpu().numpy()

# Function to predict with YOLOv8
def predict_with_yolo(model, img_path):
    # Make prediction
    results = model(img_path, conf=0.25)
    
    # Process results
    probs = np.zeros(len(CLASSES))
    pred_class = None
    confidence = 0.0
    
    # Get first result (first image)
    result = results[0]
    
    if len(result.boxes) > 0:
        # Get the box with highest confidence
        boxes = result.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        
        # Find the detection with highest confidence
        max_conf_idx = np.argmax(confidences)
        class_id = class_ids[max_conf_idx]
        confidence = confidences[max_conf_idx]
        
        # Get class name
        class_name = result.names[class_id]
        
        # Try to map to one of our classes
        if class_name in CLASSES:
            pred_class = class_name
            probs[CLASSES.index(pred_class)] = confidence
        else:
            # Handle case where YOLOv8 class name doesn't match CNN classes
            # This could happen if model was trained with different class names
            print(f"Warning: YOLOv8 detected '{class_name}' which is not in our class list")
    
    return pred_class, confidence, probs

# Combined prediction function
def predict_dessert(img_path):
    coeff_cnn = 0.5
    coeff_yolo = 1 - coeff_cnn
    # Load models if not already loaded
    try:
        global cnn_model, cnn_device, yolo_model
    except NameError:
        print("Loading models...")
        cnn_model, cnn_device = load_cnn_model()
        yolo_model = load_yolo_model()
        print("Models loaded successfully!")
    
    # Make predictions with both models
    cnn_class, cnn_conf, cnn_probs = predict_with_cnn(cnn_model, cnn_device, img_path)
    yolo_class, yolo_conf, yolo_probs = predict_with_yolo(yolo_model, img_path)
    
    print(f"CNN prediction: {cnn_class} with confidence {cnn_conf:.4f}")
    print(f"YOLOv8 prediction: {yolo_class if yolo_class else 'No detection'} with confidence {yolo_conf:.4f}")
    
    # Combine predictions with weighted average (CNN: 0.6, YOLOv8: 0.4)
    combined_probs = cnn_probs * coeff_cnn + yolo_probs * coeff_yolo
    pred_idx = np.argmax(combined_probs)
    pred_class = CLASSES[pred_idx]
    pred_conf = combined_probs[pred_idx]
    
    print(f"Combined prediction: {pred_class} with confidence {pred_conf:.4f}")
    
    # Display the image with prediction
    img = Image.open(img_path).convert('RGB')
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class} (Confidence: {pred_conf:.4f})")
    plt.axis('off')
    plt.show()
    
    # Also show YOLOv8 detection visualization
    results = yolo_model(img_path, conf=0.25)
    detection_img = results[0].plot()
    plt.figure(figsize=(8, 8))
    plt.imshow(detection_img)
    plt.title(f"YOLOv8 Detection: {yolo_class if yolo_class else 'No detection'} (Conf: {yolo_conf:.4f})")
    plt.axis('off')
    plt.show()
    
    return pred_class, pred_conf

# Function to run predictions on a directory of images
def run_on_directory(img_dir):
    # Get all image files
    img_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        img_files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    
    print(f"Found {len(img_files)} images to process")
    
    # Process each image
    for img_path in img_files:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        predict_dessert(img_path)
        input("Press Enter to continue to next image...")

# Main function to run interactively
def main():
    # Load models
    global cnn_model, cnn_device, yolo_model
    cnn_model, cnn_device = load_cnn_model()
    yolo_model = load_yolo_model()
    
    while True:
        print("\nDessert Recognition System")
        print("1. Predict a single image")
        print("2. Predict all images in a directory")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            img_path = input("Enter the path to the image: ")
            if os.path.exists(img_path):
                predict_dessert(img_path)
            else:
                print("Error: File not found")
        
        elif choice == '2':
            img_dir = input("Enter the path to the directory containing images: ")
            if os.path.isdir(img_dir):
                run_on_directory(img_dir)
            else:
                print("Error: Directory not found")
        
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
