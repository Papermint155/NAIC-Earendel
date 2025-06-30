import os
import sys
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
MAIN_DIR = os.getcwd()
CNN_MODEL_PATH = r"C:\Users\menth\Downloads\best_cnn_model.pth"#os.path.join(MAIN_DIR, "cnn_model", "best_cnn_model.pth")
YOLO_MODEL_PATH = r"C:\Users\menth\Downloads\best.pt"#os.path.join(MAIN_DIR, "yolo_model", "dessert_detector", "weights", "best.pt")
YOLO_CLS_PATH =os.path.join(MAIN_DIR, "yolo_model_cls", "dessert_detector", "weights", "best.pt")
CLASSES = [
    "kek_lapis", "Kuih_Bahulu", "kuih_kaswi_pandan", "Kuih_Ketayap",
    "Kuih_Lapis", "Kuih_Seri_Muka", "Kuih_Talam", "Kuih_Ubi_Kayu", "Onde_Onde"
]
names = ['kek_lapis', 'kuih_kaswi_pandan', 'Kuih_Ketayap','Kuih_Ubi_Kayu', 'Kuih_Lapis', 'Kuih_Seri_Muka', 'Kuih_Talam', 'Onde_Onde']
clsnames =  ['Kuih_Ketayap','Kuih_Lapis','Kuih_Seri_Muka','Kuih_Talam','Kuih_Ubi_Kayu','Onde_Onde','kek_lapis','kuih_kaswi_pandan']
def load_cnn_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"Error: CNN model not found at {CNN_MODEL_PATH}")
        sys.exit(1)
    
    model = models.resnet50(weights = None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def load_yolo_model():
    if not os.path.exists(YOLO_MODEL_PATH):

        print(f"Error: YOLO model not found at {YOLO_MODEL_PATH}")
        sys.exit(1)
    return YOLO(YOLO_MODEL_PATH)

def load_yolo_cls():
    if not os.path.exists(YOLO_CLS_PATH):
        print(f"Error: YOLO classification model not found at {YOLO_CLS_PATH}")
        sys.exit(1)
    return YOLO(YOLO_CLS_PATH)

def predict_with_cnn(model, device, img_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return None, 0.0, np.zeros(len(CLASSES))
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    pred_class = CLASSES[pred_idx.item()]
    confidence = conf.item()
    
    return pred_class, confidence, probs.squeeze().cpu().numpy()

def predict_with_yolo(model, img_path):
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return None, 0.0, np.zeros(len(names))
    
    results = model(img_path, conf=0.25)
    
    probs = np.zeros(len(names))
    pred_class = None
    confidence = 0.0
    
    result = results[0]
    
    if len(result.boxes) > 0:
        boxes = result.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        
        max_conf_idx = np.argmax(confidences)
        class_id = class_ids[max_conf_idx]
        confidence = confidences[max_conf_idx]
        
        class_name = result.names[class_id]
        
        if class_name in names:
            pred_class = class_name
            probs[names.index(pred_class)] = confidence
        else:
            print(f"Warning: YOLO detected '{class_name}' which is not in our class list")
    
    return pred_class, confidence, probs

def predict_with_yolo_cls(model, img_path):
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return None, 0.0, np.zeros(len(clsnames))
    
    results = model(img_path, conf=0.25)
    
    probs = np.zeros(len(clsnames))
    pred_class = None
    confidence = 0.0
    
    result = results[0]
    
    if hasattr(result, 'probs') and result.probs is not None:
        class_probs = result.probs.data.cpu().numpy()
        max_conf_idx = np.argmax(class_probs)
        confidence = class_probs[max_conf_idx]
        pred_class = clsnames[max_conf_idx]
        probs = class_probs

    
    return pred_class, confidence, probs

def predict_dessert(img_path):
    coeff_cls = 0.4
    coeff_cnn = 0.3
    coeff_yolo = 0.3
    
    try:
        global cnn_model, cnn_device, yolo_model, yolo_cls_model
    except NameError:
        print("Loading models...")
        cnn_model, cnn_device = load_cnn_model()
        yolo_model = load_yolo_model()
        yolo_cls_model = load_yolo_cls()
        print("Models loaded successfully!")
    
    cnn_class, cnn_conf, cnn_probs = predict_with_cnn(cnn_model, cnn_device, img_path)
    yolo_class, yolo_conf, yolo_probs = predict_with_yolo(yolo_model, img_path)
    yolo_cls_class, yolo_cls_conf, yolo_cls_probs = predict_with_yolo_cls(yolo_cls_model, img_path)
    
    print(f"CNN prediction: {cnn_class if cnn_class else 'No detection'} with confidence {cnn_conf:.4f}")
    print(f"YOLOv11 detection: {yolo_class if yolo_class else 'No detection'} with confidence {yolo_conf:.4f}")
    print(f"YOLOv11 classification: {yolo_cls_class if yolo_cls_class else 'No detection'} with confidence {yolo_cls_conf:.4f}")
    
    predictions = [cnn_class, yolo_class, yolo_cls_class]
    confidences = [cnn_conf, yolo_conf, yolo_cls_conf]
    valid_predictions = [p for p in predictions if p is not None]
    
    if not valid_predictions:
        print("No valid predictions made")
        pred_class = "No detection"
        pred_conf = 0.0
    else:
        vote_counts = {}
        for pred in valid_predictions:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
        
        max_votes = max(vote_counts.values())
        majority_classes = [cls for cls, count in vote_counts.items() if count == max_votes]
        
        if len(majority_classes) == 1:
            pred_class = majority_classes[0]
            pred_conf = np.mean([conf for pred, conf in zip(predictions, confidences) if pred == pred_class])
        else:
            valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
            max_conf_idx = valid_indices[np.argmax([confidences[i] for i in valid_indices])]
            pred_class = predictions[max_conf_idx]
            pred_conf = confidences[max_conf_idx]
    
    print(f"Combined prediction (majority vote): {pred_class} with confidence {pred_conf:.4f}")
    
    img = Image.open(img_path).convert('RGB')
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class} (Confidence: {pred_conf:.4f})")
    plt.axis('off')
    plt.show()
    
    results = yolo_model(img_path, conf=0.25)
    detection_img = results[0].plot()
    plt.figure(figsize=(8, 8))
    plt.imshow(detection_img)
    plt.title(f"YOLOv11 Detection: {yolo_class if yolo_class else 'No detection'} (Conf: {yolo_conf:.4f})")
    plt.axis('off')
    plt.show()
    
    results = yolo_cls_model(img_path, conf=0.25)
    detection_img = results[0].plot()
    plt.figure(figsize=(8, 8))
    plt.imshow(detection_img)
    plt.title(f"YOLOv11 Classification: {yolo_cls_class if yolo_cls_class else 'No detection'} (Conf: {yolo_cls_conf:.4f})")
    plt.axis('off')
    plt.show()
    
    return pred_class, pred_conf

def run_on_directory(img_dir):
    if not os.path.isdir(img_dir):
        print(f"Error: Directory not found at {img_dir}")
        return
    
    img_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        img_files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    
    print(f"Found {len(img_files)} images to process")
    
    for img_path in img_files:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        predict_dessert(img_path)
        input("Press Enter to continue to next image...")

def main():
    global cnn_model, cnn_device, yolo_model, yolo_cls_model
    cnn_model, cnn_device = load_cnn_model()
    yolo_model = load_yolo_model()
    yolo_cls_model = load_yolo_cls()
    
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
            run_on_directory(img_dir)
        
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()