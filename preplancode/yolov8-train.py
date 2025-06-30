import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Define paths
MAIN_DIR = r"C:\Coding\Earendel"
DATASET_DIR = os.path.join(MAIN_DIR, "dessert_dataset_yolo")
YOLO_MODEL_DIR = os.path.join(MAIN_DIR, "yolo_model")
os.makedirs(YOLO_MODEL_DIR, exist_ok=True)

def train_yolov8():
    print("Starting YOLOv8 training...")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    device = "0" if cuda_available else "cpu"
    
    print(f"Training on device: {'CUDA GPU' if cuda_available else 'CPU'}")
    
    # Load a YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 nano model
    
    # Data YAML path
    data_yaml = os.path.join(DATASET_DIR, "dataset.yaml")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        project=YOLO_MODEL_DIR,
        name="dessert_detector",
        device=device,
        verbose=True,
        exist_ok=True
    )
    
    print("YOLOv8 training complete!")
    return os.path.join(YOLO_MODEL_DIR, "dessert_detector")

def evaluate_yolov8(results_dir):
    """Evaluate YOLOv8 model performance and plot metrics"""
    print("Evaluating YOLOv8 model performance...")
    
    # Load the trained model
    model_path = os.path.join(results_dir, "weights", "best.pt")
    model = YOLO(model_path)
    
    # Run model validation to get metrics
    metrics = model.val(
        data=os.path.join(DATASET_DIR, "dataset.yaml"),
        split='test'
    )
    
    # Plot metrics if available
    results_csv = os.path.join(results_dir, "results.csv")
    if os.path.exists(results_csv):
        results = pd.read_csv(results_csv)
        
        # Plot training curves
        plt.figure(figsize=(15, 10))
        
        # Plot box loss
        plt.subplot(2, 2, 1)
        plt.plot(results['epoch'], results['train/box_loss'], label='train_box_loss')
        plt.plot(results['epoch'], results['val/box_loss'], label='val_box_loss')
        plt.title('Box Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot class loss
        plt.subplot(2, 2, 2)
        plt.plot(results['epoch'], results['train/cls_loss'], label='train_cls_loss')
        plt.plot(results['epoch'], results['val/cls_loss'], label='val_cls_loss')
        plt.title('Class Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot dfl loss (Distribution Focal Loss - specific to YOLOv8)
        if 'train/dfl_loss' in results.columns:
            plt.subplot(2, 2, 3)
            plt.plot(results['epoch'], results['train/dfl_loss'], label='train_dfl_loss')
            plt.plot(results['epoch'], results['val/dfl_loss'], label='val_dfl_loss')
            plt.title('DFL Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        # Plot mAP
        plt.subplot(2, 2, 4)
        plt.plot(results['epoch'], results['metrics/mAP50'], label='mAP@0.5')
        plt.plot(results['epoch'], results['metrics/mAP50-95'], label='mAP@0.5:0.95')
        plt.title('Mean Average Precision')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Final metrics
        final_epoch = results.iloc[-1]
        print("\nYOLOv8 Final Metrics:")
        print(f"mAP@0.5: {final_epoch['metrics/mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {final_epoch['metrics/mAP50-95']:.4f}")
        
        # Try to extract precision and recall if available
        if 'metrics/precision' in final_epoch and 'metrics/recall' in final_epoch:
            precision = final_epoch['metrics/precision']
            recall = final_epoch['metrics/recall']
            print(f"Precision: {precision:.4f}")
            print(f"Recall (TPR): {recall:.4f}")
            print(f"F1-Score: {2 * precision * recall / (precision + recall):.4f}")
    
    # Display confusion matrix from validation results if available
    conf_matrix_path = os.path.join(results_dir, "val", "confusion_matrix.png")
    if os.path.exists(conf_matrix_path):
        plt.figure(figsize=(10, 8))
        conf_matrix = plt.imread(conf_matrix_path)
        plt.imshow(conf_matrix)
        plt.axis('off')
        plt.title('YOLOv8 Confusion Matrix')
        plt.show()
    
    # Get class names from the dataset yaml
    import yaml
    with open(os.path.join(DATASET_DIR, "dataset.yaml"), 'r') as f:
        dataset_config = yaml.safe_load(f)
        class_names = dataset_config.get('names', [])
    
    print(f"\nDetected Classes: {', '.join(class_names)}")
    
    # Calculate and display ROC curve for each class if possible
    try:
        # Run prediction on test set to get probabilities
        test_images_dir = os.path.join(DATASET_DIR, "test", "images")
        results = model(test_images_dir, task='detect', save=False, conf=0.1)
        
        # Process predictions to get class probabilities and true labels
        all_probs = []
        all_labels = []
        
        # This is simplified and would need to be expanded based on your specific dataset structure
        print("Calculating ROC curves is complex with object detection models")
        print("This would require ground truth matching and is beyond the scope of this script")
        
        # Approximate ROC calculation would go here if implemented
        
    except Exception as e:
        print(f"Could not calculate ROC curves: {str(e)}")
    
    return metrics

if __name__ == "__main__":
    # Train YOLOv8 model
    results_dir = train_yolov8()
    
    # Wait for model training to complete and files to be written
    time.sleep(2)
    
    # Evaluate performance
    metrics = evaluate_yolov8(results_dir)
    
    print("\nYOLOv8 training and evaluation complete!")
    print(f"Model saved at: {results_dir}")
