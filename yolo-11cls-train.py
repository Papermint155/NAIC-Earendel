import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import time
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Fix the path to match what's in the error logs
MAIN_DIR = r"C:\Coding\Earendel"
WHL_DIR = os.path.join(MAIN_DIR, "ls")  # Add WHL subfolder to match error logs
DATASET_DIR = r"C:\Coding\Earendel\ls\dessert_dataset_cnn" #os.path.join(WHL_DIR, "dessert_dataset_cnn")
YOLO_MODEL_DIR = os.path.join(WHL_DIR, "yolo_model_cls")
os.makedirs(YOLO_MODEL_DIR, exist_ok=True)


def train_yolo():
    print("Start Training YOLOv11 Classification")

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = YOLO("yolo11s-cls.pt")#s
    
    # Use absolute path for the dataset.yaml file
    data_yaml = os.path.join(DATASET_DIR)
    
    # Set task explicitly to classification
    result = model.train(
        task="classify",  # <-- This is critical for classification tasks
        data= data_yaml,
        epochs=100,
        imgsz=640,
        batch=14,#12/10
        patience=50,
        project=YOLO_MODEL_DIR,
        name="dessert_detector",
        device=device,
        verbose=True,
        exist_ok=True
    )

    print("YOLOv11 training complete!")
    return os.path.join(YOLO_MODEL_DIR, "dessert_detector")


def evaluate_yolo(results_dir):
    print("Start Evaluation")

    model_path = os.path.join(results_dir, "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return None

    model = YOLO(model_path)

    metrics = model.val(
        data=os.path.join(DATASET_DIR, "dataset.yaml"),
        split="test"
    )

    # Plot metrics if available
    results_csv = os.path.join(results_dir, "results-cls.csv")
    if os.path.exists(results_csv):
        try:
            results = pd.read_csv(results_csv)
            plt.figure(figsize=(15, 10))
            
            if 'train/box_loss' in results.columns and 'val/box_loss' in results.columns:
                plt.subplot(2, 2, 1)
                plt.plot(results['epoch'], results['train/box_loss'], label='train_box_loss')
                plt.plot(results['epoch'], results['val/box_loss'], label='val_box_loss')
                plt.title('Box Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
            
            if 'train/cls_loss' in results.columns and 'val/cls_loss' in results.columns:
                plt.subplot(2, 2, 2)
                plt.plot(results['epoch'], results['train/cls_loss'], label='train_cls_loss')
                plt.plot(results['epoch'], results['val/cls_loss'], label='val_cls_loss')
                plt.title('Class Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

            if 'train/dfl_loss' in results.columns and 'val/dfl_loss' in results.columns:
                plt.subplot(2, 2, 3)
                plt.plot(results['epoch'], results['train/dfl_loss'], label='train_dfl_loss')
                plt.plot(results['epoch'], results['val/dfl_loss'], label='val_dfl_loss')
                plt.title('DFL Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
            
            if 'metrics/mAP50' in results.columns and 'metrics/mAP50-95' in results.columns:
                plt.subplot(2, 2, 4)
                plt.plot(results['epoch'], results['metrics/mAP50'], label='mAP@0.5')
                plt.plot(results['epoch'], results['metrics/mAP50-95'], label='mAP@0.5:0.95')
                plt.title('Mean Average Precision')
                plt.xlabel('Epoch')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "training_curves.png"))
            plt.show()

            final_epoch = results.iloc[-1]
            print("\nYOLOv11 Final Metrics:")
            print(f"mAP@0.5: {final_epoch['metrics/mAP50']:.4f}")
            print(f"mAP@0.5:0.95: {final_epoch['metrics/mAP50-95']:.4f}")
            
            if 'metrics/precision' in final_epoch and 'metrics/recall' in final_epoch:
                precision = final_epoch['metrics/precision']
                recall = final_epoch['metrics/recall']
                print(f"Precision: {precision:.4f}")
                print(f"Recall (TPR): {recall:.4f}")
                print(f"F1-Score: {2 * precision * recall / (precision + recall):.4f}")
        
        except Exception as e:
            print(f"Error processing results CSV: {str(e)}")

    conf_matrix_path = os.path.join(results_dir, "val", "confusion_matrix.png")
    if os.path.exists(conf_matrix_path):
        plt.figure(figsize=(10, 8))
        conf_matrix = plt.imread(conf_matrix_path)
        plt.imshow(conf_matrix)
        plt.axis('off')
        plt.title('YOLOv11 Confusion Matrix')
        plt.show()
    
    try:
        with open(os.path.join(DATASET_DIR, "dataset.yaml"), 'r') as f:
            dataset_config = yaml.safe_load(f)
            class_names = dataset_config.get('names', [])
        print(f"\nDetected Classes: {', '.join(class_names)}")
    except Exception as e:
        print(f"Error reading dataset.yaml: {str(e)}")
    
    return metrics
if __name__ == "__main__":
    results_dir = train_yolo()
    evaluate_yolo(results_dir)