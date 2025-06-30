import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import yaml
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

MAIN_DIR = os.getcwd()
DATASET_DIR = os.path.join(MAIN_DIR, "dessert_dataset_yolo")
YOLO_MODEL_DIR = os.path.join(MAIN_DIR, "yolo_model")
os.makedirs(YOLO_MODEL_DIR, exist_ok=True)

def train_yolo():
    print("Start Training YOLOv11 Detection")

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = YOLO("yolo11s.pt")
    data_yaml = os.path.join(DATASET_DIR, "dataset.yaml")

    if not os.path.exists(data_yaml):
        print(f"Error: dataset.yaml not found at {data_yaml}")
        sys.exit(1)

    # Verify dataset structure
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DATASET_DIR, split, "images")
        label_dir = os.path.join(DATASET_DIR, split, "labels")
        if not (os.path.exists(img_dir) and os.path.exists(label_dir)):
            print(f"Error: Missing {split}/images or {split}/labels directory in {DATASET_DIR}")
            sys.exit(1)

    result = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=7,
        patience=30,
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
    results_csv = os.path.join(results_dir, "results.csv")
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
    time.sleep(5)
    metrics = evaluate_yolo(results_dir)
    print("\nYOLOv11 training and evaluation complete!")
    print(f"Model saved at: {results_dir}")