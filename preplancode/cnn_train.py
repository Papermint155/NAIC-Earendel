import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import time

# Define paths
MAIN_DIR = r"C:\Coding\Earendel"
DATASET_SRC = r"C:\Users\menth\Downloads\dessert_dataset"
DATASET_CNN = os.path.join(MAIN_DIR, "dessert_dataset_cnn")
CNN_MODEL_DIR = os.path.join(MAIN_DIR, "cnn_model")
os.makedirs(CNN_MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_CNN, exist_ok=True)

# Create train, val, test directories for CNN dataset
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DATASET_CNN, split), exist_ok=True)

# Class names
CLASSES = [
    "kek_lapis", "Kuih_Bahulu", "kuih_kaswi_pandan", "Kuih_Ketayap", 
    "Kuih_Lapis", "Kuih_Seri_Muka", "Kuih_Talam", "Kuih_Ubi_Kayu", "Onde_Onde"
]

# Create class directories in each split
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DATASET_CNN, split, cls), exist_ok=True)

# Prepare CNN dataset
def prepare_cnn_dataset():
    print("Preparing CNN dataset...")
    
    for cls in CLASSES:
        src_dir = os.path.join(DATASET_SRC, cls)
        if not os.path.exists(src_dir):
            print(f"Warning: {src_dir} does not exist, skipping...")
            continue
        
        # Get all image files
        img_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            img_files.extend(Path(src_dir).glob(f"*{ext}"))
        
        if not img_files:
            print(f"Warning: No images found in {src_dir}")
            continue
            
        # Shuffle and split: 70% train, 20% val, 10% test
        np.random.shuffle(img_files)
        
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
    
    print(f"CNN dataset prepared at {DATASET_CNN}")

# Custom dataset class
class DessertDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.img_paths = []
        self.labels = []
        
        # Collect all images and their labels
        for cls in self.classes:
            cls_dir = os.path.join(img_dir, cls)
            if not os.path.exists(cls_dir):
                continue
                
            for img_file in os.listdir(cls_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(cls_dir, img_file))
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Train CNN model
def train_cnn_model():
    print("Starting CNN model training...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Dataloaders
    train_dataset = DessertDataset(os.path.join(DATASET_CNN, 'train'), data_transforms['train'])
    val_dataset = DessertDataset(os.path.join(DATASET_CNN, 'val'), data_transforms['val'])
    test_dataset = DessertDataset(os.path.join(DATASET_CNN, 'test'), data_transforms['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    num_epochs = 25
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting training loop...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset)
        
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())
        
        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(CNN_MODEL_DIR, 'best_cnn_model.pth'))
            
    print(f"Best val Acc: {best_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accs, label='Train Acc')
    plt.plot(range(1, num_epochs+1), val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CNN_MODEL_DIR, 'training_curves.png'))
    plt.show()
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(CNN_MODEL_DIR, 'best_cnn_model.pth')))
    
    return model, device, test_loader

def evaluate_cnn_model(model, device, test_loader):
    """Evaluate CNN model and show performance metrics"""
    print("Evaluating CNN model performance...")
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    print("CNN Model Performance Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (PPV): {precision:.4f}")
    print(f"Recall (TPR): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    
    # Calculate specificity/TNR and NPV for each class using one-vs-all approach
    spec_values = []
    npv_values = []
    fpr_values = []
    fdr_values = []
    
    for class_idx in range(len(CLASSES)):
        binary_labels = (all_labels == class_idx).astype(int)
        binary_preds = (all_preds == class_idx).astype(int)
        
        tn = np.sum((binary_labels == 0) & (binary_preds == 0))
        fp = np.sum((binary_labels == 0) & (binary_preds == 1))
        fn = np.sum((binary_labels == 1) & (binary_preds == 0))
        tp = np.sum((binary_labels == 1) & (binary_preds == 1))
        
        # Specificity = TN / (TN + FP)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        spec_values.append(spec)
        
        # NPV = TN / (TN + FN)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        npv_values.append(npv)
        
        # FPR = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_values.append(fpr)
        
        # FDR = FP / (FP + TP)
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
        fdr_values.append(fdr)
    
    # Average values
    avg_spec = np.mean(spec_values)
    avg_npv = np.mean(npv_values)
    avg_fpr = np.mean(fpr_values)
    avg_fdr = np.mean(fdr_values)
    
    print(f"Specificity (SPC): {avg_spec:.4f}")
    print(f"Negative Predictive Value (NPV): {avg_npv:.4f}")
    print(f"False Positive Rate (FPR): {avg_fpr:.4f}")
    print(f"False Discovery Rate (FDR): {avg_fdr:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(CNN_MODEL_DIR, 'confusion_matrix.png'))
    plt.show()
    
    # ROC curves (one-vs-all)
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(CLASSES):
        # Prepare binary labels for this class
        binary_labels = (all_labels == i).astype(int)
        
        # Get probabilities for this class
        class_probs = all_probs[:, i]
        
        # Calculate ROC
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-All)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(CNN_MODEL_DIR, 'roc_curves.png'))
    plt.show()
    
    # Save metrics for later use
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc': mcc,
        'specificity': avg_spec,
        'npv': avg_npv,
        'fpr': avg_fpr,
        'fdr': avg_fdr
    }
    
    # Save metrics to file
    with open(os.path.join(CNN_MODEL_DIR, 'metrics.txt'), 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")
    
    return metrics

if __name__ == "__main__":
    # First, prepare the dataset
    prepare_cnn_dataset()
    
    # Then train and evaluate the model
    model, device, test_loader = train_cnn_model()
    metrics = evaluate_cnn_model(model, device, test_loader)
    
    print("\nCNN model training and evaluation complete!")
    print(f"Model saved at: {os.path.join(CNN_MODEL_DIR, 'best_cnn_model.pth')}")