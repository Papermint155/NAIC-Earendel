import os
import sys
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

# Define paths and constants
MAIN_DIR = os.getcwd()
DATASET_CNN = os.path.join(MAIN_DIR, "dessert_dataset_cnn")
CNN_MODEL_DIR = os.path.join(MAIN_DIR, "cnn_model")
CLASSES = [
    "kek_lapis", "kuih_kaswi_pandan", "Kuih_Ketayap",
    "Kuih_Lapis", "Kuih_Seri_Muka", "Kuih_Talam", "Kuih_Ubi_Kayu", "Onde_Onde"
]
os.makedirs(CNN_MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_CNN, exist_ok=True)

class DessertDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.img_paths = []
        self.labels = []
        
        for cls in self.classes:
            cls_dir = os.path.join(img_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: Class directory {cls_dir} not found")
                continue
                
            for img_file in os.listdir(cls_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(cls_dir, img_file))
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None, None
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(640),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(
            .20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(640),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(640),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_cnn_model():
    print("Starting CNN model training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(DATASET_CNN, split)
        if not os.path.exists(split_dir):
            print(f"Error: {split} directory not found at {split_dir}")
            sys.exit(1)
    
    try:
        train_dataset = DessertDataset(os.path.join(DATASET_CNN, 'train'), data_transforms['train'])
        val_dataset = DessertDataset(os.path.join(DATASET_CNN, 'val'), data_transforms['val'])
        test_dataset = DessertDataset(os.path.join(DATASET_CNN, 'test'), data_transforms['test'])
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        sys.exit(1)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: One or more datasets are empty")
        sys.exit(1)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    num_epochs = 100
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting training loop...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            if inputs is None or labels is None:
                continue
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
        
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                if inputs is None or labels is None:
                    continue
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
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            try:
                torch.save(model.state_dict(), os.path.join(CNN_MODEL_DIR, 'best_cnn_model.pth'))
            except Exception as e:
                print(f"Error saving model: {str(e)}")
            
    print(f"Best val Acc: {best_acc:.4f}")
    
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
    
    try:
        model.load_state_dict(torch.load(os.path.join(CNN_MODEL_DIR, 'best_cnn_model.pth')))
    except Exception as e:
        print(f"Error loading best model: {str(e)}")
    
    return model, device, test_loader

def evaluate_cnn_model(model, device, test_loader):
    print("Evaluating CNN model performance...")
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs is None or labels is None:
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
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
        
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        spec_values.append(spec)
        
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        npv_values.append(npv)
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_values.append(fpr)
        
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
        fdr_values.append(fdr)
    
    avg_spec = np.mean(spec_values)
    avg_npv = np.mean(npv_values)
    avg_fpr = np.mean(fpr_values)
    avg_fdr = np.mean(fdr_values)
    
    print(f"Specificity (SPC): {avg_spec:.4f}")
    print(f"Negative Predictive Value (NPV): {avg_npv:.4f}")
    print(f"False Positive Rate (FPR): {avg_fpr:.4f}")
    print(f"False Discovery Rate (FDR): {avg_fdr:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(CNN_MODEL_DIR, 'confusion_matrix.png'))
    plt.show()
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(CLASSES):
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-All)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(CNN_MODEL_DIR, 'roc_curves.png'))
    plt.show()
    
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
    
    try:
        with open(os.path.join(CNN_MODEL_DIR, 'metrics.txt'), 'w') as f:
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
    except Exception as e:
        print(f"Error saving metrics: {str(e)}")
    
    return metrics

if __name__ == "__main__":
    model, device, test_loader = train_cnn_model()
    metrics = evaluate_cnn_model(model, device, test_loader)
    print("\nCNN model training and evaluation complete!")
    print(f"Model saved at: {os.path.join(CNN_MODEL_DIR, 'best_cnn_model.pth')}")