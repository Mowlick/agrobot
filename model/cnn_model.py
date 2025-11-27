# cnn_model_fixed.py  (Final Working Version – No Errors)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import datetime

# ------------------- CONFIG -------------------
DATA_ROOT = r"C:\Users\Mowlick\Downloads\inf_projects\AgroBot-Universal\plantvillagedata"
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = 128
NUM_CLASSES = 38
# ------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- Transforms ----------
    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---------- Dataset ----------
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"Dataset not found at: {DATA_ROOT}")

    full_dataset = ImageFolder(DATA_ROOT, transform=transform_train)
    print(f"Found {len(full_dataset.classes)} classes")

    # Split
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_set.dataset.transform = transform_val_test
    test_set.dataset.transform = transform_val_test

    # ---------- DataLoaders (Windows-safe) ----------
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    # ---------- Model ----------
    class PlantCNN(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),   nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),  nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = PlantCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Fixed: 'verbose' removed (not supported in older PyTorch versions)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.5
    )

    # ---------- Training ----------
    def train_one_epoch():
        model.train()
        running_loss = 0.0
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        return running_loss / len(train_loader), correct / total

    def evaluate(loader):
        model.eval()
        running_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                running_loss += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return running_loss / len(loader), correct / total

    print("\nTraining started...")
    best_acc = 0.0
    train_hist = {'loss':[], 'acc':[]}
    val_hist   = {'loss':[], 'acc':[]}

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch()
        val_loss, val_acc = evaluate(val_loader)

        train_hist['loss'].append(tr_loss); train_hist['acc'].append(tr_acc)
        val_hist['loss'].append(val_loss);   val_hist['acc'].append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  [SAVED] Epoch {epoch} – Val Acc: {val_acc:.4f}")

        print(f"Epoch {epoch:02d} | Train L:{tr_loss:.4f} A:{tr_acc:.4f} | Val L:{val_loss:.4f} A:{val_acc:.4f}")

    print(f"\nTraining complete! Best Val Accuracy: {best_acc:.4f}")

    # ---------- Test ----------
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = evaluate(test_loader)
    print(f"TEST Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # ---------- Plots ----------
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_hist['loss'], label='Train')
    plt.plot(val_hist['loss'], label='Val')
    plt.title('Loss'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_hist['acc'], label='Train')
    plt.plot(val_hist['acc'], label='Val')
    plt.title('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig('curves.png', dpi=150)
    plt.show()

    # ---------- Confusion Matrix ----------
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(y.numpy())

    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(20,16))
    sns.heatmap(cm, annot=False, cmap='Blues',
                xticklabels=full_dataset.classes,
                yticklabels=full_dataset.classes)
    plt.title('Confusion Matrix – Test Set')
    plt.tight_layout()
    plt.savefig('confusion.png', dpi=150)
    plt.show()

    # ---------- Report ----------
    report = f"""
PLANT DISEASE DETECTION – FINAL PROJECT
=======================================
Name: Mowlick
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Dataset: {DATA_ROOT}

RESULTS
- Test Accuracy: {test_acc*100:.2f}%
- Best Val Accuracy: {best_acc*100:.2f}%
- Train/Val/Test: {len(train_set)} / {len(val_set)} / {len(test_set)}

FILES GENERATED
- best_model.pth
- curves.png
- confusion.png
- this script
"""
    with open("PROJECT_REPORT.txt", "w") as f:
        f.write(report)

    print("\n" + "="*50)
    print("PROJECT SUCCESSFULLY COMPLETED!")
    print("="*50)
    print("Next: Join Microsoft Teams group")
    print("Message Eldhose: 'Hi, project is ready for review.'")

# ------------------- RUN -------------------
if __name__ == '__main__':
    main()