import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt

# ── Dataset ─────────────────────────────────────────────────────────────────
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

class CIFAR10Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_arr, label = self.data[idx]
        img = Image.fromarray(img_arr.reshape(32, 32, 3).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, label

# ── Load CIFAR-10 from KNN dataset ──────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("../5-KNN/dataset/trainLabels.csv")
labels_map = df.set_index("id")["label"].to_dict()
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

dataset_raw = []
for i in range(1, 10001):
    if i % 2000 == 0:
        print(f"  {i}/10000 loaded")
    img = Image.open(f"../5-KNN/dataset/train/{i}.png")
    pixel = np.array(img).flatten()
    label = class_to_idx[labels_map[i]]
    dataset_raw.append((pixel, label))

print("Complete load")

# ── Train/Val split (80/20) ─────────────────────────────────────────────────
split = int(len(dataset_raw) * 0.8)
train_raw = dataset_raw[:split]
val_raw   = dataset_raw[split:]

train_transform = transforms.ToTensor()
val_transform = transforms.ToTensor()

train_dataset = CIFAR10Dataset(train_raw, transform=train_transform)
val_dataset   = CIFAR10Dataset(val_raw,   transform=val_transform)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0)

# ── CNN Model ────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32→16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 16→8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 8→4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Training ─────────────────────────────────────────────────────────────────
device = torch.device("mps")
print(f"Device: {device}")

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

EPOCHS = 30
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    train_losses.append(total_loss / total)
    train_accs.append(correct / total * 100)

    # Validate
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    val_losses.append(total_loss / total)
    val_accs.append(correct / total * 100)

    print(f"Epoch [{epoch:2d}/{EPOCHS}]  "
          f"Train Loss: {train_losses[-1]:.4f}  Acc: {train_accs[-1]:.2f}%  |  "
          f"Val Loss: {val_losses[-1]:.4f}  Acc: {val_accs[-1]:.2f}%")

# ── Confusion Matrix ─────────────────────────────────────────────────────────
model.eval()
confusion = np.zeros((10, 10), dtype=int)
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        for t, p in zip(labels.numpy(), preds):
            confusion[t][p] += 1

def save_confusion(confusion, accuracy):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'CNN Confusion Matrix  |  Accuracy: {accuracy:.2f}%')
    thresh = confusion.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(confusion[i][j]), ha='center', va='center',
                    color='white' if confusion[i][j] > thresh else 'black', fontsize=8)
    plt.tight_layout()
    plt.savefig('./results/confusion_matrix/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved.")

final_acc = val_accs[-1]
save_confusion(confusion, final_acc)

# ── Loss / Accuracy Plot ─────────────────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CNN Training on CIFAR-10", fontsize=14, fontweight='bold')

axes[0].plot(epochs_range, train_losses, label='Train', color='steelblue')
axes[0].plot(epochs_range, val_losses,   label='Val',   color='tomato')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].plot(epochs_range, train_accs, label='Train', color='steelblue')
axes[1].plot(epochs_range, val_accs,   label='Val',   color='tomato')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('./results/accuracy_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Training complete. Final Val Accuracy: {final_acc:.2f}%")
print("Plots saved to ./results/")
