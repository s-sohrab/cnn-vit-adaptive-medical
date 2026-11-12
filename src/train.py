
"""Training script"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time, copy
from model import CNNViTAdaptive
from data import get_dataloaders


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader, info = get_dataloaders()
    model = CNNViTAdaptive(num_classes=11).to(DEVICE)

    EPOCHS = 10
    LR = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1}: Val Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/cnn_vit_adaptive_best.pth')
        scheduler.step()

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
