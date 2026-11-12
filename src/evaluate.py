
"""Evaluation script"""
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import CNNViTAdaptive
from data import get_dataloaders


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader, info = get_dataloaders()

    model = CNNViTAdaptive(num_classes=11).to(DEVICE)
    model.load_state_dict(torch.load('models/cnn_vit_adaptive_best.pth'))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)
            logits, _ = model(images)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = list(info['label'].values())
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    main()
