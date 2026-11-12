
"""Data loading for OrganAMNIST"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import OrganAMNIST, INFO


def get_dataloaders(batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = OrganAMNIST(split='train', download=True, size=28, transform=transform)
    val_dataset = OrganAMNIST(split='val', download=True, size=28, transform=transform)
    test_dataset = OrganAMNIST(split='test', download=True, size=28, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, INFO['organamnist']
