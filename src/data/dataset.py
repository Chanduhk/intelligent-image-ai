"""
Dataset Management Module
Handles CIFAR-10 loading, splitting, and custom image integration.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image
import json


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CLASS_CATEGORIES = {
    "airplane": "Vehicle",
    "automobile": "Vehicle",
    "bird": "Animal",
    "cat": "Animal",
    "deer": "Animal",
    "dog": "Animal",
    "frog": "Animal",
    "horse": "Animal",
    "ship": "Vehicle",
    "truck": "Vehicle"
}


def get_cifar10_transforms(augment: bool = False):
    """Return standard transforms for CIFAR-10."""
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def load_cifar10(data_dir: str = "./data/raw", batch_size: int = 128,
                 num_workers: int = 0, augment_train: bool = True):
    """
    Load CIFAR-10 dataset and return train/test DataLoaders.

    num_workers is forced to 0 on Windows to prevent multiprocessing issues.
    pin_memory is disabled when CUDA is unavailable to save RAM.

    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    os.makedirs(data_dir, exist_ok=True)

    # Force num_workers=0 on Windows for stability
    import platform
    if platform.system() == "Windows":
        num_workers = 0

    # Only pin memory if CUDA is available (saves RAM on CPU-only machines)
    use_pin_memory = torch.cuda.is_available()

    train_transform = get_cifar10_transforms(augment=augment_train)
    test_transform = get_cifar10_transforms(augment=False)

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory
    )

    return train_loader, test_loader, train_dataset, test_dataset


def get_class_name(label_idx: int) -> str:
    """Convert a CIFAR-10 integer label to its string class name."""
    return CIFAR10_CLASSES[label_idx]


def get_class_category(class_name: str) -> str:
    """Return the category (Animal / Vehicle) for a given class name."""
    return CLASS_CATEGORIES.get(class_name.lower(), "Unknown")


class SingleImageDataset(Dataset):
    """Wraps a single pre-processed image tensor so it works with DataLoader."""

    def __init__(self, image_tensor: torch.Tensor):
        self.image = image_tensor.unsqueeze(0)   # (1, C, H, W)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.image[0]


class LearningBufferDataset(Dataset):
    """
    Dataset backed by the on-disk learning buffer
    (pairs of tensors and labels stored as .pt files).
    """

    def __init__(self, buffer_dir: str = "./data/learning_buffer"):
        self.buffer_dir = buffer_dir
        os.makedirs(buffer_dir, exist_ok=True)
        self.samples = self._scan()

    def _scan(self):
        samples = []
        for fname in sorted(os.listdir(self.buffer_dir)):
            if fname.endswith(".pt"):
                samples.append(os.path.join(self.buffer_dir, fname))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx], weights_only=True)
        return data["image"], data["label"]

    def add_sample(self, image_tensor: torch.Tensor, label: int):
        """Persist a new sample to the buffer directory."""
        idx = len(self.samples)
        path = os.path.join(self.buffer_dir, f"sample_{idx:06d}.pt")
        torch.save({"image": image_tensor, "label": label}, path)
        self.samples.append(path)
        return path

    def clear(self):
        """Remove all samples from the buffer."""
        for p in self.samples:
            if os.path.exists(p):
                os.remove(p)
        self.samples = []

    def stats(self) -> dict:
        return {"total_samples": len(self.samples), "buffer_dir": self.buffer_dir}
