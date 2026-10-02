import os
import ssl
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision

# Disable SSL verification for downloading if there are cert issues
ssl._create_default_https_context = ssl._create_unverified_context

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def collect_cifar10(raw_dir: str):
    """
    Downloads the CIFAR-10 dataset which legitimately contains the exact 10 classes
    required, with standard licensing for research/educational use.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading CIFAR-10 dataset...")
    # Get both train and test to have all 60,000 images available
    trainset = torchvision.datasets.CIFAR10(root=raw_dir, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=raw_dir, train=False, download=True)
    
    # Create class directories in raw_dir
    images_dir = raw_path / "images"
    for cls in CLASSES:
        (images_dir / cls).mkdir(parents=True, exist_ok=True)
        
    print("Exporting training images...")
    for i in tqdm(range(len(trainset))):
        img, label = trainset[i]
        cls_name = CLASSES[label]
        img.save(images_dir / cls_name / f"train_{i}.png")
        
    print("Exporting testing images...")
    for i in tqdm(range(len(testset))):
        img, label = testset[i]
        cls_name = CLASSES[label]
        img.save(images_dir / cls_name / f"test_{i}.png")
        
    print(f"Collection complete. Raw images saved to {images_dir}")
    return images_dir

if __name__ == "__main__":
    collect_cifar10("data/raw_dataset")
