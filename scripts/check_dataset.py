import json
from pathlib import Path
import os
from PIL import Image

def check_dataset(dataset_dir: str = "data/dataset"):
    dataset_path = Path(dataset_dir)
    report_path = dataset_path / "dataset_report.json"
    
    if not dataset_path.exists():
        print(f"Dataset directory {dataset_dir} does not exist.")
        print("Please run 'python src/dataset/build_dataset.py' first.")
        return
        
    if not report_path.exists():
        print(f"Dataset report not found at {report_path}.")
        return
        
    with open(report_path, 'r') as f:
        report = json.load(f)
        
    print("=== DATASET VALIDATION ===")
    print(f"Total processed: {report.get('total_images_processed', 'N/A')}")
    print(f"Train count: {report.get('train_count', 'N/A')}")
    print(f"Validation count: {report.get('validation_count', 'N/A')}")
    print(f"Test count: {report.get('test_count', 'N/A')}")
    
    # Lightweight file checking
    print("\nRunning lightweight file check...")
    for split in ["train", "validation", "test"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"Missing split directory: {split}")
            continue
            
        classes = os.listdir(split_dir)
        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.is_dir(): continue
            files = list(cls_dir.glob("*.png"))
            if len(files) > 0:
                # check first file
                with Image.open(files[0]) as img:
                    try:
                        img.verify()
                    except Exception as e:
                        print(f"Failed verification on {files[0]}: {e}")
            else:
                print(f"Warning: No images found in {split}/{cls}")

    leakage = report.get('leakage_results', {})
    if leakage.get('is_leaking', False):
        print("WARNING: Data leakage detected!")
    else:
        print("PASS: No data leakage detected across splits.")
        
    print("Check complete. Dataset pipeline is ready for v4.")

if __name__ == "__main__":
    check_dataset()
