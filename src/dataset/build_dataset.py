import os
from pathlib import Path
from tqdm import tqdm
import shutil

from collect import CLASSES, collect_cifar10
from validate import is_valid_image, convert_to_rgb_and_save
from deduplicate import Deduplicator, get_exact_hash
from split import split_class_data
from statistics import generate_report, check_leakage

def build_dataset():
    raw_dir = Path("data/raw_dataset")
    out_dir = Path("data/dataset")
    
    # 1. Collect dataset if needed
    if not (raw_dir / "images").exists():
        collect_cifar10(str(raw_dir))
        
    images_dir = raw_dir / "images"
    
    # Prepare output directories
    for split in ["train", "validation", "test"]:
        for cls in CLASSES:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)
            
    stats = {
        "total_images_processed": 0,
        "valid_images": 0,
        "rejected_images": 0,
        "corrupted_images": 0,
        "duplicate_images": 0,
        "near_duplicates": 0,
        "images_per_class": {cls: 0 for cls in CLASSES},
        "train_count": 0,
        "validation_count": 0,
        "test_count": 0,
        "class_imbalance_info": "Balanced to targets",
        "leakage_results": {},
        "source_info": "CIFAR-10 Dataset (torchvision), MIT/Research License"
    }
    
    all_train_hashes = set()
    all_val_hashes = set()
    all_test_hashes = set()
    
    for cls in CLASSES:
        print(f"Processing class: {cls}")
        cls_dir = images_dir / cls
        if not cls_dir.exists():
            print(f"Directory {cls_dir} not found. Skipping.")
            continue
            
        filepaths = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
        stats["total_images_processed"] += len(filepaths)
        
        valid_paths = []
        dedup = Deduplicator()
        
        # Validation and Deduplication
        for fp in tqdm(filepaths, desc=f"Validating {cls}"):
            # Validate
            is_valid, reason = is_valid_image(fp)
            if not is_valid:
                stats["corrupted_images"] += 1
                stats["rejected_images"] += 1
                continue
                
            # Deduplicate
            is_dup, dup_reason = dedup.is_duplicate(fp)
            if is_dup:
                if dup_reason == "Exact duplicate":
                    stats["duplicate_images"] += 1
                else:
                    stats["near_duplicates"] += 1
                stats["rejected_images"] += 1
                continue
                
            dedup.add(fp)
            valid_paths.append(fp)
            
        # Splitting
        train_p, val_p, test_p, rejected = split_class_data(
            valid_paths, train_target=4000, val_target=1000, test_target=1000
        )
        
        stats["rejected_images"] += rejected
        
        # Save to final dataset
        def save_split(paths, split_name, global_hash_set):
            for i, p in enumerate(tqdm(paths, desc=f"Saving {split_name} {cls}")):
                dest = out_dir / split_name / cls / f"{cls}_{i:04d}.png"
                convert_to_rgb_and_save(p, dest)
                stats[f"{split_name}_count"] += 1
                stats["valid_images"] += 1
                stats["images_per_class"][cls] += 1
                global_hash_set.add(get_exact_hash(dest))
                
        save_split(train_p, "train", all_train_hashes)
        save_split(val_p, "validation", all_val_hashes)
        save_split(test_p, "test", all_test_hashes)

    # Check for leakage
    print("Checking for leakage across splits...")
    stats["leakage_results"] = check_leakage(all_train_hashes, all_val_hashes, all_test_hashes)
    
    # Generate report
    generate_report(stats, out_dir / "dataset_report.json")
    print("Dataset build complete.")

if __name__ == "__main__":
    build_dataset()
