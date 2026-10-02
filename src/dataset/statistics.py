import json
from pathlib import Path

def generate_report(stats: dict, output_path: Path):
    """
    Generates a dataset report in JSON format.
    stats dict should contain:
    - total_images
    - valid_images
    - rejected_images
    - corrupted_images
    - duplicate_images
    - near_duplicates
    - class_counts (dict: class -> count)
    - train_count
    - validation_count
    - test_count
    - class_imbalance_info
    - leakage_results
    - source_info
    """
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Dataset report generated at {output_path}")

def check_leakage(train_set: set, val_set: set, test_set: set) -> dict:
    """Checks for intersection between sets (leakage prevention)."""
    train_val_leak = len(train_set.intersection(val_set))
    train_test_leak = len(train_set.intersection(test_set))
    val_test_leak = len(val_set.intersection(test_set))
    
    return {
        "train_val_leak": train_val_leak,
        "train_test_leak": train_test_leak,
        "val_test_leak": val_test_leak,
        "is_leaking": (train_val_leak > 0) or (train_test_leak > 0) or (val_test_leak > 0)
    }
