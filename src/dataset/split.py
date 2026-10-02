import random
from typing import List
from pathlib import Path

def split_class_data(
    filepaths: List[Path], 
    train_target: int = 4000, 
    val_target: int = 1000, 
    test_target: int = 1000, 
    seed: int = 42
) -> tuple[List[Path], List[Path], List[Path], int]:
    """
    Deterministically shuffles and splits filepaths into train, val, test.
    Returns (train_paths, val_paths, test_paths, rejected_count)
    """
    total_target = train_target + val_target + test_target
    
    # Shuffle deterministically
    shuffled = list(filepaths)
    random.seed(seed)
    random.shuffle(shuffled)
    
    selected = shuffled[:total_target]
    rejected = len(shuffled) - len(selected)
    
    train_paths = selected[:train_target]
    val_paths = selected[train_target:train_target+val_target]
    test_paths = selected[train_target+val_target:]
    
    return train_paths, val_paths, test_paths, rejected
