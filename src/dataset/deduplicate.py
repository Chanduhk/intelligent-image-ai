import hashlib
from PIL import Image
from pathlib import Path

def get_exact_hash(filepath: Path) -> str:
    """Computes MD5 hash of the file for exact duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

class Deduplicator:
    def __init__(self):
        self.exact_hashes = set()
        
    def is_duplicate(self, filepath: Path) -> tuple[bool, str]:
        """
        Checks if an image is an exact duplicate.
        Returns (is_duplicate, reason)
        """
        exact = get_exact_hash(filepath)
        if exact in self.exact_hashes:
            return True, "Exact duplicate"
            
        return False, ""
        
    def add(self, filepath: Path):
        self.exact_hashes.add(get_exact_hash(filepath))
