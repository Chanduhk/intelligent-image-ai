from PIL import Image
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_valid_image(filepath: Path) -> tuple[bool, str]:
    """
    Validates an image according to the rules:
    1. Be readable and not corrupted
    2. Be converted to RGB (or can be)
    3. Have sufficient resolution (>= 32x32)
    """
    try:
        with Image.open(filepath) as img:
            # 1 & 3: Check if readable and not corrupted
            img.verify()
    except Exception as e:
        return False, f"Corrupted or unreadable: {e}"
        
    try:
        # Re-open for mode and size checking since verify() can close or restrict it
        with Image.open(filepath) as img:
            # 5: Minimum-resolution checking
            width, height = img.size
            if width < 32 or height < 32:
                return False, f"Resolution too low: {width}x{height}"
            
            # 4: RGB Conversion (Check if RGB)
            if img.mode != "RGB":
                # It can be converted, but we report if it wasn't RGB natively 
                # (for the pipeline we will convert it during saving if needed)
                pass 
                
    except Exception as e:
        return False, f"Error reading properties: {e}"
        
    return True, "Valid"

def convert_to_rgb_and_save(src_path: Path, dest_path: Path):
    """Converts image to RGB and saves to destination."""
    with Image.open(src_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dest_path)
