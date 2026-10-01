"""
Preprocessing Module
Converts raw images (PIL / numpy / file path) into normalised tensors
ready for the CNN Autoencoder, and un-normalises tensors for display.
"""

import numpy as np
import torch
from torchvision import transforms
from PIL import Image


# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE   = 32          # CIFAR-10 native resolution
IMG_MEAN   = (0.5, 0.5, 0.5)
IMG_STD    = (0.5, 0.5, 0.5)
IMG_ORIG_VALUES = 3 * IMG_SIZE * IMG_SIZE   # 3072


# ── Standard pipelines ───────────────────────────────────────────────────────
_to_tensor_norm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])


def preprocess_pil(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image → normalised float tensor (C, H, W) in [-1, 1].
    Handles greyscale / RGBA by converting to RGB first.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return _to_tensor_norm(pil_image)


def preprocess_file(path: str) -> torch.Tensor:
    """Load an image from *path* and apply the standard preprocessing."""
    img = Image.open(path)
    return preprocess_pil(img)


def preprocess_numpy(arr: np.ndarray) -> torch.Tensor:
    """
    Convert a uint8 HxWxC numpy array → normalised tensor.
    Accepts HxWxC (RGB) arrays.
    """
    pil = Image.fromarray(arr.astype(np.uint8))
    return preprocess_pil(pil)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalise a (C, H, W) tensor in [-1, 1] back to a
    uint8 HxWxC numpy array suitable for display / saving.
    """
    t = tensor.detach().cpu().clone()
    # Undo Normalize: x = t * std + mean  → x ∈ [0, 1]
    mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMG_STD).view(3, 1, 1)
    t = t * std + mean
    t = t.clamp(0.0, 1.0)
    # (C, H, W) → (H, W, C)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return arr


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Denormalise a (C, H, W) tensor and return a PIL Image."""
    return Image.fromarray(tensor_to_numpy(tensor))


def batch_to_numpy(batch: torch.Tensor) -> np.ndarray:
    """
    Denormalise a (N, C, H, W) batch tensor → (N, H, W, C) uint8 array.
    """
    return np.stack([tensor_to_numpy(t) for t in batch])


def validate_image(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Basic validation: check that the image is non-empty and can be
    converted to RGB.  Returns (ok, message).
    """
    try:
        rgb = pil_image.convert("RGB")
        w, h = rgb.size
        if w == 0 or h == 0:
            return False, "Image has zero dimension."
        return True, "OK"
    except Exception as exc:
        return False, str(exc)
