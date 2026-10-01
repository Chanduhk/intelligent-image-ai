"""
Quality Metrics Module
Computes MSE, PSNR, and SSIM between original and reconstructed images,
plus the compression ratio relative to the latent dimension.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as ski_ssim

from src.models.autoencoder import ORIGINAL_DIM, LATENT_DIM


# ── Per-metric helpers ────────────────────────────────────────────────────────

def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Mean-Squared Error over uint8 [0, 255] arrays.
    Input shape: (H, W, C).
    Returns a float in [0, ∞).
    """
    orig  = original.astype(np.float32)
    recon = reconstructed.astype(np.float32)
    return float(np.mean((orig - recon) ** 2))


def compute_psnr(mse: float, max_pixel: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    If MSE == 0 (perfect reconstruction) returns +inf.
    """
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((max_pixel ** 2) / mse))


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Structural Similarity Index Measure in [0, 1].
    Input shape: (H, W, C) uint8.
    """
    orig  = original.astype(np.float32) / 255.0
    recon = reconstructed.astype(np.float32) / 255.0
    # channel_axis=-1 for multi-channel images
    return float(ski_ssim(orig, recon, data_range=1.0, channel_axis=-1))


def compute_compression_ratio(original_dim: int = ORIGINAL_DIM,
                               latent_dim: int   = LATENT_DIM) -> float:
    """Return original_dim / latent_dim as a float (e.g. 3.0)."""
    return round(original_dim / latent_dim, 2)


def quality_label(psnr: float, ssim: float) -> str:
    """
    Heuristic label based on PSNR and SSIM thresholds.
    """
    if psnr == float("inf") or (psnr >= 40 and ssim >= 0.95):
        return "Excellent"
    if psnr >= 30 and ssim >= 0.85:
        return "Good"
    if psnr >= 20 and ssim >= 0.70:
        return "Fair"
    return "Poor"


# ── Main entry point ──────────────────────────────────────────────────────────

def evaluate_reconstruction(
    original_tensor: torch.Tensor,
    reconstructed_tensor: torch.Tensor,
) -> dict:
    """
    Compute all quality metrics for a single image pair.

    Args:
        original_tensor      – (C, H, W) float tensor in [-1, 1]
        reconstructed_tensor – (C, H, W) float tensor in [0, 1]  (Sigmoid output)

    Returns a dict with MSE, PSNR, SSIM, compression ratio, and quality label.
    """
    from src.data.preprocessing import tensor_to_numpy

    orig_np  = tensor_to_numpy(original_tensor)           # uint8 HxWxC

    # Decoder output is Sigmoid → [0, 1]; we need to map to display space too.
    # The decoder output is NOT normalised with -0.5/0.5 so we just scale to [0,255].
    recon_np = (reconstructed_tensor.detach().cpu()
                .clamp(0, 1)
                .permute(1, 2, 0)
                .numpy() * 255).astype(np.uint8)

    mse   = compute_mse(orig_np, recon_np)
    psnr  = compute_psnr(mse)
    ssim  = compute_ssim(orig_np, recon_np)
    cr    = compute_compression_ratio()
    label = quality_label(psnr, ssim)

    return {
        "mse":               round(mse / (255.0 ** 2), 6),   # normalised [0,1]
        "mse_raw":           round(mse, 4),
        "psnr_db":           round(psnr, 2) if psnr != float("inf") else psnr,
        "ssim":              round(ssim, 4),
        "compression_ratio": cr,
        "original_dim":      ORIGINAL_DIM,
        "latent_dim":        LATENT_DIM,
        "quality_label":     label,
    }
