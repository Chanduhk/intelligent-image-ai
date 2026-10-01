"""
Image Specification Engine
Assembles a structured, human-readable specification dict from all
module outputs (classification, quality, anomaly, similarity, versioning).
"""

from src.data.dataset import CIFAR10_CLASSES, CLASS_CATEGORIES
from src.models.autoencoder import (
    LATENT_CHANNELS, LATENT_SPATIAL, LATENT_DIM,
    ORIGINAL_DIM, COMPRESSION_RATIO
)


CIFAR10_IMG_SIZE = 32


def build_specification(
    classification: dict,
    quality:        dict,
    anomaly:        dict,
    similarity:     list,
    model_version:  str = "v1",
) -> dict:
    """
    Assemble a complete image specification from sub-module results.

    Args:
        classification – output of LatentClassifier.predict()
        quality        – output of evaluate_reconstruction()
        anomaly        – output of AnomalyDetector.predict()
        similarity     – output of SimilarityEngine.find_similar()
        model_version  – string like "v1", "v2", …

    Returns a flat dict with every field displayed in the dashboard.
    """
    pred_class  = classification.get("predicted_class", "unknown")
    confidence  = classification.get("confidence", 0.0)
    category    = CLASS_CATEGORIES.get(pred_class.lower(), "Unknown")

    spec = {
        # ── Identity ─────────────────────────────────────────────────────
        "image_type":         pred_class.capitalize(),
        "category":           category,
        "confidence_pct":     confidence,
        "top_predictions":    classification.get("top_predictions", []),

        # ── Geometry ─────────────────────────────────────────────────────
        "image_width":        CIFAR10_IMG_SIZE,
        "image_height":       CIFAR10_IMG_SIZE,
        "channels":           3,
        "color_space":        "RGB",

        # ── Compression ───────────────────────────────────────────────────
        "latent_channels":    LATENT_CHANNELS,
        "latent_spatial":     LATENT_SPATIAL,
        "latent_dim":         LATENT_DIM,
        "original_dim":       ORIGINAL_DIM,
        "compression_ratio":  quality.get("compression_ratio", COMPRESSION_RATIO),
        "compression_str":    f"{quality.get('compression_ratio', COMPRESSION_RATIO):.1f}× (1:{quality.get('compression_ratio', COMPRESSION_RATIO):.1f})",

        # ── Quality ───────────────────────────────────────────────────────
        "mse":                quality.get("mse", 0.0),
        "psnr_db":            quality.get("psnr_db", 0.0),
        "ssim":               quality.get("ssim", 0.0),
        "quality_label":      quality.get("quality_label", "N/A"),

        # ── Anomaly ───────────────────────────────────────────────────────
        "anomaly_score":      anomaly.get("anomaly_score", 0.0),
        "anomaly_status":     anomaly.get("status", "UNKNOWN"),
        "is_anomaly":         anomaly.get("is_anomaly", False),

        # ── Similarity ────────────────────────────────────────────────────
        "similar_images":     similarity,

        # ── Meta ──────────────────────────────────────────────────────────
        "model_version":      model_version,
    }
    return spec


def format_specification_text(spec: dict) -> str:
    """
    Render the specification dict as a formatted text block for display.
    """
    cr = spec["compression_ratio"]
    lines = [
        "╔══════════════════════════════════════╗",
        "║        IMAGE SPECIFICATIONS          ║",
        "╠══════════════════════════════════════╣",
        f"║ Image Type       : {spec['image_type']:<18}║",
        f"║ Category         : {spec['category']:<18}║",
        f"║ Confidence       : {spec['confidence_pct']:<17.1f}%║",
        "║                                      ║",
        f"║ Dimensions       : {spec['image_width']} × {spec['image_height']:<12}     ║",
        f"║ Channels         : {spec['channels']:<18}║",
        f"║ Color Space      : {spec['color_space']:<18}║",
        "║                                      ║",
        f"║ Latent Dim       : {spec['latent_channels']} × {spec['latent_spatial']} × {spec['latent_spatial']:<7}    ║",
        f"║ Original Values  : {spec['original_dim']:<18}║",
        f"║ Latent Values    : {spec['latent_dim']:<18}║",
        "║                                      ║",
        f"║ Compression Ratio: {cr:.1f} : 1             ║",
        f"║ MSE              : {spec['mse']:<18.6f}║",
        f"║ PSNR             : {spec['psnr_db']:<14.2f} dB  ║",
        f"║ SSIM             : {spec['ssim']:<18.4f}║",
        f"║ Quality          : {spec['quality_label']:<18}║",
        "║                                      ║",
        f"║ Anomaly          : {spec['anomaly_status']:<18}║",
        f"║ Anomaly Score    : {spec['anomaly_score']:<18.4f}║",
        f"║ Model Version    : {spec['model_version']:<18}║",
        "╚══════════════════════════════════════╝",
    ]
    return "\n".join(lines)
