"""
Reasoning / Explanation Engine
Generates evidence-based, human-readable explanations from actual
model outputs — no language model, no fabricated reasoning.
All explanations are built from measured values and rule templates.
"""

from src.models.autoencoder import LATENT_CHANNELS, LATENT_SPATIAL


# ── Thresholds used in explanation rules ─────────────────────────────────────
MSE_EXCELLENT   = 0.001
MSE_GOOD        = 0.005
MSE_FAIR        = 0.015
PSNR_GOOD       = 28.0
SSIM_GOOD       = 0.80
CONF_HIGH       = 80.0
CONF_MEDIUM     = 50.0


# ─────────────────────────────────────────────────────────────────────────────
def _explain_classification(pred_class: str, confidence: float,
                             top_predictions: list) -> str:
    if confidence >= CONF_HIGH:
        conf_text = "high"
    elif confidence >= CONF_MEDIUM:
        conf_text = "moderate"
    else:
        conf_text = "low"

    parts = [
        f"The uploaded image is classified as a {pred_class.upper()} "
        f"with {confidence:.1f}% confidence ({conf_text} certainty)."
    ]

    if len(top_predictions) >= 2:
        runner = top_predictions[1]
        parts.append(
            f"The next most likely class is '{runner['class']}' "
            f"with {runner['confidence']:.1f}% probability."
        )
    return " ".join(parts)


def _explain_compression(latent_dim: int, original_dim: int,
                          compression_ratio: float) -> str:
    return (
        f"The encoder compressed the original {original_dim}-value image "
        f"into a latent representation of {LATENT_CHANNELS} × {LATENT_SPATIAL} × {LATENT_SPATIAL} "
        f"= {latent_dim} values, achieving a {compression_ratio:.1f}× compression ratio."
    )


def _explain_quality(mse: float, psnr: float, ssim: float,
                     quality_label: str) -> str:
    if mse <= MSE_EXCELLENT:
        mse_text = "very low reconstruction error — almost perfect fidelity"
    elif mse <= MSE_GOOD:
        mse_text = "low reconstruction error — important visual features preserved"
    elif mse <= MSE_FAIR:
        mse_text = "moderate reconstruction error — general structure retained"
    else:
        mse_text = "noticeable reconstruction error — some fine detail lost"

    if psnr >= PSNR_GOOD:
        psnr_text = f"PSNR of {psnr:.1f} dB indicates good signal fidelity"
    else:
        psnr_text = f"PSNR of {psnr:.1f} dB indicates acceptable fidelity"

    ssim_text = (
        f"SSIM of {ssim:.3f} confirms that the structural content of the image "
        "is well preserved." if ssim >= SSIM_GOOD else
        f"SSIM of {ssim:.3f} suggests some structural detail was lost during compression."
    )

    return (
        f"Reconstruction quality is rated {quality_label.upper()}. "
        f"MSE = {mse:.6f} ({mse_text}). "
        f"{psnr_text}. {ssim_text}"
    )


def _explain_anomaly(anomaly_score: float, is_anomaly: bool,
                     pred_class: str) -> str:
    if is_anomaly:
        return (
            f"The latent representation of this image (score = {anomaly_score:.3f}) "
            "lies outside the learned normal-data distribution. "
            "The image may be unusual, out-of-distribution, or corrupted."
        )
    return (
        f"The image is classified as NORMAL (anomaly score = {anomaly_score:.3f}). "
        f"Its latent representation falls within the learned distribution of the reference data, "
        f"consistent with typical {pred_class} images."
    )


def _explain_similarity(similar_images: list, pred_class: str) -> str:
    if not similar_images:
        return "Similarity search database not available."
    top = similar_images[0]
    top3_classes = [s["class"] for s in similar_images[:3]]
    class_agreement = top3_classes.count(pred_class.lower())
    agreement_text = (
        f"{class_agreement} of the top 3 similar images are also {pred_class} images, "
        "supporting the classification."
        if class_agreement >= 2 else
        "The top similar images span multiple classes, suggesting the image has mixed features."
    )
    return (
        f"The closest match in latent space is a '{top['class']}' image "
        f"with {top['similarity']:.1f}% cosine similarity. "
        f"{agreement_text}"
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_explanation(spec: dict) -> str:
    """
    Generate a full, evidence-based explanation from the specification dict.

    Args:
        spec – output of build_specification() from specifications.py

    Returns:
        A multi-paragraph plain-text explanation string.
    """
    paragraphs = []

    # 1. Classification
    paragraphs.append(_explain_classification(
        spec["image_type"],
        spec["confidence_pct"],
        spec.get("top_predictions", []),
    ))

    # 2. Compression
    paragraphs.append(_explain_compression(
        spec["latent_dim"],
        spec["original_dim"],
        spec["compression_ratio"],
    ))

    # 3. Reconstruction quality
    paragraphs.append(_explain_quality(
        spec["mse"],
        spec["psnr_db"],
        spec["ssim"],
        spec["quality_label"],
    ))

    # 4. Anomaly
    paragraphs.append(_explain_anomaly(
        spec["anomaly_score"],
        spec["is_anomaly"],
        spec["image_type"],
    ))

    # 5. Similarity
    paragraphs.append(_explain_similarity(
        spec.get("similar_images", []),
        spec["image_type"],
    ))

    # 6. Closing
    paragraphs.append(
        f"This result was produced by model version {spec['model_version']}."
    )

    return "\n\n".join(paragraphs)
