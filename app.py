"""
app.py  --  Streamlit Dashboard
Intelligent Image Compression & Analysis System
AI-Based Image Compression and Intelligent Image Analysis using Autoencoders

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import gc
import io
import logging
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.analysis.quality_metrics import evaluate_reconstruction
from src.analysis.similarity import SimilarityEngine
from src.analysis.specifications import build_specification, format_specification_text
from src.data.dataset import CIFAR10_CLASSES, load_cifar10
from src.data.preprocessing import preprocess_pil, tensor_to_numpy
from src.explanation.reasoning_engine import generate_explanation
from src.models.anomaly_detector import AnomalyDetector
from src.models.autoencoder import LATENT_DIM, ORIGINAL_DIM, build_autoencoder
from src.models.classifier import LatentClassifier
from src.training.continuous_learning import ContinuousLearningEngine
from src.versioning.model_manager import ModelManager

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "logs", "app.log"),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Intelligent Image AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  .main-header {
      font-size: 1.9rem; font-weight: 700; color: #1e3a5f;
      margin-bottom: 0.1rem;
  }
  .sub-header {
      font-size: 0.95rem; color: #555; margin-bottom: 1.2rem;
  }
  .section-title {
      font-size: 1.05rem; font-weight: 600; color: #1e3a5f;
      border-bottom: 2px solid #2d6a9f;
      padding-bottom: 4px; margin-top: 1rem; margin-bottom: 0.5rem;
  }
  .status-ok   { color: #1a7a1a; font-weight: 700; }
  .status-warn { color: #cc7700; font-weight: 700; }
  .status-err  { color: #cc2222; font-weight: 700; }
  .normal-badge  { color: #1a7a1a; font-weight: 700; }
  .anomaly-badge { color: #cc2222; font-weight: 700; }
  .spec-box {
      font-family: 'Courier New', monospace;
      background: #1e1e2e; color: #cdd6f4;
      padding: 12px; border-radius: 6px;
      font-size: 0.80rem; white-space: pre; overflow-x: auto;
  }
  .explanation-box {
      background: #f8f9fa; border-radius: 8px;
      border-left: 4px solid #2d9a6a;
      padding: 14px 18px;
      font-size: 0.93rem; line-height: 1.7;
  }
  .warning-box {
      background: #fff8e1; border-radius: 8px;
      border-left: 4px solid #f0a000;
      padding: 12px 16px; font-size: 0.9rem;
  }
  .info-pill {
      display: inline-block; padding: 2px 10px;
      border-radius: 12px; font-size: 0.78rem;
      background: #e8f0fe; color: #1a56a0;
  }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
#  Cached model loader  (runs once per Streamlit session)
# =============================================================================
@st.cache_resource(show_spinner="Loading models — please wait...")
def load_all_models() -> dict:
    """
    Load every trained model artefact exactly once and cache in Streamlit's
    resource cache.  Returns a status dict so the UI can show what is
    available without crashing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading models for device=%s", device)

    manager = ModelManager(
        autoencoder_dir=os.path.join(BASE_DIR, "models", "autoencoder"),
        classifier_dir =os.path.join(BASE_DIR, "models", "classifier"),
        anomaly_dir    =os.path.join(BASE_DIR, "models", "anomaly"),
        meta_dir       =os.path.join(BASE_DIR, "models", "metadata"),
        sim_dir        =os.path.join(BASE_DIR, "models", "similarity"),
    )

    # ── Auto-recover metadata if training was interrupted ──────────────────
    if manager.get_deployed_version() is None:
        logger.warning("No deployed version in metadata — attempting disk recovery.")
        manager.recover_deployed_from_disk()

    # ── Autoencoder ────────────────────────────────────────────────────────
    model = build_autoencoder()
    ok_ae, ae_path = manager.load_best_autoencoder(model, device)
    if ok_ae:
        model.eval()
        logger.info("Autoencoder loaded: %s", ae_path)
    else:
        logger.warning("Autoencoder NOT loaded: %s", ae_path)

    # ── Classifier ────────────────────────────────────────────────────────
    classifier = LatentClassifier()
    ok_clf, clf_path = manager.load_best_classifier(classifier)
    if ok_clf:
        logger.info("Classifier loaded: %s", clf_path)
    else:
        logger.warning("Classifier NOT loaded: %s", clf_path)

    # ── Anomaly detector ──────────────────────────────────────────────────
    detector = AnomalyDetector()
    ok_anom, anom_path = manager.load_best_anomaly(detector)
    if ok_anom:
        logger.info("Anomaly detector loaded: %s", anom_path)
    else:
        logger.warning("Anomaly detector NOT loaded: %s", anom_path)

    # ── Similarity engine ─────────────────────────────────────────────────
    sim_engine = SimilarityEngine()
    try:
        sim_engine.load(manager.similarity_dir())
        ok_sim = sim_engine.built
        logger.info("Similarity DB loaded (built=%s)", ok_sim)
    except Exception as exc:
        ok_sim = False
        logger.warning("Similarity DB failed to load: %s", exc)

    # ── Version consistency check ──────────────────────────────────────────
    dv = manager.get_deployed_version()
    version_warning = None
    if dv is not None:
        if ok_clf:
            # Check that classifier filename contains the deployed version
            clf_fn = os.path.basename(clf_path)
            if f"v{dv}" not in clf_fn:
                version_warning = (
                    f"Classifier ({clf_fn}) may not match deployed "
                    f"autoencoder (v{dv})."
                )
                logger.warning(version_warning)

    return {
        "model":           model,
        "classifier":      classifier,
        "detector":        detector,
        "sim_engine":      sim_engine,
        "manager":         manager,
        "device":          device,
        "ok_ae":           ok_ae,
        "ok_clf":          ok_clf,
        "ok_anom":         ok_anom,
        "ok_sim":          ok_sim,
        "ae_path":         ae_path,
        "version_warning": version_warning,
    }


# =============================================================================
#  Core inference pipeline
# =============================================================================
@torch.no_grad()
def run_inference(pil_image: Image.Image, resources: dict,
                  original_size: tuple | None = None) -> dict:
    """
    Run the full analysis pipeline on one PIL image.
    Returns a result dict; never raises — errors are captured and returned
    as {"error": str} so the UI can show them gracefully.
    """
    try:
        model      = resources["model"]
        classifier = resources["classifier"]
        detector   = resources["detector"]
        sim_engine = resources["sim_engine"]
        manager    = resources["manager"]
        device     = resources["device"]

        # 1. Preprocess (resize to 32×32, normalise to [-1,1])
        img_tensor = preprocess_pil(pil_image).unsqueeze(0).to(device)

        # 2. Encode + Decode — with eval() and no_grad (already in decorator)
        model.eval()
        z, recon = model(img_tensor)

        # Verify latent shape
        assert z.shape[-3:] == (64, 4, 4), (
            f"Unexpected latent shape {z.shape}; expected (B,64,4,4)"
        )

        latent_flat = z.view(1, -1).cpu().numpy()[0]  # (1024,)
        recon_cpu   = recon[0].cpu()
        img_cpu     = img_tensor[0].cpu()

        # 3. Quality metrics
        quality = evaluate_reconstruction(img_cpu, recon_cpu)

        # 4. Classification
        if resources["ok_clf"] and classifier.is_trained:
            try:
                classification = classifier.predict(latent_flat)
            except Exception as exc:
                logger.warning("Classifier predict failed: %s", exc)
                classification = {
                    "predicted_class": "N/A", "confidence": 0.0,
                    "predicted_index": 0, "top_predictions": [],
                }
        else:
            classification = {
                "predicted_class": "N/A", "confidence": 0.0,
                "predicted_index": 0, "top_predictions": [],
            }

        # 5. Anomaly detection
        if resources["ok_anom"] and detector.is_trained:
            try:
                anomaly = detector.predict(latent_flat)
            except Exception as exc:
                logger.warning("Anomaly predict failed: %s", exc)
                anomaly = {"anomaly_score": 0.0, "is_anomaly": False,
                           "status": "UNAVAILABLE"}
        else:
            anomaly = {"anomaly_score": 0.0, "is_anomaly": False,
                       "status": "UNAVAILABLE"}

        # 6. Similarity search
        similar = (
            sim_engine.find_similar(latent_flat, top_k=5)
            if resources["ok_sim"] and sim_engine.built else []
        )

        # 7. Specification — pass original dimensions if available
        dv = manager.get_deployed_version()
        version_tag = manager.version_tag(dv if dv is not None else 1)
        spec = build_specification(classification, quality, anomaly, similar, version_tag)

        # Enrich spec with original upload dimensions
        if original_size:
            spec["upload_width"]  = original_size[0]
            spec["upload_height"] = original_size[1]

        # 8. Explanation
        explanation = generate_explanation(spec)

        # 9. Convert tensors to display numpy
        orig_np  = tensor_to_numpy(img_cpu)           # uint8 HxWxC at 32×32
        recon_np = (
            recon_cpu.clamp(0, 1).permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8)

        # Upscale both to 256×256 for display (nearest keeps blocky aesthetic)
        orig_display  = np.array(
            Image.fromarray(orig_np).resize((256, 256), Image.NEAREST)
        )
        recon_display = np.array(
            Image.fromarray(recon_np).resize((256, 256), Image.NEAREST)
        )

        # Free tensors
        del z, recon, img_tensor
        gc.collect()

        return {
            "orig_np":        orig_np,
            "recon_np":       recon_np,
            "orig_display":   orig_display,
            "recon_display":  recon_display,
            "img_tensor":     img_cpu,
            "latent_flat":    latent_flat,
            "classification": classification,
            "quality":        quality,
            "anomaly":        anomaly,
            "similar":        similar,
            "spec":           spec,
            "explanation":    explanation,
        }

    except Exception as exc:
        err_msg = str(exc)
        logger.error("Inference error: %s\n%s", err_msg, traceback.format_exc())
        return {"error": err_msg}


# =============================================================================
#  Matplotlib / figure helpers
# =============================================================================
def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    pil = Image.open(buf).copy()
    plt.close(fig)
    return pil


def make_comparison_figure(
    orig: np.ndarray, recon: np.ndarray,
    pred_class: str, confidence: float,
) -> Image.Image:
    """Side-by-side original vs reconstructed figure (upscaled for display)."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.patch.set_facecolor("#111827")
    for ax, img, title, border in zip(
        axes,
        [orig, recon],
        ["Original", "Reconstructed"],
        ["#4b9cd3", "#e07a3a"],
    ):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2)
    fig.suptitle(
        f"Predicted: {pred_class.upper()}   ({confidence:.1f}%)",
        color="#aaddff", fontsize=11, y=0.97,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig_to_pil(fig)


def make_confidence_chart(top_preds: list) -> Image.Image:
    classes = [p["class"].capitalize() for p in top_preds]
    confs   = [p["confidence"] for p in top_preds]
    colors  = ["#1a56a0"] + ["#7eaadb"] * (len(classes) - 1)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    bars = ax.barh(classes[::-1], confs[::-1], color=colors[::-1], height=0.55)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.set_xlim(0, 110)
    for bar, conf in zip(bars, confs[::-1]):
        ax.text(
            bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{conf:.1f}%", va="center", fontsize=9,
        )
    ax.set_title("Top Predictions", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _fig_to_pil(fig)


def make_latent_heatmap(latent_flat: np.ndarray) -> Image.Image:
    """Show latent (1024,) as a (64, 16) heat grid."""
    grid = latent_flat.reshape(64, 16)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_title(
        "Latent Representation — 64 channels x 16 spatial positions (4x4 flattened)",
        fontsize=9,
    )
    ax.set_xlabel("Spatial position (0-15)")
    ax.set_ylabel("Latent channel (0-63)")
    plt.tight_layout()
    return _fig_to_pil(fig)


def _mse_chart(history: list) -> Image.Image | None:
    """Build version MSE chart; skip recovered/placeholder versions."""
    valid = [
        v for v in history
        if v.get("metrics", {}).get("val_mse", 9999) < 100
    ]
    if len(valid) < 1:
        return None
    tags = [v["tag"] for v in valid]
    mses = [v["metrics"]["val_mse"] for v in valid]
    fig, ax = plt.subplots(figsize=(max(5, len(tags) * 1.2), 3))
    ax.plot(tags, mses, marker="o", linewidth=2, color="#1a56a0", markersize=8)
    ax.fill_between(range(len(tags)), mses, alpha=0.15, color="#1a56a0")
    ax.set_xlabel("Model Version")
    ax.set_ylabel("Validation MSE")
    ax.set_title("MSE per Model Version (lower is better)")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _fig_to_pil(fig)


# =============================================================================
#  Sidebar
# =============================================================================
def render_sidebar(resources: dict) -> None:
    manager = resources["manager"]
    dv      = manager.get_deployed_version()
    ok_all  = all([resources["ok_ae"], resources["ok_clf"],
                   resources["ok_anom"], resources["ok_sim"]])

    st.sidebar.markdown("## 🧠 System Status")
    if ok_all:
        st.sidebar.success("**READY** — All components loaded")
    elif resources["ok_ae"]:
        st.sidebar.warning("**PARTIAL** — Some components missing")
    else:
        st.sidebar.error("**NOT READY** — No trained model found")

    st.sidebar.divider()

    # Component status
    st.sidebar.markdown("### Components")
    def _icon(ok: bool) -> str:
        return "✅" if ok else "❌"

    st.sidebar.markdown(
        f"{_icon(resources['ok_ae'])} Autoencoder (CNN)\n\n"
        f"{_icon(resources['ok_clf'])} Classifier (Random Forest)\n\n"
        f"{_icon(resources['ok_anom'])} Anomaly Detector (Isolation Forest)\n\n"
        f"{_icon(resources['ok_sim'])} Similarity Database"
    )

    # Version warning
    if resources.get("version_warning"):
        st.sidebar.warning(resources["version_warning"])

    st.sidebar.divider()

    # Model info
    st.sidebar.markdown("### Model Info")
    st.sidebar.metric("Deployed Version", f"v{dv}" if dv else "None")
    st.sidebar.metric("Latent Dim",      f"{LATENT_DIM:,} values")
    st.sidebar.metric("Compression",     "3.0x (3072 → 1024)")
    st.sidebar.metric("Dataset",         "CIFAR-10")

    st.sidebar.divider()

    # CIFAR-10 classes
    st.sidebar.markdown("### CIFAR-10 Classes")
    for cls in CIFAR10_CLASSES:
        st.sidebar.caption(f"• {cls.capitalize()}")

    st.sidebar.divider()

    # Version history
    st.sidebar.markdown("### Version History")
    history = manager.version_history()
    if history:
        for v in reversed(history[-5:]):
            m   = v.get("metrics", {})
            mse = m.get("val_mse", 9999)
            dep = " ✓" if v.get("deployed") else ""
            mse_str = f"{mse:.5f}" if mse < 100 else "recovered"
            st.sidebar.caption(f"**{v['tag']}**{dep} — MSE: {mse_str}")
    else:
        st.sidebar.caption("No versions registered.")

    st.sidebar.divider()
    st.sidebar.markdown(
        "<small>Intelligent Image Compression & Analysis System<br>"
        "AI-Based using Autoencoders</small>",
        unsafe_allow_html=True,
    )


# =============================================================================
#  Section renderers (keep main() readable)
# =============================================================================
def _render_system_status(resources: dict) -> None:
    """Compact status banner at the top of main content."""
    dv = resources["manager"].get_deployed_version()
    ok_ae   = resources["ok_ae"]
    ok_clf  = resources["ok_clf"]
    ok_anom = resources["ok_anom"]
    ok_sim  = resources["ok_sim"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Version",         f"v{dv}" if dv else "—")
    c2.metric("Autoencoder",     "FOUND" if ok_ae   else "MISSING")
    c3.metric("Classifier",      "FOUND" if ok_clf  else "MISSING")
    c4.metric("Anomaly Model",   "FOUND" if ok_anom else "MISSING")
    c5.metric("Similarity DB",   "FOUND" if ok_sim  else "MISSING")


def _render_classification(cls: dict) -> None:
    st.markdown('<div class="section-title">Classification</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    pred = cls["predicted_class"]
    conf = cls["confidence"]
    c1.metric("Predicted Class",
              pred.capitalize() if pred != "N/A" else "N/A")
    c2.metric("Confidence", f"{conf:.1f}%" if conf > 0 else "N/A")
    if pred != "N/A":
        from src.data.dataset import CLASS_CATEGORIES
        cat = CLASS_CATEGORIES.get(pred.lower(), "Unknown")
        c3.metric("Category", cat)
    else:
        c3.metric("Category", "—")

    tops = cls.get("top_predictions", [])
    if tops:
        with st.expander("Top-3 Predictions"):
            for i, p in enumerate(tops[:3], 1):
                st.progress(
                    int(p["confidence"]) if p["confidence"] <= 100 else 100,
                    text=f"{i}. {p['class'].capitalize()}  —  {p['confidence']:.1f}%",
                )


def _render_quality(qual: dict) -> None:
    st.markdown('<div class="section-title">Reconstruction Quality</div>',
                unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Compression",  "3.0x")
    q2.metric("MSE",          f"{qual['mse']:.5f}")
    q3.metric("PSNR",         f"{qual['psnr_db']:.1f} dB")
    q4.metric("SSIM",         f"{qual['ssim']:.4f}")
    st.caption(
        f"Quality Rating: **{qual['quality_label']}**  |  "
        f"Original: {ORIGINAL_DIM:,} values → Latent: {LATENT_DIM:,} values"
    )


def _render_anomaly(anom: dict) -> None:
    st.markdown('<div class="section-title">Anomaly Detection</div>',
                unsafe_allow_html=True)
    if anom.get("status") == "UNAVAILABLE":
        st.caption("Anomaly detector not available.")
        return
    a1, a2 = st.columns(2)
    score = anom.get("anomaly_score", 0.0)
    a1.metric("Anomaly Score", f"{score:.4f}")
    is_anom = anom.get("is_anomaly", False)
    badge_html = (
        '<span class="anomaly-badge">⚠ ANOMALOUS</span>'
        if is_anom else
        '<span class="normal-badge">✓ NORMAL</span>'
    )
    a2.markdown(f"**Status:** {badge_html}", unsafe_allow_html=True)
    st.progress(
        min(int(score * 100), 100),
        text=f"Anomaly score: {score:.3f}  (threshold: 0.55)",
    )


def _render_similarity(sim: list) -> None:
    if not sim:
        st.caption("Similarity database not available.")
        return
    st.markdown('<div class="section-title">Top-5 Similar Images (Latent Space)</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(sim))
    for col, s in zip(cols, sim):
        col.metric(
            f"#{s['rank']} {s['class'].capitalize()}",
            f"{s['similarity']:.1f}%",
        )


def _render_model_info(manager) -> None:
    st.markdown('<div class="section-title">Model Architecture</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
| Component | Type | Details |
|-----------|------|---------|
| Encoder | Conv2D x3 + BN + ReLU + MaxPool | 3x32x32 → 64x4x4 |
| Decoder | ConvTranspose2D x3 + BN + ReLU | 64x4x4 → 3x32x32 |
| Classifier | Random Forest (300 trees) | Latent (1024) → 10 classes |
| Anomaly | Isolation Forest (200 trees) | Latent (1024) → score |
| Similarity | Cosine Similarity | Latent (1024) → top-k |
| Metrics | MSE / PSNR / SSIM | scikit-image |
| Learning | Incremental fine-tuning | Buffer + replay |
| Versioning | Metadata JSON | Validated deployment |
"""
    )

    dv = manager.get_deployed_version()
    cols = st.columns(4)
    cols[0].metric("Input",    "3 x 32 x 32")
    cols[1].metric("Latent",   "64 x 4 x 4")
    cols[2].metric("Flat Dim", "1,024 values")
    cols[3].metric("Ratio",    "3.0x")


# =============================================================================
#  Main dashboard
# =============================================================================
def main() -> None:
    resources = load_all_models()
    render_sidebar(resources)
    manager   = resources["manager"]

    # ── Page header ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-header">🧠 Intelligent Image Compression &amp; Analysis System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">AI-Based Image Compression and Intelligent Image Analysis '
        'using Autoencoders &nbsp;|&nbsp; '
        'CNN Autoencoder · Random Forest · Isolation Forest · Cosine Similarity</div>',
        unsafe_allow_html=True,
    )

    # ── System status banner ───────────────────────────────────────────────
    _render_system_status(resources)
    st.divider()

    # ── No model warning ──────────────────────────────────────────────────
    if not resources["ok_ae"]:
        st.error(
            "**No trained autoencoder found.**  "
            "Run `python train.py` to train the system first."
        )
        with st.expander("Quick start instructions"):
            st.code(
                "# 1. Install dependencies\n"
                "pip install -r requirements.txt\n\n"
                "# 2. Train (approx 90 min on CPU for 30 epochs)\n"
                "python train.py --epochs 30\n\n"
                "# 3. Evaluate\n"
                "python evaluate.py\n\n"
                "# 4. Launch dashboard\n"
                "streamlit run app.py",
                language="bash",
            )
        return

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_analyze, tab_learn, tab_versions, tab_about = st.tabs([
        "🔍  Analyze Image",
        "📚  Continuous Learning",
        "📊  Model Versions",
        "ℹ️  About",
    ])

    # =========================================================================
    #  TAB 1: ANALYZE IMAGE
    # =========================================================================
    with tab_analyze:
        col_upload, col_results = st.columns([1, 2])

        with col_upload:
            st.markdown('<div class="section-title">Upload Image</div>',
                        unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "PNG / JPG / JPEG",
                type=["png", "jpg", "jpeg", "webp"],
                key="uploader",
                help=(
                    "Upload any image. It will be resized to 32×32 for the model. "
                    "For best results use CIFAR-10-style images (32×32 objects on "
                    "simple backgrounds)."
                ),
            )
            use_demo = st.checkbox(
                "Use a random CIFAR-10 test image",
                help="Loads one image from the downloaded CIFAR-10 test set.",
            )
            if not resources["ok_clf"]:
                st.info(
                    "Classifier not available — classification will show N/A.  "
                    "Run `python train.py` to train all components."
                )

        # ── Source image ──────────────────────────────────────────────────
        pil_image     = None
        original_size = None
        demo_label    = None

        if use_demo:
            with st.spinner("Loading CIFAR-10 test image..."):
                try:
                    _, test_loader, _, _ = load_cifar10(
                        data_dir=os.path.join(BASE_DIR, "data", "raw"),
                        batch_size=1, num_workers=0, augment_train=False,
                    )
                    from src.data.preprocessing import tensor_to_pil
                    imgs, lbls = next(iter(test_loader))
                    pil_image     = tensor_to_pil(imgs[0])   # already 32×32
                    original_size = (32, 32)
                    demo_label    = CIFAR10_CLASSES[lbls[0].item()]
                    with col_upload:
                        st.caption(f"Demo image — true label: **{demo_label}**")
                except Exception as exc:
                    with col_upload:
                        st.error(f"Could not load demo image: {exc}")
                    logger.exception("Demo image load failed")

        elif uploaded is not None:
            try:
                raw_img       = Image.open(uploaded)
                original_size = raw_img.size          # (W, H) before resize
                pil_image     = raw_img.convert("RGB")
                with col_upload:
                    if original_size != (32, 32):
                        st.caption(
                            f"Original: {original_size[0]} x {original_size[1]} px — "
                            "will be resized to 32 x 32 for inference."
                        )
                    # Show the original full-size image preview
                    st.image(pil_image, caption="Uploaded image", use_container_width=True)
            except Exception as exc:
                with col_upload:
                    st.error(f"Could not read image: {exc}")
                logger.exception("Image open failed")

        # ── Run inference ─────────────────────────────────────────────────
        if pil_image is not None:
            with st.spinner("Running AI analysis..."):
                result = run_inference(pil_image, resources, original_size)

            if "error" in result:
                st.error(f"Analysis failed: {result['error']}")
                st.caption("Check `logs/app.log` for details.")
            else:
                spec = result["spec"]
                cls  = result["classification"]
                qual = result["quality"]
                anom = result["anomaly"]
                sim  = result["similar"]

                # ── Image comparison ──────────────────────────────────────
                with col_upload:
                    st.markdown('<div class="section-title">Comparison</div>',
                                unsafe_allow_html=True)
                    pred_cls = cls["predicted_class"]
                    pred_conf = cls["confidence"]
                    comp = make_comparison_figure(
                        result["orig_display"], result["recon_display"],
                        pred_cls, pred_conf,
                    )
                    st.image(comp, use_container_width=True)
                    st.caption(
                        "Left: original (32×32). "
                        "Right: reconstructed from 1,024 latent values."
                    )
                    if cls.get("top_predictions"):
                        cconf = make_confidence_chart(cls["top_predictions"])
                        st.image(cconf, use_container_width=True)

                # ── Results panel ─────────────────────────────────────────
                with col_results:
                    _render_classification(cls)
                    _render_quality(qual)
                    _render_anomaly(anom)
                    _render_similarity(sim)

                    with st.expander("📋 Full Image Specification"):
                        spec_text = format_specification_text(spec)
                        st.markdown(
                            f'<div class="spec-box">{spec_text}</div>',
                            unsafe_allow_html=True,
                        )

                    with st.expander("🔬 Latent Representation (64 × 4 × 4)"):
                        lh = make_latent_heatmap(result["latent_flat"])
                        st.image(lh, use_container_width=True)

                # ── Explanation (full width below columns) ─────────────────
                st.markdown('<div class="section-title">AI Explanation</div>',
                            unsafe_allow_html=True)
                expl_html = result["explanation"].replace("\n\n", "<br><br>")
                st.markdown(
                    f'<div class="explanation-box">{expl_html}</div>',
                    unsafe_allow_html=True,
                )

                # ── Upload-dimension note ──────────────────────────────────
                if original_size and original_size != (32, 32):
                    st.info(
                        f"**Note:** This model was trained on CIFAR-10 (32×32 images). "
                        f"Your uploaded image ({original_size[0]}×{original_size[1]}) "
                        f"was resized to 32×32 for inference. "
                        f"Predictions are most accurate for CIFAR-10-style objects."
                    )

                # ── Continuous learning buffer ────────────────────────────
                st.markdown('<div class="section-title">Continuous Learning</div>',
                            unsafe_allow_html=True)
                cl_col1, cl_col2 = st.columns([1, 2])
                with cl_col1:
                    if st.button("➕  Add to Learning Buffer"):
                        try:
                            if "learning_engine" not in st.session_state:
                                st.session_state.learning_engine = ContinuousLearningEngine(
                                    model_manager=manager,
                                    device=resources["device"],
                                    data_dir=os.path.join(BASE_DIR, "data", "raw"),
                                    buffer_dir=os.path.join(BASE_DIR, "data", "learning_buffer"),
                                    verbose=False,
                                )
                            engine  = st.session_state.learning_engine
                            lbl_idx = cls.get("predicted_index", 0)
                            decision = engine.evaluate_and_buffer(
                                result["img_tensor"], lbl_idx,
                                qual["mse"], cls["confidence"],
                            )
                            if decision["added_to_buffer"]:
                                st.success(
                                    f"Sample added to learning buffer "
                                    f"(size: {decision['buffer_size']})"
                                )
                            else:
                                st.info(
                                    f"Sample not added — classified as easy.  "
                                    f"Buffer: {decision['buffer_size']}"
                                )
                        except Exception as exc:
                            st.error(f"Buffer error: {exc}")
                            logger.exception("Learning buffer error")

                with cl_col2:
                    st.caption(
                        "Difficult samples (high MSE or low confidence) are saved to the "
                        "learning buffer. Use the **Continuous Learning** tab to retrain."
                    )
        else:
            st.info(
                "⬆️  Upload an image **or** tick 'Use a random CIFAR-10 test image' "
                "to begin analysis."
            )

    # =========================================================================
    #  TAB 2: CONTINUOUS LEARNING
    # =========================================================================
    with tab_learn:
        st.markdown("## Continuous Learning Engine")
        st.markdown(
            "Difficult samples accumulate in the learning buffer. "
            "When the buffer reaches the trigger threshold you can run "
            "incremental fine-tuning. The new model is validated before "
            "deployment — the current deployed model is never replaced blindly."
        )

        if "learning_engine" not in st.session_state:
            st.session_state.learning_engine = ContinuousLearningEngine(
                model_manager=manager,
                device=resources["device"],
                data_dir=os.path.join(BASE_DIR, "data", "raw"),
                buffer_dir=os.path.join(BASE_DIR, "data", "learning_buffer"),
                verbose=False,
            )
        engine = st.session_state.learning_engine
        stats  = engine.get_stats()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Samples Received",  stats["total_received"])
        m2.metric("In Buffer",         stats["buffer_size"])
        m3.metric("Trigger Threshold", stats["trigger_threshold"])
        m4.metric("Trainings Run",     stats["trainings_run"])

        st.divider()
        progress_val = min(
            stats["buffer_size"] / max(stats["trigger_threshold"], 1), 1.0
        )
        st.caption(
            f"Buffer fill: **{stats['buffer_size']}** / **{stats['trigger_threshold']}** "
            f"({'ready to train!' if progress_val >= 1.0 else 'accumulating...'})"
        )
        st.progress(progress_val)

        if st.button("Run Incremental Training Now", type="primary",
                     disabled=(stats["buffer_size"] == 0)):
            if stats["buffer_size"] == 0:
                st.warning("Learning buffer is empty. Analyse some images first.")
            else:
                with st.spinner(
                    "Running incremental training — this may take several minutes..."
                ):
                    try:
                        _, test_loader, _, _ = load_cifar10(
                            data_dir=os.path.join(BASE_DIR, "data", "raw"),
                            batch_size=128, num_workers=0, augment_train=False,
                        )
                        cl_result = engine.run_incremental_training(
                            resources["model"], test_loader
                        )
                        if cl_result["deployed"]:
                            st.success(
                                f"Model v{cl_result['new_version']} deployed!\n\n"
                                f"{cl_result['reason']}"
                            )
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.warning(
                                f"Candidate model was NOT deployed: {cl_result['reason']}"
                            )
                    except Exception as exc:
                        st.error(f"Training failed: {exc}")
                        logger.exception("Incremental training error")

        st.divider()
        with st.expander("How the deployment gate works"):
            st.markdown(
                """
**Before deploying a new model, the system checks:**

1. Autoencoder file exists and loads correctly
2. Forward pass produces correct latent shape (64×4×4)
3. Validation MSE is lower than the current deployed model
4. No NaN/Inf values in loss

If the candidate does not beat the current model, it is discarded and
the deployed version remains unchanged.
"""
            )

    # =========================================================================
    #  TAB 3: MODEL VERSIONS
    # =========================================================================
    with tab_versions:
        st.markdown("## Model Version History")

        history = manager.version_history()
        if not history:
            st.info("No versions registered. Train the system first.")
        else:
            import pandas as pd

            rows = []
            for v in history:
                m   = v.get("metrics", {})
                mse = m.get("val_mse", 9999)
                rows.append({
                    "Version":       v["tag"],
                    "Val MSE":       f"{mse:.6f}" if mse < 100 else "recovered",
                    "Classifier %":  f"{m.get('classifier_acc', 0):.2f}",
                    "Epochs":        str(m.get("epochs", "—")),
                    "Deployed":      "✓" if v.get("deployed") else "",
                    "Timestamp":     v["timestamp"][:19],
                    "Description":   v.get("description", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            chart = _mse_chart(history)
            if chart:
                st.markdown("### Validation MSE over Versions")
                st.image(chart, use_container_width=True)

        st.divider()
        _render_model_info(manager)

    # =========================================================================
    #  TAB 4: ABOUT
    # =========================================================================
    with tab_about:
        st.markdown("## About This System")
        st.markdown(
            """
### Intelligent Image Compression & Analysis System
**AI-Based Image Compression and Intelligent Image Analysis using Autoencoders**

This project demonstrates how a single **CNN Autoencoder** acts as the
central feature-extraction layer for multiple intelligent analysis tasks:

- **Compression:** 3,072 values → 1,024 latent values (3.0× ratio)
- **Reconstruction:** CNN decoder reproduces the original from the latent code
- **Classification:** Random Forest on the 1,024-dimensional latent vector
- **Anomaly Detection:** Isolation Forest detects out-of-distribution images
- **Similarity Search:** Cosine similarity over reference latent vectors
- **Continuous Learning:** Difficult samples drive incremental retraining

#### Architecture

```
CIFAR-10  →  Preprocessing  →  CNN Encoder  →  Latent (64×4×4)
                                                    │
              ┌─────────────────────────────────────┤
              ↓                    ↓                ↓
         CNN Decoder         Random Forest   Isolation Forest
         Reconstruction      Classification  Anomaly Detection
              ↓                    ↓                ↓
        MSE / PSNR / SSIM   Class + Conf      Normal/Anomalous
              └─────────────────────────────────────┤
                                                    ↓
                                         Specification Engine
                                                    ↓
                                         Explanation Engine
                                                    ↓
                                        Continuous Learning
```

#### Training Commands

```bash
# Full 30-epoch training (recommended, ~90 min on CPU)
python train.py --epochs 30

# Resume if interrupted
python train.py --resume --epochs 30

# Evaluate deployed model
python evaluate.py

# Run this dashboard
streamlit run app.py
```

#### Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.x |
| Dataset | CIFAR-10 (60,000 images, 10 classes) |
| Classifier | scikit-learn RandomForestClassifier |
| Anomaly | scikit-learn IsolationForest |
| Metrics | scikit-image (SSIM) |
| Dashboard | Streamlit |
| Visualisation | Matplotlib |

#### Important Notes

- The model was trained on **CIFAR-10 images (32×32 px)**. Uploaded images
  of other sizes are automatically resized.
- Classification accuracy is highest for CIFAR-10-style images.
- Reconstruction quality metrics (MSE, PSNR, SSIM) are computed in the
  32×32 normalised space.
"""
        )


if __name__ == "__main__":
    main()
