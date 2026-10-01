"""
check.py  --  Deployment Validation Script
Intelligent Image Compression & Analysis System

Verifies every component before launching the Streamlit dashboard.
Run this once after training to confirm everything is ready.

Usage:
    python check.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

# ── Path setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

PASS  = "[OK]"
FAIL  = "[FAIL]"
WARN  = "[WARN]"
SKIP  = "[SKIP]"

_results: list[tuple[str, str, str]] = []   # (status, name, detail)


def _check(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    sym = {"[OK]": "OK", "[FAIL]": "FAIL", "[WARN]": "WARN", "[SKIP]": "SKIP"}.get(status, status)
    print(f"  [{sym:<4}] {name}" + (f"  --  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
#  1. Python environment
# ============================================================
section("1. Python Environment")

import platform
_check(PASS, "Python version", platform.python_version())
_check(PASS, "Platform", platform.system() + " " + platform.release())

# ── Package imports ──────────────────────────────────────────
_pkg_ok = True
for pkg, import_name in [
    ("torch",        "torch"),
    ("torchvision",  "torchvision"),
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("sklearn",      "sklearn"),
    ("skimage",      "skimage"),
    ("matplotlib",   "matplotlib"),
    ("streamlit",    "streamlit"),
    ("PIL",          "PIL"),
    ("joblib",       "joblib"),
    ("seaborn",      "seaborn"),
]:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        _check(PASS, f"Package: {pkg}", f"v{ver}")
    except ImportError as e:
        _check(FAIL, f"Package: {pkg}", str(e))
        _pkg_ok = False

if not _pkg_ok:
    print("\n  FATAL: Missing packages.  Run:  pip install -r requirements.txt")
    sys.exit(1)

import torch
import numpy as np

# ============================================================
#  2. Project module imports
# ============================================================
section("2. Project Module Imports")

try:
    from src.models.autoencoder import (
        build_autoencoder, CNNAutoencoder,
        LATENT_DIM, LATENT_CHANNELS, LATENT_SPATIAL, ORIGINAL_DIM,
    )
    _check(PASS, "src.models.autoencoder")
except Exception as e:
    _check(FAIL, "src.models.autoencoder", str(e))
    sys.exit(1)

try:
    from src.models.classifier import LatentClassifier
    _check(PASS, "src.models.classifier")
except Exception as e:
    _check(FAIL, "src.models.classifier", str(e))

try:
    from src.models.anomaly_detector import AnomalyDetector
    _check(PASS, "src.models.anomaly_detector")
except Exception as e:
    _check(FAIL, "src.models.anomaly_detector", str(e))

try:
    from src.data.dataset import load_cifar10, CIFAR10_CLASSES
    _check(PASS, "src.data.dataset", f"{len(CIFAR10_CLASSES)} classes")
except Exception as e:
    _check(FAIL, "src.data.dataset", str(e))

try:
    from src.data.preprocessing import preprocess_pil, tensor_to_numpy
    _check(PASS, "src.data.preprocessing")
except Exception as e:
    _check(FAIL, "src.data.preprocessing", str(e))

try:
    from src.analysis.quality_metrics import evaluate_reconstruction
    _check(PASS, "src.analysis.quality_metrics")
except Exception as e:
    _check(FAIL, "src.analysis.quality_metrics", str(e))

try:
    from src.analysis.similarity import SimilarityEngine
    _check(PASS, "src.analysis.similarity")
except Exception as e:
    _check(FAIL, "src.analysis.similarity", str(e))

try:
    from src.analysis.specifications import build_specification
    _check(PASS, "src.analysis.specifications")
except Exception as e:
    _check(FAIL, "src.analysis.specifications", str(e))

try:
    from src.explanation.reasoning_engine import generate_explanation
    _check(PASS, "src.explanation.reasoning_engine")
except Exception as e:
    _check(FAIL, "src.explanation.reasoning_engine", str(e))

try:
    from src.versioning.model_manager import ModelManager
    _check(PASS, "src.versioning.model_manager")
except Exception as e:
    _check(FAIL, "src.versioning.model_manager", str(e))
    sys.exit(1)

# ============================================================
#  3. Autoencoder architecture
# ============================================================
section("3. Autoencoder Architecture")

try:
    model = build_autoencoder()
    _check(PASS, "build_autoencoder()")
except Exception as e:
    _check(FAIL, "build_autoencoder()", str(e))
    sys.exit(1)

try:
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        z, recon = model(x)
    _check(PASS, "Forward pass", f"input {list(x.shape)}")
    if z.shape[-3:] == (64, 4, 4):
        _check(PASS, "Latent shape", f"{list(z.shape)}  (64 x 4 x 4 = {LATENT_DIM})")
    else:
        _check(FAIL, "Latent shape", f"Got {list(z.shape)}, expected (B,64,4,4)")

    if recon.shape == (1, 3, 32, 32):
        _check(PASS, "Reconstruction shape", f"{list(recon.shape)}")
    else:
        _check(FAIL, "Reconstruction shape", f"Got {list(recon.shape)}, expected (1,3,32,32)")

    flat = z.view(1, -1)
    if flat.shape[1] == LATENT_DIM:
        _check(PASS, "Flattened latent dim", f"{flat.shape[1]} == {LATENT_DIM}")
    else:
        _check(FAIL, "Flattened latent dim", f"{flat.shape[1]} != {LATENT_DIM}")

except Exception as e:
    _check(FAIL, "Architecture forward pass", str(e))
    traceback.print_exc()

# ============================================================
#  4. Model files on disk
# ============================================================
section("4. Model Files on Disk")

manager = ModelManager(
    autoencoder_dir=os.path.join(BASE_DIR, "models", "autoencoder"),
    classifier_dir =os.path.join(BASE_DIR, "models", "classifier"),
    anomaly_dir    =os.path.join(BASE_DIR, "models", "anomaly"),
    meta_dir       =os.path.join(BASE_DIR, "models", "metadata"),
    sim_dir        =os.path.join(BASE_DIR, "models", "similarity"),
)

# Auto-recover if needed
dv = manager.get_deployed_version()
if dv is None:
    _check(WARN, "metadata.json deployed_version", "None — attempting disk recovery")
    recovered = manager.recover_deployed_from_disk()
    dv = manager.get_deployed_version()
    if recovered and dv is not None:
        _check(PASS, "Auto-recovery", f"Recovered v{dv} from disk")
    else:
        _check(FAIL, "No deployed model found",
               "Run: python train.py")
else:
    _check(PASS, "metadata.json", f"deployed_version = v{dv}")

if dv is not None:
    # Autoencoder file
    ae_path = manager.autoencoder_path(dv)
    if os.path.exists(ae_path):
        size_kb = os.path.getsize(ae_path) / 1024
        _check(PASS, f"autoencoder_v{dv}.pth", f"{size_kb:.0f} KB  at  {ae_path}")
    else:
        _check(FAIL, f"autoencoder_v{dv}.pth", f"NOT FOUND at {ae_path}")

    # Classifier file
    clf_path = manager.classifier_path(dv)
    if os.path.exists(clf_path):
        size_kb = os.path.getsize(clf_path) / 1024
        _check(PASS, f"classifier_v{dv}.pkl", f"{size_kb:.0f} KB")
    else:
        _check(WARN, f"classifier_v{dv}.pkl", f"NOT FOUND — classification unavailable")

    # Anomaly file
    anom_path = manager.anomaly_path(dv)
    if os.path.exists(anom_path):
        size_kb = os.path.getsize(anom_path) / 1024
        _check(PASS, f"anomaly_v{dv}.pkl", f"{size_kb:.0f} KB")
    else:
        _check(WARN, f"anomaly_v{dv}.pkl", f"NOT FOUND — anomaly detection unavailable")

    # Similarity DB
    sim_vec  = os.path.join(manager.similarity_dir(), "sim_vectors.npy")
    sim_lbl  = os.path.join(manager.similarity_dir(), "sim_labels.json")
    if os.path.exists(sim_vec) and os.path.exists(sim_lbl):
        size_mb = os.path.getsize(sim_vec) / (1024 * 1024)
        _check(PASS, "similarity DB", f"sim_vectors.npy ({size_mb:.1f} MB)")
    else:
        _check(WARN, "similarity DB", "NOT FOUND — similarity search unavailable")

    # Checkpoint dir
    ckpt_dir = os.path.join(BASE_DIR, "models", "autoencoder", "checkpoints")
    if os.path.isdir(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        _check(PASS, "checkpoints/", f"{len(ckpts)} checkpoint files found")
    else:
        _check(WARN, "checkpoints/", "Directory missing — resume training unavailable")

# Version history
history = manager.version_history()
_check(PASS, "Version history", f"{len(history)} version(s) registered")
for v in history:
    m   = v.get("metrics", {})
    mse = m.get("val_mse", 9999)
    dep = " [DEPLOYED]" if v.get("deployed") else ""
    mse_str = f"MSE={mse:.5f}" if mse < 100 else "MSE=recovered"
    _check(PASS, f"  {v['tag']}{dep}", f"{mse_str}  epochs={m.get('epochs','?')}")

# ============================================================
#  5. Model loading test
# ============================================================
section("5. Model Loading Test")

device = torch.device("cpu")

if dv is not None:
    # Load autoencoder
    try:
        model_loaded = build_autoencoder()
        ok, path = manager.load_best_autoencoder(model_loaded, device)
        if ok:
            model_loaded.eval()
            _check(PASS, "Autoencoder load", path)
        else:
            _check(FAIL, "Autoencoder load", path)
    except Exception as e:
        _check(FAIL, "Autoencoder load", str(e))
        traceback.print_exc()

    # Load classifier
    try:
        clf = LatentClassifier()
        ok, path = manager.load_best_classifier(clf)
        if ok:
            _check(PASS, "Classifier load", f"is_trained={clf.is_trained}  {path}")
        else:
            _check(WARN, "Classifier load", f"NOT LOADED: {path}")
    except Exception as e:
        _check(WARN, "Classifier load", str(e))

    # Load anomaly detector
    try:
        det = AnomalyDetector()
        ok, path = manager.load_best_anomaly(det)
        if ok:
            _check(PASS, "Anomaly detector load", f"is_trained={det.is_trained}  {path}")
        else:
            _check(WARN, "Anomaly detector load", f"NOT LOADED: {path}")
    except Exception as e:
        _check(WARN, "Anomaly detector load", str(e))

    # Load similarity engine
    try:
        sim = SimilarityEngine()
        sim.load(manager.similarity_dir())
        if sim.built:
            _check(PASS, "Similarity DB load",
                   f"{len(sim.labels)} vectors, dim={sim.vectors.shape[1] if sim.vectors is not None else '?'}")
        else:
            _check(WARN, "Similarity DB load", "NOT LOADED")
    except Exception as e:
        _check(WARN, "Similarity DB load", str(e))

# ============================================================
#  6. Full inference pipeline with synthetic image
# ============================================================
section("6. Inference Pipeline Test (synthetic 32x32 image)")

try:
    from PIL import Image as _PIL
    from src.data.preprocessing import preprocess_pil as _prep, tensor_to_numpy as _t2n

    # Create a synthetic colourful image
    synth_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    synth_pil = _PIL.fromarray(synth_arr)

    # Preprocess
    t = _prep(synth_pil)
    assert t.shape == (3, 32, 32), f"Bad tensor shape: {t.shape}"
    _check(PASS, "preprocess_pil()", f"output shape {list(t.shape)}")

    # Autoencoder forward pass
    with torch.no_grad():
        z, recon = model_loaded(t.unsqueeze(0))
    assert z.shape[-3:] == (64, 4, 4)
    assert recon.shape == (1, 3, 32, 32)
    _check(PASS, "Autoencoder forward", f"latent {list(z.shape)}")

    # Latent vector
    latent_flat = z.view(1, -1).cpu().numpy()[0]
    assert latent_flat.shape == (1024,)
    assert not np.any(np.isnan(latent_flat)), "NaN in latent!"
    assert not np.any(np.isinf(latent_flat)), "Inf in latent!"
    _check(PASS, "Latent vector", f"shape={latent_flat.shape}, range=[{latent_flat.min():.3f}, {latent_flat.max():.3f}]")

    # Quality metrics
    metrics = evaluate_reconstruction(t, recon[0].cpu())
    assert "mse" in metrics and "psnr_db" in metrics and "ssim" in metrics
    _check(PASS, "Quality metrics",
           f"MSE={metrics['mse']:.5f}  PSNR={metrics['psnr_db']:.1f}dB  SSIM={metrics['ssim']:.4f}")

    # Classifier inference
    if clf.is_trained:
        pred = clf.predict(latent_flat)
        assert "predicted_class" in pred
        assert "confidence" in pred
        _check(PASS, "Classifier inference",
               f"class={pred['predicted_class']}  conf={pred['confidence']:.1f}%")
    else:
        _check(WARN, "Classifier inference", "Classifier not trained — skipped")

    # Anomaly inference
    if det.is_trained:
        anom = det.predict(latent_flat)
        assert "anomaly_score" in anom
        _check(PASS, "Anomaly inference",
               f"score={anom['anomaly_score']:.4f}  status={anom['status']}")
    else:
        _check(WARN, "Anomaly inference", "Anomaly detector not trained — skipped")

    # Similarity search
    if sim.built:
        results = sim.find_similar(latent_flat, top_k=5)
        assert len(results) > 0
        _check(PASS, "Similarity search", f"top-1: {results[0]['class']} @ {results[0]['similarity']:.1f}%")
    else:
        _check(WARN, "Similarity search", "Similarity DB not loaded — skipped")

    # Specification engine
    clf_out  = pred if clf.is_trained else {"predicted_class": "N/A", "confidence": 0.0,
                                             "predicted_index": 0, "top_predictions": []}
    anom_out = anom if det.is_trained else {"anomaly_score": 0.0, "is_anomaly": False, "status": "UNAVAILABLE"}
    spec = build_specification(clf_out, metrics, anom_out, results if sim.built else [], "v3")
    assert "image_type" in spec and "mse" in spec
    _check(PASS, "Specification engine", f"keys={len(spec)}")

    # Explanation engine
    expl = generate_explanation(spec)
    assert len(expl) > 50
    _check(PASS, "Explanation engine", f"{len(expl)} chars generated")

except Exception as e:
    _check(FAIL, "Inference pipeline", str(e))
    traceback.print_exc()

# ============================================================
#  7. CIFAR-10 real image test
# ============================================================
section("7. Real CIFAR-10 Image Test")

try:
    cifar_dir = os.path.join(BASE_DIR, "data", "raw")
    if not os.path.isdir(cifar_dir):
        _check(WARN, "CIFAR-10 data dir", f"Not found at {cifar_dir}")
    else:
        _, test_loader, _, _ = load_cifar10(
            data_dir=cifar_dir, batch_size=1, num_workers=0, augment_train=False
        )
        imgs, lbls = next(iter(test_loader))
        true_class = CIFAR10_CLASSES[lbls[0].item()]
        _check(PASS, "CIFAR-10 test loader", f"true_class={true_class}")

        with torch.no_grad():
            z, recon = model_loaded(imgs.to(device))
        latent_flat = z.view(1, -1).cpu().numpy()[0]
        metrics = evaluate_reconstruction(imgs[0].cpu(), recon[0].cpu())

        _check(PASS, "Real image inference",
               f"MSE={metrics['mse']:.5f}  PSNR={metrics['psnr_db']:.1f}dB  SSIM={metrics['ssim']:.4f}")

        if clf.is_trained:
            pred = clf.predict(latent_flat)
            correct = pred["predicted_class"] == true_class
            _check(
                PASS if correct else WARN,
                "Real image classification",
                f"predicted={pred['predicted_class']}  true={true_class}  "
                f"conf={pred['confidence']:.1f}%  {'CORRECT' if correct else 'wrong'}",
            )
        else:
            _check(WARN, "Real image classification", "Classifier not loaded")

except Exception as e:
    _check(WARN, "CIFAR-10 real image test", str(e))

# ============================================================
#  8. Image preprocessing — various formats
# ============================================================
section("8. Image Preprocessing")

try:
    from PIL import Image as _PIL
    from src.data.preprocessing import preprocess_pil as _prep

    for size, mode in [
        ((32, 32),   "RGB"),
        ((64, 64),   "RGB"),
        ((224, 224), "RGB"),
        ((100, 150), "L"),        # grayscale
        ((50, 50),   "RGBA"),     # with alpha
    ]:
        img = _PIL.new(mode, size)
        try:
            t = _prep(img)
            assert t.shape == (3, 32, 32), f"Expected (3,32,32) got {t.shape}"
            _check(PASS, f"preprocess_pil  {size} {mode}", f"-> {list(t.shape)}")
        except Exception as e:
            _check(FAIL, f"preprocess_pil  {size} {mode}", str(e))
except Exception as e:
    _check(FAIL, "Preprocessing suite", str(e))

# ============================================================
#  9. Streamlit import
# ============================================================
section("9. Streamlit Import")

try:
    import streamlit as st
    _check(PASS, "streamlit", f"v{st.__version__}")
except Exception as e:
    _check(FAIL, "streamlit", str(e))

try:
    import ast
    with open(os.path.join(BASE_DIR, "app.py"), encoding="utf-8") as f:
        src = f.read()
    ast.parse(src)
    _check(PASS, "app.py syntax", "No syntax errors")
except SyntaxError as e:
    _check(FAIL, "app.py syntax", str(e))
except Exception as e:
    _check(FAIL, "app.py read", str(e))

# ============================================================
#  Summary
# ============================================================
section("DEPLOYMENT CHECK SUMMARY")

fails  = [r for r in _results if r[0] == FAIL]
warns  = [r for r in _results if r[0] == WARN]
passes = [r for r in _results if r[0] == PASS]

print(f"\n  Total checks : {len(_results)}")
print(f"  Passed       : {len(passes)}")
print(f"  Warnings     : {len(warns)}")
print(f"  Failures     : {len(fails)}")

# Accuracy/single-image warnings are non-blocking
_non_blocking = {"real image classification", "classifier inference"}
blocking_warns = [
    r for r in warns
    if not any(nb in r[1].lower() for nb in _non_blocking)
]

if fails:
    print("\n  FAILED CHECKS:")
    for _, name, detail in fails:
        print(f"    [FAIL] {name}  --  {detail}")
    print(f"\n  STATUS: NOT READY FOR DEPLOYMENT")
    print(f"  Fix the failures above before running:  streamlit run app.py")
    sys.exit(1)
elif blocking_warns:
    print("\n  WARNINGS (some components unavailable):")
    for _, name, detail in warns:
        print(f"    [WARN] {name}  --  {detail}")
    print(f"\n  STATUS: PARTIAL — dashboard will run, some features unavailable.")
    print(f"  Run:  streamlit run app.py")
else:
    if warns:
        print("\n  NON-BLOCKING NOTES:")
        for _, name, detail in warns:
            print(f"    [NOTE] {name}  --  {detail}")
    print(f"\n  STATUS: FULLY READY FOR DEPLOYMENT")
    print(f"  All critical components verified successfully.")
    print(f"  Run:  streamlit run app.py")
