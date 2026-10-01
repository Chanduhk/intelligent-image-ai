"""
evaluate.py — Standalone evaluation script
Runs the full analysis pipeline on the test set and prints a summary report.

Usage:
    python evaluate.py [--data-dir ./data/raw] [--batch-size 128]
"""

import os
import sys
import argparse
import traceback
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from src.data.dataset import load_cifar10, CIFAR10_CLASSES
from src.models.autoencoder import build_autoencoder
from src.models.classifier  import LatentClassifier
from src.models.anomaly_detector import AnomalyDetector
from src.analysis.quality_metrics import evaluate_reconstruction
from src.analysis.similarity import SimilarityEngine
from src.versioning.model_manager import ModelManager


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   type=str, default="./data/raw")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-samples",  type=int, default=1000,
                   help="Number of test images to evaluate metrics over")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("   INTELLIGENT IMAGE AI — EVALUATION")
    print("=" * 60)
    print(f"Device           : {device}")
    sys.stdout.flush()

    # ── Model manager ────────────────────────────────────────────────────
    manager = ModelManager()

    # Auto-recover metadata if training completed but metadata was not written
    # (e.g. process was killed after model file was saved but before register_version)
    deployed_v = manager.get_deployed_version()
    if deployed_v is None:
        print("\n  [INFO] No deployed version in metadata.")
        recovered = manager.recover_deployed_from_disk()
        if recovered:
            deployed_v = manager.get_deployed_version()
            print(f"  [INFO] Auto-recovered deployed version from disk: v{deployed_v}")
        else:
            print("  ERROR: No trained models found.")
            print("  Please run:  python train.py")
            print("=" * 60)
            sys.stdout.flush()
            sys.exit(1)

    print(f"Deployed version : v{deployed_v}")
    sys.stdout.flush()

    # ── Load autoencoder ─────────────────────────────────────────────────
    model = build_autoencoder()
    ok_ae, ae_path = manager.load_best_autoencoder(model, device)
    if not ok_ae:
        print(f"\n  ERROR: Autoencoder not found or failed to load.")
        print(f"  Expected path : {ae_path}")
        print(f"  Please run    : python train.py")
        print("=" * 60)
        sys.stdout.flush()
        sys.exit(1)
    print(f"Autoencoder      : FOUND  ({ae_path})")
    sys.stdout.flush()

    # ── Load classifier ──────────────────────────────────────────────────
    classifier = LatentClassifier()
    ok_clf, clf_path = manager.load_best_classifier(classifier)
    if ok_clf:
        print(f"Classifier       : FOUND  ({clf_path})")
    else:
        print(f"Classifier       : NOT FOUND  ({clf_path})")
    sys.stdout.flush()

    # ── Load anomaly detector ────────────────────────────────────────────
    detector = AnomalyDetector()
    ok_anom, anom_path = manager.load_best_anomaly(detector)
    if ok_anom:
        print(f"Anomaly Model    : FOUND  ({anom_path})")
    else:
        print(f"Anomaly Model    : NOT FOUND  ({anom_path})")
    sys.stdout.flush()

    # ── Load similarity engine ───────────────────────────────────────────
    sim_engine = SimilarityEngine()
    try:
        sim_engine.load(manager.similarity_dir())
        if sim_engine.built:
            print(f"Similarity DB    : FOUND  ({manager.similarity_dir()})")
        else:
            print(f"Similarity DB    : NOT FOUND  ({manager.similarity_dir()})")
    except Exception as e:
        print(f"Similarity DB    : ERROR  ({e})")
    sys.stdout.flush()

    # ── Load test data ────────────────────────────────────────────────────
    print(f"\nLoading test data …")
    sys.stdout.flush()
    _, test_loader, _, _ = load_cifar10(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=0
    )

    # ── Evaluate reconstruction quality ───────────────────────────────────
    print(f"Evaluating reconstruction on up to {args.n_samples} samples …")
    sys.stdout.flush()
    model.eval()
    mse_list, psnr_list, ssim_list = [], [], []
    correct = total = 0
    count   = 0

    try:
        with torch.no_grad():
            for imgs, labels in test_loader:
                if count >= args.n_samples:
                    break
                imgs = imgs.to(device)
                z, recon = model(imgs)

                for i in range(len(imgs)):
                    if count >= args.n_samples:
                        break
                    metrics_i = evaluate_reconstruction(imgs[i].cpu(), recon[i].cpu())
                    mse_list.append(metrics_i["mse"])
                    psnr_val = metrics_i["psnr_db"]
                    psnr_list.append(psnr_val if psnr_val != float("inf") else 60.0)
                    ssim_list.append(metrics_i["ssim"])

                    if ok_clf and classifier.is_trained:
                        fv   = z[i].view(1, -1).cpu().numpy()
                        pred = classifier.predict(fv)
                        if pred["predicted_index"] == labels[i].item():
                            correct += 1
                        total += 1
                    count += 1

    except Exception as exc:
        print(f"\n  [ERROR] During evaluation loop: {exc}")
        traceback.print_exc()
        sys.stdout.flush()

    if not mse_list:
        print("\n  ERROR: No samples were evaluated.")
        sys.stdout.flush()
        sys.exit(1)

    mean_mse  = float(np.mean(mse_list))
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    clf_acc   = (correct / total * 100) if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("   EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Deployed version      : v{deployed_v}")
    print(f"  Samples evaluated     : {count}")
    print(f"  Mean MSE              : {mean_mse:.6f}")
    print(f"  Mean PSNR             : {mean_psnr:.2f} dB")
    print(f"  Mean SSIM             : {mean_ssim:.4f}")
    if total > 0:
        print(f"  Classifier Accuracy   : {clf_acc:.2f}%  ({correct}/{total})")
    else:
        print(f"  Classifier Accuracy   : N/A (classifier not loaded)")
    print(f"  Compression Ratio     : 3.0 : 1 (3072 -> 1024)")
    print()
    try:
        print(manager.summary())
    except Exception:
        pass
    sys.stdout.flush()

    # ── Save metrics distribution plots ──────────────────────────────────
    try:
        os.makedirs("outputs/graphs", exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(mse_list,  bins=50, color="steelblue", edgecolor="white")
        axes[0].set_title("MSE Distribution"); axes[0].set_xlabel("MSE")
        axes[1].hist(psnr_list, bins=50, color="darkorange", edgecolor="white")
        axes[1].set_title("PSNR Distribution (dB)"); axes[1].set_xlabel("PSNR (dB)")
        axes[2].hist(ssim_list, bins=50, color="seagreen", edgecolor="white")
        axes[2].set_title("SSIM Distribution"); axes[2].set_xlabel("SSIM")
        plt.suptitle("Reconstruction Quality Distribution", fontsize=13)
        plt.tight_layout()
        fig.savefig("outputs/graphs/metrics_distribution.png", dpi=150)
        plt.close(fig)
        print("  Metrics plot saved : outputs/graphs/metrics_distribution.png")
    except Exception as exc:
        print(f"  [WARN] Could not save plots: {exc}")

    print("\n  Evaluation completed successfully.")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
