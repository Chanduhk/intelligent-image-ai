"""
train.py — Main training entry point
Run this script once to train all models in the correct order.

Usage:
    python train.py [--epochs 30] [--batch-size 128] [--data-dir ./data/raw]
    python train.py --resume          # resume autoencoder training from last checkpoint
"""

import os
import sys
import argparse
import traceback
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.data.dataset import load_cifar10
from src.models.autoencoder import build_autoencoder
from src.training.train_autoencoder import train_autoencoder, save_reconstruction_samples, CHECKPOINT_DIR
from src.training.train_classifier import (
    train_classifier, train_anomaly_detector, build_similarity_db
)
from src.versioning.model_manager import ModelManager


def parse_args():
    p = argparse.ArgumentParser(description="Train all models for the Intelligent Image AI System")
    p.add_argument("--epochs",      type=int,   default=30,           help="Autoencoder training epochs")
    p.add_argument("--batch-size",  type=int,   default=128,          help="Batch size")
    p.add_argument("--data-dir",    type=str,   default="./data/raw", help="CIFAR-10 data directory")
    p.add_argument("--sim-samples", type=int,   default=10000,        help="Max samples for similarity DB")
    p.add_argument("--no-gpu",      action="store_true",              help="Disable CUDA even if available")
    p.add_argument("--resume",      action="store_true",              help="Resume autoencoder training from last checkpoint")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda")

    print("\n" + "=" * 60)
    print("   INTELLIGENT IMAGE AI — FULL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Resume     : {args.resume}")
    sys.stdout.flush()

    # ── Step 1: Load CIFAR-10 ─────────────────────────────────────────────
    print("\n[STEP 1] Loading CIFAR-10 …")
    sys.stdout.flush()
    train_loader, val_loader, _, _ = load_cifar10(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        augment_train=True,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    sys.stdout.flush()

    # ── Step 2: Build model manager & version ─────────────────────────────
    manager = ModelManager()

    # If resuming, try to reuse the version from the latest checkpoint
    if args.resume:
        from src.training.train_autoencoder import _find_latest_checkpoint, CHECKPOINT_DIR
        import torch as _torch
        ckpt_path, ckpt_epoch = _find_latest_checkpoint(CHECKPOINT_DIR)
        if ckpt_path:
            _ckpt_data = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _ckpt_save_path = _ckpt_data.get("save_path", "")
            # Extract version from the save_path (e.g. autoencoder_v3.pth -> 3)
            import re as _re
            _m = _re.search(r"autoencoder_v(\d+)\.pth", _ckpt_save_path)
            if _m:
                new_version = int(_m.group(1))
                print(f"  [RESUME] Continuing version v{new_version} from checkpoint epoch {ckpt_epoch}")
            else:
                new_version = manager.next_version()
        else:
            print("  [RESUME] No checkpoint found - starting fresh.")
            new_version = manager.next_version()
    else:
        new_version = manager.next_version()

    ae_path  = manager.autoencoder_path(new_version)
    clf_path = manager.classifier_path(new_version)
    anom_path = manager.anomaly_path(new_version)

    print(f"\n[STEP 2] Model version : v{new_version}")
    sys.stdout.flush()

    # ── Step 3: Train Autoencoder ─────────────────────────────────────────
    print("\n[STEP 3] Training CNN Autoencoder …")
    sys.stdout.flush()
    model = build_autoencoder()

    try:
        history = train_autoencoder(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            epochs       = args.epochs,
            device       = device,
            save_path    = ae_path,
            resume       = args.resume,
        )
    except Exception as exc:
        print("\n" + "=" * 60)
        print("  AUTOENCODER TRAINING FAILED")
        print("=" * 60)
        print(f"  Exception: {exc}")
        traceback.print_exc()
        print("\n  To resume training from the last checkpoint, run:")
        print("    python train.py --resume")
        print("=" * 60)
        sys.stdout.flush()
        sys.exit(1)

    print(f"\n  Autoencoder training complete.")
    print(f"  Best Val MSE : {history['best_val_mse']:.6f} (epoch {history['best_epoch']})")
    print(f"  Model saved  : {ae_path}")
    sys.stdout.flush()

    # ── Step 4: Save reconstruction samples ──────────────────────────────
    print("\n[STEP 4] Saving reconstruction samples …")
    sys.stdout.flush()
    try:
        model.load(ae_path, device)
        save_reconstruction_samples(model, val_loader, device)
    except Exception as exc:
        print(f"  [WARN] Could not save reconstruction samples: {exc}")
    sys.stdout.flush()

    # ── Step 5: Train Random Forest Classifier ────────────────────────────
    print("\n[STEP 5] Training Random Forest Classifier …")
    sys.stdout.flush()
    try:
        clf_results = train_classifier(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            device       = device,
            save_path    = clf_path,
        )
        print(f"  Classifier saved : {clf_path}")
        print(f"  Accuracy         : {clf_results['accuracy']:.2f}%")
    except Exception as exc:
        print(f"  [ERROR] Classifier training failed: {exc}")
        traceback.print_exc()
        clf_results = {"accuracy": 0.0}
    sys.stdout.flush()

    # ── Step 6: Train Isolation Forest Anomaly Detector ──────────────────
    print("\n[STEP 6] Training Isolation Forest Anomaly Detector …")
    sys.stdout.flush()
    try:
        train_anomaly_detector(
            model        = model,
            train_loader = train_loader,
            device       = device,
            save_path    = anom_path,
        )
        print(f"  Anomaly model saved : {anom_path}")
    except Exception as exc:
        print(f"  [ERROR] Anomaly detector training failed: {exc}")
        traceback.print_exc()
    sys.stdout.flush()

    # ── Step 7: Build Similarity Database ────────────────────────────────
    print("\n[STEP 7] Building Similarity Database …")
    sys.stdout.flush()
    try:
        build_similarity_db(
            model        = model,
            train_loader = train_loader,
            device       = device,
            save_dir     = manager.similarity_dir(),
            max_samples  = args.sim_samples,
        )
        print(f"  Similarity DB saved : {manager.similarity_dir()}")
    except Exception as exc:
        print(f"  [ERROR] Similarity DB build failed: {exc}")
        traceback.print_exc()
    sys.stdout.flush()

    # ── Step 8: Register & deploy ─────────────────────────────────────────
    print("\n[STEP 8] Registering and deploying model …")
    sys.stdout.flush()
    metrics = {
        "val_mse":        history["best_val_mse"],
        "classifier_acc": clf_results.get("accuracy", 0.0),
        "epochs":         args.epochs,
    }
    try:
        manager.register_version(new_version, metrics,
                                  description="Initial training on CIFAR-10")
        deploy_ok, reason = manager.should_deploy(metrics)
        if deploy_ok:
            manager.deploy(new_version)
            print(f"  [OK] Model v{new_version} deployed - {reason}")
        else:
            print(f"  [--] Not deployed - {reason}")
    except Exception as exc:
        print(f"  [ERROR] Registration/deployment failed: {exc}")
        traceback.print_exc()
    sys.stdout.flush()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Val MSE          : {history['best_val_mse']:.6f}")
    print(f"  Classifier Acc   : {clf_results.get('accuracy', 0.0):.2f}%")
    deployed_v = manager.get_deployed_version()
    print(f"  Deployed version : v{deployed_v}" if deployed_v else "  Deployed version : (none)")
    print()
    try:
        print(manager.summary())
    except Exception:
        pass
    print()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
