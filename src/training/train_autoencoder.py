"""
Autoencoder Training Module
Full training loop with:
  - configurable epochs / batch size
  - MSE loss + Adam optimiser
  - validation after every epoch
  - learning-rate scheduling
  - per-epoch checkpointing to survive OOM/process kills
  - resume-from-checkpoint support
  - memory-efficient: explicit garbage collection, no accumulated tensors
  - loss history saved to outputs/graphs/
  - best-model checkpoint
"""

import os
import sys
import gc
import json
import time
import traceback
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.autoencoder import CNNAutoencoder, build_autoencoder
from src.data.dataset import load_cifar10


# ── Default hyper-parameters ──────────────────────────────────────────────────
DEFAULT_EPOCHS       = 30
DEFAULT_BATCH_SIZE   = 128
DEFAULT_LR           = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
GRAPHS_DIR           = "outputs/graphs"
RECON_DIR            = "outputs/reconstructions"
CHECKPOINT_DIR       = "models/autoencoder/checkpoints"
CHECKPOINT_INTERVAL  = 1     # save a checkpoint every N epochs


# ── MSE loss on Sigmoid output ────────────────────────────────────────────────
def _target_from_input(imgs: torch.Tensor) -> torch.Tensor:
    """
    The decoder outputs values in [0, 1] (Sigmoid).
    The input images are normalised to [-1, 1].
    We un-normalise the inputs to [0, 1] so the MSE target matches.
    """
    return (imgs * 0.5 + 0.5).clamp(0, 1)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_path(checkpoint_dir: str, epoch: int) -> str:
    return os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")


def _save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: CNNAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loss: float,
    val_loss: float,
    best_val_mse: float,
    best_epoch: int,
    train_losses: list,
    val_losses: list,
    save_path: str,
):
    """Save a full resumable checkpoint for the given epoch."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_loss":      train_loss,
        "val_loss":        val_loss,
        "best_val_mse":    best_val_mse,
        "best_epoch":      best_epoch,
        "train_losses":    train_losses,
        "val_losses":      val_losses,
        "save_path":       save_path,
    }
    path = _checkpoint_path(checkpoint_dir, epoch)
    torch.save(ckpt, path)
    return path


def _find_latest_checkpoint(checkpoint_dir: str):
    """
    Scan checkpoint_dir for checkpoint_epoch_NNN.pth files and return
    the path + epoch number of the most recent one, or (None, 0) if none found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None, 0
    best_epoch = 0
    best_path  = None
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("checkpoint_epoch_") and fname.endswith(".pth"):
            try:
                ep = int(fname.replace("checkpoint_epoch_", "").replace(".pth", ""))
                if ep > best_epoch:
                    best_epoch = ep
                    best_path  = os.path.join(checkpoint_dir, fname)
            except ValueError:
                pass
    return best_path, best_epoch


def load_checkpoint(
    checkpoint_dir: str,
    model: CNNAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> dict:
    """
    Load the latest checkpoint if one exists.
    Returns a dict with resume state, or empty dict if none found.
    """
    path, epoch = _find_latest_checkpoint(checkpoint_dir)
    if path is None:
        return {}
    print(f"  [RESUME] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
def train_autoencoder(
    model:           CNNAutoencoder,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    epochs:          int   = DEFAULT_EPOCHS,
    lr:              float = DEFAULT_LR,
    weight_decay:    float = DEFAULT_WEIGHT_DECAY,
    device:          torch.device | None = None,
    save_path:       str   = "models/autoencoder/autoencoder_best.pth",
    graphs_dir:      str   = GRAPHS_DIR,
    checkpoint_dir:  str   = CHECKPOINT_DIR,
    resume:          bool  = False,
    verbose:         bool  = True,
) -> dict:
    """
    Train the CNN Autoencoder and return a history dict.

    Features:
      - Per-epoch checkpoints in checkpoint_dir/checkpoint_epoch_NNN.pth
      - Resume from latest checkpoint with resume=True
      - Explicit memory cleanup each epoch to survive on low-RAM machines
      - Full traceback on any training error (never silent)
      - sys.stdout.flush() after every epoch print

    Returns:
        {
          "train_losses": [...],
          "val_losses":   [...],
          "best_val_mse": float,
          "best_epoch":   int,
        }
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    train_losses: list = []
    val_losses:   list = []
    best_val_mse = float("inf")
    best_epoch   = 0
    start_epoch  = 1

    # ── Optionally resume from checkpoint ─────────────────────────────────
    if resume:
        ckpt = load_checkpoint(checkpoint_dir, model, optimizer, scheduler, device)
        if ckpt:
            start_epoch  = ckpt["epoch"] + 1
            best_val_mse = ckpt["best_val_mse"]
            best_epoch   = ckpt["best_epoch"]
            train_losses = ckpt["train_losses"]
            val_losses   = ckpt["val_losses"]
            if verbose:
                print(f"  [RESUME] Continuing from epoch {start_epoch}/{epochs}")
        else:
            if verbose:
                print("  [RESUME] No checkpoint found — starting from scratch.")

    if verbose:
        print("=" * 56)
        print("       CNN AUTOENCODER TRAINING")
        print("=" * 56)
        print(f"Device        : {device}")
        print(f"Epochs        : {epochs}  (start={start_epoch})")
        print(f"Batch Size    : {train_loader.batch_size}")
        print(f"Optimizer     : Adam  lr={lr}")
        print(f"Loss Function : MSELoss")
        print(f"Checkpoint    : {checkpoint_dir}")
        print("-" * 56)
        sys.stdout.flush()

    if start_epoch > epochs:
        if verbose:
            print(f"Training already completed ({epochs} epochs). Nothing to do.")
        return {
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "best_val_mse": best_val_mse,
            "best_epoch":   best_epoch,
        }

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # ── Training ──────────────────────────────────────────────────────
        try:
            model.train()
            running_loss = 0.0
            for batch_idx, (imgs, _) in enumerate(train_loader):
                imgs   = imgs.to(device)
                target = _target_from_input(imgs)

                optimizer.zero_grad()
                _, recon = model(imgs)
                loss = criterion(recon, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Accumulate loss value only (not tensor) to avoid holding graph
                running_loss += loss.item() * imgs.size(0)

                # Explicit cleanup to keep RAM low
                del imgs, target, recon, loss
                if batch_idx % 100 == 99:
                    gc.collect()

            train_loss = running_loss / len(train_loader.dataset)

            # ── Validation ────────────────────────────────────────────────
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs   = imgs.to(device)
                    target = _target_from_input(imgs)
                    _, recon = model(imgs)
                    val_running += criterion(recon, target).item() * imgs.size(0)
                    del imgs, target, recon
            val_loss = val_running / len(val_loader.dataset)

            scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Track best model
            if val_loss < best_val_mse:
                best_val_mse = val_loss
                best_epoch   = epoch
                model.save(save_path)

            # ── Save epoch checkpoint ──────────────────────────────────────
            if epoch % CHECKPOINT_INTERVAL == 0:
                ckpt_path = _save_checkpoint(
                    checkpoint_dir, epoch, model, optimizer, scheduler,
                    train_loss, val_loss, best_val_mse, best_epoch,
                    train_losses, val_losses, save_path,
                )

            elapsed = time.time() - t0

            # Check for NaN/Inf
            if not (train_loss == train_loss) or train_loss == float("inf"):
                print(f"\n[ERROR] NaN/Inf detected in train loss at epoch {epoch}!")
                print(f"  train_loss = {train_loss}")
                sys.stdout.flush()
                raise ValueError(f"NaN/Inf train loss at epoch {epoch}")

            if verbose:
                print(f"Epoch {epoch:>3}/{epochs}  "
                      f"Train MSE: {train_loss:.6f}  "
                      f"Val MSE: {val_loss:.6f}  "
                      f"({elapsed:.1f}s)")
                sys.stdout.flush()

            # Explicit garbage collection every epoch on low-RAM machines
            gc.collect()

        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] Training interrupted at epoch {epoch}.")
            print(f"  Last checkpoint: {_find_latest_checkpoint(checkpoint_dir)[0]}")
            sys.stdout.flush()
            raise

        except Exception as exc:
            print("\n" + "=" * 56)
            print("  TRAINING ERROR")
            print("=" * 56)
            print(f"  Epoch     : {epoch}/{epochs}")
            print(f"  Exception : {exc}")
            print("-" * 56)
            traceback.print_exc()
            print("-" * 56)
            ckpt_file, ckpt_epoch = _find_latest_checkpoint(checkpoint_dir)
            print(f"  Last good checkpoint : {ckpt_file} (epoch {ckpt_epoch})")
            print(f"  Resume command       : python train.py --resume")
            print("=" * 56)
            sys.stdout.flush()
            raise

    if verbose:
        print("-" * 56)
        print(f"Best Val MSE : {best_val_mse:.6f}  (epoch {best_epoch})")
        print(f"Model saved  : {save_path}")
        print("=" * 56)
        sys.stdout.flush()

    # ── Plot loss curves ──────────────────────────────────────────────────
    try:
        _plot_losses(train_losses, val_losses, graphs_dir)
    except Exception as e:
        print(f"  [WARN] Could not save loss plot: {e}")

    return {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "best_val_mse": best_val_mse,
        "best_epoch":   best_epoch,
    }


def _plot_losses(train_losses: list, val_losses: list, graphs_dir: str):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train MSE", linewidth=2)
    ax.plot(epochs, val_losses,   label="Val MSE",   linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(graphs_dir, "training_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    del fig


def save_reconstruction_samples(
    model: CNNAutoencoder,
    data_loader: DataLoader,
    device: torch.device,
    save_dir: str = RECON_DIR,
    n_samples: int = 8,
):
    """Save a grid of original vs reconstructed images."""
    import numpy as np
    from src.data.preprocessing import tensor_to_numpy

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    imgs, _ = next(iter(data_loader))
    imgs = imgs[:n_samples].to(device)

    with torch.no_grad():
        _, recon = model(imgs)

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    for i in range(n_samples):
        orig_np  = tensor_to_numpy(imgs[i].cpu())
        recon_np = (recon[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        axes[0, i].imshow(orig_np);   axes[0, i].axis("off")
        axes[1, i].imshow(recon_np);  axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original",      fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    plt.suptitle("Original vs Reconstructed")
    plt.tight_layout()
    path = os.path.join(save_dir, "sample_reconstructions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
