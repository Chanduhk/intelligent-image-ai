"""
Classifier and Anomaly Detector Training
Extracts latent features from the trained encoder and fits:
  1. Random Forest classifier
  2. Isolation Forest anomaly detector
Also builds the similarity database.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.autoencoder import CNNAutoencoder
from src.models.classifier  import LatentClassifier
from src.models.anomaly_detector import AnomalyDetector
from src.analysis.similarity import SimilarityEngine
from src.data.dataset import CIFAR10_CLASSES


GRAPHS_DIR = "outputs/graphs"


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _extract_all_features(model: CNNAutoencoder, loader, device) -> tuple:
    """Extract (features, labels) arrays from a full DataLoader pass."""
    model.eval()
    feats, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        z, _ = model(imgs)
        feats.append(z.view(z.size(0), -1).cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def train_classifier(
    model:        CNNAutoencoder,
    train_loader,
    val_loader,
    device:       torch.device,
    save_path:    str = "models/classifier/classifier_v1.pkl",
    verbose:      bool = True,
) -> dict:
    """
    Extract latent features from the encoder and train a Random Forest.

    Returns:
        {"accuracy": float, "classification_report": dict, ...}
    """
    if verbose:
        print("\n--- Training Random Forest Classifier ---")
        print("Extracting train features …")
    train_feats, train_labels = _extract_all_features(model, train_loader, device)

    if verbose:
        print("Extracting val features …")
    val_feats, val_labels = _extract_all_features(model, val_loader, device)

    classifier = LatentClassifier()
    if verbose:
        print("Fitting Random Forest …")
    classifier.fit(train_feats, train_labels)

    results = classifier.evaluate(val_feats, val_labels)
    if verbose:
        print(f"Validation Accuracy: {results['accuracy']:.2f}%")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    classifier.save(save_path)
    if verbose:
        print(f"Classifier saved : {save_path}")

    _plot_confusion_matrix(results["confusion_matrix"])
    return results


def train_anomaly_detector(
    model:        CNNAutoencoder,
    train_loader,
    device:       torch.device,
    save_path:    str = "models/anomaly/anomaly_v1.pkl",
    verbose:      bool = True,
) -> AnomalyDetector:
    """Extract features and fit an Isolation Forest."""
    if verbose:
        print("\n--- Training Isolation Forest Anomaly Detector ---")
        print("Extracting features …")
    feats, _ = _extract_all_features(model, train_loader, device)

    detector = AnomalyDetector()
    if verbose:
        print("Fitting Isolation Forest …")
    detector.fit(feats)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    detector.save(save_path)
    if verbose:
        print(f"Anomaly detector saved : {save_path}")
    return detector


def build_similarity_db(
    model:         CNNAutoencoder,
    train_loader,
    device:        torch.device,
    save_dir:      str = "models/similarity",
    max_samples:   int = 10000,
    verbose:       bool = True,
) -> SimilarityEngine:
    """Build and persist the cosine-similarity reference database."""
    if verbose:
        print(f"\n--- Building Similarity Database (max {max_samples} samples) ---")
    sim = SimilarityEngine()
    sim.build_from_loader(model, train_loader, device, max_samples=max_samples)
    sim.save(save_dir)
    if verbose:
        print(f"Similarity DB saved  : {save_dir}")
    return sim


def _plot_confusion_matrix(cm: list):
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        sns.heatmap(
            cm_arr, annot=True, fmt="d", ax=ax,
            xticklabels=CIFAR10_CLASSES,
            yticklabels=CIFAR10_CLASSES,
            cmap="Blues",
        )
    except ImportError:
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_xticks(range(10)); ax.set_xticklabels(CIFAR10_CLASSES, rotation=45)
        ax.set_yticks(range(10)); ax.set_yticklabels(CIFAR10_CLASSES)

    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Classifier Confusion Matrix (CIFAR-10)")
    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
