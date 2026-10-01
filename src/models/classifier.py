"""
Random Forest Classifier on Latent Features
Wraps sklearn RandomForestClassifier with helpers to
extract latent features from the CNN Autoencoder and produce
class predictions with per-class probability estimates.
"""

import os
import numpy as np
import joblib
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data.dataset import CIFAR10_CLASSES


# ── Default hyper-parameters ──────────────────────────────────────────────────
DEFAULT_N_ESTIMATORS = 300
DEFAULT_MAX_DEPTH    = None
DEFAULT_N_JOBS       = -1
DEFAULT_RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
class LatentClassifier:
    """
    Random Forest classifier that operates on flattened latent vectors
    produced by the CNN Autoencoder encoder.
    """

    def __init__(self,
                 n_estimators: int = DEFAULT_N_ESTIMATORS,
                 max_depth=DEFAULT_MAX_DEPTH,
                 random_state: int = DEFAULT_RANDOM_STATE):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=DEFAULT_N_JOBS,
            random_state=random_state,
            verbose=0,
        )
        self.is_trained = False
        self.classes    = CIFAR10_CLASSES

    # ── feature extraction ────────────────────────────────────────────────
    @staticmethod
    @torch.no_grad()
    def extract_features(encoder, data_loader, device: torch.device) -> tuple:
        """
        Run the encoder over a DataLoader and collect:
          - latent_vectors: (N, 1024) float32 numpy array
          - labels:         (N,)     int numpy array
        """
        encoder.eval()
        features, labels = [], []
        for batch in data_loader:
            imgs, lbls = batch
            imgs = imgs.to(device)
            z = encoder(imgs)
            if isinstance(z, tuple):
                z = z[0]                  # (B, 64, 4, 4)
            features.append(z.view(z.size(0), -1).cpu().numpy())
            labels.append(lbls.numpy())
        return np.concatenate(features), np.concatenate(labels)

    # ── training ──────────────────────────────────────────────────────────
    def fit(self, features: np.ndarray, labels: np.ndarray):
        """Train the Random Forest on pre-extracted features."""
        self.model.fit(features, labels)
        self.is_trained = True

    # ── inference ─────────────────────────────────────────────────────────
    def predict(self, feature_vector: np.ndarray) -> dict:
        """
        Predict the class for a single latent vector.

        Args:
            feature_vector: shape (1024,) or (1, 1024)

        Returns dict with keys:
            predicted_class, confidence, top_predictions
        """
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained yet.")

        fv = feature_vector.reshape(1, -1)
        pred_idx   = self.model.predict(fv)[0]
        proba      = self.model.predict_proba(fv)[0]

        top_idx    = np.argsort(proba)[::-1][:3]
        top_preds  = [
            {"class": self.classes[i], "confidence": round(float(proba[i]) * 100, 2)}
            for i in top_idx
        ]

        return {
            "predicted_class": self.classes[pred_idx],
            "predicted_index": int(pred_idx),
            "confidence":      round(float(proba[pred_idx]) * 100, 2),
            "top_predictions": top_preds,
            "all_probabilities": {
                self.classes[i]: round(float(p) * 100, 2)
                for i, p in enumerate(proba)
            }
        }

    # ── evaluation ────────────────────────────────────────────────────────
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """Return accuracy, classification report, and confusion matrix."""
        preds = self.model.predict(features)
        return {
            "accuracy":              round(accuracy_score(labels, preds) * 100, 2),
            "classification_report": classification_report(
                labels, preds, target_names=self.classes, output_dict=True
            ),
            "confusion_matrix":      confusion_matrix(labels, preds).tolist(),
        }

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "is_trained": self.is_trained}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model      = data["model"]
        self.is_trained = data["is_trained"]
