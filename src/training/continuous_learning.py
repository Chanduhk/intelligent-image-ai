"""
Continuous Learning Engine
Implements the Collect → Select → Retrain → Validate → Deploy loop.

Key design decisions:
  • Samples are only added to the learning buffer if they are "difficult"
    (high reconstruction error OR low classifier confidence).
  • Retraining combines replay data (random subset of CIFAR-10) with
    buffer samples to avoid catastrophic forgetting.
  • A new model version is deployed only when validation MSE improves.
"""

import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.autoencoder import CNNAutoencoder, build_autoencoder
from src.models.classifier  import LatentClassifier
from src.models.anomaly_detector import AnomalyDetector
from src.data.dataset import LearningBufferDataset, load_cifar10
from src.versioning.model_manager import ModelManager
from src.training.train_autoencoder import train_autoencoder
from src.training.train_classifier  import (
    train_classifier, train_anomaly_detector, build_similarity_db
)


# ── Difficulty thresholds ─────────────────────────────────────────────────────
HIGH_MSE_THRESHOLD   = 0.010   # normalised MSE above this → difficult
LOW_CONF_THRESHOLD   = 60.0    # confidence below this → difficult
BUFFER_TRIGGER_SIZE  = 30      # start retraining when buffer reaches N samples
REPLAY_SIZE          = 2000    # number of CIFAR-10 samples for replay
FINE_TUNE_EPOCHS     = 5       # epochs for incremental fine-tuning


# ─────────────────────────────────────────────────────────────────────────────
class ContinuousLearningEngine:
    """
    Manages the full continuous-learning lifecycle.
    """

    def __init__(self,
                 model_manager: ModelManager,
                 device:        torch.device,
                 data_dir:      str = "./data/raw",
                 buffer_dir:    str = "./data/learning_buffer",
                 verbose:       bool = True):
        self.manager    = model_manager
        self.device     = device
        self.data_dir   = data_dir
        self.buffer_dir = buffer_dir
        self.verbose    = verbose
        self.buffer     = LearningBufferDataset(buffer_dir)
        self.stats      = {
            "total_received":    0,
            "added_to_buffer":   0,
            "trainings_run":     0,
            "versions_deployed": 0,
        }

    # ── sample evaluation ─────────────────────────────────────────────────
    def is_difficult_sample(self, mse: float, confidence: float) -> bool:
        """
        A sample is "difficult" if the model struggles with it —
        high reconstruction error OR low classification confidence.
        """
        return (mse > HIGH_MSE_THRESHOLD) or (confidence < LOW_CONF_THRESHOLD)

    def add_to_buffer(self, image_tensor: torch.Tensor, label: int) -> str:
        """Add one sample to the on-disk learning buffer."""
        path = self.buffer.add_sample(image_tensor.cpu(), label)
        self.stats["added_to_buffer"] += 1
        return path

    def evaluate_and_buffer(self,
                             image_tensor: torch.Tensor,
                             label: int,
                             mse: float,
                             confidence: float) -> dict:
        """
        Decide whether to add the sample to the buffer.
        Returns a dict describing the decision.
        """
        self.stats["total_received"] += 1
        difficult = self.is_difficult_sample(mse, confidence)
        if difficult:
            path = self.add_to_buffer(image_tensor, label)
            added = True
        else:
            path, added = None, False

        return {
            "is_difficult":   difficult,
            "added_to_buffer": added,
            "buffer_size":    len(self.buffer),
            "reason":         (
                f"MSE={mse:.5f} > {HIGH_MSE_THRESHOLD} or "
                f"confidence={confidence:.1f}% < {LOW_CONF_THRESHOLD}%"
                if difficult else "Sample is easy — not added to buffer."
            ),
        }

    # ── incremental retraining ────────────────────────────────────────────
    def should_retrain(self) -> bool:
        """Trigger retraining when buffer reaches the threshold."""
        return len(self.buffer) >= BUFFER_TRIGGER_SIZE

    def run_incremental_training(self,
                                 current_model: CNNAutoencoder,
                                 val_loader:    DataLoader) -> dict:
        """
        Fine-tune a copy of the current model on buffer + replay data.
        Returns a result dict including whether the new model was deployed.
        """
        if self.verbose:
            print("\n" + "=" * 56)
            print("       CONTINUOUS LEARNING — INCREMENTAL TRAINING")
            print("=" * 56)
            print(f"Buffer size   : {len(self.buffer)} samples")

        # ── Build combined train loader ────────────────────────────────
        _, _, cifar_train, _ = load_cifar10(
            data_dir=self.data_dir, batch_size=128, num_workers=0
        )
        replay_indices = random.sample(range(len(cifar_train)),
                                       min(REPLAY_SIZE, len(cifar_train)))
        from torch.utils.data import Subset
        replay_subset = Subset(cifar_train, replay_indices)
        combined = ConcatDataset([replay_subset, self.buffer])
        combo_loader = DataLoader(combined, batch_size=128,
                                  shuffle=True, num_workers=0)
        if self.verbose:
            print(f"Combined data : {len(combined)} samples "
                  f"({len(replay_subset)} replay + {len(self.buffer)} buffer)")

        # ── Create candidate model (deep copy) ────────────────────────
        candidate = copy.deepcopy(current_model)
        candidate.to(self.device)
        new_version = self.manager.next_version()
        candidate_path = self.manager.autoencoder_path(new_version)

        history = train_autoencoder(
            model        = candidate,
            train_loader = combo_loader,
            val_loader   = val_loader,
            epochs       = FINE_TUNE_EPOCHS,
            device       = self.device,
            save_path    = candidate_path,
            verbose      = self.verbose,
        )
        self.stats["trainings_run"] += 1

        candidate_metrics = {
            "val_mse":           history["best_val_mse"],
            "fine_tune_epochs":  FINE_TUNE_EPOCHS,
            "buffer_size_used":  len(self.buffer),
        }

        # ── Validation gate ───────────────────────────────────────────
        deploy, reason = self.manager.should_deploy(candidate_metrics)
        if self.verbose:
            print(f"\nValidation : {reason}")

        if deploy:
            # Register + deploy
            self.manager.register_version(new_version, candidate_metrics,
                                          description="Incremental fine-tuning")
            self.manager.deploy(new_version)
            self.stats["versions_deployed"] += 1

            # Re-train downstream models on new encoder
            clf_path = self.manager.classifier_path(new_version)
            anom_path = self.manager.anomaly_path(new_version)
            train_classifier(candidate, combo_loader, val_loader,
                             self.device, save_path=clf_path,
                             verbose=self.verbose)
            train_anomaly_detector(candidate, combo_loader,
                                   self.device, save_path=anom_path,
                                   verbose=self.verbose)
            build_similarity_db(candidate, combo_loader, self.device,
                                save_dir=self.manager.similarity_dir(),
                                verbose=self.verbose)

            # Clear the buffer after successful deployment
            self.buffer.clear()
            if self.verbose:
                print(f"\n[OK] Model v{new_version} deployed successfully.")
                print(f"  Buffer cleared.")
        else:
            if self.verbose:
                print(f"[--] Candidate model NOT deployed - keeping current version.")
            # Remove candidate checkpoint to save disk space
            if os.path.exists(candidate_path):
                os.remove(candidate_path)

        return {
            "deployed":           deploy,
            "reason":             reason,
            "new_version":        new_version if deploy else None,
            "candidate_val_mse":  history["best_val_mse"],
            "deployed_version":   self.manager.get_deployed_version(),
            "stats":              self.stats.copy(),
        }

    # ── stats summary ─────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        return {
            **self.stats,
            "buffer_size":      len(self.buffer),
            "deployed_version": self.manager.get_deployed_version(),
            "trigger_threshold": BUFFER_TRIGGER_SIZE,
        }
