"""
Anomaly Detection Module
Uses Isolation Forest on latent representations to flag
images that deviate from the learned data distribution.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


# ── Default hyper-parameters ──────────────────────────────────────────────────
DEFAULT_N_ESTIMATORS   = 200
DEFAULT_CONTAMINATION  = 0.05    # expected fraction of anomalies in training set
DEFAULT_MAX_SAMPLES    = "auto"
DEFAULT_RANDOM_STATE   = 42

# Score thresholds – tuned so a normalised score in [0, 1] maps cleanly
ANOMALY_THRESHOLD      = 0.55    # normalised score above this → ANOMALOUS


# ─────────────────────────────────────────────────────────────────────────────
class AnomalyDetector:
    """
    Wraps IsolationForest and provides a clean interface that returns
    normalised anomaly scores in [0, 1] (1 = most anomalous).
    """

    def __init__(self,
                 n_estimators: int   = DEFAULT_N_ESTIMATORS,
                 contamination       = DEFAULT_CONTAMINATION,
                 random_state: int   = DEFAULT_RANDOM_STATE):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=DEFAULT_MAX_SAMPLES,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_trained  = False
        self._score_min  = None    # used for min-max normalisation
        self._score_max  = None

    # ── training ──────────────────────────────────────────────────────────
    def fit(self, features: np.ndarray):
        """
        Train on (N, latent_dim) feature matrix.
        Computes score min/max on the training set for later normalisation.
        """
        self.model.fit(features)
        raw_scores      = self.model.score_samples(features)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())
        self.is_trained = True

    # ── inference ─────────────────────────────────────────────────────────
    def _normalise(self, raw_score: float) -> float:
        """
        Isolation Forest's score_samples returns negative values;
        lower (more negative) = more anomalous.
        We flip and normalise to [0, 1] so 1 = most anomalous.
        """
        if self._score_min is None or self._score_max is None:
            return 0.5
        denom = self._score_max - self._score_min
        if denom == 0:
            return 0.0
        norm = (raw_score - self._score_min) / denom   # [0, 1], high = normal
        return round(1.0 - float(norm), 4)             # flip: high = anomalous

    def predict(self, feature_vector: np.ndarray) -> dict:
        """
        Predict anomaly status for a single latent vector.

        Returns dict with keys:
            anomaly_score  – float in [0, 1]   (higher → more anomalous)
            is_anomaly     – bool
            status         – "NORMAL" | "ANOMALOUS"
            label          – sklearn raw label (+1 normal, -1 anomaly)
        """
        if not self.is_trained:
            raise RuntimeError("AnomalyDetector has not been trained yet.")

        fv         = feature_vector.reshape(1, -1)
        raw_score  = float(self.model.score_samples(fv)[0])
        norm_score = self._normalise(raw_score)
        label      = int(self.model.predict(fv)[0])
        is_anomaly = norm_score >= ANOMALY_THRESHOLD

        return {
            "anomaly_score": norm_score,
            "is_anomaly":    is_anomaly,
            "status":        "ANOMALOUS" if is_anomaly else "NORMAL",
            "raw_label":     label,
        }

    # ── batch evaluation ──────────────────────────────────────────────────
    def predict_batch(self, features: np.ndarray) -> list:
        return [self.predict(features[i]) for i in range(len(features))]

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "model":       self.model,
            "is_trained":  self.is_trained,
            "score_min":   self._score_min,
            "score_max":   self._score_max,
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model       = data["model"]
        self.is_trained  = data["is_trained"]
        self._score_min  = data["score_min"]
        self._score_max  = data["score_max"]
