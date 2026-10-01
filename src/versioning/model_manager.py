"""
Model Versioning & Management
Tracks all saved model files, writes/reads metadata.json,
and decides whether a new candidate model is better than
the current deployed model.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime


METADATA_FILE = "metadata.json"

# ── Default directories ───────────────────────────────────────────────────────
DEFAULT_AUTOENCODER_DIR = "models/autoencoder"
DEFAULT_CLASSIFIER_DIR  = "models/classifier"
DEFAULT_ANOMALY_DIR     = "models/anomaly"
DEFAULT_META_DIR        = "models/metadata"
DEFAULT_SIM_DIR         = "models/similarity"


# ─────────────────────────────────────────────────────────────────────────────
class ModelManager:
    """
    Centralised model-version registry.

    Responsibilities:
      • Generate versioned file paths for model artefacts
      • Persist and load metadata.json
      • Compare candidate vs current model metrics
      • Promote a candidate to the active deployment

    Version numbers are always positive integers (1, 2, 3, …).
    The deployed version is stored as an integer in metadata.
    get_deployed_version() returns None (not 0) if nothing is deployed.
    load_best_* methods return (False, path) and print a clear error when
    no deployed model exists — they NEVER fabricate a fallback v0 path.
    """

    def __init__(self,
                 autoencoder_dir: str = DEFAULT_AUTOENCODER_DIR,
                 classifier_dir:  str = DEFAULT_CLASSIFIER_DIR,
                 anomaly_dir:     str = DEFAULT_ANOMALY_DIR,
                 meta_dir:        str = DEFAULT_META_DIR,
                 sim_dir:         str = DEFAULT_SIM_DIR):
        self.autoencoder_dir = autoencoder_dir
        self.classifier_dir  = classifier_dir
        self.anomaly_dir     = anomaly_dir
        self.meta_dir        = meta_dir
        self.sim_dir         = sim_dir

        for d in [autoencoder_dir, classifier_dir, anomaly_dir, meta_dir, sim_dir]:
            os.makedirs(d, exist_ok=True)

        self.metadata = self._load_metadata()

    # ── metadata persistence ──────────────────────────────────────────────
    def _meta_path(self) -> str:
        return os.path.join(self.meta_dir, METADATA_FILE)

    def _load_metadata(self) -> dict:
        p = self._meta_path()
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            # Migrate: ensure deployed_version is int or None (never a string)
            dv = data.get("deployed_version")
            if isinstance(dv, str):
                # strip leading 'v' if present
                try:
                    data["deployed_version"] = int(dv.lstrip("v"))
                except (ValueError, AttributeError):
                    data["deployed_version"] = None
            return data
        return {
            "current_version":  0,
            "versions":         [],
            "deployed_version": None,
        }

    def save_metadata(self):
        with open(self._meta_path(), "w") as f:
            json.dump(self.metadata, f, indent=2)

    # ── version number helpers ────────────────────────────────────────────
    @property
    def current_version(self) -> int:
        return self.metadata.get("current_version", 0)

    def next_version(self) -> int:
        """
        Return the next version number.
        Always at least 1 — never returns 0.
        """
        return max(1, self.current_version + 1)

    def version_tag(self, v: int | None = None) -> str:
        """Return a 'vN' tag string. Returns 'vNone' only when v is explicitly None."""
        if v is None:
            return "vNone"
        return f"v{v}"

    # ── path generators ───────────────────────────────────────────────────
    def autoencoder_path(self, version: int) -> str:
        """Return the canonical path for autoencoder_vN.pth."""
        tag = self.version_tag(version)
        return os.path.join(self.autoencoder_dir, f"autoencoder_{tag}.pth")

    def classifier_path(self, version: int) -> str:
        tag = self.version_tag(version)
        return os.path.join(self.classifier_dir, f"classifier_{tag}.pkl")

    def anomaly_path(self, version: int) -> str:
        tag = self.version_tag(version)
        return os.path.join(self.anomaly_dir, f"anomaly_{tag}.pkl")

    def similarity_dir(self) -> str:
        return self.sim_dir

    # ── registering a new version ─────────────────────────────────────────
    def register_version(self, version: int, metrics: dict,
                         description: str = ""):
        """
        Record a trained model version in metadata.
        metrics should include at minimum {"val_mse": float}.
        """
        entry = {
            "version":     version,
            "tag":         self.version_tag(version),
            "timestamp":   datetime.now().isoformat(),
            "metrics":     metrics,
            "description": description,
            "deployed":    False,
        }
        self.metadata["versions"].append(entry)
        self.metadata["current_version"] = version
        self.save_metadata()

    # ── deployment logic ──────────────────────────────────────────────────
    def get_deployed_version(self) -> int | None:
        """
        Return the deployed version number as int, or None if not deployed.
        Never returns 0 or a string.
        """
        dv = self.metadata.get("deployed_version")
        if dv is None or dv == 0:
            return None
        try:
            return int(dv)
        except (TypeError, ValueError):
            return None

    def get_deployed_metrics(self) -> dict:
        dv = self.get_deployed_version()
        if dv is None:
            return {}
        for v in self.metadata["versions"]:
            if v["version"] == dv:
                return v.get("metrics", {})
        return {}

    def should_deploy(self, candidate_metrics: dict) -> tuple[bool, str]:
        """
        Decide whether candidate_metrics justify deployment.
        Lower val_mse is better.

        Returns (deploy: bool, reason: str).
        """
        deployed_metrics = self.get_deployed_metrics()
        if not deployed_metrics:
            return True, "No deployed model exists — deploying first version."

        current_mse   = deployed_metrics.get("val_mse", float("inf"))
        candidate_mse = candidate_metrics.get("val_mse", float("inf"))

        if candidate_mse < current_mse:
            improvement = (current_mse - candidate_mse) / current_mse * 100
            return True, (
                f"Candidate MSE ({candidate_mse:.6f}) is lower than deployed "
                f"MSE ({current_mse:.6f}) — {improvement:.1f}% improvement."
            )
        return False, (
            f"Candidate MSE ({candidate_mse:.6f}) is NOT better than deployed "
            f"MSE ({current_mse:.6f}). Keeping current model."
        )

    def deploy(self, version: int):
        """
        Mark a version as deployed and update metadata.
        Validates that the model file exists before deploying.
        """
        ae_path = self.autoencoder_path(version)
        if not os.path.exists(ae_path):
            raise FileNotFoundError(
                f"Cannot deploy v{version}: model file not found at {ae_path}"
            )
        for v in self.metadata["versions"]:
            v["deployed"] = v["version"] == version
        self.metadata["deployed_version"] = version
        self.save_metadata()

    # ── loading helpers ───────────────────────────────────────────────────
    def load_best_autoencoder(self, autoencoder, device) -> tuple[bool, str]:
        """
        Load the deployed autoencoder weights.

        Returns:
            (True, path)  — loaded successfully
            (False, path) — file missing or no deployed version

        NEVER fabricates a v0 fallback. If no model is deployed,
        returns (False, '<no deployed version>') with a clear message.
        """
        dv = self.get_deployed_version()
        if dv is None:
            msg = "<no deployed version — run train.py first>"
            print(f"  [ModelManager] No deployed version in metadata.")
            return False, msg

        path = self.autoencoder_path(dv)
        if not os.path.exists(path):
            print(f"  [ModelManager] Autoencoder file not found: {path}")
            return False, path

        try:
            autoencoder.load(path, device)
            return True, path
        except Exception as e:
            print(f"  [ModelManager] Failed to load autoencoder from {path}: {e}")
            return False, path

    def load_best_classifier(self, classifier) -> tuple[bool, str]:
        dv = self.get_deployed_version()
        if dv is None:
            return False, "<no deployed version>"
        path = self.classifier_path(dv)
        if not os.path.exists(path):
            return False, path
        try:
            classifier.load(path)
            return True, path
        except Exception as e:
            print(f"  [ModelManager] Failed to load classifier from {path}: {e}")
            return False, path

    def load_best_anomaly(self, anomaly_detector) -> tuple[bool, str]:
        dv = self.get_deployed_version()
        if dv is None:
            return False, "<no deployed version>"
        path = self.anomaly_path(dv)
        if not os.path.exists(path):
            return False, path
        try:
            anomaly_detector.load(path)
            return True, path
        except Exception as e:
            print(f"  [ModelManager] Failed to load anomaly detector from {path}: {e}")
            return False, path

    # ── recovery helper ───────────────────────────────────────────────────
    def recover_deployed_from_disk(self) -> bool:
        """
        Scan the autoencoder directory for autoencoder_vN.pth files and
        register + deploy the highest version found.  Used to recover
        after a training run that completed model-save but was killed
        before metadata was written.

        Returns True if a model was recovered and deployed.
        """
        ae_dir = self.autoencoder_dir
        if not os.path.isdir(ae_dir):
            return False

        best_version = 0
        for fname in os.listdir(ae_dir):
            if fname.startswith("autoencoder_v") and fname.endswith(".pth"):
                try:
                    v = int(fname.replace("autoencoder_v", "").replace(".pth", ""))
                    if v > best_version:
                        best_version = v
                except ValueError:
                    pass

        if best_version == 0:
            return False

        print(f"  [ModelManager] Recovering deployed model: v{best_version} "
              f"(found on disk, not in metadata)")

        # Only register if not already registered
        existing_versions = [e["version"] for e in self.metadata["versions"]]
        if best_version not in existing_versions:
            self.register_version(
                best_version,
                # Use a large val_mse so any fully-trained model will supersede this
                {"val_mse": 9999.0, "classifier_acc": 0.0, "recovered": True},
                description="Auto-recovered from disk (training was interrupted after model save)",
            )

        # Force deploy without file check (we already confirmed it exists)
        for v in self.metadata["versions"]:
            v["deployed"] = v["version"] == best_version
        self.metadata["deployed_version"] = best_version
        self.save_metadata()
        return True

    # ── summary ───────────────────────────────────────────────────────────
    def version_history(self) -> list:
        return self.metadata.get("versions", [])

    def summary(self) -> str:
        lines = [
            f"{'Version':<10} {'MSE':<12} {'Acc%':<10} {'Deployed':<10} Timestamp"
        ]
        for v in self.version_history():
            m   = v.get("metrics", {})
            dep = "[deployed]" if v.get("deployed") else ""
            lines.append(
                f"{v['tag']:<10} "
                f"{m.get('val_mse', 0.0):<12.6f} "
                f"{m.get('classifier_acc', 0.0):<10.2f} "
                f"{dep:<10} "
                f"{v['timestamp'][:19]}"
            )
        return "\n".join(lines)
