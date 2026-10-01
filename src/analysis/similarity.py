"""
Similarity Engine
Stores a reference database of latent vectors and retrieves
the most similar entries for a query vector using cosine similarity.
"""

import os
import json
import numpy as np
import torch

from src.data.dataset import CIFAR10_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
class SimilarityEngine:
    """
    In-memory latent vector database with cosine-similarity search.

    The reference database is built from training samples once the
    encoder is trained.  It can be persisted to / loaded from disk.
    """

    def __init__(self):
        self.vectors: np.ndarray | None = None   # (N, latent_dim)
        self.labels:  list[int]         = []
        self.built    = False

    # ── building / indexing ───────────────────────────────────────────────
    def build(self, vectors: np.ndarray, labels: list[int] | np.ndarray):
        """
        Populate the database.

        Args:
            vectors: (N, latent_dim) float32 array (already flattened)
            labels:  (N,) integer class indices
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        self.vectors = (vectors / norms).astype(np.float32)
        self.labels  = list(labels)
        self.built   = True

    @torch.no_grad()
    def build_from_loader(self, encoder, data_loader, device: torch.device,
                          max_samples: int = 10000):
        """
        Convenience method: run encoder over data_loader and populate db.
        max_samples limits memory usage (CIFAR-10 has 50k train images).
        """
        encoder.eval()
        vecs, lbls = [], []
        count = 0
        for imgs, labels in data_loader:
            if count >= max_samples:
                break
            imgs = imgs.to(device)
            z = encoder(imgs)
            if isinstance(z, tuple):
                z = z[0]
            flat = z.view(z.size(0), -1).cpu().numpy()
            vecs.append(flat)
            lbls.extend(labels.tolist())
            count += len(labels)
        self.build(np.concatenate(vecs)[:max_samples],
                   lbls[:max_samples])

    # ── querying ──────────────────────────────────────────────────────────
    def find_similar(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Return the top-k most similar reference entries.

        Args:
            query_vector: (latent_dim,) or (1, latent_dim) float32 array

        Returns list of dicts:
            [{"rank": 1, "class": "dog", "similarity": 94.3}, ...]
        """
        if not self.built or self.vectors is None:
            return []

        q = query_vector.flatten().astype(np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            norm = 1e-9
        q = q / norm

        # cosine similarities (higher = more similar)
        sims = self.vectors @ q        # (N,)
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            results.append({
                "rank":       rank,
                "class":      CIFAR10_CLASSES[self.labels[idx]],
                "similarity": round(float(sims[idx]) * 100, 1),
            })
        return results

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        if self.vectors is not None:
            np.save(os.path.join(directory, "sim_vectors.npy"), self.vectors)
        with open(os.path.join(directory, "sim_labels.json"), "w") as f:
            json.dump({"labels": self.labels}, f)

    def load(self, directory: str):
        vecs_path   = os.path.join(directory, "sim_vectors.npy")
        labels_path = os.path.join(directory, "sim_labels.json")
        if not os.path.exists(vecs_path):
            return
        self.vectors = np.load(vecs_path)
        with open(labels_path) as f:
            self.labels = json.load(f)["labels"]
        self.built = True
