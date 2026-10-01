"""
CNN Autoencoder Model
Architecture:
  Encoder: Conv2D × 3 with ReLU + MaxPool → Latent (64 × 4 × 4 = 1024 values)
  Decoder: ConvTranspose2D × 3 with ReLU → Sigmoid → 3 × 32 × 32
"""

import torch
import torch.nn as nn


# ── Latent geometry (must stay consistent across all modules) ─────────────────
LATENT_CHANNELS = 64
LATENT_SPATIAL  = 4                              # 4 × 4
LATENT_DIM      = LATENT_CHANNELS * LATENT_SPATIAL * LATENT_SPATIAL  # 1024
ORIGINAL_DIM    = 3 * 32 * 32                    # 3072
COMPRESSION_RATIO = ORIGINAL_DIM / LATENT_DIM   # 3.0


# ─────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    3 × 32 × 32  →  64 × 4 × 4
    Convolution stages with batch-norm for stable training.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Stage 1: 3 × 32 × 32 → 32 × 16 × 16
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # → 32 × 16 × 16

            # Stage 2: 32 × 16 × 16 → 64 × 8 × 8
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # → 64 × 8 × 8

            # Stage 3: 64 × 8 × 8 → 64 × 4 × 4
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # → 64 × 4 × 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    """
    64 × 4 × 4  →  3 × 32 × 32
    Three ConvTranspose stages followed by a Sigmoid.
    """

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # Stage 1: 64 × 4 × 4 → 64 × 8 × 8
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Stage 2: 64 × 8 × 8 → 32 × 16 × 16
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Stage 3: 32 × 16 × 16 → 3 × 32 × 32
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ─────────────────────────────────────────────────────────────────────────────
class CNNAutoencoder(nn.Module):
    """
    Full autoencoder that exposes both the latent representation and the
    reconstructed image so every downstream module can use the same object.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw latent tensor (B, 64, 4, 4)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent tensor (B, 64, 4, 4)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            latent       – (B, 64, 4, 4)
            reconstructed – (B, 3, 32, 32) in [0, 1] (Sigmoid output)
        """
        z     = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    def get_latent_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a 1-D latent vector (B, 1024) for downstream ML models.
        """
        z = self.encode(x)
        return z.view(z.size(0), -1)

    # ── persistence helpers ───────────────────────────────────────────────
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device = torch.device("cpu")):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.to(device)
        self.eval()


# ─────────────────────────────────────────────────────────────────────────────
def build_autoencoder() -> CNNAutoencoder:
    """Factory – returns a freshly initialised CNNAutoencoder."""
    return CNNAutoencoder()
