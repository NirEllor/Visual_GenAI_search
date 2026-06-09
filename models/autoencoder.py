import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        if latent_dim % 16 != 0:
            raise ValueError(
                "latent_dim must be divisible by 16, because latent shape is (C, 4, 4)."
            )

        self.latent_dim = latent_dim
        self.latent_channels = latent_dim // 16

        # ------------------------------------------------------------------
        # Encoder
        # Input: (B, 3, 32, 32)
        # ------------------------------------------------------------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # (B,16,16,16)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B,32,8,8)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B,64,4,4)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Deterministic latent projection
        self.encoder_latent = nn.Conv2d(
            in_channels=64,
            out_channels=self.latent_channels,
            kernel_size=1
        )

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        self.decoder_proj = nn.Conv2d(
            in_channels=self.latent_channels,
            out_channels=64,
            kernel_size=1
        )

        # Important refinement block at 4x4
        self.decoder_refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),  # (B,32,8,8)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),  # (B,16,16,16)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                16, 3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )   # (B,3,32,32)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns deterministic latent vector.
        Shape: (B, latent_dim)
        """
        h = self.encoder_conv(x)          # (B,64,4,4)
        z = self.encoder_latent(h)        # (B,C,4,4)
        return torch.flatten(z, 1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Returns reconstruction logits.
        No sigmoid is applied here.
        """
        x = latent.view(-1, self.latent_channels, 4, 4)
        x = self.decoder_proj(x)          # (B,64,4,4)
        x = self.decoder_refine(x)        # (B,64,4,4)
        return self.decoder_conv(x)       # (B,3,32,32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns reconstruction logits.
        """
        z = self.encode(x)
        return self.decode(z)