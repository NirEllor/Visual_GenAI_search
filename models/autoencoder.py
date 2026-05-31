import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int):
        super(ConvAutoencoder, self).__init__()

        if latent_dim % 16 != 0:
            raise ValueError("latent_dim must be divisible by 16, because latent shape is (C, 4, 4).")

        self.latent_dim = latent_dim
        self.latent_channels = latent_dim // 16  # because 4 * 4 = 16

        # Input: (B, 3, 32, 32)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),    # (B, 16, 16, 16)
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   # (B, 32, 8, 8)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # (B, 64, 4, 4)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # Replaces Flatten + Linear(1024 -> latent_dim)
        # Keeps spatial structure: (B, 64, 4, 4) -> (B, C, 4, 4)
        self.encoder_proj = nn.Conv2d(
            in_channels=64,
            out_channels=self.latent_channels,
            kernel_size=1
        )

        # Replaces Linear(latent_dim -> 1024)
        # Restores channels: (B, C, 4, 4) -> (B, 64, 4, 4)
        self.decoder_proj = nn.Conv2d(
            in_channels=self.latent_channels,
            out_channels=64,
            kernel_size=1
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, 8, 8)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 16, 16, 16)
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # (B, 3, 32, 32)
        )

    def encode(self, x):
        x = self.encoder_conv(x)          # (B, 64, 4, 4)
        x = self.encoder_proj(x)          # (B, C, 4, 4)
        return torch.flatten(x, start_dim=1)  # (B, latent_dim)

    def decode(self, latent):
        x = latent.view(-1, self.latent_channels, 4, 4)  # (B, C, 4, 4)
        x = self.decoder_proj(x)                         # (B, 64, 4, 4)
        return self.decoder_conv(x)

    def forward(self, x):
        return self.decode(self.encode(x))