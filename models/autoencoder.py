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

        # VAE: separate mean and logvar heads — (B, 64, 4, 4) → (B, C, 4, 4)
        self.encoder_mean   = nn.Conv2d(64, self.latent_channels, kernel_size=1)
        self.encoder_logvar = nn.Conv2d(64, self.latent_channels, kernel_size=1)

        # Decoder projection: (B, C, 4, 4) → (B, 64, 4, 4)
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

    def encode(self, x: torch.Tensor, sample: bool = True):
        """
        Returns (z_flat, mean_flat, logvar_flat).
        sample=True  → reparameterized draw, used during training
        sample=False → mean only, deterministic, used for latent extraction
        """
        h      = self.encoder_conv(x)                          # (B, 64, 4, 4)
        mean   = self.encoder_mean(h)                          # (B, C, 4, 4)
        logvar = self.encoder_logvar(h).clamp(-30, 20)        # (B, C, 4, 4)
        if sample:
            z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        else:
            z = mean
        return torch.flatten(z, 1), torch.flatten(mean, 1), torch.flatten(logvar, 1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent.view(-1, self.latent_channels, 4, 4)  # (B, C, 4, 4)
        x = self.decoder_proj(x)                         # (B, 64, 4, 4)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor):
        """Returns (recon_logits, mean_flat, logvar_flat) for training."""
        z, mean, logvar = self.encode(x, sample=True)
        return self.decode(z), mean, logvar
