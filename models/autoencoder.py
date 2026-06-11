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
        # ------------------------------------------------------------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # VAE heads
        self.encoder_mean = nn.Conv2d(
            in_channels=64,
            out_channels=self.latent_channels,
            kernel_size=1
        )

        self.encoder_logvar = nn.Conv2d(
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

        # Important refinement block at 4×4
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
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                16, 3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
        )

    def encode(self, x: torch.Tensor, sample: bool = True):
        """
        Returns:
            z_flat      : (B, latent_dim)
            mean_flat   : (B, latent_dim)
            logvar_flat : (B, latent_dim)

        sample=True  -> reparameterization trick
        sample=False -> deterministic mean
        """
        h = self.encoder_conv(x)

        mean = self.encoder_mean(h)
        logvar = self.encoder_logvar(h).clamp(-30, 20)

        if sample:
            eps = torch.randn_like(mean)
            z = mean + eps * torch.exp(0.5 * logvar)
        else:
            z = mean

        return (
            torch.flatten(z, 1),
            torch.flatten(mean, 1),
            torch.flatten(logvar, 1),
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Returns reconstruction logits.
        No sigmoid is applied here.
        """
        x = latent.view(-1, self.latent_channels, 4, 4)

        x = self.decoder_proj(x)
        x = self.decoder_refine(x)

        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            recon_logits, mean, logvar
        """
        z, mean, logvar = self.encode(x, sample=True)
        recon_logits = self.decode(z)

        return recon_logits, mean, logvar