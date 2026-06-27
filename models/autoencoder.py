import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        if latent_dim % 16 != 0:
            raise ValueError(
                "latent_dim must be divisible by 16, because latent shape is (C, 4, 4)."
            )

        self.latent_dim = latent_dim
        self.latent_channels = latent_dim // 16

        self.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(64, 128, 256),
            latent_channels=self.latent_channels,
            sample_size=32,
            scaling_factor=1.0,
        )

    def encode(self, x: torch.Tensor, sample: bool = False):
        posterior = self.vae.encode(x).latent_dist

        mean = posterior.mean
        logvar = posterior.logvar.clamp(-30, 20)

        z = posterior.sample() if sample else mean

        return (
            torch.flatten(z, 1),
            torch.flatten(mean, 1),
            torch.flatten(logvar, 1),
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Returns reconstruction logits.
        """
        z_spatial = latent.view(-1, self.latent_channels, 4, 4)

        recon_logits = self.vae.decode(z_spatial).sample
        return recon_logits

    def forward(self, x: torch.Tensor):
        z, mean, logvar = self.encode(x, sample=False)
        recon_logits = self.decode(z)
        return recon_logits, mean, logvar