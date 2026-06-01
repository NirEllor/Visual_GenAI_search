"""
Velocity networks for latent flow matching (PyTorch).

TeacherDenoiser : 4 conv residual blocks, hidden_channels=256
StudentDenoiser : 2 conv residual blocks, hidden_channels=128

Both process latents as (C, 4, 4) spatial maps with FiLM time conditioning,
then flatten back to (B, latent_dim).  Interface is identical to the old MLP:

    out = model(x_t, t)
where
    x_t : (B, latent_dim)  interpolated latent at time t
    t   : (B,)             continuous time in [0, 1]
    out : (B, latent_dim)  predicted velocity v_theta(x_t, t)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sinusoidal time embedding ─────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Transformer-style sinusoidal embedding for scalar timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)      # (B, dim)
        return emb


# ── Convolutional residual block with FiLM time conditioning ─────────────────

class ConvResBlock(nn.Module):
    """
    Conv residual block operating on (B, channels, H, W) spatial feature maps.

    FiLM conditioning: scale and shift from time embedding, applied after the first conv.
    """

    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        groups = min(32, channels)
        self.norm1     = nn.GroupNorm(groups, channels)
        self.conv1     = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2     = nn.GroupNorm(groups, channels)
        self.conv2     = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, 2 * channels)  # → scale + shift

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)   # each (B, channels)
        scale = scale.unsqueeze(-1).unsqueeze(-1)                # (B, channels, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = F.gelu(self.conv1(self.norm1(x))) * (1 + scale) + shift
        h = self.conv2(self.norm2(h))
        return x + h


# ── Convolutional denoiser ────────────────────────────────────────────────────

class ConvDenoiser(nn.Module):
    """
    Velocity network that processes latents as (C, 4, 4) spatial maps.

    Architecture
    ------------
    time → SinPosEmb → MLP → t_emb
    x_t (B, latent_dim)
      → reshape (B, C, 4, 4)
      → Conv2d 1×1 input projection → (B, hidden_channels, 4, 4)
      → n × ConvResBlock(hidden_channels, t_emb)
      → GroupNorm → Conv2d 1×1 output projection → (B, C, 4, 4)
      → flatten → (B, latent_dim)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int,
        n_blocks: int,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim      = latent_dim
        self.latent_channels = latent_dim // 16  # C, since 4*4=16

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_proj = nn.Conv2d(self.latent_channels, hidden_channels, kernel_size=1)

        self.blocks = nn.ModuleList(
            [ConvResBlock(hidden_channels, time_emb_dim) for _ in range(n_blocks)]
        )

        groups = min(32, hidden_channels)
        self.output_head = nn.Sequential(
            nn.GroupNorm(groups, hidden_channels),
            nn.Conv2d(hidden_channels, self.latent_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, latent_dim)  interpolated latent
        t : (B,)             continuous time in [0, 1]

        Returns
        -------
        v : (B, latent_dim)  predicted velocity
        """
        t_emb = self.time_embed(t * 1000)                          # (B, time_emb_dim)
        h = x.view(-1, self.latent_channels, 4, 4)                 # (B, C, 4, 4)
        h = self.input_proj(h)                                     # (B, hidden_channels, 4, 4)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_head(h).flatten(1)                      # (B, latent_dim)


# ── Concrete models ───────────────────────────────────────────────────────────

class TeacherDenoiser(ConvDenoiser):
    """Large teacher: 4 conv residual blocks, hidden_channels=256."""

    def __init__(self, latent_dim: int, hidden_channels: int = 256, n_blocks: int = 4):
        super().__init__(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            n_blocks=n_blocks,
        )


class StudentDenoiser(ConvDenoiser):
    """Small student: 2 conv residual blocks, hidden_channels=128 (~4× fewer params)."""

    def __init__(self, latent_dim: int, hidden_channels: int = 128, n_blocks: int = 2):
        super().__init__(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            n_blocks=n_blocks,
        )


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_teacher(ckpt_path: str, latent_dim: int, device: str = "cpu") -> TeacherDenoiser:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = TeacherDenoiser(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_student(ckpt_path: str, latent_dim: int, device: str = "cpu") -> StudentDenoiser:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = StudentDenoiser(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def param_count(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{n:,}"
