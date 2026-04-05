"""
MLP-based denoiser models for latent diffusion (PyTorch).

TeacherDenoiser : 4 residual blocks, hidden_dim=512  (~large)
StudentDenoiser : 2 residual blocks, hidden_dim=256  (~4x fewer params)

Both share the same interface:
    out = model(x_t, t)
where
    x_t : (B, latent_dim)  noisy latent at timestep t
    t   : (B,)             integer timestep indices
    out : (B, latent_dim)  predicted noise ε
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ── Sinusoidal time embedding ─────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Transformer-style sinusoidal embedding for scalar timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t : (B,) integer or float
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)      # (B, dim)
        return emb


# ── Residual block ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    One residual MLP block.

    x  →  LayerNorm  →  Linear(dim)  →  GELU  →  + time_proj(t_emb)  →  Linear(dim)  →  + x
    """

    def __init__(self, dim: int, time_emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_emb_dim, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.lin1(h))
        h = h + self.time_proj(t_emb)
        h = self.lin2(h)
        return x + h


# ── Base MLP denoiser ─────────────────────────────────────────────────────────

class MLPDenoiser(nn.Module):
    """
    Generic MLP denoiser.  Subclasses set hidden_dim and n_blocks.

    Architecture
    ------------
    time → SinPosEmb(time_emb_dim) → MLP → t_emb
    x_t  → Linear(hidden_dim)      → n × ResBlock(hidden_dim, t_emb)
           → LayerNorm → Linear(latent_dim)   [ε prediction]
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_blocks: int,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, time_emb_dim) for _ in range(n_blocks)]
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, latent_dim)   noisy latent
        t : (B,)              integer timestep in [0, T-1]

        Returns
        -------
        eps : (B, latent_dim)  predicted noise
        """
        t_emb = self.time_embed(t)             # (B, time_emb_dim)
        h = self.input_proj(x)                 # (B, hidden_dim)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_head(h)             # (B, latent_dim)


# ── Concrete models ───────────────────────────────────────────────────────────

class TeacherDenoiser(MLPDenoiser):
    """Large teacher: 4 residual blocks, hidden_dim=512."""

    def __init__(self, latent_dim: int, hidden_dim: int = 512, n_blocks: int = 4):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
        )


class StudentDenoiser(MLPDenoiser):
    """Small student: 2 residual blocks, hidden_dim=256 (~4× fewer params)."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256, n_blocks: int = 2):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
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
