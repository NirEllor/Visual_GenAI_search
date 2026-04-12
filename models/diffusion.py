"""
DDPM noise schedule and sampling utilities (PyTorch).

Supports:
  - Cosine noise schedule (Nichol & Dhariwal 2021)
  - Forward process  q(x_t | x_0)
  - DDPM reverse sampling (stochastic, T steps)
  - DDIM reverse sampling (deterministic, configurable n_steps)

All operations work on flat 1-D latent vectors (B, latent_dim) — no
spatial structure is assumed.
"""

import math
import torch


# ── Noise schedule ────────────────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine variance schedule.  Returns betas of shape (T,).
    From: Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal 2021)
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    f = f / f[0]
    betas = 1.0 - f[1:] / f[:-1]
    return betas.clamp(1e-4, 0.9999).float()


# ── Schedule class ────────────────────────────────────────────────────────────

class DiffusionSchedule:
    """
    Precomputed cosine diffusion schedule.

    Indexing convention: t ∈ {0, …, T-1}
      t=0  → very slightly noisy
      t=T-1 → almost pure Gaussian
    """

    def __init__(self, T: int = 1000, device: str = "cpu"):
        self.T = T
        self.device = device

        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev

        # Forward process coefficients
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

        # x_0 recovery coefficients
        self.sqrt_recip_alpha_bar = (1.0 / alpha_bar).sqrt()
        self.sqrt_recipm1_alpha_bar = (1.0 / alpha_bar - 1.0).sqrt()

        # Posterior q(x_{t-1} | x_t, x_0) coefficients
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.posterior_var = posterior_var
        self.log_posterior_var = posterior_var.clamp(min=1e-20).log()
        self.posterior_mean_x0_coef = (
            betas * alpha_bar_prev.sqrt() / (1.0 - alpha_bar)
        )
        self.posterior_mean_xt_coef = (
            (1.0 - alpha_bar_prev) * alphas.sqrt() / (1.0 - alpha_bar)
        )

    # ── Forward process ───────────────────────────────────────────────────────

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        """
        Sample x_t ~ q(x_t | x_0) by adding noise.

        Parameters
        ----------
        x_0 : (B, D)
        t   : (B,) int64  timestep indices

        Returns
        -------
        x_t   : (B, D)  noisy sample
        noise : (B, D)  the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        s_ab = self.sqrt_alpha_bar[t].view(-1, 1)
        s_om = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        x_t = s_ab * x_0 + s_om * noise
        return x_t, noise

    # ── x_0 / epsilon conversion ──────────────────────────────────────────────

    def predict_x0_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """Recover x̂_0 from a predicted epsilon."""
        sr = self.sqrt_recip_alpha_bar[t].view(-1, 1)
        srm1 = self.sqrt_recipm1_alpha_bar[t].view(-1, 1)
        return sr * x_t - srm1 * eps

    def predict_eps_from_x0(
        self, x_t: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor
    ) -> torch.Tensor:
        """Recover ε from a predicted x̂_0."""
        sr = self.sqrt_recip_alpha_bar[t].view(-1, 1)
        srm1 = self.sqrt_recipm1_alpha_bar[t].view(-1, 1)
        return (sr * x_t - x_0) / srm1.clamp(min=1e-8)

    # ── DDPM single reverse step ──────────────────────────────────────────────

    @torch.no_grad()
    def ddpm_step(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t_val: int,
    ) -> torch.Tensor:
        """One DDPM reverse step: x_t → x_{t-1}."""
        B = x_t.shape[0]
        t = torch.full((B,), t_val, device=self.device, dtype=torch.long)

        eps_pred = model(x_t, t)
        x_0_pred = self.predict_x0_from_eps(x_t, t, eps_pred).clamp(-3.0, 3.0)

        if t_val == 0:
            return x_0_pred

        mean = (
            self.posterior_mean_x0_coef[t_val] * x_0_pred
            + self.posterior_mean_xt_coef[t_val] * x_t
        )
        noise = torch.randn_like(x_t)
        std = self.log_posterior_var[t_val].exp().sqrt()
        return mean + std * noise

    # ── DDIM sampling ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        n_steps: int = 50,
        eta: float = 0.0,
        clip_x0: bool = True,
    ) -> torch.Tensor:
        """
        DDIM sampling.

        Parameters
        ----------
        model   : denoiser  f(x_t, t) → ε_pred
        shape   : (n_samples, latent_dim)
        n_steps : number of DDIM steps (< T)
        eta     : stochasticity (0 = deterministic DDIM, 1 = DDPM-like)

        Returns
        -------
        x_0 : (n_samples, latent_dim)
        """
        device = self.device
        x = torch.randn(shape, device=device)

        # Evenly spaced timesteps descending from T-1 to 0
        ts = torch.linspace(self.T - 1, 0, n_steps + 1).long()

        for i in range(n_steps):
            t_cur = ts[i].item()
            t_nxt = ts[i + 1].item()

            B = x.shape[0]
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

            eps_pred = model(x, t_batch)
            x_0_pred = self.predict_x0_from_eps(x, t_batch, eps_pred)
            if clip_x0:
                x_0_pred = x_0_pred.clamp(-3.0, 3.0)

            ab_cur = self.alpha_bar[t_cur]
            ab_nxt = self.alpha_bar[t_nxt] if t_nxt >= 0 else torch.tensor(1.0, device=device)

            # DDIM sigma
            sigma = (
                eta
                * ((1.0 - ab_nxt) / (1.0 - ab_cur) * (1.0 - ab_cur / ab_nxt)).sqrt()
            )

            # Direction pointing to x_t
            pred_dir = (1.0 - ab_nxt - sigma ** 2).clamp(min=0.0).sqrt() * eps_pred

            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
            x = ab_nxt.sqrt() * x_0_pred + pred_dir + noise

        return x

    # ── Full DDPM sampling (for reference / teacher eval) ─────────────────────

    @torch.no_grad()
    def ddpm_sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
    ) -> torch.Tensor:
        """Full DDPM reverse diffusion (T steps, slow)."""
        device = self.device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            x = self.ddpm_step(model, x, t)
        return x
