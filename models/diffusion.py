"""
Rectified Flow / Flow Matching utilities (PyTorch).

Straight-line interpolation between data x_0 and noise x_1:
    x_t = (1 - t) * x_0  +  t * x_1        t in [0, 1]
    v   = x_1 - x_0                         constant target velocity

Training:
    t ~ U[0, 1]
    x_1 ~ N(0, I)
    x_t = (1-t)*x_0 + t*x_1
    loss = MSE(v_theta(x_t, t), x_1 - x_0)

Sampling (Euler, n_steps):
    x <- N(0, I)            # start at t=1 (pure noise)
    dt = 1 / n_steps
    for t = 1, 1-dt, ..., dt:
        x <- x - v_theta(x, t) * dt
    return x                # t=0 = data
"""

import torch


class FlowMatching:
    def __init__(self, device: str = "cpu"):
        self.device = device

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x_0: torch.Tensor):
        """
        Sample a noisy interpolation and return it with the target velocity.

        Parameters
        ----------
        x_0 : (B, D)  clean data latents

        Returns
        -------
        x_t : (B, D)  interpolated sample
        t   : (B,)    continuous time in [0, 1]
        v   : (B, D)  target velocity = x_1 - x_0
        """
        B = x_0.shape[0]
        x_1 = torch.randn_like(x_0)
        t   = torch.rand(B, device=x_0.device)                  # U[0, 1]
        t_  = t.view(-1, 1)
        x_t = (1.0 - t_) * x_0 + t_ * x_1
        v   = x_1 - x_0
        return x_t, t, v

    # ── sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def euler_sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        n_steps: int = 50,
    ) -> torch.Tensor:
        """
        Euler ODE integration from t=1 (noise) to t=0 (data).

        Parameters
        ----------
        model   : velocity network  v_theta(x_t, t) -> (B, D)
        shape   : (n_samples, latent_dim)
        n_steps : number of Euler steps

        Returns
        -------
        x : (n_samples, latent_dim)  generated samples
        """
        device = self.device
        x  = torch.randn(shape, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_val = 1.0 - i * dt                                 # 1.0 → dt
            t = torch.full((shape[0],), t_val, device=device)
            v = model(x, t)
            x = x - v * dt

        return x
