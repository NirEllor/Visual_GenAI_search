"""
Step 2 — Train DDPM teacher diffusion models in latent space.

For each latent dimension (64, 128, 256, 384):
  1. Load the saved latent vectors from step 1.
  2. Normalise latents to zero-mean unit-variance; save stats for later use.
  3. Train a TeacherDenoiser (MLP, 4 residual blocks, hidden=512) with the
     standard DDPM epsilon-prediction loss.
  4. Save the trained model checkpoint.

Output files
------------
models/teacher_64.pt
models/teacher_128.pt
models/teacher_256.pt
models/teacher_384.pt
latents/latents_{dim}_norm_stats.npy   (mean, std for denormalisation)
"""

import os
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.diffusion import DiffusionSchedule
from models.denoiser import TeacherDenoiser, param_count

# ── hyper-parameters ──────────────────────────────────────────────────────────
LATENT_DIMS  = [64, 128, 256, 384]
T            = 1000          # diffusion timesteps
EPOCHS       = 200
BATCH_SIZE   = 256
LR           = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
SAVE_EVERY   = 50            # save an intermediate checkpoint every N epochs

LATENT_DIR   = "latents"
MODEL_DIR    = "models"
LOG_INTERVAL = 10            # print loss every N batches

# ── utilities ─────────────────────────────────────────────────────────────────

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalise_latents(latents: np.ndarray, dim: int):
    """
    Standardise latents to N(0,1).  Save stats so step 3/4 can denormalise.

    Returns
    -------
    latents_norm : float32 np.ndarray  (N, latent_dim)
    mean, std    : scalar float64
    """
    mean = latents.mean()
    std  = latents.std()
    latents_norm = ((latents - mean) / (std + 1e-8)).astype(np.float32)

    stats_path = Path(LATENT_DIR) / f"latents_{dim}_norm_stats.npy"
    np.save(stats_path, np.array([mean, std], dtype=np.float64))
    print(f"  Norm stats saved → {stats_path}  (mean={mean:.4f}, std={std:.4f})")
    return latents_norm, mean, std


def train_one_epoch(
    model: TeacherDenoiser,
    loader: DataLoader,
    diffusion: DiffusionSchedule,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_idx, (x_0,) in enumerate(loader):
        x_0 = x_0.to(device)
        B = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long)

        # Forward process: add noise
        x_t, noise = diffusion.q_sample(x_0, t)

        # Predict noise
        eps_pred = model(x_t, t)
        loss = F.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            avg = total_loss / (batch_idx + 1)
            print(f"    [epoch {epoch:03d}  step {batch_idx+1:04d}]  loss = {avg:.5f}", flush=True)

    return total_loss / len(loader)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    diffusion = DiffusionSchedule(T=T, device=device)

    for dim in LATENT_DIMS:
        out_path = Path(MODEL_DIR) / f"teacher_{dim}.pt"
        if out_path.exists():
            print(f"\n[skip] teacher_{dim}.pt already exists.")
            continue

        print(f"\n{'='*60}")
        print(f"  Training teacher  latent_dim = {dim}")

        # Load & normalise latents
        latents_raw = np.load(Path(LATENT_DIR) / f"latents_{dim}.npy")
        latents, mean, std = normalise_latents(latents_raw, dim)

        dataset = TensorDataset(torch.from_numpy(latents))
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=2, pin_memory=(device == "cuda"))

        # Build model
        model = TeacherDenoiser(latent_dim=dim).to(device)
        print(f"  Model params: {param_count(model)}")

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

        best_loss = float("inf")
        history   = []

        for epoch in tqdm(range(1, EPOCHS + 1), desc=f"dim={dim}"):
            avg_loss = train_one_epoch(model, loader, diffusion, optimizer, device, epoch)
            scheduler.step()
            history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if epoch % SAVE_EVERY == 0:
                interim = Path(MODEL_DIR) / f"teacher_{dim}_ep{epoch:03d}.pt"
                torch.save({"model_state_dict": model.state_dict()}, interim)

            print(f"  epoch {epoch:03d}  avg_loss={avg_loss:.5f}  best={best_loss:.5f}")

        # Final checkpoint — includes normalisation stats so step 3/4 can load them
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "latent_dim":       dim,
                "latent_mean":      float(mean),
                "latent_std":       float(std),
                "T":                T,
                "loss_history":     history,
            },
            out_path,
        )
        print(f"  Saved → {out_path}  (best_loss={best_loss:.5f})")

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
