"""
Step 2 — Train DDPM teacher diffusion models in latent space.
"""

import argparse
import numpy as np
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.diffusion import DiffusionSchedule
from models.denoiser import TeacherDenoiser, param_count

LATENT_DIMS = [64, 128, 256, 384]
T            = 1000
EPOCHS       = 1000
BATCH_SIZE   = 256
LR           = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
SAVE_EVERY   = 50
EMA_DECAY    = 0.9999

LATENT_DIR   = "latents"
MODEL_DIR    = "models"
LOG_INTERVAL = 10


def get_device(dim: int = None) -> str:
    if torch.cuda.is_available():
        if dim is not None:
            gpu_id = LATENT_DIMS.index(dim) % torch.cuda.device_count()
            return f"cuda:{gpu_id}"
        return "cuda"
    return "cpu"


def normalise_latents(latents: np.ndarray, dim: int):
    mean = latents.mean()
    std  = latents.std()
    latents_norm = ((latents - mean) / (std + 1e-8)).astype(np.float32)
    stats_path = Path(LATENT_DIR) / f"latents_{dim}_norm_stats.npy"
    np.save(stats_path, np.array([mean, std], dtype=np.float64))
    print(f"  Norm stats saved → {stats_path}  (mean={mean:.4f}, std={std:.4f})")
    return latents_norm, mean, std


def create_ema(model, device):
    ema = deepcopy(model).to(device)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def update_ema(ema_model, model, decay=EMA_DECAY):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def plot_teacher_loss(history: list, dim: int, results_dir: str = "results") -> None:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history) + 1), history, color="royalblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Teacher Denoiser — Training Loss  (latent_dim={dim})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(results_dir) / f"teacher_loss_{dim}.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Loss curve → {out}")


def train_one_epoch(model, ema_model, loader, diffusion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (x_0,) in enumerate(loader):
        x_0 = x_0.to(device)
        B = x_0.shape[0]

        t = torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long)
        x_t, noise = diffusion.q_sample(x_0, t)
        eps_pred = model(x_t, t)
        loss = F.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        update_ema(ema_model, model)  # ← EMA update

        total_loss += loss.item()

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            avg = total_loss / (batch_idx + 1)
            print(f"    [epoch {epoch:03d}  step {batch_idx+1:04d}]  loss = {avg:.5f}", flush=True)

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS, default=None)
    args = parser.parse_args()

    dims = [args.dim] if args.dim is not None else LATENT_DIMS
    device = get_device(args.dim)
    print(f"Device: {device}  |  dims: {dims}")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    diffusion = DiffusionSchedule(T=T, device=device)

    for dim in dims:
        out_path = Path(MODEL_DIR) / f"teacher_{dim}.pt"
        if out_path.exists():
            print(f"\n[skip] teacher_{dim}.pt already exists.")
            continue

        print(f"\n{'='*60}")
        print(f"  Training teacher  latent_dim = {dim}")

        latents_raw = np.load(Path(LATENT_DIR) / f"latents_{dim}.npy")
        latents, mean, std = normalise_latents(latents_raw, dim)

        dataset = TensorDataset(torch.from_numpy(latents))
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=2, pin_memory=(device == "cuda"))

        model     = TeacherDenoiser(latent_dim=dim).to(device)
        ema_model = create_ema(model, device)  # ← create EMA
        print(f"  Model params: {param_count(model)}")

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

        best_loss = float("inf")
        history   = []

        for epoch in tqdm(range(1, EPOCHS + 1), desc=f"dim={dim}"):
            avg_loss = train_one_epoch(model, ema_model, loader, diffusion, optimizer, device, epoch)
            scheduler.step()
            history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if epoch % SAVE_EVERY == 0:
                interim = Path(MODEL_DIR) / f"teacher_{dim}_ep{epoch:03d}.pt"
                torch.save({"model_state_dict": ema_model.state_dict()}, interim)  # ← EMA

            print(f"  epoch {epoch:03d}  avg_loss={avg_loss:.5f}  best={best_loss:.5f}")

        torch.save(
            {
                "model_state_dict": ema_model.state_dict(),  # ← EMA
                "latent_dim":       dim,
                "latent_mean":      float(mean),
                "latent_std":       float(std),
                "T":                T,
                "loss_history":     history,
            },
            out_path,
        )
        print(f"  Saved → {out_path}  (best_loss={best_loss:.5f})")
        plot_teacher_loss(history, dim)

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()