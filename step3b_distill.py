"""
Step 3b — Train 24 student flow matching models on synthetic teacher datasets.

For each latent dim × dataset size (6 × 4 = 24 students):
  - Loads synthetic x_0 dataset from step3a
  - Trains a StudentDenoiser with pure flow matching loss
  - x_1 and t are sampled ONCE per batch and used consistently
    for both the interpolation and the velocity target — no hidden resampling
  Saves: models/student_{dim}_{n_samples}.pt

Supports --dim and --size for fine-grained parallel GPU runs.
Skips any student checkpoint that already exists — safe to restart.

Usage:
    python step3b_distill.py                        # all 24 students sequentially
    python step3b_distill.py --dim 128              # all 4 sizes for dim=128
    python step3b_distill.py --dim 128 --size 500000  # single student
"""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.denoiser import StudentDenoiser, param_count

LATENT_DIMS   = [64, 128, 256, 384, 512, 1024]
DATASET_SIZES = [250_000, 500_000, 1_000_000, 2_000_000]
EPOCHS        = 100
BATCH_SIZE    = 256
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
EMA_DECAY     = 0.9999
LOG_INTERVAL  = 50

MODEL_DIR     = Path("models")
SYNTHETIC_DIR = Path("synthetic")
LATENT_DIR    = Path("latents")
RESULTS_DIR   = Path("results/trained_AE")


def get_device(dim: int = None) -> torch.device:
    if torch.cuda.is_available():
        if dim is not None:
            gpu_id = LATENT_DIMS.index(dim) % torch.cuda.device_count()
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def create_ema(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    ema = deepcopy(model).to(device)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def update_ema(ema: torch.nn.Module, model: torch.nn.Module) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema.parameters(), model.parameters()):
            ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)


def plot_loss(history: list, dim: int, n_samples: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history) + 1), history, color="royalblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow Matching Loss")
    ax.set_title(f"Student Loss  (dim={dim}, n={n_samples:,})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / f"student_loss_{dim}_{n_samples}.png"
    plt.savefig(str(out), dpi=150)
    plt.close()


def train_student(dim: int, n_samples: int, device: torch.device,
                  load_to_ram: bool = True) -> None:
    out_path = MODEL_DIR / f"student_{dim}_{n_samples}.pt"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists.")
        return

    data_path = SYNTHETIC_DIR / str(dim) / f"synthetic_{dim}_{n_samples}.npy"
    if not data_path.exists():
        print(f"  [ERROR] {data_path} not found — run step3a first.")
        return

    # Load norm stats so the student checkpoint can denormalise at eval time
    stats_path = LATENT_DIR / f"latents_{dim}_norm_stats.npy"
    if not stats_path.exists():
        print(f"  [ERROR] {stats_path} not found — run step2 first.")
        return
    stats = np.load(stats_path)
    lat_mean, lat_std = float(stats[0]), float(stats[1])

    # ── dataset ───────────────────────────────────────────────────────────────
    size_gb = n_samples * dim * 4 / 1e9
    print(f"\n  dim={dim}  n={n_samples:,}  device={device}")
    print(f"  Dataset size: {size_gb:.2f} GB")

    x0_data = np.memmap(str(data_path), dtype="float32", mode="r", shape=(n_samples, dim))

    if load_to_ram:
        try:
            print(f"  Loading dataset into RAM ({size_gb:.2f} GB) …")
            x0_tensor = torch.from_numpy(np.array(x0_data))
            del x0_data
            shuffle = True
            print("  Loaded. shuffle=True (random permutation each epoch)")
        except MemoryError:
            print(f"  [warning] MemoryError — falling back to memmap "
                  f"(file-backed, shuffle=True via random index access, slower on HDD)")
            x0_tensor = torch.from_numpy(x0_data)
            shuffle = True
    else:
        print("  Using memmap (file-backed). shuffle=True via random index access.")
        x0_tensor = torch.from_numpy(x0_data)
        shuffle = True

    dataset = TensorDataset(x0_tensor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                         num_workers=2, pin_memory=True)

    steps_per_epoch = len(loader)
    total_steps     = EPOCHS * steps_per_epoch
    print(f"  Batch shape      : ({BATCH_SIZE}, {dim})")
    print(f"  Steps per epoch  : {steps_per_epoch:,}")
    print(f"  Total opt. steps : {total_steps:,}")

    # ── model ─────────────────────────────────────────────────────────────────
    student     = StudentDenoiser(latent_dim=dim).to(device)
    ema_student = create_ema(student, device)
    print(f"  Student params   : {param_count(student)}")

    optimizer = AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    best_loss      = float("inf")
    history        = []
    sanity_checked = False

    for epoch in tqdm(range(1, EPOCHS + 1), desc=f"    dim={dim} n={n_samples:,}"):
        student.train()
        total_loss = 0.0

        for batch_idx, (x_0,) in enumerate(loader):
            x_0 = x_0.to(device)
            B   = x_0.shape[0]

            # ── sample x_1 and t ONCE — used for both x_t and v_target ───────
            x_1 = torch.randn_like(x_0)
            t   = torch.rand(B, device=device)
            t_  = t.view(-1, 1)

            x_t      = (1.0 - t_) * x_0 + t_ * x_1
            v_target = x_1 - x_0                 # same x_1 as above

            # ── one-time sanity check: verify x_t and v_target share x_1/t ───
            if not sanity_checked:
                residual = (x_t - (1.0 - t_) * x_0 - t_ * x_1).abs().max().item()
                assert residual < 1e-5, \
                    f"x_t construction inconsistency — max residual={residual:.2e}"
                assert torch.allclose(v_target, x_1 - x_0, atol=1e-6), \
                    "v_target does not equal x_1 - x_0 from the same x_1"
                print(f"  [sanity] x_t/v_target consistency check passed "
                      f"(max residual={residual:.2e})")
                sanity_checked = True

            v_pred = student(x_t, t)
            loss   = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            optimizer.step()
            update_ema(ema_student, student)

            total_loss += loss.item()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"      [ep {epoch:03d} step {batch_idx+1:05d}]  loss={avg:.5f}", flush=True)

        scheduler.step()
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        print(f"    epoch {epoch:03d}  loss={avg_loss:.5f}  best={best_loss:.5f}")

    torch.save(
        {
            "model_state_dict": ema_student.state_dict(),
            "latent_dim":       dim,
            "n_samples":        n_samples,
            "latent_mean":      lat_mean,
            "latent_std":       lat_std,
            "loss_history":     history,
        },
        out_path,
    )
    print(f"  Saved → {out_path}")
    plot_loss(history, dim, n_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS,
                        help="Latent dim to distil (omit for all)")
    parser.add_argument("--size", type=int, choices=DATASET_SIZES,
                        help="Synthetic dataset size to use (omit for all sizes)")
    parser.add_argument("--load-to-ram", dest="load_to_ram",
                        action="store_true", default=True,
                        help="Copy dataset into RAM before training (default: on)")
    parser.add_argument("--no-load-to-ram", dest="load_to_ram", action="store_false",
                        help="Keep dataset file-backed via memmap (lower RAM, slower)")
    args = parser.parse_args()

    dims  = [args.dim]  if args.dim  else LATENT_DIMS
    sizes = [args.size] if args.size else DATASET_SIZES

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for dim in dims:
        device = get_device(dim)
        print(f"\n{'='*60}")
        print(f"Distilling students  latent_dim={dim}  device={device}")
        print(f"{'='*60}")
        for n_samples in sizes:
            train_student(dim, n_samples, device, load_to_ram=args.load_to_ram)

    print("\nStep 3b complete.")


if __name__ == "__main__":
    main()
