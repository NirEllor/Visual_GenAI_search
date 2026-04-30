"""
Step 3 — Distil each flow matching teacher into a smaller student model.
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

from models.diffusion import FlowMatching
from models.denoiser import TeacherDenoiser, StudentDenoiser, load_teacher, param_count

LATENT_DIMS = [64, 128, 256, 384, 512, 1024]
EPOCHS             = 700
BATCH_SIZE         = 256
LR                 = 1e-4
WEIGHT_DECAY       = 1e-4
GRAD_CLIP          = 1.0
ALPHA              = 0.5
STUDENT_DDIM_STEPS = 20
EMA_DECAY          = 0.9999
LOG_INTERVAL       = 10

LATENT_DIR         = "latents"
MODEL_DIR          = "models"


def get_device(dim: int = None) -> str:
    if torch.cuda.is_available():
        if dim is not None:
            gpu_id = LATENT_DIMS.index(dim) % torch.cuda.device_count()
            return f"cuda:{gpu_id}"
        return "cuda"
    return "cpu"


def plot_combined_losses(
    teacher_history: list,
    student_history: dict,
    dim: int,
    results_dir: str = "results/trained_AE",
) -> None:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Teacher subplot
    ax1.plot(range(1, len(teacher_history) + 1), teacher_history,
             color="royalblue", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title(f"Teacher — Training Loss  (dim={dim})")
    ax1.grid(True, alpha=0.3)

    # Student subplot
    epochs_s = range(1, len(student_history["total"]) + 1)
    ax2.plot(epochs_s, student_history["total"],   label="Total",   color="crimson",   linewidth=1.5)
    ax2.plot(epochs_s, student_history["flow"],    label="Flow",    color="darkorange", linewidth=1.2, linestyle="--")
    ax2.plot(epochs_s, student_history["distill"], label="Distill", color="seagreen",   linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title(f"Student — Training Loss  (dim={dim})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Training Loss Curves — latent_dim={dim}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(results_dir) / f"loss_curves_{dim}.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Combined loss curve → {out}")


def load_latents_normalised(dim: int):
    latents = np.load(Path(LATENT_DIR) / f"latents_{dim}.npy")
    stats   = np.load(Path(LATENT_DIR) / f"latents_{dim}_norm_stats.npy")
    mean, std = float(stats[0]), float(stats[1])
    latents_norm = ((latents - mean) / (std + 1e-8)).astype(np.float32)
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


def train_one_epoch(student, ema_student, teacher, loader, flow, optimizer, device, epoch):
    student.train()
    total_loss = total_flow_loss = total_dist_loss = 0.0

    for batch_idx, (x_0,) in enumerate(loader):
        x_0 = x_0.to(device)

        x_t, t, v_target = flow.forward(x_0)

        v_student = student(x_t, t)
        flow_loss = F.mse_loss(v_student, v_target)

        with torch.no_grad():
            v_teacher = teacher(x_t, t)
        dist_loss = F.mse_loss(v_student, v_teacher)

        loss = ALPHA * flow_loss + (1.0 - ALPHA) * dist_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()
        update_ema(ema_student, student)

        total_loss      += loss.item()
        total_flow_loss += flow_loss.item()
        total_dist_loss += dist_loss.item()

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            n = batch_idx + 1
            print(f"    [ep {epoch:03d} step {n:04d}]"
                  f"  total={total_loss/n:.5f}"
                  f"  flow={total_flow_loss/n:.5f}"
                  f"  dist={total_dist_loss/n:.5f}", flush=True)

    n = len(loader)
    return total_loss / n, total_flow_loss / n, total_dist_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS, default=None)
    args = parser.parse_args()

    dims = [args.dim] if args.dim is not None else LATENT_DIMS
    device = get_device(args.dim)
    print(f"Device: {device}  |  dims: {dims}")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    flow = FlowMatching(device=device)

    for dim in dims:
        out_path = Path(MODEL_DIR) / f"student_{dim}.pt"
        if out_path.exists():
            print(f"\n[skip] student_{dim}.pt already exists.")
            continue

        teacher_path = Path(MODEL_DIR) / f"teacher_{dim}.pt"
        if not teacher_path.exists():
            print(f"[ERROR] teacher_{dim}.pt not found — run step2 first.")
            continue

        print(f"\n{'='*60}")
        print(f"  Distilling student  latent_dim = {dim}")

        latents_norm, mean, std = load_latents_normalised(dim)
        dataset = TensorDataset(torch.from_numpy(latents_norm))
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=2, pin_memory=(device == "cuda"))

        raw_teacher_ckpt = torch.load(str(teacher_path), map_location="cpu", weights_only=False)
        teacher_loss_history = raw_teacher_ckpt.get("loss_history", [])

        teacher = load_teacher(str(teacher_path), latent_dim=dim, device=device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        student     = StudentDenoiser(latent_dim=dim).to(device)
        ema_student = create_ema(student, device)  # ← create EMA
        print(f"  Teacher params : {param_count(teacher)}")
        print(f"  Student params : {param_count(student)}")

        optimizer = AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

        best_loss = float("inf")
        history   = {"total": [], "flow": [], "distill": []}

        for epoch in tqdm(range(1, EPOCHS + 1), desc=f"dim={dim}"):
            avg_total, avg_flow, avg_dist = train_one_epoch(
                student, ema_student, teacher, loader, flow, optimizer, device, epoch
            )
            scheduler.step()

            history["total"].append(avg_total)
            history["flow"].append(avg_flow)
            history["distill"].append(avg_dist)

            if avg_total < best_loss:
                best_loss = avg_total

            print(f"  epoch {epoch:03d}  total={avg_total:.5f}"
                  f"  flow={avg_flow:.5f}  dist={avg_dist:.5f}"
                  f"  best={best_loss:.5f}")

        torch.save(
            {
                "model_state_dict":      ema_student.state_dict(),
                "latent_dim":            dim,
                "latent_mean":           float(mean),
                "latent_std":            float(std),
                "student_euler_steps":   STUDENT_DDIM_STEPS,
                "alpha":                 ALPHA,
                "loss_history":          history,
            },
            out_path,
        )
        print(f"  Saved → {out_path}  (best_loss={best_loss:.5f})")
        plot_combined_losses(teacher_loss_history, history, dim)

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()