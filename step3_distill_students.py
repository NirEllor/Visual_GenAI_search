"""
Step 3 — Distil each teacher into a smaller student model.

Distillation strategy
---------------------
A combined loss is used so that:
  (a) The student learns the data distribution on its own (DDPM loss).
  (b) The student matches the teacher's x₀-predictions at every timestep
      (distillation loss).

    L = α · L_DDPM  +  (1-α) · L_distill

where
    L_DDPM    = MSE(ε_student, ε_true)          [standard epsilon-prediction]
    L_distill = MSE(x̂₀_student, x̂₀_teacher)   [match teacher's x₀ estimate]
    α = ALPHA  (default 0.5)

After training, the student uses DDIM with STUDENT_DDIM_STEPS (default 4) at
inference time while the teacher requires the full T=1000 DDPM steps.

Output files
------------
models/student_64.pt
models/student_128.pt
models/student_256.pt
models/student_384.pt
"""

import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.diffusion import DiffusionSchedule
from models.denoiser import TeacherDenoiser, StudentDenoiser, load_teacher, param_count

# ── hyper-parameters ──────────────────────────────────────────────────────────
LATENT_DIMS        = [64, 128, 256, 384]
T                  = 1000
EPOCHS             = 150
BATCH_SIZE         = 256
LR                 = 1e-4
WEIGHT_DECAY       = 1e-4
GRAD_CLIP          = 1.0
ALPHA              = 0.5      # weight of DDPM loss  (1-ALPHA = distill weight)
STUDENT_DDIM_STEPS = 4        # how many steps the student uses at inference
LOG_INTERVAL       = 10

LATENT_DIR         = "latents"
MODEL_DIR          = "models"

# ── utilities ─────────────────────────────────────────────────────────────────

def get_device(dim: int = None) -> str:
    if torch.cuda.is_available():
        if dim is not None:
            gpu_id = LATENT_DIMS.index(dim)
            return f"cuda:{gpu_id}"
        return "cuda"
    return "cpu"


def load_latents_normalised(dim: int):
    """Load latents and return normalised array + stats."""
    latents = np.load(Path(LATENT_DIR) / f"latents_{dim}.npy")
    stats   = np.load(Path(LATENT_DIR) / f"latents_{dim}_norm_stats.npy")
    mean, std = float(stats[0]), float(stats[1])
    latents_norm = ((latents - mean) / (std + 1e-8)).astype(np.float32)
    return latents_norm, mean, std


def train_one_epoch(
    student: StudentDenoiser,
    teacher: TeacherDenoiser,
    loader: DataLoader,
    diffusion: DiffusionSchedule,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> tuple:
    student.train()

    total_loss       = 0.0
    total_ddpm_loss  = 0.0
    total_dist_loss  = 0.0

    for batch_idx, (x_0,) in enumerate(loader):
        x_0 = x_0.to(device)
        B   = x_0.shape[0]

        t = torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long)
        x_t, noise = diffusion.q_sample(x_0, t)

        # ── DDPM loss (student trains on true data noise) ──────────────────
        eps_student = student(x_t, t)
        ddpm_loss   = F.mse_loss(eps_student, noise)

        # ── Distillation loss (match teacher's x₀ prediction) ─────────────
        with torch.no_grad():
            eps_teacher = teacher(x_t, t)
            x0_teacher  = diffusion.predict_x0_from_eps(x_t, t, eps_teacher)

        x0_student  = diffusion.predict_x0_from_eps(x_t, t, eps_student)
        dist_loss   = F.mse_loss(x0_student, x0_teacher)

        # ── Combined loss ──────────────────────────────────────────────────
        loss = ALPHA * ddpm_loss + (1.0 - ALPHA) * dist_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss      += loss.item()
        total_ddpm_loss += ddpm_loss.item()
        total_dist_loss += dist_loss.item()

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            n = batch_idx + 1
            print(
                f"    [ep {epoch:03d} step {n:04d}]"
                f"  total={total_loss/n:.5f}"
                f"  ddpm={total_ddpm_loss/n:.5f}"
                f"  dist={total_dist_loss/n:.5f}",
                flush=True,
            )

    n = len(loader)
    return total_loss / n, total_ddpm_loss / n, total_dist_loss / n


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS, default=None,
                        help="Single latent dim to distil (for parallel GPU runs). "
                             "Omit to distil all dims sequentially.")
    args = parser.parse_args()

    dims = [args.dim] if args.dim is not None else LATENT_DIMS
    device = get_device(args.dim)
    print(f"Device: {device}  |  dims: {dims}")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    diffusion = DiffusionSchedule(T=T, device=device)

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

        # Load data
        latents_norm, mean, std = load_latents_normalised(dim)
        dataset = TensorDataset(torch.from_numpy(latents_norm))
        loader  = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=(device == "cuda"),
        )

        # Load frozen teacher
        teacher = load_teacher(str(teacher_path), latent_dim=dim, device=device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # Build student
        student = StudentDenoiser(latent_dim=dim).to(device)
        print(f"  Teacher params : {param_count(teacher)}")
        print(f"  Student params : {param_count(student)}")

        optimizer = AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

        best_loss = float("inf")
        history   = {"total": [], "ddpm": [], "distill": []}

        for epoch in tqdm(range(1, EPOCHS + 1), desc=f"dim={dim}"):
            avg_total, avg_ddpm, avg_dist = train_one_epoch(
                student, teacher, loader, diffusion, optimizer, device, epoch
            )
            scheduler.step()

            history["total"].append(avg_total)
            history["ddpm"].append(avg_ddpm)
            history["distill"].append(avg_dist)

            if avg_total < best_loss:
                best_loss = avg_total

            print(
                f"  epoch {epoch:03d}  total={avg_total:.5f}"
                f"  ddpm={avg_ddpm:.5f}  dist={avg_dist:.5f}"
                f"  best={best_loss:.5f}"
            )

        torch.save(
            {
                "model_state_dict":  student.state_dict(),
                "latent_dim":        dim,
                "latent_mean":       float(mean),
                "latent_std":        float(std),
                "T":                 T,
                "student_ddim_steps": STUDENT_DDIM_STEPS,
                "alpha":             ALPHA,
                "loss_history":      history,
            },
            out_path,
        )
        print(f"  Saved → {out_path}  (best_loss={best_loss:.5f})")

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
