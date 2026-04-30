"""
Step 3a — Generate synthetic latent datasets from trained teacher models.

For each latent dim:
  - Runs teacher Euler sampling to produce 4 synthetic x_0 datasets:
      250k, 500k, 1M, 2M samples (normalized latents, float32)
    Saves: synthetic/{dim}/synthetic_{dim}_{n}.npy
  - Generates a trajectory dataset for TRAJ_SAMPLES samples:
      shape (TRAJ_SAMPLES, EULER_STEPS+1, dim), float16
      traj[:, 0] = x_1 (start noise), traj[:, -1] = x_0 (final data)
    Saves: synthetic/{dim}/trajectories_{dim}.npy

All outputs are in normalized latent space (same space the teacher was trained in).
Skips any file that already exists — safe to restart.

Usage:
    python step3a_generate.py              # all dims sequentially
    python step3a_generate.py --dim 128    # single dim (for parallel GPU runs)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.diffusion import FlowMatching
from models.denoiser import load_teacher

LATENT_DIMS     = [64, 128, 256, 384, 512, 1024]
DATASET_SIZES   = [250_000, 500_000, 1_000_000, 2_000_000]
TRAJ_SAMPLES    = 50_000    # trajectory dataset size (storage-bounded)
EULER_STEPS     = 50
GEN_BATCH       = 2_048
MODEL_DIR       = Path("models")
SYNTHETIC_DIR   = Path("synthetic")


def get_device(dim: int = None) -> torch.device:
    if torch.cuda.is_available():
        if dim is not None:
            gpu_id = LATENT_DIMS.index(dim) % torch.cuda.device_count()
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def generate_dataset(
    model: torch.nn.Module,
    flow: FlowMatching,
    dim: int,
    n_samples: int,
    out_path: Path,
) -> None:
    """Generate n_samples via Euler sampling and save as float32 memmap."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.memmap(str(out_path), dtype="float32", mode="w+", shape=(n_samples, dim))

    generated = 0
    with tqdm(total=n_samples, desc=f"    generating {n_samples:,}", unit="samples") as pbar:
        while generated < n_samples:
            batch_n = min(GEN_BATCH, n_samples - generated)
            z = flow.euler_sample(model, (batch_n, dim), n_steps=EULER_STEPS)
            out[generated : generated + batch_n] = z.cpu().numpy()
            generated += batch_n
            pbar.update(batch_n)

    del out  # flush memmap to disk
    print(f"    Saved → {out_path}  ({n_samples:,} × {dim}  float32)")


def generate_trajectories(
    model: torch.nn.Module,
    flow: FlowMatching,
    dim: int,
    out_path: Path,
) -> None:
    """
    Generate TRAJ_SAMPLES trajectories with all EULER_STEPS+1 intermediate states.
    Saved as float16 memmap of shape (TRAJ_SAMPLES, EULER_STEPS+1, dim).
    traj[:, 0] = starting noise x_1, traj[:, -1] = final sample x_0.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = EULER_STEPS + 1
    out = np.memmap(
        str(out_path), dtype="float16", mode="w+",
        shape=(TRAJ_SAMPLES, n_frames, dim),
    )

    generated = 0
    with tqdm(total=TRAJ_SAMPLES, desc=f"    trajectories", unit="samples") as pbar:
        while generated < TRAJ_SAMPLES:
            batch_n = min(GEN_BATCH, TRAJ_SAMPLES - generated)
            _, traj = flow.euler_sample(
                model, (batch_n, dim),
                n_steps=EULER_STEPS,
                return_trajectory=True,
            )
            out[generated : generated + batch_n] = traj.astype(np.float16)
            generated += batch_n
            pbar.update(batch_n)

    del out
    size_gb = TRAJ_SAMPLES * n_frames * dim * 2 / 1e9
    print(f"    Saved → {out_path}  ({TRAJ_SAMPLES:,} × {n_frames} × {dim}  float16  ~{size_gb:.1f} GB)")


def generate_for_dim(dim: int, device: torch.device) -> None:
    print(f"\n{'='*60}")
    print(f"Generating synthetic data  latent_dim={dim}  device={device}")
    print(f"{'='*60}")

    teacher_path = MODEL_DIR / f"teacher_{dim}.pt"
    if not teacher_path.exists():
        print(f"[ERROR] {teacher_path} not found — run step2 first.")
        return

    model = load_teacher(str(teacher_path), latent_dim=dim, device=str(device))
    flow  = FlowMatching(device=str(device))

    dim_dir = SYNTHETIC_DIR / str(dim)

    # ── synthetic x_0 datasets ────────────────────────────────────────────────
    for n in DATASET_SIZES:
        out_path = dim_dir / f"synthetic_{dim}_{n}.npy"
        if out_path.exists():
            print(f"  [skip] {out_path.name} already exists.")
            continue
        print(f"  Generating dataset  n={n:,} …")
        generate_dataset(model, flow, dim, n, out_path)

    # ── trajectory dataset (250k only, float16) ───────────────────────────────
    traj_path = dim_dir / f"trajectories_{dim}.npy"
    if traj_path.exists():
        print(f"  [skip] {traj_path.name} already exists.")
    else:
        print(f"  Generating trajectories  n={TRAJ_SAMPLES:,}  steps={EULER_STEPS} …")
        generate_trajectories(model, flow, dim, traj_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS,
                        help="Single latent dim to generate (omit for all dims sequentially)")
    args = parser.parse_args()

    dims = [args.dim] if args.dim else LATENT_DIMS
    for dim in dims:
        generate_for_dim(dim, get_device(dim))

    print("\nStep 3a complete.")


if __name__ == "__main__":
    main()
