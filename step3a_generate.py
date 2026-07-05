"""
Step 3a — Generate synthetic latent datasets from trained teacher models.

For each latent dim:
  - Runs teacher Euler sampling to produce 4 synthetic x_0 datasets.
    Saves: results/<exp_name>/latents/teacher/dim_<dim>/synthetic_<dim>_<n>.npy
  - Generates a trajectory dataset for TRAJ_SAMPLES samples.
    Saves: results/<exp_name>/latents/teacher/dim_<dim>/trajectories_<dim>.npy

All outputs are in normalised latent space (same space the teacher was trained in).
Skips any file that already exists by default — pass --overwrite to regenerate.

Usage:
    python step3a_generate.py                             # all dims sequentially
    python step3a_generate.py --dim 128                   # single dim
    python step3a_generate.py --dim 128 --overwrite       # force regeneration
    python step3a_generate.py --exp-name my_run           # custom experiment
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.diffusion import FlowMatching
from models.denoiser import load_teacher
from exp_config import (
    get_paths, add_exp_arg, add_teacher_ckpt_arg,
    get_teacher_ckpt_path, print_exp_summary, ExpPaths,
)

LATENT_DIMS     = [64, 128, 256, 384, 512, 1024]
DATASET_SIZES   = [50_000, 100_000, 150_000, 200_000]
TRAJ_SAMPLES    = 200_000    # trajectory dataset size (storage-bounded)
EULER_STEPS     = 200
GEN_BATCH       = 128


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
    overwrite: bool,
) -> None:
    """Generate n_samples via Euler sampling and save as float32 memmap."""
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path.name} already exists.")
        return
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
    overwrite: bool,
) -> None:
    """
    Generate TRAJ_SAMPLES trajectories with all EULER_STEPS+1 intermediate states.
    Saved as float16 memmap of shape (TRAJ_SAMPLES, EULER_STEPS+1, dim).
    traj[:, 0] = starting noise x_1, traj[:, -1] = final sample x_0.
    """
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path.name} already exists.")
        return
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


def generate_for_dim(dim: int, device: torch.device,
                     paths: ExpPaths, overwrite: bool,
                     teacher_ckpt: str = "best_fid") -> None:
    print(f"\n{'='*60}")
    print(f"Generating synthetic data  latent_dim={dim}  device={device}")
    print(f"{'='*60}")

    teacher_path = get_teacher_ckpt_path(paths, dim, teacher_ckpt)
    if not teacher_path.exists():
        print(f"[ERROR] {teacher_path} not found — run step2 first.")
        return

    print(f"  Teacher checkpoint : {teacher_path.name}  (--teacher-ckpt={teacher_ckpt})")
    model = load_teacher(str(teacher_path), latent_dim=dim, device=str(device))
    flow  = FlowMatching(device=str(device))

    dim_dir = paths.teacher_latent_dir / f"dim_{dim}"

    for n in DATASET_SIZES:
        out_path = dim_dir / f"synthetic_{dim}_{n}.npy"
        print(f"  Generating dataset  n={n:,} …")
        generate_dataset(model, flow, dim, n, out_path, overwrite)

    traj_path = dim_dir / f"trajectories_{dim}.npy"
    print(f"  Generating trajectories  n={TRAJ_SAMPLES:,}  steps={EULER_STEPS} …")
    generate_trajectories(model, flow, dim, traj_path, overwrite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS,
                        help="Single latent dim to generate (omit for all dims sequentially)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate even if output files already exist")
    add_exp_arg(parser)
    add_teacher_ckpt_arg(parser)
    args = parser.parse_args()

    paths = get_paths(args.exp_name)

    print_exp_summary(
        paths,
        ckpt_path=paths.ckpt_dir,
        latent_path=paths.teacher_latent_dir,
    )

    dims = [args.dim] if args.dim else LATENT_DIMS
    for dim in dims:
        generate_for_dim(dim, get_device(dim), paths, args.overwrite, args.teacher_ckpt)

    print("\nStep 3a complete.")


if __name__ == "__main__":
    main()
