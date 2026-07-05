import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.diffusion import FlowMatching
from models.denoiser import load_teacher
from exp_config import (
    get_paths, add_exp_arg, add_teacher_ckpt_arg,
    get_teacher_ckpt_path,
)


LATENT_DIMS = [64, 128, 256, 384, 512, 1024]


def load_stats(dim: int, real_latent_dir: Path):
    stats_path = real_latent_dir / f"latents_{dim}_norm_stats.npy"
    mean, std = np.load(stats_path)
    return mean.astype(np.float32), std.astype(np.float32)


def summarize(arr):
    norms = np.linalg.norm(arr, axis=1)
    return {
        "mean":       float(arr.mean()),
        "std":        float(arr.std()),
        "min":        float(arr.min()),
        "max":        float(arr.max()),
        "norm_mean":  float(norms.mean()),
        "norm_std":   float(norms.std()),
        "norm_p50":   float(np.percentile(norms, 50)),
        "norm_p90":   float(np.percentile(norms, 90)),
        "norm_p99":   float(np.percentile(norms, 99)),
    }


@torch.no_grad()
def sample_teacher(model, dim, device, steps, n_samples, batch_size):
    flow = FlowMatching(device=str(device))
    out = []
    generated = 0
    while generated < n_samples:
        b = min(batch_size, n_samples - generated)
        z = flow.euler_sample(model, (b, dim), n_steps=steps)
        out.append(z.cpu().numpy())
        generated += b
    return np.concatenate(out, axis=0).astype(np.float32)


def analyze_dim(dim, steps_list, n_samples, batch_size, device, paths, teacher_ckpt):
    print(f"\n=== dim={dim} device={device} ===")

    teacher_path = get_teacher_ckpt_path(paths, dim, teacher_ckpt)
    if not teacher_path.exists():
        raise FileNotFoundError(f"{teacher_path} not found — run step2 first.")

    print(f"  Teacher checkpoint : {teacher_path.name}  (--teacher-ckpt={teacher_ckpt})")

    real = np.load(paths.real_latent_dir / f"latents_{dim}.npy").astype(np.float32)
    mean, std = load_stats(dim, paths.real_latent_dir)
    real_norm = ((real - mean) / (std + 1e-8)).astype(np.float32)

    model = load_teacher(str(teacher_path), latent_dim=dim, device=str(device))
    model.eval()

    result = {
        "exp_name":       paths.exp_name,
        "teacher_ckpt":   teacher_ckpt,
        "checkpoint":     str(teacher_path),
        "real_norm":      summarize(real_norm),
        "samples":        {},
    }

    print("real_norm:", result["real_norm"])

    out_dir = paths.exp_dir / "analysis" / "teacher_sampling"
    out_dir.mkdir(parents=True, exist_ok=True)

    for steps in steps_list:
        print(f"  sampling Euler-{steps} ...")
        z_norm = sample_teacher(
            model=model, dim=dim, device=device,
            steps=steps, n_samples=n_samples, batch_size=batch_size,
        )
        z_orig = z_norm * std[None, :] + mean[None, :]

        result["samples"][str(steps)] = {
            "z_norm": summarize(z_norm),
            "z_orig": summarize(z_orig),
        }
        print(f"  Euler-{steps} z_norm:", result["samples"][str(steps)]["z_norm"])

    out_path = out_dir / f"teacher_sampling_dim_{dim}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    add_exp_arg(parser)
    add_teacher_ckpt_arg(parser)
    args = parser.parse_args()

    paths  = get_paths(args.exp_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    analyze_dim(
        dim=args.dim,
        steps_list=args.steps,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
        paths=paths,
        teacher_ckpt=args.teacher_ckpt,
    )


if __name__ == "__main__":
    main()
