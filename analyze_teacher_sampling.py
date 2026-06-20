import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.diffusion import FlowMatching
from models.denoiser import load_teacher


LATENT_DIMS = [64, 128, 256, 384, 512, 1024]
MODEL_DIR = Path("models")
LATENT_DIR = Path("latents")
OUT_DIR = Path("results/teacher_sampling_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_stats(dim):
    mean, std = np.load(LATENT_DIR / f"latents_{dim}_norm_stats.npy")
    return mean.astype(np.float32), std.astype(np.float32)


def summarize(arr):
    norms = np.linalg.norm(arr, axis=1)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_p50": float(np.percentile(norms, 50)),
        "norm_p90": float(np.percentile(norms, 90)),
        "norm_p99": float(np.percentile(norms, 99)),
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


def analyze_dim(dim, steps_list, n_samples, batch_size, device):
    print(f"\n=== dim={dim} device={device} ===")

    teacher_path = MODEL_DIR / f"teacher_{dim}.pt"
    if not teacher_path.exists():
        raise FileNotFoundError(f"{teacher_path} not found")

    real = np.load(LATENT_DIR / f"latents_{dim}.npy").astype(np.float32)
    mean, std = load_stats(dim)
    real_norm = ((real - mean) / (std + 1e-8)).astype(np.float32)

    model = load_teacher(str(teacher_path), latent_dim=dim, device=str(device))
    model.eval()

    result = {
        "real_norm": summarize(real_norm),
        "samples": {}
    }

    print("real_norm:", result["real_norm"])

    for steps in steps_list:
        print(f"  sampling Euler-{steps} ...")
        z_norm = sample_teacher(
            model=model,
            dim=dim,
            device=device,
            steps=steps,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        z_orig = z_norm * std[None, :] + mean[None, :]

        result["samples"][str(steps)] = {
            "z_norm": summarize(z_norm),
            "z_orig": summarize(z_orig),
        }

        print(f"  Euler-{steps} z_norm:", result["samples"][str(steps)]["z_norm"])

    out_path = OUT_DIR / f"teacher_sampling_dim_{dim}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    analyze_dim(
        dim=args.dim,
        steps_list=args.steps,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()