import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.autoencoder import ConvAutoencoder

LATENT_DIMS = [64, 128, 256, 384, 512, 1024]
N_SAMPLES = 10_000
DECODE_BATCH = 256

LATENT_DIR = Path("latents")
CKPT_DIR = Path("checkpoints")
OUT_DIR = Path("results/gaussian_baseline")


def compute_fid(gen_dir: str) -> float:
    from cleanfid import fid
    return float(fid.compute_fid(
        gen_dir,
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split="train",
        verbose=False,
    ))


def run_dim(dim: int, device: str):
    print(f"\n=== Gaussian baseline dim={dim} ===")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen_dir = OUT_DIR / f"generated_gaussian_{dim}"
    z_path = OUT_DIR / f"z_gaussian_{dim}.npy"
    metrics_path = OUT_DIR / f"metrics_gaussian_{dim}.json"

    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)

    latents = np.load(LATENT_DIR / f"latents_{dim}.npy").astype(np.float32)
    mu = latents.mean(axis=0)
    sigma = latents.std(axis=0) + 1e-8

    rng = np.random.default_rng(0)
    z = rng.normal(
        loc=mu[None, :],
        scale=sigma[None, :],
        size=(N_SAMPLES, dim),
    ).astype(np.float32)

    np.save(z_path, z)
    print(f"Saved latents → {z_path}")

    ckpt = torch.load(CKPT_DIR / f"ae_{dim}.pt", map_location=device, weights_only=True)
    ae = ConvAutoencoder(latent_dim=dim).to(device)
    ae.load_state_dict(ckpt["state_dict"])
    ae.eval()

    z_t = torch.from_numpy(z)

    img_idx = 0
    with torch.no_grad():
        for start in tqdm(range(0, N_SAMPLES, DECODE_BATCH), desc=f"Decoding dim={dim}"):
            z_b = z_t[start:start + DECODE_BATCH].to(device)

            recon_logits = ae.decode(z_b)
            recon = (
                torch.sigmoid(recon_logits)
                .clamp(0, 1)
                .mul(255)
                .byte()
                .cpu()
                .numpy()
            )
            recon = recon.transpose(0, 2, 3, 1)

            for img_arr in recon:
                Image.fromarray(img_arr).save(gen_dir / f"{img_idx:05d}.png")
                img_idx += 1

    fid_score = compute_fid(str(gen_dir))
    print(f"Gaussian baseline FID dim={dim}: {fid_score:.2f}")

    with open(metrics_path, "w") as f:
        json.dump({
            "dim": dim,
            "n_samples": N_SAMPLES,
            "fid": fid_score,
            "latent_mean": float(z.mean()),
            "latent_std": float(z.std()),
        }, f, indent=2)

    print(f"Saved metrics → {metrics_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS)
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.dim is not None:
        dims = [args.dim]
    elif args.dims is not None:
        dims = args.dims
    else:
        dims = [128, 256, 512, 1024]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dim in dims:
        run_dim(dim, device)


if __name__ == "__main__":
    main()