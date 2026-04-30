"""
Step 1 — Encode all 50k CIFAR-10 training images with trained ConvAutoencoders.

Requires: checkpoints/ae_{dim}.pt  (produced by step0)
Saves:    latents/latents_{dim}.npy  — shape (50000, dim), float32

Usage:
    python step1_extract_latents.py            # all dims sequentially
    python step1_extract_latents.py --dim 64   # single dim
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.autoencoder import ConvAutoencoder

LATENT_DIMS = [64, 128, 256, 384, 512, 1024]
BATCH_SIZE  = 512
CKPT_DIR    = Path("checkpoints")
LATENT_DIR  = Path("latents")


def load_ae(dim: int, device: torch.device) -> ConvAutoencoder:
    ckpt_path = CKPT_DIR / f"ae_{dim}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — run step0 first.")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = ConvAutoencoder(latent_dim=dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def extract_latents(dim: int, device: torch.device) -> None:
    print(f"\n{'='*60}")
    print(f"Extracting latents  latent_dim={dim}  device={device}")
    print(f"{'='*60}")

    LATENT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LATENT_DIR / f"latents_{dim}.npy"

    tf = transforms.Compose([transforms.ToTensor()])  # → [0, 1], matches step0 training
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=tf)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)

    model = load_ae(dim, device)

    all_latents = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=f"  Encoding (dim={dim})"):
            imgs = imgs.to(device)
            z    = model.encode(imgs)
            all_latents.append(z.cpu().numpy())

    latents = np.concatenate(all_latents, axis=0).astype(np.float32)
    print(f"  Latents shape : {latents.shape}")
    print(f"  Value range   : [{latents.min():.3f}, {latents.max():.3f}]")
    np.save(out_path, latents)
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS,
                        help="Single latent dim to extract (omit for all dims sequentially)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims   = [args.dim] if args.dim else LATENT_DIMS

    for dim in dims:
        extract_latents(dim, device)


if __name__ == "__main__":
    main()
