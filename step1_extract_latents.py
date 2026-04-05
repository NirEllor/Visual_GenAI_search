"""
Step 1 — Extract latents from frozen JAX/Flax autoencoders.

For each of the 4 autoencoder checkpoints (latent dims 64 / 128 / 256 / 384):
  1. Download the checkpoint (if not already present).
  2. Load the pretrained JAX/Flax autoencoder.
  3. Encode the full CIFAR-10 training set (50,000 images).
  4. Save the latent vectors as  latents/latents_{dim}.npy   shape (50000, dim).

Output files
------------
latents/latents_64.npy
latents/latents_128.npy
latents/latents_256.npy
latents/latents_384.npy
"""

import numpy as np
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T

# ── paths ─────────────────────────────────────────────────────────────────────
LATENT_DIMS   = [64, 128, 256, 384]
DATA_DIR      = "data"
CKPT_DIR      = "checkpoints"
LATENT_DIR    = "latents"
BATCH_SIZE    = 256

# ── data ──────────────────────────────────────────────────────────────────────

def get_cifar10_numpy() -> np.ndarray:
    """
    Load CIFAR-10 training set and return as float32 numpy array.

    Returns
    -------
    images : (50000, 32, 32, 3)  float32 in [-1, 1]
             channels-last (JAX convention)
    """
    # torchvision stores images as (C, H, W) in [0,1] after ToTensor
    transform = T.Compose([T.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    all_images = []
    for imgs, _ in loader:
        # (B, 3, 32, 32) float32 in [0,1]  →  (B, 32, 32, 3) in [-1, 1]
        imgs = imgs.permute(0, 2, 3, 1).numpy()   # channels last
        imgs = (imgs * 2.0) - 1.0                 # [0,1] → [-1,1]
        all_images.append(imgs)

    return np.concatenate(all_images, axis=0).astype(np.float32)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    from models.autoencoder_jax import (
        download_checkpoints,
        load_autoencoder,
        encode_dataset,
        CHECKPOINT_NAMES,
    )

    Path(LATENT_DIR).mkdir(parents=True, exist_ok=True)

    # Download checkpoints once
    download_checkpoints(CKPT_DIR)

    # Load CIFAR-10 once (shared across all dims)
    print("\nLoading CIFAR-10 training set …")
    images = get_cifar10_numpy()
    print(f"  images shape: {images.shape}  range: [{images.min():.2f}, {images.max():.2f}]")

    for dim in LATENT_DIMS:
        out_path = Path(LATENT_DIR) / f"latents_{dim}.npy"
        if out_path.exists():
            print(f"\n[skip] latents_{dim}.npy already exists.")
            continue

        ckpt_path = Path(CKPT_DIR) / CHECKPOINT_NAMES[dim]
        print(f"\n{'='*60}")
        print(f"  latent_dim = {dim}")
        print(f"  checkpoint  = {ckpt_path}")

        # Load pretrained autoencoder (weights frozen — only inference)
        model, params = load_autoencoder(str(ckpt_path), latent_dim=dim)

        # Encode
        print("  Encoding CIFAR-10 …")
        latents = encode_dataset(model, params, images, batch_size=BATCH_SIZE)

        print(f"  latents shape : {latents.shape}")
        print(f"  latents range : [{latents.min():.3f}, {latents.max():.3f}]")
        print(f"  latents mean  : {latents.mean():.4f}  std: {latents.std():.4f}")

        np.save(out_path, latents)
        print(f"  Saved → {out_path}")

        # Sanity-check: decode one batch and verify shapes
        from models.autoencoder_jax import decode_latents
        sample_latents = latents[:8]
        recon = decode_latents(model, params, sample_latents, batch_size=8)
        assert recon.shape == (8, 32, 32, 3), f"Unexpected decoded shape: {recon.shape}"
        print(f"  Decode sanity check passed: {recon.shape} uint8")

    print("\nStep 1 complete. Latent files saved in:", LATENT_DIR)


if __name__ == "__main__":
    main()
