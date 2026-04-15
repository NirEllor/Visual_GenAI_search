"""
Step 1 — Extract latents from frozen JAX/Flax autoencoders.

For each latent dimension:
  1. Download/load the checkpoint.
  2. Load the pretrained JAX/Flax autoencoder.
  3. Encode the full CIFAR-10 training set (50,000 images).
  4. Save the latent vectors as  latents/latents_{dim}.npy

Output files
------------
latents/latents_{dim}.npy   shape (50000, dim)
"""

import argparse
import numpy as np
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T

LATENT_DIMS   = [8, 16, 32, 64, 128, 256, 384]
DATA_DIR      = "data"
CKPT_DIR      = "checkpoints"
LATENT_DIR    = "latents"
BATCH_SIZE    = 256

CHECKPOINT_NAMES = {
    8:   "cifar10_8_custom.ckpt",
    16:  "cifar10_16_custom.ckpt",
    32:  "cifar10_32_custom.ckpt",
    64:  "cifar10_64.ckpt",
    128: "cifar10_128.ckpt",
    256: "cifar10_256.ckpt",
    384: "cifar10_384.ckpt",
}


def get_cifar10_numpy() -> np.ndarray:
    transform = T.Compose([T.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    all_images = []
    for imgs, _ in loader:
        imgs = imgs.permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 2.0) - 1.0
        all_images.append(imgs)
    return np.concatenate(all_images, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS, default=None,
                        help="Single latent dim to extract. Omit to extract all.")
    args = parser.parse_args()

    dims = [args.dim] if args.dim is not None else LATENT_DIMS

    from models.autoencoder_jax import (
        download_checkpoints,
        load_autoencoder,
        encode_dataset,
    )

    Path(LATENT_DIR).mkdir(parents=True, exist_ok=True)

    # Download phlippe checkpoints (64/128/256/384 only)
    download_checkpoints(CKPT_DIR)

    # Load CIFAR-10 once
    print("\nLoading CIFAR-10 training set …")
    images = get_cifar10_numpy()
    print(f"  images shape: {images.shape}  range: [{images.min():.2f}, {images.max():.2f}]")

    for dim in dims:
        out_path = Path(LATENT_DIR) / f"latents_{dim}.npy"
        if out_path.exists():
            print(f"\n[skip] latents_{dim}.npy already exists.")
            continue

        ckpt_path = Path(CKPT_DIR) / CHECKPOINT_NAMES[dim]
        if not ckpt_path.exists():
            print(f"\n[ERROR] {ckpt_path} not found — run step0 first for dim={dim}.")
            continue

        print(f"\n{'='*60}")
        print(f"  latent_dim = {dim}")
        print(f"  checkpoint  = {ckpt_path}")

        model, params = load_autoencoder(str(ckpt_path), latent_dim=dim)

        print("  Encoding CIFAR-10 …")
        latents = encode_dataset(model, params, images, batch_size=BATCH_SIZE)

        print(f"  latents shape : {latents.shape}")
        print(f"  latents range : [{latents.min():.3f}, {latents.max():.3f}]")

        np.save(out_path, latents)
        print(f"  Saved → {out_path}")

        # Sanity check
        from models.autoencoder_jax import decode_latents
        recon = decode_latents(model, params, latents[:8], batch_size=8)
        assert recon.shape == (8, 32, 32, 3)
        print(f"  Decode sanity check passed: {recon.shape} uint8")

    print("\nStep 1 complete.")


if __name__ == "__main__":
    main()