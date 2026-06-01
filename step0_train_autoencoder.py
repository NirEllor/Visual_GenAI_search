"""
Step 0 — Train ConvAutoencoder from scratch on CIFAR-10 for each latent dim.

Saves: checkpoints/ae_{dim}.pt  (state_dict + latent_dim)

Usage:
    python step0_train_autoencoder.py            # all dims sequentially
    python step0_train_autoencoder.py --dim 64   # single dim (for parallel runs)
"""

import argparse
import lpips
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from models.autoencoder import ConvAutoencoder
import torch.nn.functional as F

LATENT_DIMS  = [64, 128, 256, 384, 512, 1024]
EPOCHS       = 1000
BATCH_SIZE   = 128
LR           = 2e-3
WEIGHT_DECAY = 0
GRAD_CLIP    = 5.0
LPIPS_WEIGHT = 1.0
L1_WEIGHT    = 0.0
KL_WEIGHT    = 0.001  # β-KL: forces latent space toward N(0,I) for flow matching
CKPT_DIR     = Path("checkpoints")


def get_cifar10_loader(batch_size: int) -> DataLoader:
    tf = transforms.Compose([
        transforms.ToTensor(),   # → [0, 1]
    ])
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=tf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=2, pin_memory=True)


def train_one_dim(dim: int, device: torch.device) -> None:
    print(f"\n{'='*60}")
    print(f"Training ConvAutoencoder  latent_dim={dim}  device={device}")
    print(f"{'='*60}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CKPT_DIR / f"ae_{dim}.pt"

    loader = get_cifar10_loader(BATCH_SIZE)
    model  = ConvAutoencoder(latent_dim=dim).to(device)
    opt      = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched    = CosineAnnealingLR(opt, T_max=EPOCHS)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn.eval()  # frozen AlexNet backbone — only AE weights train
    for p in lpips_fn.parameters():
        p.requires_grad = False
    l1_fn    = nn.L1Loss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for imgs, _ in tqdm(loader, desc=f"  Epoch {epoch}/{EPOCHS}", leave=False):
            imgs = imgs.to(device)

            recon_logits, mean, logvar = model(imgs)
            recon_for_loss = torch.sigmoid(recon_logits)

            recon_lpips = F.interpolate(
                recon_for_loss,
                size=(64, 64),
                mode="bilinear",
                align_corners=False
            )

            imgs_lpips = F.interpolate(
                imgs,
                size=(64, 64),
                mode="bilinear",
                align_corners=False
            )

            lpips_loss = lpips_fn(
                recon_lpips * 2 - 1,
                imgs_lpips * 2 - 1
            ).mean()

            l1_loss = l1_fn(recon_for_loss, imgs)

            kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean()

            loss = LPIPS_WEIGHT * lpips_loss + L1_WEIGHT * l1_loss + KL_WEIGHT * kl_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        sched.step()
        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"latent_dim": dim, "state_dict": model.state_dict()}, save_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss(lpips+l1)={avg_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}"
                  f"{'  [saved]' if avg_loss == best_loss else ''}")

    print(f"Done. Best loss={best_loss:.6f}  →  {save_path}")


def get_device(dim: int = None) -> torch.device:
    if torch.cuda.is_available():
        if dim is not None:
            gpu_id = LATENT_DIMS.index(dim) % torch.cuda.device_count()
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS,
                        help="Single latent dim to train (omit for all dims sequentially)")
    args = parser.parse_args()

    dims = [args.dim] if args.dim else LATENT_DIMS

    for dim in dims:
        train_one_dim(dim, get_device(dim))


if __name__ == "__main__":
    main()
