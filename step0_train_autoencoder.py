"""
Step 0 — Train ConvAutoencoder from scratch on CIFAR-10 for each latent dim.

Saves: checkpoints/ae_{dim}.pt  (state_dict + latent_dim)

Usage:
    python step0_train_autoencoder.py            # all dims sequentially
    python step0_train_autoencoder.py --dim 64   # single dim (for parallel runs)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.autoencoder import ConvAutoencoder

LATENT_DIMS  = [64, 128, 256, 384, 512, 1024]
EPOCHS       = 150
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
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
    opt    = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched  = CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for imgs, _ in tqdm(loader, desc=f"  Epoch {epoch}/{EPOCHS}", leave=False):
            imgs = imgs.to(device)
            recon = model(imgs)
            loss  = loss_fn(recon, imgs)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        sched.step()
        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"latent_dim": dim, "state_dict": model.state_dict()}, save_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.6f}  lr={sched.get_last_lr()[0]:.2e}"
                  f"{'  [saved]' if avg_loss == best_loss else ''}")

    print(f"Done. Best loss={best_loss:.6f}  →  {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS,
                        help="Single latent dim to train (omit for all dims sequentially)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims   = [args.dim] if args.dim else LATENT_DIMS

    for dim in dims:
        train_one_dim(dim, device)


if __name__ == "__main__":
    main()
