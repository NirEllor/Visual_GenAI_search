"""
Step 0b — Evaluate trained ConvAutoencoders on CIFAR-10 test set.

For each latent dim:
  1. Encode + decode the 10k CIFAR-10 test images
  2. Save reconstructed PNGs  →  results/ae_eval/reconstructed_{dim}/
  3. Save a visual sample grid (original vs reconstructed)
  4. Compute FID between reconstructions and real CIFAR-10 test images
  5. Save per-dim metrics  →  results/ae_eval/metrics_{dim}.json

--plot-only  merges all per-dim JSONs and plots AE FID vs latent dim.

Usage:
    python step0b_eval_ae.py               # all dims sequentially
    python step0b_eval_ae.py --dim 128     # single dim
    python step0b_eval_ae.py --plot-only   # plot from existing metrics
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.autoencoder import ConvAutoencoder

LATENT_DIMS  = [64, 128, 256, 384, 512, 1024]
BATCH_SIZE   = 256
CKPT_DIR     = Path("checkpoints")
RESULTS_DIR  = Path("results/ae_eval")
N_GRID       = 8   # grid is N_GRID x N_GRID pairs


def load_ae(dim: int, device: torch.device) -> ConvAutoencoder:
    ckpt_path = CKPT_DIR / f"ae_{dim}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — run step0 first.")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = ConvAutoencoder(latent_dim=dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def eval_one_dim(dim: int, device: torch.device) -> float:
    print(f"\n{'='*60}")
    print(f"Evaluating AE  latent_dim={dim}  device={device}")
    print(f"{'='*60}")

    model   = load_ae(dim, device)
    out_dir = RESULTS_DIR / f"reconstructed_{dim}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tf      = transforms.ToTensor()
    testset = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    loader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)

    all_orig, all_recon = [], []
    img_idx = 0

    # 🔹 MSE accumulators (on raw outputs!)
    mse_sum = 0.0
    n_pixels = 0

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=f"  Reconstructing (dim={dim})"):
            imgs  = imgs.to(device)
            recon = model(imgs)

            # ✅ true MSE (before clamp!)
            mse_sum += torch.sum((recon - imgs) ** 2).item()
            n_pixels += imgs.numel()

            # 🔹 only for visualization / FID
            recon_vis = recon.clamp(0, 1)

            orig_np  = (imgs.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            recon_np = (recon_vis.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

            all_orig.append(orig_np)
            all_recon.append(recon_np)

            for img_arr in recon_np:
                Image.fromarray(img_arr).save(out_dir / f"{img_idx:05d}.png")
                img_idx += 1

    print(f"  Saved {img_idx} reconstructed images → {out_dir}/")

    # ── sample grid ─────────────────────────────────────────────────────────
    orig_all  = np.concatenate(all_orig,  axis=0)
    recon_all = np.concatenate(all_recon, axis=0)
    save_sample_grid(orig_all, recon_all, dim)

    # ── MSE (correct computation) ───────────────────────────────────────────
    mse_01  = mse_sum / n_pixels
    mse_255 = mse_01 * (255 ** 2)

    print(f"  MSE (raw, 0-1 scale):   {mse_01:.6f}")
    print(f"  MSE (raw, 0-255 scale): {mse_255:.4f}")

    # ── FID ─────────────────────────────────────────────────────────────────
    fid_score = compute_fid(str(out_dir))
    print(f"  FID: {fid_score:.2f}")

    metrics_path = RESULTS_DIR / f"metrics_{dim}.json"
    with open(metrics_path, "w") as f:
        json.dump({str(dim): {"fid": fid_score, "mse": mse_255}}, f, indent=2)

    print(f"  Metrics saved → {metrics_path}")

    return fid_score

def save_sample_grid(orig: np.ndarray, recon: np.ndarray, dim: int) -> None:
    n = N_GRID
    fig, axes = plt.subplots(n * 2, n, figsize=(n * 1.5, n * 3))
    fig.suptitle(f"AE Reconstructions  (latent_dim={dim})\n"
                 f"Top: original  |  Bottom: reconstructed", fontsize=10)

    indices = np.random.choice(len(orig), size=n * n, replace=False)
    for col in range(n):
        for row in range(n):
            idx = indices[row * n + col]
            axes[row * 2,     col].imshow(orig[idx])
            axes[row * 2 + 1, col].imshow(recon[idx])
            axes[row * 2,     col].axis("off")
            axes[row * 2 + 1, col].axis("off")

    plt.tight_layout()
    grid_path = RESULTS_DIR / f"sample_grid_{dim}.png"
    plt.savefig(str(grid_path), dpi=120)
    plt.close()
    print(f"  Sample grid saved → {grid_path}")


def compute_fid(gen_dir: str) -> float:
    try:
        from cleanfid import fid
        return float(fid.compute_fid(gen_dir, dataset_name="cifar10",
                                     dataset_res=32, dataset_split="test",
                                     verbose=False))
    except ImportError:
        print("  [warning] clean-fid not installed — FID not computed.")
        return -1.0


def plot_only() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for dim in LATENT_DIMS:
        p = RESULTS_DIR / f"metrics_{dim}.json"
        if not p.exists():
            print(f"[warning] {p} not found — skipping dim {dim}.")
            continue
        with open(p) as f:
            metrics[dim] = json.load(f)[str(dim)]

    if not metrics:
        print("[ERROR] No per-dim metrics found. Run without --plot-only first.")
        return

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    valid_dims = [d for d in LATENT_DIMS if d in metrics]
    fid_scores = [metrics[d]["fid"] for d in valid_dims]
    mse_scores = [metrics[d]["mse"] for d in valid_dims]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(valid_dims, fid_scores, marker="o", linewidth=2, color="royalblue")
    axes[0].set_xlabel("Latent Dim")
    axes[0].set_ylabel("FID (↓)")
    axes[0].set_title("AE Reconstruction FID vs Latent Dim")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(valid_dims, mse_scores, marker="s", linewidth=2, color="darkorange")
    axes[1].set_xlabel("Latent Dim")
    axes[1].set_ylabel("MSE (pixel, 0-255 scale) (↓)")
    axes[1].set_title("AE Reconstruction MSE vs Latent Dim")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    out_png = RESULTS_DIR / "ae_fid_vs_dim.png"
    plt.savefig(str(out_png), dpi=150)
    plt.close()
    print(f"Plot saved → {out_png}")

    print("\nAE evaluation summary:")
    for dim in valid_dims:
        print(f"  dim={dim:4d}  FID={metrics[dim]['fid']:7.2f}  MSE={metrics[dim]['mse']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS,
                        help="Single latent dim to evaluate (omit for all dims sequentially)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Merge existing per-dim JSONs and plot — no GPU needed")
    args = parser.parse_args()

    if args.plot_only:
        plot_only()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims   = [args.dim] if args.dim else LATENT_DIMS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for dim in dims:
        eval_one_dim(dim, device)

    if not args.dim:
        plot_only()


if __name__ == "__main__":
    main()
