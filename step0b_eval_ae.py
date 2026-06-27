"""
Step 0b — Evaluate trained ConvAutoencoders on CIFAR-10 test set.

For each latent dim:
  1. Encode + decode the 10k CIFAR-10 test images
  2. Save reconstructed PNGs  →  results/<exp_name>/ae_recon/dim_<dim>/
  3. Save a visual sample grid  →  results/<exp_name>/plots/ae_grid_<dim>.png
  4. Compute FID between reconstructions and real CIFAR-10 test images
  5. Save per-dim metrics  →  results/<exp_name>/metrics/ae_metrics_<dim>.json

--plot-only  merges all per-dim JSONs and plots AE FID vs latent dim.

Usage:
    python step0b_eval_ae.py                           # all dims sequentially
    python step0b_eval_ae.py --dim 128                 # single dim
    python step0b_eval_ae.py --plot-only               # plot from existing metrics
    python step0b_eval_ae.py --dim 128 --overwrite     # regenerate even if output exists
    python step0b_eval_ae.py --exp-name my_run         # custom experiment
"""

import argparse
import json
from pathlib import Path
import lpips as lpips_lib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.autoencoder import ConvAutoencoder

from exp_config import (
    get_paths, add_exp_arg, print_exp_summary,
    maybe_clear_dir, ExpPaths,
)

LATENT_DIMS  = [64, 128, 256, 384, 512, 1024]
BATCH_SIZE   = 256
N_GRID       = 8   # grid is N_GRID × N_GRID pairs


def load_ae(dim: int, ckpt_dir: Path, device: torch.device) -> ConvAutoencoder:
    ckpt_path = ckpt_dir / f"ae_{dim}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — run step0 first.")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = ConvAutoencoder(latent_dim=dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def eval_one_dim(
    dim: int,
    device: torch.device,
    paths: ExpPaths,
    overwrite: bool,
) -> float:
    print(f"\n{'='*60}")
    print(f"Evaluating AE  latent_dim={dim}  device={device}")
    print(f"{'='*60}")

    model    = load_ae(dim, paths.ckpt_dir, device)
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    out_dir = paths.ae_recon_dir / f"dim_{dim}"
    metrics_file = paths.metrics_dir / f"ae_metrics_{dim}.json"

    if not maybe_clear_dir(out_dir, overwrite, label=f"ae_recon/dim_{dim}"):
        # Directory exists and overwrite=False; check if metrics exist too
        if metrics_file.exists():
            with open(metrics_file) as fh:
                saved = json.load(fh)
            return saved.get(str(dim), {}).get("fid", -1.0)
        # Images exist but metrics don't — fall through to compute metrics only
        print("  [info] Images exist; recomputing metrics only.")
        fid_score = compute_fid(str(out_dir))
        _save_metrics(metrics_file, dim, fid_score, 0.0, 0.0)
        return fid_score

    tf      = transforms.ToTensor()
    testset = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    loader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)

    all_orig, all_recon = [], []
    img_idx = 0

    l1_sum         = 0.0
    n_pixels       = 0
    lpips_sum      = 0.0
    n_batches_eval = 0

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=f"  Reconstructing (dim={dim})"):
            imgs  = imgs.to(device)
            recon_logits, _, _ = model(imgs)
            recon = torch.sigmoid(recon_logits)

            l1_sum += torch.sum(torch.abs(recon - imgs)).item()
            n_pixels += imgs.numel()

            recon_eval = recon.clamp(0, 1)

            recon_lpips = F.interpolate(recon_eval, size=(64, 64),
                                        mode="bilinear", align_corners=False)
            imgs_lpips  = F.interpolate(imgs, size=(64, 64),
                                        mode="bilinear", align_corners=False)
            lpips_sum += lpips_fn(recon_lpips * 2 - 1, imgs_lpips * 2 - 1).mean().item()
            n_batches_eval += 1

            orig_np  = (imgs.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            recon_np = (recon_eval.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            all_orig.append(orig_np)
            all_recon.append(recon_np)

            for img_arr in recon_np:
                Image.fromarray(img_arr).save(out_dir / f"{img_idx:05d}.png")
                img_idx += 1

    n_saved = len(list(out_dir.glob("*.png")))
    assert n_saved == len(testset), f"Expected {len(testset)}, got {n_saved}"
    print(f"  Saved {img_idx} reconstructed images → {out_dir}/")

    orig_all  = np.concatenate(all_orig,  axis=0)
    recon_all = np.concatenate(all_recon, axis=0)
    save_sample_grid(orig_all, recon_all, dim, paths.plots_dir)

    l1_01     = l1_sum / n_pixels
    l1_255    = l1_01 * 255
    avg_lpips = lpips_sum / n_batches_eval

    print(f"  L1 (raw, 0-1 scale):   {l1_01:.6f}")
    print(f"  L1 (raw, 0-255 scale): {l1_255:.4f}")
    print(f"  LPIPS (AlexNet):        {avg_lpips:.4f}")

    fid_score = compute_fid(str(out_dir))
    print(f"  FID: {fid_score:.2f}")

    _save_metrics(metrics_file, dim, fid_score, l1_255, avg_lpips)
    print(f"  Metrics saved → {metrics_file}")

    return fid_score


def _save_metrics(metrics_file: Path, dim: int, fid: float,
                  l1: float, lpips_val: float) -> None:
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as fh:
        json.dump({str(dim): {"fid": fid, "l1": l1, "lpips": lpips_val}}, fh, indent=2)


def save_sample_grid(orig: np.ndarray, recon: np.ndarray,
                     dim: int, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
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
    grid_path = plots_dir / f"ae_grid_{dim}.png"
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


def plot_only(paths: ExpPaths) -> None:
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for dim in LATENT_DIMS:
        p = paths.metrics_dir / f"ae_metrics_{dim}.json"
        if not p.exists():
            print(f"[warning] {p} not found — skipping dim {dim}.")
            continue
        with open(p) as fh:
            metrics[dim] = json.load(fh)[str(dim)]

    if not metrics:
        print("[ERROR] No per-dim metrics found. Run without --plot-only first.")
        return

    unified_path = paths.metrics_dir / "ae_metrics.json"
    with open(unified_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  Unified metrics → {unified_path}")

    valid_dims = [d for d in LATENT_DIMS if d in metrics]
    fid_scores = [metrics[d]["fid"] for d in valid_dims]
    l1_scores  = [metrics[d]["l1"]  for d in valid_dims]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(valid_dims, fid_scores, marker="o", linewidth=2, color="royalblue")
    axes[0].set_xlabel("Latent Dim")
    axes[0].set_ylabel("FID (↓)")
    axes[0].set_title("AE Reconstruction FID vs Latent Dim")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(valid_dims, l1_scores, marker="s", linewidth=2, color="darkorange")
    axes[1].set_xlabel("Latent Dim")
    axes[1].set_ylabel("L1 (pixel, 0-255 scale) (↓)")
    axes[1].set_title("AE Reconstruction L1 vs Latent Dim")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    out_png = paths.plots_dir / "ae_fid_vs_dim.png"
    plt.savefig(str(out_png), dpi=150)
    plt.close()
    print(f"Plot saved → {out_png}")

    print("\nAE evaluation summary:")
    for dim in valid_dims:
        print(f"  dim={dim:4d}  FID={metrics[dim]['fid']:7.2f}  L1={metrics[dim]['l1']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS,
                        help="Single latent dim to evaluate (omit for all dims sequentially)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Merge existing per-dim JSONs and plot — no GPU needed")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete and regenerate existing output directories")
    add_exp_arg(parser)
    args = parser.parse_args()

    paths = get_paths(args.exp_name)

    print_exp_summary(
        paths,
        ckpt_path=paths.ckpt_dir,
        gen_path=paths.ae_recon_dir,
        metrics_path=paths.metrics_dir,
    )

    if args.plot_only:
        plot_only(paths)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims   = [args.dim] if args.dim else LATENT_DIMS

    for dim in dims:
        eval_one_dim(dim, device, paths, args.overwrite)

    if not args.dim:
        plot_only(paths)


if __name__ == "__main__":
    main()
