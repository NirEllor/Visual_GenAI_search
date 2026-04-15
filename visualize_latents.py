"""
visualize_latents.py — Project latent spaces to 2D and compare with pixel space.

Loads latents/latents_{dim}.npy for each available latent dimension, along with
raw CIFAR-10 pixel data, reduces everything to 2D, and saves a single figure
with one panel per space, each point coloured by CIFAR-10 class label.

Usage
-----
  python visualize_latents.py                        # PCA, 5000 samples, all dims
  python visualize_latents.py --method tsne          # t-SNE (slower, better clusters)
  python visualize_latents.py --method umap          # UMAP  (pip install umap-learn)
  python visualize_latents.py --dims 64 128 256 384  # specific dims only
  python visualize_latents.py --n-samples 2000       # fewer points (faster t-SNE)

Output
------
  results/latent_space_2d_{method}.png
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torchvision

# ── constants ─────────────────────────────────────────────────────────────────

LATENT_DIMS = [8, 16, 32, 64, 128, 256, 384]
LATENT_DIR  = "latents"
DATA_DIR    = "data"
RESULTS_DIR = "results"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Visually distinct palette (one colour per class)
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#808080",
]


# ── data loading ──────────────────────────────────────────────────────────────

def load_cifar10_labels(data_dir: str) -> np.ndarray:
    """Return the 50 000 training labels as an int32 array."""
    torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
    cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
    labels = []
    for i in range(1, 6):
        path = os.path.join(cifar_dir, f"data_batch_{i}")
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        labels.extend(d["labels"])
    return np.array(labels, dtype=np.int32)


def load_pixel_space(data_dir: str) -> np.ndarray:
    """Return CIFAR-10 training images flattened to (50000, 3072) float32 in [-1, 1]."""
    cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
    batches = []
    for i in range(1, 6):
        path = os.path.join(cifar_dir, f"data_batch_{i}")
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        batches.append(d["data"])  # (10000, 3072) uint8
    pixels = np.concatenate(batches, axis=0).astype(np.float32) / 127.5 - 1.0
    return pixels  # (50000, 3072)


# ── dimensionality reduction ──────────────────────────────────────────────────

def reduce_2d(X: np.ndarray, method: str) -> np.ndarray:
    """StandardScale X, then project to 2D.  Returns (N, 2) float array."""
    X_scaled = StandardScaler().fit_transform(X)

    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=2, perplexity=40, n_iter=1000,
            random_state=42, n_jobs=-1,
        ).fit_transform(X_scaled)

    elif method == "umap":
        try:
            import umap as umap_lib
        except ImportError:
            raise ImportError(
                "umap-learn is not installed.\n"
                "Install it with:  pip install umap-learn"
            )
        return umap_lib.UMAP(n_components=2, random_state=42).fit_transform(X_scaled)

    raise ValueError(f"Unknown method: {method!r}")


# ── plotting helpers ──────────────────────────────────────────────────────────

def draw_panel(ax: plt.Axes, xy: np.ndarray, labels: np.ndarray, title: str) -> None:
    for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
        mask = labels == cls_idx
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=3, alpha=0.45, linewidths=0,
            color=PALETTE[cls_idx], label=cls_name, rasterized=True,
        )
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise latent space distributions in 2D."
    )
    parser.add_argument(
        "--method", choices=["pca", "tsne", "umap"], default="pca",
        help="Dimensionality-reduction method (default: pca).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5000,
        help="Points to sample per space (default 5000; use fewer for faster t-SNE).",
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", default=None,
        help="Latent dims to include (default: all with a saved .npy file).",
    )
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # ── which dims are available ───────────────────────────────────────────────
    candidate_dims = args.dims or LATENT_DIMS
    dims_to_plot = []
    for d in candidate_dims:
        p = Path(LATENT_DIR) / f"latents_{d}.npy"
        if p.exists():
            dims_to_plot.append(d)
        else:
            print(f"  [skip] latents_{d}.npy not found — run step1 first.")

    if not dims_to_plot:
        print("No latent files found. Run step1_extract_latents.py first.")
        return

    # ── shared sample indices (same rows across all spaces for fair comparison) ─
    print("Loading CIFAR-10 labels...")
    labels_full = load_cifar10_labels(DATA_DIR)   # (50000,)
    N = len(labels_full)
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(args.n_samples, N), replace=False)
    labels = labels_full[idx]

    print(f"Method: {args.method.upper()}  |  Samples per space: {len(idx)}")

    # ── figure layout ─────────────────────────────────────────────────────────
    # +1 panel for pixel space
    n_panels = len(dims_to_plot) + 1
    n_cols   = min(4, n_panels)
    n_rows   = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 4.5 * n_rows),
    )
    axes_flat = np.array(axes).reshape(-1)

    panel = 0

    # ── pixel space ───────────────────────────────────────────────────────────
    print("\n[pixel space]  3072D")
    pixels = load_pixel_space(DATA_DIR)[idx]
    xy_px  = reduce_2d(pixels, args.method)
    draw_panel(
        axes_flat[panel], xy_px, labels,
        f"Pixel Space  (3072D → 2D  {args.method.upper()})",
    )
    panel += 1
    del pixels, xy_px

    # ── latent spaces ─────────────────────────────────────────────────────────
    for d in dims_to_plot:
        print(f"[latent  dim={d}]  {d}D")
        latents = np.load(Path(LATENT_DIR) / f"latents_{d}.npy")[idx]
        xy      = reduce_2d(latents, args.method)
        draw_panel(
            axes_flat[panel], xy, labels,
            f"Latent  dim={d}  ({d}D → 2D  {args.method.upper()})",
        )
        panel += 1
        del latents, xy

    # Hide unused axes
    for ax in axes_flat[panel:]:
        ax.set_visible(False)

    # ── shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=PALETTE[i], markersize=8,
            label=CIFAR10_CLASSES[i],
        )
        for i in range(10)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=5, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle(
        f"Latent Space Distribution — 2D Projection  "
        f"({args.method.upper()}, n={len(idx)} per space)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out = Path(RESULTS_DIR) / f"latent_space_2d_{args.method}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
