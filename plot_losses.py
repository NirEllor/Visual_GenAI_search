"""
Plot training loss curves — 6 figures, one per latent dim.

Each figure has two subplots:
  Left  : teacher loss over its full training run
  Right : 4 student losses (one per synthetic dataset size) overlaid

Saves: results/trained_AE/losses_{dim}.png

Usage:
    python plot_losses.py           # all dims
    python plot_losses.py --dim 128 # single dim
"""

import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATENT_DIMS   = [64, 128, 256, 384, 512, 1024]
DATASET_SIZES = [250_000, 500_000, 1_000_000, 2_000_000]
SIZE_LABELS   = {250_000: "250k", 500_000: "500k",
                 1_000_000: "1M",  2_000_000: "2M"}
SIZE_COLORS   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

MODEL_DIR   = Path("models")
RESULTS_DIR = Path("results/trained_AE")


def load_history(ckpt_path: Path) -> list:
    if not ckpt_path.exists():
        return []
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    h = ckpt.get("loss_history", [])
    # student history may be a dict {"total": [...], "flow": [...], ...}
    return h if isinstance(h, list) else h.get("total", [])


def plot_dim(dim: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    teacher_history = load_history(MODEL_DIR / f"teacher_{dim}.pt")
    student_histories = {
        size: load_history(MODEL_DIR / f"student_{dim}_{size}.pt")
        for size in DATASET_SIZES
    }

    if not teacher_history and all(len(h) == 0 for h in student_histories.values()):
        print(f"  [skip] dim={dim}: no loss histories found.")
        return

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Loss Curves — latent_dim={dim}", fontsize=13, fontweight="bold")

    # ── left: teacher ─────────────────────────────────────────────────────────
    if teacher_history:
        epochs = range(1, len(teacher_history) + 1)
        ax_t.plot(epochs, teacher_history, color="black", linewidth=1.5)
        ax_t.set_title(f"Teacher  ({len(teacher_history)} epochs)")
    else:
        ax_t.text(0.5, 0.5, "No teacher checkpoint found",
                  ha="center", va="center", transform=ax_t.transAxes, color="grey")
        ax_t.set_title("Teacher")
    ax_t.set_xlabel("Epoch")
    ax_t.set_ylabel("Flow Matching Loss")
    ax_t.grid(True, alpha=0.3)

    # ── right: students ───────────────────────────────────────────────────────
    any_student = False
    for i, size in enumerate(DATASET_SIZES):
        h = student_histories[size]
        if not h:
            continue
        any_student = True
        epochs = range(1, len(h) + 1)
        ax_s.plot(epochs, h, color=SIZE_COLORS[i], linewidth=1.5,
                  label=f"n={SIZE_LABELS[size]}")

    if not any_student:
        ax_s.text(0.5, 0.5, "No student checkpoints found",
                  ha="center", va="center", transform=ax_s.transAxes, color="grey")

    ax_s.set_title(f"Students  (4 synthetic dataset sizes)")
    ax_s.set_xlabel("Epoch")
    ax_s.set_ylabel("Flow Matching Loss")
    ax_s.legend(title="Dataset size", loc="upper right")
    ax_s.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / f"losses_{dim}.png"
    plt.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS,
                        help="Single dim to plot (omit for all)")
    args = parser.parse_args()

    dims = [args.dim] if args.dim else LATENT_DIMS
    for dim in dims:
        print(f"Plotting losses  dim={dim} …")
        plot_dim(dim)


if __name__ == "__main__":
    main()
