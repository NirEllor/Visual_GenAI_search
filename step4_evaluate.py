"""
Step 4 — Generate images, compute metrics, and plot FID vs latent dim.
Split into four phases:
  --generate-only DIM : PyTorch only, saves z_orig_{dim}.npy
  --decode-only DIM   : JAX only, decodes latents to images
  --metrics-only DIM  : PyTorch only, computes FID/IS on saved images
  --plot-only         : merges JSONs and plots
"""

import argparse
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ── configuration ─────────────────────────────────────────────────────────────
LATENT_DIMS        = [64, 128, 256, 384]
N_SAMPLES          = 10_000
STUDENT_DDIM_STEPS = 4
DECODE_BATCH       = 256
CKPT_DIR           = "checkpoints"
MODEL_DIR          = "models"
RESULTS_DIR        = "results"


# ── phase 1: PyTorch only ─────────────────────────────────────────────────────

def generate_only(dim: int) -> None:
    """Generate latents with student model and save to disk. PyTorch only."""
    import torch
    from models.diffusion import DiffusionSchedule
    from models.denoiser import load_student

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[generate] dim={dim} device={device}")

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    student_path = Path(MODEL_DIR) / f"student_{dim}.pt"
    if not student_path.exists():
        print(f"[ERROR] {student_path} not found — run step3 first.")
        return

    diffusion  = DiffusionSchedule(T=1000, device=device)
    student    = load_student(str(student_path), latent_dim=dim, device=device)
    ckpt       = torch.load(str(student_path), map_location="cpu", weights_only=True)
    lat_mean   = float(ckpt["latent_mean"])
    lat_std    = float(ckpt["latent_std"])
    ddim_steps = int(ckpt.get("student_ddim_steps", STUDENT_DDIM_STEPS))

    print(f"  Generating {N_SAMPLES} samples with DDIM-{ddim_steps} …")
    all_latents = []
    generated   = 0
    while generated < N_SAMPLES:
        n_batch = min(512, N_SAMPLES - generated)
        z = diffusion.ddim_sample(student, (n_batch, dim), n_steps=ddim_steps, eta=0.0)
        all_latents.append(z.cpu().numpy())
        generated += n_batch

    z_norm = np.concatenate(all_latents, axis=0)
    z_orig = (z_norm * lat_std + lat_mean).astype(np.float32)
    print(f"  Latent range: [{z_orig.min():.3f}, {z_orig.max():.3f}]")

    out_path = Path(RESULTS_DIR) / f"z_orig_{dim}.npy"
    np.save(out_path, z_orig)
    print(f"  Saved → {out_path}")


# ── phase 2: JAX only ─────────────────────────────────────────────────────────

def decode_only(dim: int) -> None:
    """Decode latents to images using JAX autoencoder. JAX + NumPy>=2 only."""
    from models.autoencoder_jax import load_autoencoder, decode_latents, CHECKPOINT_NAMES

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    z_path = Path(RESULTS_DIR) / f"z_orig_{dim}.npy"
    if not z_path.exists():
        print(f"[ERROR] {z_path} not found — run --generate-only first.")
        return

    z_orig = np.load(z_path)
    print(f"[decode] dim={dim} latents shape: {z_orig.shape}")

    ckpt_path = Path(CKPT_DIR) / CHECKPOINT_NAMES[dim]
    ae_model, ae_params = load_autoencoder(str(ckpt_path), latent_dim=dim)

    print("  Decoding latents → pixel images …")
    images_uint8 = decode_latents(ae_model, ae_params, z_orig, batch_size=DECODE_BATCH)
    print(f"  Decoded shape: {images_uint8.shape}")

    gen_dir = Path(RESULTS_DIR) / f"generated_{dim}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(tqdm(images_uint8, desc="Saving images", leave=False)):
        Image.fromarray(img).save(gen_dir / f"{i:05d}.png")
    print(f"  Images saved → {gen_dir}/")


# ── phase 3: PyTorch only ─────────────────────────────────────────────────────

def metrics_only(dim: int) -> None:
    """Compute FID/IS on saved images. PyTorch only."""
    gen_dir = Path(RESULTS_DIR) / f"generated_{dim}"
    if not gen_dir.exists():
        print(f"[ERROR] {gen_dir} not found — run --decode-only first.")
        return

    print(f"[metrics] dim={dim}")
    print("  Computing FID …")
    fid = compute_fid(str(gen_dir))
    print(f"  FID = {fid:.2f}")

    print("  Computing IS …")
    is_score = compute_inception_score(str(gen_dir))
    print(f"  IS  = {is_score:.2f}")

    dim_metrics_path = Path(RESULTS_DIR) / f"metrics_{dim}.json"
    with open(dim_metrics_path, "w") as f:
        json.dump({str(dim): {"fid": fid, "is": is_score}}, f, indent=2)
    print(f"  Metrics saved → {dim_metrics_path}")


# ── metrics helpers ───────────────────────────────────────────────────────────

def compute_fid(gen_dir: str) -> float:
    try:
        from cleanfid import fid
        return float(fid.compute_fid(gen_dir, dataset_name="cifar10",
                                     dataset_res=32, dataset_split="train",
                                     verbose=True))
    except ImportError:
        print("  [warning] clean-fid not installed.")
        return -1.0


def compute_inception_score(gen_dir: str) -> float:
    try:
        import torch_fidelity
        metrics = torch_fidelity.calculate_metrics(input1=gen_dir, isc=True, verbose=False)
        return float(metrics["inception_score_mean"])
    except ImportError:
        print("  [warning] torch-fidelity not installed.")
        return -1.0


# ── phase 4: plot ─────────────────────────────────────────────────────────────

def plot_only() -> None:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    metrics = {}
    for dim in LATENT_DIMS:
        p = Path(RESULTS_DIR) / f"metrics_{dim}.json"
        if not p.exists():
            print(f"[warning] {p} not found — skipping dim {dim}.")
            continue
        with open(p) as f:
            metrics[dim] = json.load(f)[str(dim)]

    if not metrics:
        print("[ERROR] No per-dim metrics found.")
        return

    with open(Path(RESULTS_DIR) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    valid_dims = [d for d in LATENT_DIMS if d in metrics]
    fid_scores = [metrics[d]["fid"] for d in valid_dims]
    is_scores  = [metrics[d]["is"]  for d in valid_dims]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(valid_dims, fid_scores, marker="o", linewidth=2, color="royalblue")
    axes[0].set_xlabel("Latent Dim"); axes[0].set_ylabel("FID (↓)")
    axes[0].set_title("FID vs Latent Dim"); axes[0].grid(True, alpha=0.4)

    valid_is = [v for v in is_scores if v >= 0]
    if valid_is:
        axes[1].plot(valid_dims, is_scores, marker="s", linewidth=2, color="darkorange")
        axes[1].set_xlabel("Latent Dim"); axes[1].set_ylabel("IS (↑)")
        axes[1].set_title("IS vs Latent Dim"); axes[1].grid(True, alpha=0.4)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(Path(RESULTS_DIR) / "fid_vs_dim.png"), dpi=150)
    print("Plot saved → results/fid_vs_dim.png")

    # ── latent space 2D visualisation ─────────────────────────────────────────
    try:
        from visualize_latents import run as visualize_latents
        print("\n[phase 4] Generating latent space 2D visualisation (PCA) …")
        visualize_latents(method="pca", n_samples=5000)
    except Exception as e:
        print(f"[warning] Latent space visualisation skipped: {e}")

    print("\nStep 4 complete.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate-only", type=int, choices=LATENT_DIMS, metavar="DIM",
                       help="Phase 1: generate latents (PyTorch). Saves z_orig_{dim}.npy")
    group.add_argument("--decode-only", type=int, choices=LATENT_DIMS, metavar="DIM",
                       help="Phase 2: decode latents to images (JAX). Needs z_orig_{dim}.npy")
    group.add_argument("--metrics-only", type=int, choices=LATENT_DIMS, metavar="DIM",
                       help="Phase 3: compute FID/IS (PyTorch). Needs generated_{dim}/")
    group.add_argument("--plot-only", action="store_true",
                       help="Phase 4: merge JSONs and plot.")
    args = parser.parse_args()

    if args.generate_only is not None:
        generate_only(args.generate_only)
    elif args.decode_only is not None:
        decode_only(args.decode_only)
    elif args.metrics_only is not None:
        metrics_only(args.metrics_only)
    elif args.plot_only:
        plot_only()
    else:
        print("Specify --generate-only, --decode-only, --metrics-only, or --plot-only")


if __name__ == "__main__":
    main()