"""
Step 4 — Generate images, compute metrics, and plot FID vs latent dim.

For each student model (dims 64 / 128 / 256 / 384):
  1. Sample N_SAMPLES latent vectors using DDIM (STUDENT_DDIM_STEPS steps).
  2. Denormalise latents using the stats saved in step 2.
  3. Decode latents → pixel images via the frozen JAX autoencoder.
  4. Save images to  results/generated_{dim}/.
  5. Compute FID (clean-fid) and IS (torch-fidelity).

After all dims are evaluated, plot FID vs latent dimension and save:
  results/fid_vs_dim.png
  results/metrics.json

Requirements
------------
pip install clean-fid torch-fidelity lpips
"""

import argparse
import json
import numpy as np
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from models.diffusion import DiffusionSchedule
from models.denoiser import load_student

# ── configuration ─────────────────────────────────────────────────────────────
LATENT_DIMS        = [64, 128, 256, 384]
N_SAMPLES          = 10_000
STUDENT_DDIM_STEPS = 4
DECODE_BATCH       = 256
SAVE_BATCH         = 256
CKPT_DIR           = "checkpoints"
MODEL_DIR          = "models"
RESULTS_DIR        = "results"

# ── utilities ─────────────────────────────────────────────────────────────────

def get_device(dim: int = None) -> str:
    if torch.cuda.is_available():
        if dim is not None:
            gpu_id = LATENT_DIMS.index(dim) % torch.cuda.device_count()
            return f"cuda:{gpu_id}"
        return "cuda"
    return "cpu"


def generate_latents(
    student: torch.nn.Module,
    diffusion: DiffusionSchedule,
    latent_dim: int,
    n_samples: int,
    ddim_steps: int,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Generate latents with the student model using DDIM sampling.

    Returns float32 numpy array of shape (n_samples, latent_dim)
    in the *normalised* latent space (i.e. before denormalisation).
    """
    all_latents = []
    generated   = 0

    while generated < n_samples:
        n_batch = min(batch_size, n_samples - generated)
        shape   = (n_batch, latent_dim)
        z = diffusion.ddim_sample(student, shape, n_steps=ddim_steps, eta=0.0)
        all_latents.append(z.cpu().numpy())
        generated += n_batch

    return np.concatenate(all_latents, axis=0)


def save_images(images_uint8: np.ndarray, out_dir: Path) -> None:
    """Save (N, 32, 32, 3) uint8 numpy array as individual PNG files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(tqdm(images_uint8, desc="Saving images", leave=False)):
        Image.fromarray(img).save(out_dir / f"{i:05d}.png")


def compute_fid(gen_dir: str, dataset_name: str = "cifar10") -> float:
    """Compute FID against CIFAR-10 using clean-fid."""
    try:
        from cleanfid import fid
        score = fid.compute_fid(
            gen_dir,
            dataset_name=dataset_name,
            dataset_res=32,
            dataset_split="train",
            verbose=True,
        )
        return float(score)
    except ImportError:
        print("  [warning] clean-fid not installed. FID set to -1.")
        return -1.0


def compute_inception_score(gen_dir: str) -> float:
    """Compute Inception Score using torch-fidelity."""
    try:
        import torch_fidelity
        metrics = torch_fidelity.calculate_metrics(
            input1=gen_dir,
            isc=True,
            verbose=False,
        )
        return float(metrics["inception_score_mean"])
    except ImportError:
        print("  [warning] torch-fidelity not installed. IS set to -1.")
        return -1.0


def compute_lpips_vs_real(gen_dir: str, real_dir: str) -> float:
    """
    Optional: mean LPIPS between generated images and nearest real images.
    Skipped if lpips is not installed or real_dir is empty.
    """
    try:
        import lpips
    except ImportError:
        return -1.0

    gen_files  = sorted(Path(gen_dir).glob("*.png"))[:1000]
    real_files = sorted(Path(real_dir).glob("*.png"))[:1000]
    if not real_files:
        return -1.0

    loss_fn = lpips.LPIPS(net="alex")
    loss_fn.eval()

    import torchvision.transforms.functional as TF

    total, count = 0.0, 0
    for gf, rf in zip(gen_files, real_files):
        g = TF.to_tensor(Image.open(gf)).unsqueeze(0) * 2 - 1
        r = TF.to_tensor(Image.open(rf)).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            total += loss_fn(g, r).item()
        count += 1

    return total / max(count, 1)


def plot_results(dims: list, metrics: dict, out_path: str) -> None:
    """Plot FID (and optionally IS) vs latent dimension."""
    fid_scores = [metrics[d]["fid"] for d in dims]
    is_scores  = [metrics[d]["is"]  for d in dims]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # FID
    ax = axes[0]
    ax.plot(dims, fid_scores, marker="o", linewidth=2, markersize=8, color="royalblue")
    ax.set_xlabel("Latent Space Dimension", fontsize=13)
    ax.set_ylabel("FID Score (↓ better)", fontsize=13)
    ax.set_title("FID vs Latent Dimensionality\n(Distilled Diffusion on CIFAR-10)", fontsize=13)
    ax.set_xticks(dims)
    ax.grid(True, alpha=0.4)
    for x, y in zip(dims, fid_scores):
        if y >= 0:
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=10)

    # IS
    ax = axes[1]
    valid_is = [v for v in is_scores if v >= 0]
    if valid_is:
        ax.plot(dims, is_scores, marker="s", linewidth=2, markersize=8, color="darkorange")
        ax.set_xlabel("Latent Space Dimension", fontsize=13)
        ax.set_ylabel("Inception Score (↑ better)", fontsize=13)
        ax.set_title("IS vs Latent Dimensionality\n(Distilled Diffusion on CIFAR-10)", fontsize=13)
        ax.set_xticks(dims)
        ax.grid(True, alpha=0.4)
    else:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate_dim(dim: int, device: str) -> None:
    """Evaluate a single latent dim: generate, decode, save images, compute metrics."""
    from models.autoencoder_jax import load_autoencoder, decode_latents, CHECKPOINT_NAMES

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Evaluating student  latent_dim = {dim}  device = {device}")

    student_path = Path(MODEL_DIR) / f"student_{dim}.pt"
    if not student_path.exists():
        print(f"  [ERROR] {student_path} not found — run step3 first.")
        return

    diffusion = DiffusionSchedule(T=1000, device=device)

    student  = load_student(str(student_path), latent_dim=dim, device=device)
    ckpt     = torch.load(str(student_path), map_location="cpu", weights_only=True)
    lat_mean = float(ckpt["latent_mean"])
    lat_std  = float(ckpt["latent_std"])
    ddim_steps = int(ckpt.get("student_ddim_steps", STUDENT_DDIM_STEPS))

    print(f"  Generating {N_SAMPLES} samples with DDIM-{ddim_steps} …")
    z_norm = generate_latents(student, diffusion, dim, N_SAMPLES, ddim_steps, device)
    z_orig = (z_norm * lat_std + lat_mean).astype(np.float32)
    print(f"  Latent range: [{z_orig.min():.3f}, {z_orig.max():.3f}]")

    ckpt_path = Path(CKPT_DIR) / CHECKPOINT_NAMES[dim]
    ae_model, ae_params = load_autoencoder(str(ckpt_path), latent_dim=dim)

    print("  Decoding latents → pixel images …")
    images_uint8 = decode_latents(ae_model, ae_params, z_orig, batch_size=DECODE_BATCH)
    print(f"  Decoded shape: {images_uint8.shape}  dtype: {images_uint8.dtype}")

    gen_dir = Path(RESULTS_DIR) / f"generated_{dim}"
    save_images(images_uint8, gen_dir)
    print(f"  Images saved → {gen_dir}/")

    print("  Computing FID …")
    fid = compute_fid(str(gen_dir))
    print(f"  FID = {fid:.2f}")

    print("  Computing IS …")
    is_score = compute_inception_score(str(gen_dir))
    print(f"  IS  = {is_score:.2f}")

    # Save per-dim metrics so the plot step can merge them
    dim_metrics_path = Path(RESULTS_DIR) / f"metrics_{dim}.json"
    with open(dim_metrics_path, "w") as f:
        json.dump({str(dim): {"fid": fid, "is": is_score}}, f, indent=2)
    print(f"  Metrics saved → {dim_metrics_path}")


def plot_only() -> None:
    """Merge per-dim metrics JSONs and produce the final plot. CPU-only."""
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
        print("[ERROR] No per-dim metrics found. Run evaluation first.")
        return

    # Write combined metrics.json
    metrics_path = Path(RESULTS_DIR) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")
    print(json.dumps(metrics, indent=2))

    valid_dims = [d for d in LATENT_DIMS if d in metrics]
    plot_results(valid_dims, metrics, str(Path(RESULTS_DIR) / "fid_vs_dim.png"))
    print("\nStep 4 complete.")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dim", type=int, choices=LATENT_DIMS, default=None,
                       help="Evaluate a single latent dim (for parallel GPU runs). "
                            "Saves results/metrics_{dim}.json.")
    group.add_argument("--plot-only", action="store_true",
                       help="Merge per-dim JSONs and plot. No GPU needed.")
    args = parser.parse_args()

    if args.plot_only:
        plot_only()
    elif args.dim is not None:
        evaluate_dim(args.dim, get_device(args.dim))
    else:
        # Sequential fallback: evaluate all dims then plot
        for dim in LATENT_DIMS:
            evaluate_dim(dim, get_device())
        plot_only()


if __name__ == "__main__":
    main()
