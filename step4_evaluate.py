"""
Step 4 — Generate images, compute metrics, and plot FID vs latent dim.
Split into four phases:
  --generate-only DIM : generate latents with student, saves z_orig_{dim}.npy
  --decode-only DIM   : decode latents → PNG images using trained PyTorch AE
  --metrics-only DIM  : compute FID/IS on saved images
  --plot-only         : merge per-dim JSONs and produce fid_vs_dim.png
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
LATENT_DIMS        = [64, 128, 256, 384, 512, 1024]
N_SAMPLES          = 10_000
STUDENT_DDIM_STEPS = 4
DECODE_BATCH       = 256
CKPT_DIR           = Path("checkpoints")
MODEL_DIR          = Path("models")
RESULTS_DIR        = Path("results/trained_AE")


# ── phase 1: generate latents ─────────────────────────────────────────────────

def generate_only(dim: int) -> None:
    import torch
    from models.diffusion import FlowMatching
    from models.denoiser import load_student

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[generate] dim={dim}  device={device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    student_path = MODEL_DIR / f"student_{dim}.pt"
    if not student_path.exists():
        print(f"[ERROR] {student_path} not found — run step3 first.")
        return

    flow        = FlowMatching(device=device)
    student     = load_student(str(student_path), latent_dim=dim, device=device)
    ckpt        = torch.load(str(student_path), map_location="cpu", weights_only=True)
    lat_mean    = float(ckpt["latent_mean"])
    lat_std     = float(ckpt["latent_std"])
    euler_steps = int(ckpt.get("student_euler_steps", STUDENT_DDIM_STEPS))

    print(f"  Generating {N_SAMPLES} samples with Euler-{euler_steps} …")
    all_latents, generated = [], 0
    while generated < N_SAMPLES:
        n_batch = min(512, N_SAMPLES - generated)
        z = flow.euler_sample(student, (n_batch, dim), n_steps=euler_steps)
        all_latents.append(z.cpu().numpy())
        generated += n_batch

    z_norm = np.concatenate(all_latents, axis=0)
    z_orig = (z_norm * lat_std + lat_mean).astype(np.float32)
    print(f"  Latent range: [{z_orig.min():.3f}, {z_orig.max():.3f}]")

    out_path = RESULTS_DIR / f"z_orig_{dim}.npy"
    np.save(out_path, z_orig)
    print(f"  Saved → {out_path}")


# ── phase 2: decode latents → images (PyTorch AE) ────────────────────────────

def decode_only(dim: int) -> None:
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from models.autoencoder import ConvAutoencoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[decode] dim={dim}  device={device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    z_path = RESULTS_DIR / f"z_orig_{dim}.npy"
    if not z_path.exists():
        print(f"[ERROR] {z_path} not found — run --generate-only first.")
        return

    ae_path = CKPT_DIR / f"ae_{dim}.pt"
    if not ae_path.exists():
        print(f"[ERROR] {ae_path} not found — run step0 first.")
        return

    ckpt  = torch.load(ae_path, map_location=device, weights_only=True)
    model = ConvAutoencoder(latent_dim=dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ── decode generated latents → PNG images ────────────────────────────────
    z_orig  = torch.from_numpy(np.load(z_path))
    gen_dir = RESULTS_DIR / f"generated_{dim}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Decoding {len(z_orig)} generated latents …")
    img_idx = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(z_orig), DECODE_BATCH), desc="  Decoding", leave=False):
            z_batch = z_orig[start:start + DECODE_BATCH].to(device)
            recon   = model.decode(z_batch)                         # (B, 3, 32, 32) in [0,1]
            recon   = (recon.clamp(0, 1) * 255).byte().cpu().numpy()
            recon   = recon.transpose(0, 2, 3, 1)                  # (B, H, W, 3)
            for img_arr in recon:
                Image.fromarray(img_arr).save(gen_dir / f"{img_idx:05d}.png")
                img_idx += 1
    print(f"  Images saved → {gen_dir}/  ({img_idx} files)")

    # ── AE reconstruction FID: encode/decode CIFAR-10 test set ───────────────
    print("  Encoding CIFAR-10 test set for AE reconstruction FID …")
    tf      = transforms.ToTensor()
    testset = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    loader  = DataLoader(testset, batch_size=DECODE_BATCH, shuffle=False, num_workers=2)

    ae_dir = RESULTS_DIR / f"ae_reconstructed_{dim}"
    ae_dir.mkdir(parents=True, exist_ok=True)

    img_idx = 0
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="  AE recon", leave=False):
            imgs  = imgs.to(device)
            recon = model(imgs)                                     # encode → decode
            recon = (recon.clamp(0, 1) * 255).byte().cpu().numpy()
            recon = recon.transpose(0, 2, 3, 1)
            for img_arr in recon:
                Image.fromarray(img_arr).save(ae_dir / f"{img_idx:05d}.png")
                img_idx += 1
    print(f"  AE reconstructions saved → {ae_dir}/  ({img_idx} files)")


# ── phase 3: compute FID / IS ─────────────────────────────────────────────────

def metrics_only(dim: int) -> None:
    gen_dir = RESULTS_DIR / f"generated_{dim}"
    if not gen_dir.exists():
        print(f"[ERROR] {gen_dir} not found — run --decode-only first.")
        return

    print(f"[metrics] dim={dim}")
    fid    = compute_fid(str(gen_dir))
    is_val = compute_inception_score(str(gen_dir))
    print(f"  FID={fid:.2f}  IS={is_val:.2f}")

    ae_fid = -1.0
    ae_dir = RESULTS_DIR / f"ae_reconstructed_{dim}"
    if ae_dir.exists():
        ae_fid = compute_fid(str(ae_dir))
        print(f"  AE-FID={ae_fid:.2f}")

    out = RESULTS_DIR / f"metrics_{dim}.json"
    with open(out, "w") as f:
        json.dump({str(dim): {"fid": fid, "is": is_val, "ae_fid": ae_fid}}, f, indent=2)
    print(f"  Saved → {out}")


def compute_fid(gen_dir: str) -> float:
    try:
        from cleanfid import fid
        return float(fid.compute_fid(gen_dir, dataset_name="cifar10",
                                     dataset_res=32, dataset_split="train", verbose=True))
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
        print("[ERROR] No per-dim metrics found.")
        return

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    valid_dims    = [d for d in LATENT_DIMS if d in metrics]
    fid_scores    = [metrics[d]["fid"]             for d in valid_dims]
    is_scores     = [metrics[d]["is"]              for d in valid_dims]
    ae_fid_scores = [metrics[d].get("ae_fid", -1) for d in valid_dims]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(valid_dims, fid_scores, marker="o", linewidth=2, color="royalblue")
    axes[0].set_xlabel("Latent Dim"); axes[0].set_ylabel("FID (↓)")
    axes[0].set_title("Diffusion FID vs Latent Dim"); axes[0].grid(True, alpha=0.4)

    if any(v >= 0 for v in is_scores):
        axes[1].plot(valid_dims, is_scores, marker="s", linewidth=2, color="darkorange")
        axes[1].set_xlabel("Latent Dim"); axes[1].set_ylabel("IS (↑)")
        axes[1].set_title("IS vs Latent Dim"); axes[1].grid(True, alpha=0.4)
    else:
        axes[1].set_visible(False)

    valid_ae = [(d, s) for d, s in zip(valid_dims, ae_fid_scores) if s >= 0]
    if valid_ae:
        ae_dims, ae_fids = zip(*valid_ae)
        axes[2].plot(ae_dims, ae_fids, marker="^", linewidth=2, color="seagreen")
        axes[2].set_xlabel("Latent Dim"); axes[2].set_ylabel("FID (↓)")
        axes[2].set_title("AE Reconstruction FID vs Latent Dim"); axes[2].grid(True, alpha=0.4)
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    out_png = RESULTS_DIR / "fid_vs_dim.png"
    plt.savefig(str(out_png), dpi=150)
    print(f"Plot saved → {out_png}")
    print("\nStep 4 complete.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate-only", type=int, choices=LATENT_DIMS, metavar="DIM",
                       help="Phase 1: generate latents with student DDIM.")
    group.add_argument("--decode-only", type=int, choices=LATENT_DIMS, metavar="DIM",
                       help="Phase 2: decode latents → PNG images via PyTorch AE.")
    group.add_argument("--metrics-only", type=int, choices=LATENT_DIMS, metavar="DIM",
                       help="Phase 3: compute FID/IS on saved images.")
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
        print("Specify --generate-only, --decode-only, --metrics-only, or --plot-only.")


if __name__ == "__main__":
    main()
