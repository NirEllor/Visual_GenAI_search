"""
Step 4 — Generate images, compute FID/IS, and plot results for all 24 students.

Four phases, each independently restartable:

  --generate  --dim D --size N   : Euler-sample 10k latents from student_{D}_{N}.pt
  --decode    --dim D --size N   : decode latents → PNGs via AE; also saves AE recon
  --metrics   --dim D --size N   : compute FID + IS on generated_{D}_{N}/
  --plot                         : merge all per-(dim,size) JSONs and produce:
                                     fid_vs_size.png  — FID vs dataset size, one line per dim
                                     fid_vs_dim.png   — FID vs latent dim, one line per size

Usage examples:
    # run one student
    python step4_evaluate.py --generate --dim 128 --size 500000
    python step4_evaluate.py --decode   --dim 128 --size 500000
    python step4_evaluate.py --metrics  --dim 128 --size 500000

    # batch across all 24 students (run in parallel via SLURM array jobs)
    for dim in 64 128 256 384 512 1024; do
      for size in 250000 500000 1000000 2000000; do
        python step4_evaluate.py --generate --dim $dim --size $size &
      done
    done

    # after all jobs finish
    python step4_evaluate.py --plot
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
LATENT_DIMS   = [64, 128, 256, 384, 512, 1024]
DATASET_SIZES = [250_000, 500_000, 1_000_000, 2_000_000]
N_SAMPLES     = 10_000
EULER_STEPS   = 50
DECODE_BATCH  = 256
CKPT_DIR      = Path("checkpoints")
MODEL_DIR     = Path("models")
RESULTS_DIR   = Path("results/trained_AE")

SIZE_LABELS   = {250_000: "250k", 500_000: "500k",
                 1_000_000: "1M",  2_000_000: "2M"}
DIM_COLORS    = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SIZE_COLORS   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ── helpers ───────────────────────────────────────────────────────────────────

def z_path(dim: int, size: int) -> Path:
    return RESULTS_DIR / f"z_orig_{dim}_{size}.npy"

def gen_dir(dim: int, size: int) -> Path:
    return RESULTS_DIR / f"generated_{dim}_{size}"

def ae_recon_dir(dim: int) -> Path:
    return RESULTS_DIR / f"ae_reconstructed_{dim}"

def metrics_path(dim: int, size: int) -> Path:
    return RESULTS_DIR / f"metrics_{dim}_{size}.json"


# ── phase 1: generate latents ─────────────────────────────────────────────────

def generate(dim: int, size: int) -> None:
    import torch
    from models.diffusion import FlowMatching
    from models.denoiser import load_student

    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out = z_path(dim, size)
    if out.exists():
        print(f"[generate] [skip] {out.name} already exists.")
        return

    student_ckpt = MODEL_DIR / f"student_{dim}_{size}.pt"
    if not student_ckpt.exists():
        print(f"[generate] [ERROR] {student_ckpt} not found — run step3b first.")
        return

    print(f"[generate] dim={dim}  size={SIZE_LABELS[size]}  device={device}")
    flow        = FlowMatching(device=device)
    student     = load_student(str(student_ckpt), latent_dim=dim, device=device)
    ckpt        = torch.load(str(student_ckpt), map_location="cpu", weights_only=True)
    lat_mean    = float(ckpt["latent_mean"])
    lat_std     = float(ckpt["latent_std"])
    euler_steps = int(ckpt.get("student_euler_steps", EULER_STEPS))

    print(f"  Generating {N_SAMPLES:,} samples with Euler-{euler_steps} …")
    all_latents, generated = [], 0
    while generated < N_SAMPLES:
        n_batch = min(512, N_SAMPLES - generated)
        z = flow.euler_sample(student, (n_batch, dim), n_steps=euler_steps)
        all_latents.append(z.cpu().numpy())
        generated += n_batch

    z_norm = np.concatenate(all_latents, axis=0)
    z_orig = (z_norm * lat_std + lat_mean).astype(np.float32)
    print(f"  Latent range: [{z_orig.min():.3f}, {z_orig.max():.3f}]")
    np.save(str(out), z_orig)
    print(f"  Saved → {out}")


# ── phase 2: decode latents → images ─────────────────────────────────────────

def decode(dim: int, size: int) -> None:
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from models.autoencoder import ConvAutoencoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[decode] dim={dim}  size={SIZE_LABELS[size]}  device={device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    zp = z_path(dim, size)
    if not zp.exists():
        print(f"  [ERROR] {zp} not found — run --generate first.")
        return

    ae_ckpt = CKPT_DIR / f"ae_{dim}.pt"
    if not ae_ckpt.exists():
        print(f"  [ERROR] {ae_ckpt} not found — run step0 first.")
        return

    ckpt  = torch.load(ae_ckpt, map_location=device, weights_only=True)
    ae    = ConvAutoencoder(latent_dim=dim).to(device)
    ae.load_state_dict(ckpt["state_dict"])
    ae.eval()

    # ── decode student-generated latents → PNGs ───────────────────────────────
    gdir = gen_dir(dim, size)
    gdir.mkdir(parents=True, exist_ok=True)
    z_orig  = torch.from_numpy(np.load(str(zp)))
    img_idx = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(z_orig), DECODE_BATCH),
                          desc="  Decoding generated", leave=False):
            z_b   = z_orig[start:start + DECODE_BATCH].to(device)
            recon = (ae.decode(z_b).clamp(0, 1) * 255).byte().cpu().numpy()
            recon = recon.transpose(0, 2, 3, 1)
            for img_arr in recon:
                Image.fromarray(img_arr).save(gdir / f"{img_idx:05d}.png")
                img_idx += 1
    print(f"  Generated images → {gdir}/  ({img_idx} files)")

    # ── AE reconstruction (shared per dim, skip if already done) ─────────────
    aedir = ae_recon_dir(dim)
    if aedir.exists():
        print(f"  AE reconstruction already exists at {aedir}/ — skipping.")
        return

    print("  Encoding CIFAR-10 test set for AE-FID …")
    tf      = transforms.ToTensor()
    testset = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    loader  = DataLoader(testset, batch_size=DECODE_BATCH, shuffle=False, num_workers=2)
    aedir.mkdir(parents=True, exist_ok=True)
    img_idx = 0
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="  AE recon", leave=False):
            recon = (ae(imgs.to(device)).clamp(0, 1) * 255).byte().cpu().numpy()
            recon = recon.transpose(0, 2, 3, 1)
            for img_arr in recon:
                Image.fromarray(img_arr).save(aedir / f"{img_idx:05d}.png")
                img_idx += 1
    print(f"  AE reconstructions → {aedir}/  ({img_idx} files)")


# ── phase 3: metrics ──────────────────────────────────────────────────────────

def metrics(dim: int, size: int) -> None:
    gdir = gen_dir(dim, size)
    if not gdir.exists():
        print(f"[metrics] [ERROR] {gdir} not found — run --decode first.")
        return

    out = metrics_path(dim, size)
    if out.exists():
        print(f"[metrics] [skip] {out.name} already exists.")
        return

    print(f"[metrics] dim={dim}  size={SIZE_LABELS[size]}")
    fid    = compute_fid(str(gdir))
    is_val = compute_inception_score(str(gdir))
    print(f"  FID={fid:.2f}  IS={is_val:.2f}")

    ae_fid = -1.0
    aedir  = ae_recon_dir(dim)
    if aedir.exists():
        ae_fid = compute_fid(str(aedir))
        print(f"  AE-FID={ae_fid:.2f}")

    with open(out, "w") as f:
        json.dump({"fid": fid, "is": is_val, "ae_fid": ae_fid}, f, indent=2)
    print(f"  Saved → {out}")


def compute_fid(gen_dir: str) -> float:
    try:
        from cleanfid import fid
        return float(fid.compute_fid(gen_dir, dataset_name="cifar10",
                                     dataset_res=32, dataset_split="train",
                                     verbose=False))
    except ImportError:
        print("  [warning] clean-fid not installed.")
        return -1.0


def compute_inception_score(gen_dir: str) -> float:
    try:
        import torch_fidelity
        m = torch_fidelity.calculate_metrics(input1=gen_dir, isc=True, verbose=False)
        return float(m["inception_score_mean"])
    except ImportError:
        print("  [warning] torch-fidelity not installed.")
        return -1.0


# ── phase 4: plot ─────────────────────────────────────────────────────────────

def plot() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load all available per-(dim, size) metrics ────────────────────────────
    all_metrics = {}
    for dim in LATENT_DIMS:
        for size in DATASET_SIZES:
            p = metrics_path(dim, size)
            if p.exists():
                with open(p) as f:
                    all_metrics.setdefault(dim, {})[size] = json.load(f)
            else:
                print(f"[plot] [warning] {p.name} not found — skipping.")

    if not all_metrics:
        print("[plot] [ERROR] No metrics found. Run --metrics first.")
        return

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump({str(d): {str(s): v for s, v in sv.items()}
                   for d, sv in all_metrics.items()}, f, indent=2)

    # ── plot 1: FID vs dataset size, one line per dim ─────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i, dim in enumerate(LATENT_DIMS):
        if dim not in all_metrics:
            continue
        sizes = sorted(all_metrics[dim].keys())
        fids  = [all_metrics[dim][s]["fid"] for s in sizes]
        valid = [(s, f) for s, f in zip(sizes, fids) if f >= 0]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax1.plot([SIZE_LABELS[x] for x in xs], ys,
                 marker="o", linewidth=2, label=f"dim={dim}",
                 color=DIM_COLORS[i % len(DIM_COLORS)])

    ax1.set_xlabel("Synthetic Dataset Size")
    ax1.set_ylabel("FID (↓)")
    ax1.set_title("FID vs Synthetic Dataset Size  (one line per latent dim)")
    ax1.legend(title="Latent dim", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax1.grid(True, alpha=0.4)
    plt.tight_layout()
    out1 = RESULTS_DIR / "fid_vs_size.png"
    fig1.savefig(str(out1), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Plot saved → {out1}")

    # ── plot 2: FID vs latent dim, one line per dataset size ──────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i, size in enumerate(DATASET_SIZES):
        dims_with_data = [d for d in LATENT_DIMS if d in all_metrics and size in all_metrics[d]]
        fids = [all_metrics[d][size]["fid"] for d in dims_with_data]
        valid = [(d, f) for d, f in zip(dims_with_data, fids) if f >= 0]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax2.plot(xs, ys,
                 marker="s", linewidth=2, label=SIZE_LABELS[size],
                 color=SIZE_COLORS[i % len(SIZE_COLORS)])

    ax2.set_xlabel("Latent Dim")
    ax2.set_ylabel("FID (↓)")
    ax2.set_title("FID vs Latent Dim  (one line per synthetic dataset size)")
    ax2.legend(title="Dataset size", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    out2 = RESULTS_DIR / "fid_vs_dim.png"
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot saved → {out2}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\nFID summary (rows=dim, cols=size):")
    header = f"{'dim':>6}  " + "  ".join(f"{SIZE_LABELS[s]:>7}" for s in DATASET_SIZES)
    print(header)
    for dim in LATENT_DIMS:
        row = f"{dim:>6}  "
        for size in DATASET_SIZES:
            fid = all_metrics.get(dim, {}).get(size, {}).get("fid", -1)
            row += f"  {fid:>7.2f}" if fid >= 0 else f"  {'—':>7}"
        print(row)

    print("\nStep 4 complete.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    phase  = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--generate", action="store_true", help="Phase 1: generate latents")
    phase.add_argument("--decode",   action="store_true", help="Phase 2: decode latents → images")
    phase.add_argument("--metrics",  action="store_true", help="Phase 3: compute FID/IS")
    phase.add_argument("--plot",     action="store_true", help="Phase 4: plot results")

    parser.add_argument("--dim",  type=int, choices=LATENT_DIMS,
                        help="Latent dim (required for --generate/--decode/--metrics)")
    parser.add_argument("--size", type=int, choices=DATASET_SIZES,
                        help="Synthetic dataset size (required for --generate/--decode/--metrics)")
    args = parser.parse_args()

    if args.plot:
        plot()
        return

    if args.dim is None or args.size is None:
        parser.error("--dim and --size are required for --generate, --decode, and --metrics")

    if args.generate:
        generate(args.dim, args.size)
    elif args.decode:
        decode(args.dim, args.size)
    elif args.metrics:
        metrics(args.dim, args.size)


if __name__ == "__main__":
    main()
