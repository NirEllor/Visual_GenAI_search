"""
Step 4 — Generate images, compute FID/IS, and plot results for teachers + 24 students.

Four phases, each independently restartable:

  --generate  --dim D [--size N | --teacher]
  --decode    --dim D [--size N | --teacher]
  --metrics   --dim D [--size N | --teacher]
  --plot  →  fid_vs_size.png, fid_vs_dim.png, metrics_all.json

Omit --size and --teacher to run a student. Use --teacher to evaluate the teacher.

Usage:
    python step4_evaluate.py --generate --dim 128 --size 500000
    python step4_evaluate.py --generate --dim 128 --teacher
    python step4_evaluate.py --plot
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional

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
AE_EVAL_DIR   = Path("results/ae_eval")

SIZE_LABELS = {250_000: "250k", 500_000: "500k",
               1_000_000: "1M",  2_000_000: "2M"}
DIM_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SIZE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ── path helpers (size=None → teacher) ───────────────────────────────────────

def _tag(size: Optional[int]) -> str:
    return "teacher" if size is None else str(size)

def _label(size: Optional[int]) -> str:
    return "teacher" if size is None else SIZE_LABELS[size]

def z_path(dim: int, size: Optional[int]) -> Path:
    return RESULTS_DIR / f"z_orig_{dim}_{_tag(size)}.npy"

def gen_dir(dim: int, size: Optional[int]) -> Path:
    return RESULTS_DIR / f"generated_{dim}_{_tag(size)}"

def ae_recon_dir(dim: int) -> Path:
    return RESULTS_DIR / f"ae_reconstructed_{dim}"

def metrics_path(dim: int, size: Optional[int]) -> Path:
    return RESULTS_DIR / f"metrics_{dim}_{_tag(size)}.json"


# ── phase 1: generate latents ─────────────────────────────────────────────────

def generate(dim: int, size: Optional[int]) -> None:
    import torch
    from models.diffusion import FlowMatching
    from models.denoiser import load_student, load_teacher

    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out = z_path(dim, size)
    if out.exists():
        print(f"[generate] [skip] {out.name} already exists.")
        return

    is_teacher = size is None
    if is_teacher:
        ckpt_path = MODEL_DIR / f"teacher_{dim}.pt"
        if not ckpt_path.exists():
            print(f"[generate] [ERROR] {ckpt_path} not found — run step2 first.")
            return
        print(f"[generate] dim={dim}  model=teacher  device={device}")
        model       = load_teacher(str(ckpt_path), latent_dim=dim, device=device)
        euler_steps = EULER_STEPS
    else:
        ckpt_path = MODEL_DIR / f"student_{dim}_{size}.pt"
        if not ckpt_path.exists():
            print(f"[generate] [ERROR] {ckpt_path} not found — run step3b first.")
            return
        print(f"[generate] dim={dim}  size={_label(size)}  device={device}")
        model       = load_student(str(ckpt_path), latent_dim=dim, device=device)
        euler_steps = EULER_STEPS

    ckpt     = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    lat_mean = float(ckpt["latent_mean"])
    lat_std  = float(ckpt["latent_std"])
    flow     = FlowMatching(device=device)

    print(f"  Generating {N_SAMPLES:,} samples with Euler-{euler_steps} …")
    all_latents, generated = [], 0
    while generated < N_SAMPLES:
        n_batch = min(512, N_SAMPLES - generated)
        z = flow.euler_sample(model, (n_batch, dim), n_steps=euler_steps)
        all_latents.append(z.cpu().numpy())
        generated += n_batch

    z_norm = np.concatenate(all_latents, axis=0)
    z_orig = (z_norm * lat_std + lat_mean).astype(np.float32)
    print(f"  Latent range: [{z_orig.min():.3f}, {z_orig.max():.3f}]")
    np.save(str(out), z_orig)
    print(f"  Saved → {out}")


# ── phase 2: decode latents → images ─────────────────────────────────────────

def decode(dim: int, size: Optional[int]) -> None:
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from models.autoencoder import ConvAutoencoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[decode] dim={dim}  model={_label(size)}  device={device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    zp = z_path(dim, size)
    if not zp.exists():
        print(f"  [ERROR] {zp} not found — run --generate first.")
        return

    ae_ckpt = CKPT_DIR / f"ae_{dim}.pt"
    if not ae_ckpt.exists():
        print(f"  [ERROR] {ae_ckpt} not found — run step0 first.")
        return

    ckpt = torch.load(ae_ckpt, map_location=device, weights_only=True)
    ae   = ConvAutoencoder(latent_dim=dim).to(device)
    ae.load_state_dict(ckpt["state_dict"])
    ae.eval()

    # ── decode generated latents → PNGs ──────────────────────────────────────
    gdir    = gen_dir(dim, size)
    gdir.mkdir(parents=True, exist_ok=True)
    z_orig  = torch.from_numpy(np.load(str(zp)))
    img_idx = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(z_orig), DECODE_BATCH),
                          desc="  Decoding", leave=False):
            z_b   = z_orig[start:start + DECODE_BATCH].to(device)
            recon = (ae.decode(z_b).clamp(0, 1) * 255).byte().cpu().numpy()
            recon = recon.transpose(0, 2, 3, 1)
            for img_arr in recon:
                Image.fromarray(img_arr).save(gdir / f"{img_idx:05d}.png")
                img_idx += 1
    print(f"  Images → {gdir}/  ({img_idx} files)")

    # ── AE reconstruction (shared per dim, computed once) ─────────────────────
    aedir = ae_recon_dir(dim)
    if aedir.exists():
        print(f"  AE recon already exists at {aedir}/ — skipping.")
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
    print(f"  AE recon → {aedir}/  ({img_idx} files)")


# ── phase 3: metrics ──────────────────────────────────────────────────────────

def metrics(dim: int, size: Optional[int]) -> None:
    gdir = gen_dir(dim, size)
    if not gdir.exists():
        print(f"[metrics] [ERROR] {gdir} not found — run --decode first.")
        return

    out = metrics_path(dim, size)
    if out.exists():
        print(f"[metrics] [skip] {out.name} already exists.")
        return

    print(f"[metrics] dim={dim}  model={_label(size)}")
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


# ── phase 4: plot + unified JSON ──────────────────────────────────────────────

def plot() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load student metrics ──────────────────────────────────────────────────
    student_metrics = {}
    for dim in LATENT_DIMS:
        for size in DATASET_SIZES:
            p = metrics_path(dim, size)
            if p.exists():
                with open(p) as f:
                    student_metrics.setdefault(dim, {})[size] = json.load(f)
            else:
                print(f"[plot] [warning] {p.name} missing — skipping.")

    # ── load teacher metrics ──────────────────────────────────────────────────
    teacher_metrics = {}
    for dim in LATENT_DIMS:
        p = metrics_path(dim, None)
        if p.exists():
            with open(p) as f:
                teacher_metrics[dim] = json.load(f)
        else:
            print(f"[plot] [warning] metrics_{dim}_teacher.json missing — skipping.")

    # ── load AE metrics (from step0b) ─────────────────────────────────────────
    ae_metrics = {}
    ae_metrics_file = AE_EVAL_DIR / "metrics.json"
    if ae_metrics_file.exists():
        with open(ae_metrics_file) as f:
            ae_metrics = json.load(f)
    else:
        print(f"[plot] [warning] {ae_metrics_file} missing — AE metrics excluded.")

    if not student_metrics and not teacher_metrics:
        print("[plot] [ERROR] No metrics found.")
        return

    # ── unified JSON ─────────────────────────────────────────────────────────
    unified = {
        "ae":      ae_metrics,
        "teacher": {str(d): v for d, v in teacher_metrics.items()},
        "student": {str(d): {str(s): v for s, v in sv.items()}
                    for d, sv in student_metrics.items()},
    }
    unified_path = RESULTS_DIR / "metrics_all.json"
    with open(unified_path, "w") as f:
        json.dump(unified, f, indent=2)
    print(f"  Unified metrics → {unified_path}")

    # ── plot 1: FID vs dataset size, one line per dim + teacher dashes ────────
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for i, dim in enumerate(LATENT_DIMS):
        color = DIM_COLORS[i % len(DIM_COLORS)]

        # student line
        if dim in student_metrics:
            sizes = sorted(student_metrics[dim].keys())
            valid = [(s, student_metrics[dim][s]["fid"])
                     for s in sizes if student_metrics[dim][s]["fid"] >= 0]
            if valid:
                xs, ys = zip(*valid)
                ax1.plot([SIZE_LABELS[x] for x in xs], ys,
                         marker="o", linewidth=2, color=color, label=f"dim={dim}")

        # teacher horizontal dashed line
        if dim in teacher_metrics and teacher_metrics[dim]["fid"] >= 0:
            t_fid = teacher_metrics[dim]["fid"]
            ax1.axhline(t_fid, color=color, linewidth=1, linestyle="--", alpha=0.6)

    # legend entry for teacher style
    ax1.plot([], [], color="grey", linewidth=1, linestyle="--", alpha=0.6,
             label="teacher (per dim)")
    ax1.set_xlabel("Synthetic Dataset Size")
    ax1.set_ylabel("FID (↓)")
    ax1.set_title("FID vs Synthetic Dataset Size\n(dashed = teacher baseline per dim)")
    ax1.legend(title="Latent dim", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax1.grid(True, alpha=0.4)
    plt.tight_layout()
    out1 = RESULTS_DIR / "fid_vs_size.png"
    fig1.savefig(str(out1), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Plot saved → {out1}")

    # ── plot 2: FID vs latent dim, one line per dataset size + teacher ─────────
    fig2, ax2 = plt.subplots(figsize=(11, 6))

    for i, size in enumerate(DATASET_SIZES):
        dims_avail = [d for d in LATENT_DIMS
                      if d in student_metrics and size in student_metrics[d]]
        valid = [(d, student_metrics[d][size]["fid"])
                 for d in dims_avail if student_metrics[d][size]["fid"] >= 0]
        if valid:
            xs, ys = zip(*valid)
            ax2.plot(xs, ys, marker="s", linewidth=2,
                     color=SIZE_COLORS[i % len(SIZE_COLORS)],
                     label=SIZE_LABELS[size])

    # teacher line across dims
    teacher_pts = [(d, teacher_metrics[d]["fid"])
                   for d in LATENT_DIMS
                   if d in teacher_metrics and teacher_metrics[d]["fid"] >= 0]
    if teacher_pts:
        tx, ty = zip(*teacher_pts)
        ax2.plot(tx, ty, marker="*", markersize=12, linewidth=2,
                 color="black", linestyle="--", label="teacher")

    ax2.set_xlabel("Latent Dim")
    ax2.set_ylabel("FID (↓)")
    ax2.set_title("FID vs Latent Dim\n(one line per synthetic dataset size + teacher)")
    ax2.legend(title="Dataset size", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    out2 = RESULTS_DIR / "fid_vs_dim.png"
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot saved → {out2}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\nFID summary  (— = missing)")
    cols  = ["teacher"] + [SIZE_LABELS[s] for s in DATASET_SIZES]
    header = f"{'dim':>6}  " + "  ".join(f"{c:>8}" for c in cols)
    print(header)
    for dim in LATENT_DIMS:
        t_fid = teacher_metrics.get(dim, {}).get("fid", -1)
        t_str = f"{t_fid:>8.2f}" if t_fid >= 0 else f"{'—':>8}"
        row = f"{dim:>6}  {t_str}"
        for size in DATASET_SIZES:
            fid = student_metrics.get(dim, {}).get(size, {}).get("fid", -1)
            row += f"  {fid:>8.2f}" if fid >= 0 else f"  {'—':>8}"
        print(row)

    print("\nStep 4 complete.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--generate", action="store_true")
    phase.add_argument("--decode",   action="store_true")
    phase.add_argument("--metrics",  action="store_true")
    phase.add_argument("--plot",     action="store_true")

    parser.add_argument("--dim",     type=int, choices=LATENT_DIMS)
    parser.add_argument("--size",    type=int, choices=DATASET_SIZES)
    parser.add_argument("--teacher", action="store_true",
                        help="Evaluate the teacher model instead of a student")
    args = parser.parse_args()

    if args.plot:
        plot()
        return

    if args.dim is None:
        parser.error("--dim is required for --generate, --decode, and --metrics")
    if not args.teacher and args.size is None:
        parser.error("provide --size N or --teacher")
    if args.teacher and args.size is not None:
        parser.error("--teacher and --size are mutually exclusive")

    size = None if args.teacher else args.size

    if args.generate:
        generate(args.dim, size)
    elif args.decode:
        decode(args.dim, size)
    elif args.metrics:
        metrics(args.dim, size)


if __name__ == "__main__":
    main()
