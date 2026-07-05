"""
Step 4 — Generate images, compute FID/IS, and plot results for teachers + 24 students.

Four phases, each independently restartable:

  --generate  --dim D [--size N | --teacher]   → latents/<tag>/dim_D/z_eval[_nN].npy
  --decode    --dim D [--size N | --teacher]   → generated/<tag>/dim_D[/n_N]/*.png
  --metrics   --dim D [--size N | --teacher]   → metrics/metrics_<tag>_dim<D>[_n<N>].json
  --plot                                       → metrics/metrics_all.json + plots/*.png

Omit --size and --teacher to run a student. Use --teacher to evaluate the teacher.
Pass --overwrite to regenerate outputs that already exist.

Usage:
    python step4_evaluate.py --generate --dim 128 --size 500000
    python step4_evaluate.py --generate --dim 128 --teacher
    python step4_evaluate.py --decode   --dim 128 --size 500000 --overwrite
    python step4_evaluate.py --plot     --exp-name my_run
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

from exp_config import (
    get_paths, add_exp_arg, add_teacher_ckpt_arg,
    get_teacher_ckpt_path, print_exp_summary,
    maybe_clear_dir, ExpPaths,
)

# ── configuration ─────────────────────────────────────────────────────────────
LATENT_DIMS   = [64, 128, 256, 384, 512, 1024]
DATASET_SIZES = [50_000, 100_000, 150_000, 200_000]
N_SAMPLES     = 10_000
EULER_STEPS   = 200
DECODE_BATCH  = 256

SIZE_LABELS = {50_000: "50k", 100_000: "100k",
               150_000: "150k", 200_000: "200k"}
DIM_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SIZE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ── experiment-scoped path helpers ────────────────────────────────────────────

def _tag(size: Optional[int]) -> str:
    return "teacher" if size is None else f"student_n{size}"

def _label(size: Optional[int]) -> str:
    return "teacher" if size is None else SIZE_LABELS[size]

def _z_path(paths: ExpPaths, dim: int, size: Optional[int]) -> Path:
    """Intermediate latent file (z_orig) for the generate→decode pipeline."""
    if size is None:
        return paths.teacher_latent_dir / f"dim_{dim}" / "z_eval.npy"
    return paths.student_latent_dir / f"dim_{dim}" / f"z_eval_n{size}.npy"

def _gen_dir(paths: ExpPaths, dim: int, size: Optional[int]) -> Path:
    """Directory that holds decoded PNG images for FID computation."""
    if size is None:
        return paths.gen_dir / "teacher" / f"dim_{dim}"
    return paths.gen_dir / "student" / f"dim_{dim}" / f"n_{size}"

def _ae_recon_dir(paths: ExpPaths, dim: int) -> Path:
    return paths.ae_recon_dir / f"dim_{dim}"

def _metrics_path(paths: ExpPaths, dim: int, size: Optional[int]) -> Path:
    return paths.metrics_dir / f"metrics_{_tag(size)}_dim{dim}.json"


# ── phase 1: generate latents ─────────────────────────────────────────────────

def generate(dim: int, size: Optional[int],
             paths: ExpPaths, overwrite: bool,
             teacher_ckpt: str = "best_fid") -> None:
    import torch
    from models.diffusion import FlowMatching
    from models.denoiser import load_student, load_teacher

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = _z_path(paths, dim, size)
    if out.exists() and not overwrite:
        print(f"[generate] [skip] {out.name} already exists.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    is_teacher = size is None
    if is_teacher:
        ckpt_path = get_teacher_ckpt_path(paths, dim, teacher_ckpt)
        if not ckpt_path.exists():
            print(f"[generate] [ERROR] {ckpt_path} not found — run step2 first.")
            return
        print(f"[generate] dim={dim}  model=teacher  ckpt={ckpt_path.name}  device={device}")
        model = load_teacher(str(ckpt_path), latent_dim=dim, device=device)
        print(f"  Generating {N_SAMPLES:,} samples with Euler-{EULER_STEPS} …")
    else:
        ckpt_path = paths.ckpt_dir / f"student_{dim}_{size}.pt"
        if not ckpt_path.exists():
            print(f"[generate] [ERROR] {ckpt_path} not found — run step3b first.")
            return
        print(f"[generate] dim={dim}  size={_label(size)}  device={device}")
        model = load_student(str(ckpt_path), latent_dim=dim, device=device)
        print(f"  Generating {N_SAMPLES:,} samples with 1-Step Generation …")

    ckpt     = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    lat_mean = ckpt["latent_mean"].cpu().numpy()
    lat_std  = ckpt["latent_std"].cpu().numpy()
    flow     = FlowMatching(device=device)

    all_latents, generated = [], 0
    while generated < N_SAMPLES:
        n_batch = min(512, N_SAMPLES - generated)
        if is_teacher:
            z = flow.euler_sample(model, (n_batch, dim), n_steps=EULER_STEPS)
        else:
            z = flow.single_step_sample(model, (n_batch, dim))
        all_latents.append(z.cpu().numpy())
        generated += n_batch

    z_norm = np.concatenate(all_latents, axis=0)
    z_orig = (z_norm * lat_std + lat_mean).astype(np.float32)
    print(f"  Latent range: [{z_orig.min():.3f}, {z_orig.max():.3f}]")
    np.save(str(out), z_orig)
    print(f"  Saved → {out}")

    print_exp_summary(paths, latent_path=out)


# ── phase 2: decode latents → images ─────────────────────────────────────────

def decode(dim: int, size: Optional[int],
           paths: ExpPaths, overwrite: bool) -> None:
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from models.autoencoder import ConvAutoencoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[decode] dim={dim}  model={_label(size)}  device={device}")

    zp = _z_path(paths, dim, size)
    if not zp.exists():
        print(f"  [ERROR] {zp} not found — run --generate first.")
        return

    ae_ckpt = paths.ckpt_dir / f"ae_{dim}.pt"
    if not ae_ckpt.exists():
        print(f"  [ERROR] {ae_ckpt} not found — run step0 first.")
        return

    ckpt = torch.load(ae_ckpt, map_location=device, weights_only=True)
    ae   = ConvAutoencoder(latent_dim=dim).to(device)
    ae.load_state_dict(ckpt["state_dict"])
    ae.eval()

    # ── decode generated latents → PNGs ──────────────────────────────────────
    gdir = _gen_dir(paths, dim, size)
    if not maybe_clear_dir(gdir, overwrite, label=f"generated/{_tag(size)}/dim_{dim}"):
        print(f"  [skip] Image dir already exists — skipping decode.")
        # Still fall through to AE recon step below
    else:
        z_orig  = torch.from_numpy(np.load(str(zp)))
        img_idx = 0
        with torch.no_grad():
            for start in tqdm(range(0, len(z_orig), DECODE_BATCH),
                              desc="  Decoding", leave=False):
                z_b = z_orig[start:start + DECODE_BATCH].to(device)
                recon_logits = ae.decode(z_b)
                recon = (
                    torch.sigmoid(recon_logits)
                    .clamp(0, 1)
                    .mul(255)
                    .byte()
                    .cpu()
                    .numpy()
                )
                recon = recon.transpose(0, 2, 3, 1)
                for img_arr in recon:
                    Image.fromarray(img_arr).save(gdir / f"{img_idx:05d}.png")
                    img_idx += 1
        print(f"  Images → {gdir}/  ({img_idx} files)")

    # ── AE reconstruction (shared per dim, computed once) ─────────────────────
    aedir = _ae_recon_dir(paths, dim)
    if not maybe_clear_dir(aedir, overwrite, label=f"ae_recon/dim_{dim}"):
        print(f"  [info] AE recon dir already exists — reusing.")
    else:
        print("  Encoding CIFAR-10 test set for AE-FID …")
        tf      = transforms.ToTensor()
        testset = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
        loader  = DataLoader(testset, batch_size=DECODE_BATCH, shuffle=False, num_workers=2)
        img_idx = 0
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc="  AE recon", leave=False):
                z, _, _      = ae.encode(imgs.to(device), sample=False)
                recon_logits = ae.decode(z)
                recon        = (torch.sigmoid(recon_logits).clamp(0, 1) * 255).byte().cpu().numpy()
                recon        = recon.transpose(0, 2, 3, 1)
                for img_arr in recon:
                    Image.fromarray(img_arr).save(aedir / f"{img_idx:05d}.png")
                    img_idx += 1
        print(f"  AE recon → {aedir}/  ({img_idx} files)")

    print_exp_summary(paths, gen_path=gdir)


# ── phase 3: metrics ──────────────────────────────────────────────────────────

def metrics(dim: int, size: Optional[int],
            paths: ExpPaths, overwrite: bool,
            teacher_ckpt: str = "best_fid") -> None:
    gdir = _gen_dir(paths, dim, size)
    if not gdir.exists():
        print(f"[metrics] [ERROR] {gdir} not found — run --decode first.")
        return

    out = _metrics_path(paths, dim, size)
    if out.exists() and not overwrite:
        print(f"[metrics] [skip] {out.name} already exists.")
        return

    is_teacher = size is None
    print(f"[metrics] dim={dim}  model={_label(size)}")
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)

    fid_val = compute_fid(str(gdir))
    is_val  = compute_inception_score(str(gdir))
    print(f"  FID={fid_val:.2f}  IS={is_val:.2f}")

    ae_fid = -1.0
    aedir  = _ae_recon_dir(paths, dim)
    if aedir.exists():
        ae_fid = compute_fid(str(aedir))
        print(f"  AE-FID={ae_fid:.2f}")

    # Resolve checkpoint metadata for teachers
    ckpt_type_str  = teacher_ckpt if is_teacher else None
    ckpt_path_str  = None
    if is_teacher:
        ckpt_path_str = str(get_teacher_ckpt_path(paths, dim, teacher_ckpt))

    with open(out, "w") as fh:
        json.dump(
            {
                "fid":              fid_val,
                "is":               is_val,
                "ae_fid":           ae_fid,
                "exp_name":         paths.exp_name,
                "latent_dim":       dim,
                "model_type":       "teacher" if is_teacher else "student",
                "n_samples":        None if is_teacher else size,
                "checkpoint_type":  ckpt_type_str,
                "checkpoint_path":  ckpt_path_str,
            },
            fh,
            indent=2,
        )
    print(f"  Saved → {out}")

    print_exp_summary(paths, metrics_path=out)


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

def plot(paths: ExpPaths) -> None:
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)

    # ── load student metrics ──────────────────────────────────────────────────
    student_metrics: dict = {}
    for dim in LATENT_DIMS:
        for size in DATASET_SIZES:
            p = _metrics_path(paths, dim, size)
            if p.exists():
                with open(p) as fh:
                    student_metrics.setdefault(dim, {})[size] = json.load(fh)
            else:
                print(f"[plot] [warning] {p.name} missing — skipping.")

    # ── load teacher metrics ──────────────────────────────────────────────────
    teacher_metrics: dict = {}
    for dim in LATENT_DIMS:
        p = _metrics_path(paths, dim, None)
        if p.exists():
            with open(p) as fh:
                teacher_metrics[dim] = json.load(fh)
        else:
            print(f"[plot] [warning] {p.name} missing — skipping.")

    # ── load AE metrics (from step0b) ─────────────────────────────────────────
    ae_metrics: dict = {}
    ae_metrics_file = paths.metrics_dir / "ae_metrics.json"
    if ae_metrics_file.exists():
        with open(ae_metrics_file) as fh:
            ae_metrics = json.load(fh)
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
    unified_path = paths.metrics_dir / "metrics_all.json"
    with open(unified_path, "w") as fh:
        json.dump(unified, fh, indent=2)
    print(f"  Unified metrics → {unified_path}")

    # ── plot 1: FID vs dataset size, one line per dim + teacher dashes ────────
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for i, dim in enumerate(LATENT_DIMS):
        color = DIM_COLORS[i % len(DIM_COLORS)]

        if dim in student_metrics:
            sizes = sorted(student_metrics[dim].keys())
            valid = [(s, student_metrics[dim][s]["fid"])
                     for s in sizes if student_metrics[dim][s]["fid"] >= 0]
            if valid:
                xs, ys = zip(*valid)
                ax1.plot([SIZE_LABELS[x] for x in xs], ys,
                         marker="o", linewidth=2, color=color, label=f"dim={dim}")

        if dim in teacher_metrics and teacher_metrics[dim]["fid"] >= 0:
            t_fid = teacher_metrics[dim]["fid"]
            ax1.axhline(t_fid, color=color, linewidth=1, linestyle="--", alpha=0.6)

    ax1.plot([], [], color="grey", linewidth=1, linestyle="--", alpha=0.6,
             label="teacher (per dim)")
    ax1.set_xlabel("Synthetic Dataset Size")
    ax1.set_ylabel("FID (↓)")
    ax1.set_title("FID vs Synthetic Dataset Size\n(dashed = teacher baseline per dim)")
    ax1.legend(title="Latent dim", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax1.grid(True, alpha=0.4)
    plt.tight_layout()
    out1 = paths.plots_dir / "fid_vs_size.png"
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
    out2 = paths.plots_dir / "fid_vs_dim.png"
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot saved → {out2}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\nFID summary  (— = missing)")
    cols   = ["teacher"] + [SIZE_LABELS[s] for s in DATASET_SIZES]
    header = f"{'dim':>6}  " + "  ".join(f"{c:>8}" for c in cols)
    print(header)
    for dim in LATENT_DIMS:
        t_fid = teacher_metrics.get(dim, {}).get("fid", -1)
        t_str = f"{t_fid:>8.2f}" if t_fid >= 0 else f"{'—':>8}"
        row   = f"{dim:>6}  {t_str}"
        for size in DATASET_SIZES:
            fid_val = student_metrics.get(dim, {}).get(size, {}).get("fid", -1)
            row += f"  {fid_val:>8.2f}" if fid_val >= 0 else f"  {'—':>8}"
        print(row)

    print_exp_summary(paths, metrics_path=unified_path)
    print("\nStep 4 complete.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--generate", action="store_true")
    phase.add_argument("--decode",   action="store_true")
    phase.add_argument("--metrics",  action="store_true")
    phase.add_argument("--plot",     action="store_true")

    parser.add_argument("--dim",      type=int, choices=LATENT_DIMS)
    parser.add_argument("--size",     type=int, choices=DATASET_SIZES)
    parser.add_argument("--teacher",  action="store_true",
                        help="Evaluate the teacher model instead of a student")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete and regenerate existing output files/directories")
    add_exp_arg(parser)
    add_teacher_ckpt_arg(parser)
    args = parser.parse_args()

    paths = get_paths(args.exp_name)

    if args.plot:
        plot(paths)
        return

    if args.dim is None:
        parser.error("--dim is required for --generate, --decode, and --metrics")
    if not args.teacher and args.size is None:
        parser.error("provide --size N or --teacher")
    if args.teacher and args.size is not None:
        parser.error("--teacher and --size are mutually exclusive")

    size = None if args.teacher else args.size

    if args.generate:
        generate(args.dim, size, paths, args.overwrite, args.teacher_ckpt)
    elif args.decode:
        decode(args.dim, size, paths, args.overwrite)
    elif args.metrics:
        metrics(args.dim, size, paths, args.overwrite, args.teacher_ckpt)


if __name__ == "__main__":
    main()
