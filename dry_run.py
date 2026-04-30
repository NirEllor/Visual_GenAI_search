"""
Dry-run pipeline checker.

Audits all expected artifacts across every pipeline step and reports:
  - Which files exist / are missing
  - File sizes for existing artifacts
  - Estimated RAM per training job (step3b --load-to-ram)
  - All output paths for evaluation (step4)
  - A final summary with next-step recommendations

Usage:
    python dry_run.py
"""

from pathlib import Path

LATENT_DIMS   = [64, 128, 256, 384, 512, 1024]
DATASET_SIZES = [250_000, 500_000, 1_000_000, 2_000_000]
SIZE_LABELS   = {250_000: "250k", 500_000: "500k",
                 1_000_000: "1M", 2_000_000: "2M"}

CKPT_DIR      = Path("checkpoints")
MODEL_DIR     = Path("models")
LATENT_DIR    = Path("latents")
SYNTHETIC_DIR = Path("synthetic")
RESULTS_DIR   = Path("results/trained_AE")

OK   = "✓"
MISS = "✗"


# ── formatting helpers ────────────────────────────────────────────────────────

def fmt_bytes(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.1f} GB"
    if n >= 1e6:
        return f"{n/1e6:.1f} MB"
    if n >= 1e3:
        return f"{n/1e3:.1f} KB"
    return f"{n} B"


def check(path: Path, extra: str = "") -> tuple[bool, str]:
    if path.exists():
        size = fmt_bytes(path.stat().st_size)
        tag  = f"{OK}  {size}"
        if extra:
            tag += f"  {extra}"
        return True, tag
    return False, MISS


def check_dir(path: Path) -> tuple[bool, str]:
    if path.exists() and path.is_dir():
        n = len(list(path.glob("*.png")))
        return True, f"{OK}  ({n} PNGs)"
    return False, MISS


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── counters ──────────────────────────────────────────────────────────────────

counts = {}   # key → (found, total)

def tally(key: str, found: bool) -> None:
    f, t = counts.get(key, (0, 0))
    counts[key] = (f + found, t + 1)


# ── checks ────────────────────────────────────────────────────────────────────

def check_step0():
    section("Step 0 — AE Checkpoints  (checkpoints/ae_{dim}.pt)")
    for dim in LATENT_DIMS:
        p = CKPT_DIR / f"ae_{dim}.pt"
        ok, tag = check(p)
        print(f"  dim={dim:>4}  {str(p):<45}  {tag}")
        tally("ae_ckpt", ok)


def check_step1():
    section("Step 1 — Latent Files  (latents/latents_{dim}.npy)")
    for dim in LATENT_DIMS:
        p = LATENT_DIR / f"latents_{dim}.npy"
        ok, tag = check(p)
        print(f"  dim={dim:>4}  {str(p):<45}  {tag}")
        tally("latents", ok)

    print()
    print("  Norm stats  (latents/latents_{dim}_norm_stats.npy)")
    for dim in LATENT_DIMS:
        p = LATENT_DIR / f"latents_{dim}_norm_stats.npy"
        ok, tag = check(p)
        print(f"  dim={dim:>4}  {str(p):<45}  {tag}")
        tally("norm_stats", ok)


def check_step2():
    section("Step 2 — Teacher Checkpoints  (models/teacher_{dim}.pt)")
    for dim in LATENT_DIMS:
        p = MODEL_DIR / f"teacher_{dim}.pt"
        ok, tag = check(p)
        print(f"  dim={dim:>4}  {str(p):<45}  {tag}")
        tally("teacher_ckpt", ok)


def check_step3a():
    section("Step 3a — Synthetic Datasets  (synthetic/{dim}/synthetic_{dim}_{n}.npy)")
    for dim in LATENT_DIMS:
        for size in DATASET_SIZES:
            p = SYNTHETIC_DIR / str(dim) / f"synthetic_{dim}_{size}.npy"
            ok, tag = check(p)
            lbl = SIZE_LABELS[size]
            print(f"  dim={dim:>4}  n={lbl:>4}  {str(p):<55}  {tag}")
            tally("synthetic", ok)

    print()
    print("  Trajectory Files  (synthetic/{dim}/trajectories_{dim}.npy)")
    for dim in LATENT_DIMS:
        p = SYNTHETIC_DIR / str(dim) / f"trajectories_{dim}.npy"
        ok, tag = check(p)
        print(f"  dim={dim:>4}  {str(p):<55}  {tag}")
        tally("trajectories", ok)


def check_step3b():
    section("Step 3b — Student Checkpoints  (models/student_{dim}_{n}.pt)")
    for dim in LATENT_DIMS:
        for size in DATASET_SIZES:
            p = MODEL_DIR / f"student_{dim}_{size}.pt"
            ok, tag = check(p)
            lbl = SIZE_LABELS[size]
            print(f"  dim={dim:>4}  n={lbl:>4}  {str(p):<55}  {tag}")
            tally("student_ckpt", ok)

    print()
    print("  Estimated RAM per training job  (--load-to-ram)")
    print(f"  {'dim':>4}  {'size':>4}  {'dataset':>9}  {'model+opt':>9}  {'total':>9}")
    for dim in LATENT_DIMS:
        for size in DATASET_SIZES:
            dataset_bytes = size * dim * 4
            model_bytes   = 200 * 1024 * 1024          # ~200 MB overhead (model + optimizer)
            total_bytes   = dataset_bytes + model_bytes
            lbl = SIZE_LABELS[size]
            print(f"  {dim:>4}  {lbl:>4}  "
                  f"{fmt_bytes(dataset_bytes):>9}  "
                  f"{fmt_bytes(model_bytes):>9}  "
                  f"{fmt_bytes(total_bytes):>9}")


def check_step4():
    section("Step 4 — Evaluation Outputs")

    print("  Generated latents  (results/trained_AE/z_orig_{dim}_{n}.npy)")
    for dim in LATENT_DIMS:
        for size in DATASET_SIZES:
            p = RESULTS_DIR / f"z_orig_{dim}_{size}.npy"
            ok, tag = check(p)
            lbl = SIZE_LABELS[size]
            print(f"  dim={dim:>4}  n={lbl:>4}  {str(p):<60}  {tag}")
            tally("z_orig", ok)

    print()
    print("  Decoded image dirs  (results/trained_AE/generated_{dim}_{n}/)")
    for dim in LATENT_DIMS:
        for size in DATASET_SIZES:
            p = RESULTS_DIR / f"generated_{dim}_{size}"
            ok, tag = check_dir(p)
            lbl = SIZE_LABELS[size]
            print(f"  dim={dim:>4}  n={lbl:>4}  {str(p):<60}  {tag}")
            tally("gen_dir", ok)

    print()
    print("  AE reconstruction dirs  (shared per dim, results/trained_AE/ae_reconstructed_{dim}/)")
    for dim in LATENT_DIMS:
        p = RESULTS_DIR / f"ae_reconstructed_{dim}"
        ok, tag = check_dir(p)
        print(f"  dim={dim:>4}  {str(p):<60}  {tag}")
        tally("ae_recon_dir", ok)

    print()
    print("  Metrics files  (results/trained_AE/metrics_{dim}_{n}.json)")
    for dim in LATENT_DIMS:
        for size in DATASET_SIZES:
            p = RESULTS_DIR / f"metrics_{dim}_{size}.json"
            ok, tag = check(p)
            lbl = SIZE_LABELS[size]
            print(f"  dim={dim:>4}  n={lbl:>4}  {str(p):<60}  {tag}")
            tally("metrics", ok)

    print()
    print("  Plot outputs")
    for name in ["fid_vs_size.png", "fid_vs_dim.png", "metrics.json"]:
        p = RESULTS_DIR / name
        ok, tag = check(p)
        print(f"  {str(p):<65}  {tag}")
        tally("plots", ok)


# ── summary ───────────────────────────────────────────────────────────────────

def summary():
    section("Summary")

    labels = {
        "ae_ckpt":      "AE checkpoints      (step0)",
        "latents":      "Latent files         (step1)",
        "norm_stats":   "Norm stat files      (step1)",
        "teacher_ckpt": "Teacher checkpoints  (step2)",
        "synthetic":    "Synthetic datasets   (step3a)",
        "trajectories": "Trajectory files     (step3a)",
        "student_ckpt": "Student checkpoints  (step3b)",
        "z_orig":       "Generated latents    (step4 generate)",
        "gen_dir":      "Decoded image dirs   (step4 decode)",
        "ae_recon_dir": "AE recon dirs        (step4 decode)",
        "metrics":      "Metrics files        (step4 metrics)",
        "plots":        "Plot outputs         (step4 plot)",
    }

    any_missing = False
    for key, label in labels.items():
        f, t = counts.get(key, (0, 0))
        status = OK if f == t else MISS
        if f < t:
            any_missing = True
        print(f"  {status}  {label:<45}  {f}/{t}")

    print()
    if not any_missing:
        print("  All artifacts present. Pipeline is complete.")
    else:
        print("  Next steps:")
        missing_checks = [
            ("ae_ckpt",      "  python step0_train_autoencoder.py --dim <dim>"),
            ("latents",      "  python step1_extract_latents.py --dim <dim>"),
            ("norm_stats",   "  python step2_train_teachers.py --dim <dim>   (normalises latents)"),
            ("teacher_ckpt", "  python step2_train_teachers.py --dim <dim>"),
            ("synthetic",    "  python step3a_generate.py --dim <dim>"),
            ("trajectories", "  python step3a_generate.py --dim <dim>"),
            ("student_ckpt", "  python step3b_distill.py --dim <dim> --size <n>"),
            ("z_orig",       "  python step4_evaluate.py --generate --dim <dim> --size <n>"),
            ("gen_dir",      "  python step4_evaluate.py --decode   --dim <dim> --size <n>"),
            ("metrics",      "  python step4_evaluate.py --metrics  --dim <dim> --size <n>"),
            ("plots",        "  python step4_evaluate.py --plot"),
        ]
        for key, cmd in missing_checks:
            f, t = counts.get(key, (0, 0))
            if f < t:
                print(f"    {cmd}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Pipeline Dry Run")
    print("=" * 60)

    check_step0()
    check_step1()
    check_step2()
    check_step3a()
    check_step3b()
    check_step4()
    summary()

    print()


if __name__ == "__main__":
    main()
