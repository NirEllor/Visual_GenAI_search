"""
Central experiment path configuration.

All pipeline scripts import from here so that every run lives under a clean
results/<exp_name>/ directory, preventing outputs from different experiments
from being mixed together.

Usage in each script:
    from exp_config import get_paths, add_exp_arg, save_config, print_exp_summary

    parser = argparse.ArgumentParser()
    add_exp_arg(parser)
    args = parser.parse_args()
    paths = get_paths(args.exp_name)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_EXP_NAME = "ae_kl_lpips_only_v1"


@dataclass
class ExpPaths:
    exp_name: str
    exp_dir: Path

    # ── model weights ─────────────────────────────────────────────────────────
    ckpt_dir: Path          # ae_{dim}.pt, teacher_{dim}.pt, student_{dim}_{n}.pt

    # ── latent arrays ─────────────────────────────────────────────────────────
    real_latent_dir: Path   # latents/real/  — encoded CIFAR-10 + norm stats
    teacher_latent_dir: Path  # latents/teacher/  — teacher synthetic latents (step3a)
    student_latent_dir: Path  # latents/student/  — student eval latents (step4)

    # ── decoded image directories ─────────────────────────────────────────────
    ae_recon_dir: Path      # ae_recon/dim_{dim}/   — AE encode→decode reconstructions
    gen_dir: Path           # generated/teacher|student/dim_{dim}[/n_{size}]/

    # ── outputs ───────────────────────────────────────────────────────────────
    metrics_dir: Path
    plots_dir: Path


def get_paths(exp_name: str = DEFAULT_EXP_NAME) -> ExpPaths:
    """Return all experiment-scoped paths derived from exp_name."""
    exp_dir = Path("results") / exp_name
    return ExpPaths(
        exp_name=exp_name,
        exp_dir=exp_dir,
        ckpt_dir=exp_dir / "checkpoints",
        real_latent_dir=exp_dir / "latents" / "real",
        teacher_latent_dir=exp_dir / "latents" / "teacher",
        student_latent_dir=exp_dir / "latents" / "student",
        ae_recon_dir=exp_dir / "ae_recon",
        gen_dir=exp_dir / "generated",
        metrics_dir=exp_dir / "metrics",
        plots_dir=exp_dir / "plots",
    )


def add_exp_arg(parser) -> None:
    """Attach --exp-name to any argparse.ArgumentParser."""
    parser.add_argument(
        "--exp-name",
        default=DEFAULT_EXP_NAME,
        help=(
            f"Experiment name — all outputs go to results/<exp_name>/  "
            f"(default: {DEFAULT_EXP_NAME})"
        ),
    )


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_config(paths: ExpPaths, extra: Optional[dict] = None) -> None:
    """
    Write results/<exp_name>/config.json at experiment start.
    Idempotent — existing file is overwritten with latest metadata.
    """
    paths.exp_dir.mkdir(parents=True, exist_ok=True)
    cfg: dict = {
        "exp_name": paths.exp_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_hash(),
    }
    if extra:
        cfg.update(extra)
    cfg_path = paths.exp_dir / "config.json"
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"  Config saved  → {cfg_path}")


def maybe_clear_dir(dirpath: Path, overwrite: bool, label: str = "") -> bool:
    """
    Ensure dirpath is ready for writing.

    Returns True  → caller should proceed (directory is empty / freshly created).
    Returns False → caller should skip (directory exists, overwrite=False).

    If overwrite=True and the directory already exists it is deleted before
    returning True, so the caller starts with a clean slate.
    """
    import shutil
    tag = label or str(dirpath)
    if dirpath.exists():
        if not overwrite:
            print(f"  [skip] {tag} already exists — pass --overwrite to regenerate.")
            return False
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return True


def print_exp_summary(
    paths: ExpPaths,
    ckpt_path: Optional[Path] = None,
    latent_path: Optional[Path] = None,
    gen_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
) -> None:
    bar = "─" * 54
    print(f"\n{bar}")
    print(f"  Experiment : {paths.exp_name}")
    print(f"  Exp dir    : {paths.exp_dir}")
    if ckpt_path:
        print(f"  Checkpoint : {ckpt_path}")
    if latent_path:
        print(f"  Latents    : {latent_path}")
    if gen_path:
        print(f"  Images     : {gen_path}")
    if metrics_path:
        print(f"  Metrics    : {metrics_path}")
    print(f"{bar}\n")
