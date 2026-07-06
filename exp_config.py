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


def save_config(paths: ExpPaths, section: str, extra: Optional[dict] = None) -> None:
    """
    Merge `extra` into results/<exp_name>/config.json under cfg[section].

    Each pipeline step owns one section (e.g. "autoencoder", "teacher") so
    that steps writing at different times — or in parallel, one process per
    --dim — don't clobber each other's entries. Top-level exp_name/timestamp/
    git_commit are refreshed on every call.
    """
    paths.exp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = paths.exp_dir / "config.json"

    cfg: dict = {}
    if cfg_path.exists():
        with open(cfg_path) as fh:
            cfg = json.load(fh)

    cfg["exp_name"] = paths.exp_name
    cfg["timestamp"] = datetime.now().isoformat(timespec="seconds")
    cfg["git_commit"] = _git_hash()

    cfg.setdefault(section, {})
    if extra:
        cfg[section].update(extra)

    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"  Config saved  → {cfg_path}  [section: {section}]")


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


def get_teacher_ckpt_path(paths: "ExpPaths", dim: int, ckpt_type: str = "best_fid") -> Path:
    """
    Resolve a teacher checkpoint path.

    ckpt_type: "best_fid" | "best_loss" | "latest"

    Resolution order:
      1. results/<exp_name>/checkpoints/teacher_<dim>_<ckpt_type>.pt  (new scheme)
      2. results/<exp_name>/checkpoints/teacher_<dim>.pt              (legacy fallback)

    Returns the resolved path even if it does not exist (caller handles the error).
    """
    valid = {"best_fid", "best_loss", "latest"}
    if ckpt_type not in valid:
        raise ValueError(f"ckpt_type must be one of {valid}, got {ckpt_type!r}")

    new_path = paths.ckpt_dir / f"teacher_{dim}_{ckpt_type}.pt"
    if new_path.exists():
        return new_path

    legacy = paths.ckpt_dir / f"teacher_{dim}.pt"
    if legacy.exists():
        print(
            f"  [ckpt] {new_path.name} not found — falling back to legacy "
            f"{legacy.name}"
        )
        return legacy

    return new_path  # doesn't exist; caller emits the error


def add_teacher_ckpt_arg(parser) -> None:
    """Attach --teacher-ckpt to any argparse.ArgumentParser."""
    parser.add_argument(
        "--teacher-ckpt",
        choices=["best_fid", "best_loss", "latest"],
        default="best_fid",
        dest="teacher_ckpt",
        help=(
            "Which teacher checkpoint to load: best_fid (default), "
            "best_loss, or latest"
        ),
    )


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
