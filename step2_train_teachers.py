"""
Step 2 — Train flow matching teacher models in latent space.

Requires: results/<exp_name>/latents/real/latents_<dim>.npy  (produced by step1)
          results/<exp_name>/checkpoints/ae_<dim>.pt          (produced by step0, for FID)

Saves (per dim, all under results/<exp_name>/checkpoints/):
    teacher_<dim>_best_loss.pt   — EMA weights at lowest training loss
    teacher_<dim>_best_fid.pt    — EMA weights at best image-space FID (primary)
    teacher_<dim>_latest.pt      — EMA weights at the most recent checkpoint

Periodic FID evaluation (every SAVE_EVERY epochs):
    1. Sample latents from the EMA teacher.
    2. Denormalise with saved latent_mean / latent_std.
    3. Decode through the frozen AE decoder.
    4. Compute FID against CIFAR-10 train via clean-fid.
    5. Update best_fid checkpoint if improved.

Usage:
    python step2_train_teachers.py                        # all dims sequentially
    python step2_train_teachers.py --dim 128              # single dim
    python step2_train_teachers.py --exp-name my_run      # custom experiment
"""

import argparse
import shutil
import numpy as np
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from models.diffusion import FlowMatching
from models.denoiser import TeacherDenoiser, param_count
from models.autoencoder import ConvAutoencoder

from exp_config import get_paths, add_exp_arg, print_exp_summary, ExpPaths

LATENT_DIMS = [64, 128, 256, 384, 512, 1024]
EPOCHS       = 1000
BATCH_SIZE   = 256
LR           = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
SAVE_EVERY   = 50
EMA_DECAY    = 0.9999
EARLY_STOP_PATIENCE  = 300
EARLY_STOP_MIN_DELTA = 1e-4
LOG_INTERVAL = 10

# FID evaluation settings (run every SAVE_EVERY epochs using the frozen AE)
FID_EVAL_SAMPLES = 5_000
FID_EULER_STEPS  = 50
FID_DECODE_BATCH = 256


def get_device(dim: int = None) -> str:
    if torch.cuda.is_available():
        if dim is not None:
            gpu_id = LATENT_DIMS.index(dim) % torch.cuda.device_count()
            return f"cuda:{gpu_id}"
        return "cuda"
    return "cpu"


def normalise_latents(latents: np.ndarray, dim: int, real_latent_dir: Path):
    mean = latents.mean(axis=0, keepdims=True)
    std  = latents.std(axis=0, keepdims=True)
    latents_norm = ((latents - mean) / (std + 1e-8)).astype(np.float32)

    stats_path = real_latent_dir / f"latents_{dim}_norm_stats.npy"
    np.save(stats_path, np.stack([mean.squeeze(), std.squeeze()]))

    print(
        f"  Norm stats saved → {stats_path}  "
        f"(mean avg={mean.mean():.4f}, std avg={std.mean():.4f}, "
        f"std min={std.min():.4f}, std max={std.max():.4f})"
    )

    return latents_norm, mean.squeeze(), std.squeeze()


def create_ema(model, device):
    ema = deepcopy(model).to(device)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def update_ema(ema_model, model, decay=EMA_DECAY):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def load_ae(dim: int, ckpt_dir: Path, device: str):
    """Load frozen AE decoder for FID evaluation. Returns None if not found."""
    ae_path = ckpt_dir / f"ae_{dim}.pt"
    if not ae_path.exists():
        print(f"  [fid-eval] AE checkpoint not found at {ae_path} — FID skipped.")
        return None
    try:
        ckpt = torch.load(str(ae_path), map_location=device, weights_only=True)
        ae = ConvAutoencoder(latent_dim=dim).to(device)
        ae.load_state_dict(ckpt["state_dict"])
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        print(f"  [fid-eval] AE loaded from {ae_path.name}")
        return ae
    except Exception as exc:
        print(f"  [fid-eval] Failed to load AE: {exc} — FID skipped.")
        return None


def eval_teacher_fid(
    ema_model: torch.nn.Module,
    flow: FlowMatching,
    ae: ConvAutoencoder,
    device: str,
    dim: int,
    lat_mean: np.ndarray,
    lat_std: np.ndarray,
    paths: ExpPaths,
) -> float:
    """
    Generate FID_EVAL_SAMPLES images from the EMA teacher, decode with the
    frozen AE, and compute FID against CIFAR-10 train.

    Returns the FID value, or inf if clean-fid is unavailable.
    """
    try:
        from cleanfid import fid as cleanfid_fid
    except ImportError:
        print("    [fid-eval] clean-fid not installed — skipping FID eval.")
        return float("inf")

    tmp_dir = paths.ckpt_dir / "_fid_tmp"
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    lat_mean_t = torch.from_numpy(lat_mean.astype(np.float32)).to(device)
    lat_std_t  = torch.from_numpy(lat_std.astype(np.float32)).to(device)

    ema_model.eval()
    generated = 0
    img_idx   = 0

    with torch.no_grad():
        while generated < FID_EVAL_SAMPLES:
            n_batch = min(FID_DECODE_BATCH, FID_EVAL_SAMPLES - generated)
            z_norm  = flow.euler_sample(ema_model, (n_batch, dim),
                                        n_steps=FID_EULER_STEPS)
            z_orig  = z_norm * lat_std_t + lat_mean_t
            imgs    = (
                torch.sigmoid(ae.decode(z_orig))
                .clamp(0, 1).mul(255).byte().cpu().numpy()
            )
            imgs = imgs.transpose(0, 2, 3, 1)
            for img_arr in imgs:
                Image.fromarray(img_arr).save(tmp_dir / f"{img_idx:05d}.png")
                img_idx += 1
            generated += n_batch

    fid_val = float(
        cleanfid_fid.compute_fid(
            str(tmp_dir),
            dataset_name="cifar10",
            dataset_res=32,
            dataset_split="train",
            verbose=False,
        )
    )

    shutil.rmtree(str(tmp_dir))
    return fid_val


def _save_teacher_ckpt(
    path: Path,
    ema_model: torch.nn.Module,
    dim: int,
    mean: np.ndarray,
    std: np.ndarray,
    loss_history: list,
    fid_history: list,
    best_loss: float,
    best_loss_epoch: int,
    best_fid: float,
    best_fid_epoch: int,
    exp_name: str,
) -> None:
    torch.save(
        {
            "model_state_dict": ema_model.state_dict(),
            "latent_dim":       dim,
            "latent_mean":      torch.from_numpy(mean.astype(np.float32)),
            "latent_std":       torch.from_numpy(std.astype(np.float32)),
            "loss_history":     loss_history,
            "fid_history":      fid_history,
            "best_loss":        best_loss,
            "best_loss_epoch":  best_loss_epoch,
            "best_fid":         best_fid,
            "best_fid_epoch":   best_fid_epoch,
            "exp_name":         exp_name,
        },
        path,
    )


def plot_teacher_loss(history: list, dim: int, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history) + 1), history, color="royalblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Teacher Denoiser — Training Loss  (latent_dim={dim})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = plots_dir / f"teacher_loss_{dim}.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Loss curve → {out}")


def plot_teacher_fid(fid_history: list, dim: int, plots_dir: Path) -> None:
    if not fid_history:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs_eval, fids = zip(*fid_history)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_eval, fids, color="darkorange", linewidth=1.5,
            marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID (↓)")
    ax.set_title(f"Teacher — Image-Space FID per Checkpoint  (latent_dim={dim})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = plots_dir / f"teacher_fid_{dim}.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  FID curve  → {out}")


def train_one_epoch(model, ema_model, loader, flow, optimizer, device, epoch, ema_decay):
    model.train()
    total_loss = 0.0
    for batch_idx, (x_0,) in enumerate(loader):
        x_0 = x_0.to(device)

        x_t, t, v_target = flow.forward(x_0)
        v_pred = model(x_t, t)
        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        update_ema(ema_model, model, decay=ema_decay)

        total_loss += loss.item()

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            avg = total_loss / (batch_idx + 1)
            print(f"    [epoch {epoch:03d}  step {batch_idx+1:04d}]  loss = {avg:.5f}",
                  flush=True)

    return total_loss / len(loader)


def get_teacher_hparams(dim: int):
    if dim <= 128:
        return {"epochs": 1000, "batch_size": 256, "lr": 3e-4, "ema_decay": 0.999}
    elif dim <= 512:
        return {"epochs": 1500, "batch_size": 256, "lr": 2e-4, "ema_decay": 0.999}
    else:
        return {"epochs": 2000, "batch_size": 128, "lr": 1e-4, "ema_decay": 0.999}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=LATENT_DIMS, default=None)
    add_exp_arg(parser)
    args = parser.parse_args()

    paths = get_paths(args.exp_name)

    dims   = [args.dim] if args.dim is not None else LATENT_DIMS
    device = get_device(args.dim)
    print(f"Device: {device}  |  dims: {dims}")

    print_exp_summary(
        paths,
        ckpt_path=paths.ckpt_dir,
        latent_path=paths.real_latent_dir,
    )

    paths.ckpt_dir.mkdir(parents=True, exist_ok=True)

    flow = FlowMatching(device=device)

    for dim in dims:
        # ── skip check ────────────────────────────────────────────────────────
        ckpt_best_fid  = paths.ckpt_dir / f"teacher_{dim}_best_fid.pt"
        ckpt_best_loss = paths.ckpt_dir / f"teacher_{dim}_best_loss.pt"
        ckpt_latest    = paths.ckpt_dir / f"teacher_{dim}_latest.pt"
        legacy_ckpt    = paths.ckpt_dir / f"teacher_{dim}.pt"

        if ckpt_best_fid.exists() or ckpt_best_loss.exists() or legacy_ckpt.exists():
            print(f"\n[skip] teacher_{dim} checkpoint(s) already exist.")
            continue

        print(f"\n{'='*60}")
        print(f"  Training teacher  latent_dim = {dim}")
        hp = get_teacher_hparams(dim)

        epochs     = hp["epochs"]
        batch_size = hp["batch_size"]
        lr         = hp["lr"]
        ema_decay  = hp["ema_decay"]

        print(f"  hparams: epochs={epochs}, batch_size={batch_size}, "
              f"lr={lr}, ema={ema_decay}")

        # ── load latents ─────────────────────────────────────────────────────
        latents_path = paths.real_latent_dir / f"latents_{dim}.npy"
        if not latents_path.exists():
            print(f"  [ERROR] {latents_path} not found — run step1 first.")
            continue

        latents_raw = np.load(latents_path)
        latents, mean, std = normalise_latents(latents_raw, dim, paths.real_latent_dir)

        dataset = TensorDataset(torch.from_numpy(latents))
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=device.startswith("cuda"),
        )

        # ── model ─────────────────────────────────────────────────────────────
        model     = TeacherDenoiser(latent_dim=dim).to(device)
        ema_model = create_ema(model, device)
        print(f"  Model params: {param_count(model)}")

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        # ── load AE for FID evaluation ────────────────────────────────────────
        ae = load_ae(dim, paths.ckpt_dir, device)

        # ── tracking variables ────────────────────────────────────────────────
        best_loss  = float("inf")
        best_loss_epoch = 0
        best_loss_state = None

        best_fid   = float("inf")
        best_fid_epoch  = -1
        best_fid_state  = None

        epochs_without_improvement = 0
        loss_history = []
        fid_history  = []   # list of (epoch, fid_val) tuples

        # ── training loop ─────────────────────────────────────────────────────
        for epoch in tqdm(range(1, epochs + 1), desc=f"dim={dim}"):
            avg_loss = train_one_epoch(
                model, ema_model, loader, flow, optimizer, device, epoch, ema_decay
            )
            scheduler.step()
            loss_history.append(avg_loss)

            # Update best loss
            if avg_loss < best_loss - EARLY_STOP_MIN_DELTA:
                best_loss  = avg_loss
                best_loss_epoch = epoch
                epochs_without_improvement = 0
                best_loss_state = deepcopy(ema_model.state_dict())
                # Save best-loss checkpoint immediately
                _save_teacher_ckpt(
                    ckpt_best_loss, ema_model, dim, mean, std,
                    loss_history, fid_history,
                    best_loss, best_loss_epoch,
                    best_fid,  best_fid_epoch,
                    args.exp_name,
                )
            else:
                epochs_without_improvement += 1

            # Periodic checkpoint
            if epoch % SAVE_EVERY == 0:
                # Save latest
                _save_teacher_ckpt(
                    ckpt_latest, ema_model, dim, mean, std,
                    loss_history, fid_history,
                    best_loss, best_loss_epoch,
                    best_fid,  best_fid_epoch,
                    args.exp_name,
                )

                # FID evaluation
                if ae is not None:
                    print(f"  [fid-eval] epoch {epoch} — generating {FID_EVAL_SAMPLES:,} images …",
                          flush=True)
                    fid_val = eval_teacher_fid(
                        ema_model, flow, ae, device, dim, mean, std, paths
                    )
                    fid_history.append((epoch, fid_val))
                    print(
                        f"  [fid-eval] epoch {epoch:03d}  "
                        f"FID={fid_val:.2f}  best_FID={best_fid:.2f}"
                    )

                    if fid_val < best_fid:
                        best_fid  = fid_val
                        best_fid_epoch = epoch
                        best_fid_state = deepcopy(ema_model.state_dict())
                        _save_teacher_ckpt(
                            ckpt_best_fid, ema_model, dim, mean, std,
                            loss_history, fid_history,
                            best_loss, best_loss_epoch,
                            best_fid,  best_fid_epoch,
                            args.exp_name,
                        )
                        print(f"  [fid-eval] ★ new best FID={best_fid:.2f}  → {ckpt_best_fid.name}")

            print(
                f"  epoch {epoch:03d}  avg_loss={avg_loss:.5f}  "
                f"best_loss={best_loss:.5f}  "
                f"best_fid={best_fid:.2f}"
            )

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"  Early stopping at epoch {epoch:03d}. "
                    f"best_loss_epoch={best_loss_epoch:03d}  "
                    f"best_fid_epoch={best_fid_epoch}"
                )
                break

        # ── final checkpoint saves ─────────────────────────────────────────────

        # best_loss — reload best-loss weights and do final save with full history
        if best_loss_state is not None:
            ema_model.load_state_dict(best_loss_state)
        _save_teacher_ckpt(
            ckpt_best_loss, ema_model, dim, mean, std,
            loss_history, fid_history,
            best_loss, best_loss_epoch,
            best_fid,  best_fid_epoch,
            args.exp_name,
        )
        print(f"  Saved → {ckpt_best_loss.name}  (best_loss={best_loss:.5f})")

        # best_fid — reload best-FID weights; fall back to best-loss if FID never ran
        if best_fid_state is not None:
            ema_model.load_state_dict(best_fid_state)
        elif best_loss_state is not None:
            ema_model.load_state_dict(best_loss_state)
        _save_teacher_ckpt(
            ckpt_best_fid, ema_model, dim, mean, std,
            loss_history, fid_history,
            best_loss, best_loss_epoch,
            best_fid,  best_fid_epoch,
            args.exp_name,
        )
        fid_note = f"best_fid={best_fid:.2f}" if best_fid < float("inf") else "FID not evaluated"
        print(f"  Saved → {ckpt_best_fid.name}  ({fid_note})")

        # latest — keep current state (best_fid weights or best_loss if no FID)
        _save_teacher_ckpt(
            ckpt_latest, ema_model, dim, mean, std,
            loss_history, fid_history,
            best_loss, best_loss_epoch,
            best_fid,  best_fid_epoch,
            args.exp_name,
        )
        print(f"  Saved → {ckpt_latest.name}")

        plot_teacher_loss(loss_history, dim, paths.plots_dir)
        plot_teacher_fid(fid_history,  dim, paths.plots_dir)

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
