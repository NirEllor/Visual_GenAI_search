# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is a research project on knowledge distillation studying how **latent space dimensionality** and **synthetic dataset size** affect the FID score of distilled flow matching models on CIFAR-10. All pipeline scripts are implemented. The virtual environment is in `.venv/`.

## Repository

- GitHub: https://github.com/NirEllor/Distillation_Research
- Clone: `git clone https://github.com/NirEllor/Distillation_Research.git`

## Committing Changes

After every task, commit and push all changes:

```bash
git add .
git commit -m "<clear description of what was done>"
git push
```

## Environment

- Python virtual environment: `.venv/`
- Activate: `source .venv/Scripts/activate` (Windows/bash)
- Install dependencies: `pip install -r requirements.txt`
- Run scripts: `python <script>.py`

---

## Research Plan: Knowledge Distillation Pipeline for Latent Flow Matching on CIFAR-10

**Goal:** Study how latent space dimensionality (64, 128, 256, 384, 512, 1024) and synthetic dataset size (50k, 100k, 150k, 200k) affect the FID score of distilled flow matching models.

**Distillation approach:** Dataset distillation — the teacher generates synthetic latent datasets via Euler sampling; students train on cached (noise, sample) endpoint pairs from these trajectories using a one-step velocity-regression loss. No explicit teacher forward pass or KD loss is used during student training; the distillation effect is implicit via the teacher-generated synthetic data.

### Generative Model: Rectified Flow / Flow Matching

All generative models use **Flow Matching** (Rectified Flow), not DDPM. Implemented in `models/diffusion.py`:

- **Training**: straight-line interpolation `x_t = (1-t)*x_0 + t*x_1`, constant velocity target `v = x_1 - x_0`, loss `MSE(v_theta(x_t, t), v)`
- **Sampling**: Euler ODE integration from `t=1` (noise) → `t=0` (data), 50 steps
- Both teacher and student predict velocity `v_theta(x_t, t) → (B, D)`

### Autoencoder

Custom PyTorch `ConvAutoencoder` (`models/autoencoder.py`):
- Encoder: `Conv2d(3→128,k3)` → `down1` (ResNetBlock(128) + stride-2 conv 128→128, 32x32→16x16) → `down2` (ResNetBlock(128) + stride-2 conv 128→256, 16x16→8x8) → `down3` (ResNetBlock(256) + stride-2 conv 256→256, 8x8→4x4) → `encoder_out` (ResNetBlock(256) + GroupNorm + GELU + 1×1 Conv 256→`latent_channels`)
- Decoder: mirrors encoder with ConvTranspose2d + ResNetBlocks, final output Conv2d to 3 channels
- `latent_channels = latent_dim // 16`; **`latent_dim` must be divisible by 16**
- **No FC bottleneck** — bottleneck is the 1×1 conv projection at the spatial (4,4) bottleneck
- Deterministic autoencoder: `encode()` returns VAE-style 3-tuple `(z, mean, logvar)` for interface compatibility, but `logvar` is always zeros
- Trained from scratch in Step 0 using LPIPS (VGG backbone, frozen)

### File Structure

```
Guided_Research/
├── requirements.txt
├── data/                            # Auto-downloaded CIFAR-10
├── models/
│   ├── autoencoder.py               # PyTorch ConvAutoencoder
│   ├── diffusion.py                 # FlowMatching class (forward + euler_sample)
│   └── denoiser.py                  # TeacherDenoiser, StudentDenoiser, MLPTeacherDenoiser (legacy)
├── exp_config.py                    # ExpPaths, add_exp_arg, get_paths, save_config
├── step0_train_autoencoder.py       # Train 6 ConvAutoencoders from scratch
├── step0b_eval_ae.py                # Evaluate AE reconstruction quality
├── step1_extract_latents.py         # Encode 50k CIFAR-10 images → latents
├── step2_train_teachers.py          # Train 6 flow matching teachers
├── step3a_generate.py               # Teacher generates synthetic latent trajectories
├── step3b_distill.py                # Train 24 students on cached endpoint pairs
├── step4_evaluate.py                # Generate images, compute FID/IS, plot
├── slurm/                           # Cluster job scripts (run_step0.sh, run_step1.sh, ...)
├── results/
│   ├── ae_kl_lpips_only_v1/         # Default experiment output (--exp-name controls this)
│   │   ├── checkpoints/             # ae_{dim}.pt, teacher_{dim}_{best_fid,best_loss,latest}.pt, student_{dim}_{n}.pt
│   │   ├── latents/
│   │   │   ├── real/                # latents_{dim}.npy, latents_{dim}_norm_stats.npy
│   │   │   ├── teacher/dim_{dim}/   # synthetic_{dim}_{n}.npy, trajectories_{dim}.npy, paired_endpoints_{dim}.npz
│   │   │   └── student/             # student eval latents (step4)
│   │   ├── ae_recon/dim_{dim}/      # AE encode→decode reconstructions
│   │   ├── generated/{teacher,student}/dim_{dim}[/n_{size}]/  # PNG images
│   │   ├── metrics/                 # metrics_{dim}_{n}.json, metrics_all.json
│   │   ├── plots/                   # fid_vs_size.png, fid_vs_dim.png, student_loss_*.png
│   │   └── config.json              # git hash, timestamp, hyperparams
│   └── trained_AE/                  # Legacy results directory
└── (empty/vestigial top-level dirs: checkpoints/, latents/, synthetic/)
```

Also at top level: `conditional_flow_matching.py`, `optimal_transport.py` (model extensions);
`analyze_teacher_sampling.py`, `analyze_student_latents.py`, `visualize_latents.py`,
`plot_losses.py`, `dry_run.py`, `sanity_flow.py`, `gaussian_baseline_decode.py` (analysis scripts);
`architecture.txt`, `RESET_PIPELINE.md`, `Dockerfile`, `.dockerignore` (docs/config).

### Pipeline Steps

**Step 0 — `step0_train_autoencoder.py`**
- Train a `ConvAutoencoder` from scratch on CIFAR-10 for each of 6 latent dims
- Loss: 3-term weighted sum (MSE + LPIPS + KL) with MSE_WEIGHT=0.0, KL_WEIGHT=0.0, LPIPS_WEIGHT=1.0 → effectively LPIPS only (VGG backbone, frozen)
- 1000 epochs, AdamW lr=1e-4, batch_size=128, cosine LR decay, grad_clip=5.0
- Save: `results/<exp_name>/checkpoints/ae_{dim}.pt` (all 6 dims)

**Step 0b — `step0b_eval_ae.py`**
- Evaluate AE reconstruction quality (FID, visual samples)
- Results: `results/ae_eval/`

**Step 1 — `step1_extract_latents.py`**
- Load each trained `ConvAutoencoder` from `results/<exp_name>/checkpoints/ae_{dim}.pt`
- Encode all 50k CIFAR-10 training images (batch_size=512) through the frozen encoder
- Save: `results/<exp_name>/latents/real/latents_{dim}.npy` — shape `(50000, dim)`, dtype float32

**Step 2 — `step2_train_teachers.py`**
- Normalise latents to zero-mean unit-variance → saves `results/<exp_name>/latents/real/latents_{dim}_norm_stats.npy`
- Train a flow matching `TeacherDenoiser` (ConvDenoiser with FiLM time conditioning) on normalised latents; hyperparams scale per dim tier:
  - dim ≤ 128: 1000 epochs, lr=3e-4, batch_size=256
  - dim ≤ 512: 1500 epochs, lr=2e-4, batch_size=256
  - dim=1024: 2000 epochs, lr=1e-4, batch_size=128
- AdamW weight_decay=1e-4, cosine LR decay, grad_clip=1.0, EMA (decay=0.999, note: not 0.9999)
- Periodic clean-fid eval every 50 epochs against CIFAR-10 train; early stopping patience=300
- Save three checkpoints per dim: `results/<exp_name>/checkpoints/teacher_{dim}_{best_fid,best_loss,latest}.pt` (or legacy fallback `teacher_{dim}.pt`)

**Step 3a — `step3a_generate.py`**
- Load each teacher; run Euler sampling (200 steps, batch=128) to produce trajectories
- 4 synthetic dataset sizes per dim: 50k, 100k, 150k, 200k samples (cached endpoint pairs from trajectories)
- Trajectory dataset: 200k × 201 frames × dim (float16 memmap, one trajectory per teacher seed)
- Save: `results/<exp_name>/latents/teacher/dim_{dim}/synthetic_{dim}_{n}.npy` (endpoint pairs), `.../trajectories_{dim}.npy` (full trajectories), `.../paired_endpoints_{dim}.npz` (cached on first run for speed)

**Step 3b — `step3b_distill.py`**
- Train 24 students (6 dims × 4 sizes) — each untimed `StudentDenoiser` (default 4 res blocks, hidden_channels=max(256, latent_channels*8))
- Training: extract cached endpoint pairs `(x1, x0)` from step 3a's trajectory memmap; for each batch: `x_t = x_1`, `v_target = x_1 - x_0`, `loss = MSE(student(x_t), v_target)`
- One-step velocity regression: no time sampling, no teacher forward pass, no KD loss — distillation is purely via the teacher-generated synthetic data
- 500 epochs, AdamW lr=3e-4, weight_decay=1e-4, cosine LR decay, EMA (decay=0.9999), batch_size=256, grad_clip=1.0
- Skips if checkpoint exists; supports `--dim` and `--size` for parallel GPU runs
- Save: `results/<exp_name>/checkpoints/student_{dim}_{n}.pt` (all 24 combinations)

**Step 4 — `step4_evaluate.py`**
- Four independently restartable phases: `--generate`, `--decode`, `--metrics`, `--plot`
- `--generate`: sample 10k latents (teachers use Euler 200 steps; students use single-step sampling) → denormalise using cached norm stats → `.npy`
- `--decode`: PyTorch AE decoder → PNG images; also encodes CIFAR-10 test set through AE and decodes for AE-FID
- `--metrics`: compute FID (clean-fid vs CIFAR-10 train split), IS (torch-fidelity), AE-FID (AE-recon vs CIFAR-10 train)
- `--plot`: produces `fid_vs_size.png` (FID vs dataset size, one line per dim) and `fid_vs_dim.png` (FID vs dim, one line per size), unified `metrics_all.json`
- Evaluates both teachers (`--teacher` flag) and students (`--size N` flag); supports `--dim` for single-dim runs

### Key Design Decisions

- Autoencoders are **frozen** after Step 0 — used only for encode (Step 1) and decode (Step 4)
- Entire pipeline is **PyTorch only** (no JAX/Flax)
- Generative model is **Flow Matching**, not DDPM
- Distillation is **dataset distillation**: teacher generates synthetic data, student trains on it (no combined KD loss)
- Latents are normalised to zero-mean unit-variance before training; denormalised at eval time using saved stats
- Both teacher and student use **EMA weights** for inference

### Execution Order

```bash
pip install -r requirements.txt
python step0_train_autoencoder.py    # ~few hrs per dim (GPU)
python step0b_eval_ae.py             # optional AE quality check
python step1_extract_latents.py      # ~minutes (GPU encoding)
python step2_train_teachers.py       # ~hours per dim (GPU)
python step3a_generate.py            # ~hours per dim (GPU sampling)
python step3b_distill.py             # ~hours per dim×size (GPU)
# then for each dim+size:
python step4_evaluate.py --generate --dim D --size N
python step4_evaluate.py --decode   --dim D --size N
python step4_evaluate.py --metrics  --dim D --size N
python step4_evaluate.py --plot
```

#### Parallel GPU execution

Steps 0, 2, 3a, 3b, and 4 support `--dim` (and `--size` for 3b/4) to pin a run to a specific GPU.
All output directories are controlled by `--exp-name` (default `ae_kl_lpips_only_v1`).
Cluster jobs for SLURM are pre-built in `slurm/` (e.g. `run_step2.sh`, `run_step3a.sh`, etc.).

```bash
# Step 2 — train all 6 teachers in parallel (round-robin across available GPUs)
for dim in 64 128 256 384 512 1024; do
    python step2_train_teachers.py --dim $dim &
done
wait

# Step 3a — generate synthetic trajectories
for dim in 64 128 256 384 512 1024; do
    python step3a_generate.py --dim $dim &
done
wait

# Step 3b — train all 24 students
for dim in 64 128 256 384 512 1024; do
    python step3b_distill.py --dim $dim &
done
wait
```

### Model Architectures

**TeacherDenoiser** (conv-based, operates on (C,4,4) spatial maps):
- Processes latent as (C,4,4) where C=latent_dim//16, applies ConvResBlocks with FiLM time conditioning
- Time embedding: sinusoidal(dim=256) → 2-layer MLP → per-block FiLM (scale/shift via Linear(256→2*channels))
- Blocks scale per dim tier: dim≤128 → 8 blocks, hidden=256; dim≤512 → 10 blocks, hidden=384; else → 12 blocks, hidden=512
- Each ConvResBlock: GroupNorm → Conv3×3 → FiLM scale/shift → GELU → GroupNorm → Conv3×3 → residual add

**StudentDenoiser** (untimed conv, no time conditioning):
- Same spatial processing (C,4,4), but blocks ignore time argument `t` entirely
- Default 4 ConvResBlocks, hidden_channels=max(256, latent_channels*8)
- Blocks: GroupNorm → Conv3×3 → GELU → GroupNorm → Conv3×3 → residual add
- One-step sampling only (no time interpolation during training or inference)

Use `param_count(model)` in `models/denoiser.py` to compute parameter counts for any dim.

### Pitfalls

| Risk | Mitigation |
|------|-----------|
| AE latent scale mismatch | Normalise latents to zero-mean unit-variance before training; denormalise at eval using saved `latents_{dim}_norm_stats.npy` |
| Large trajectory data | Trajectories are the main memory artifact: 200k×201×1024×2 bytes ≈ 82 GB for dim=1024 (float16 memmap); use on disk, not RAM. Paired endpoints cache (`paired_endpoints_{dim}.npz`) is built once per dim on first step 3b run |
| Stale checkpoint | Step 3b skips if checkpoint already exists — delete `results/<exp_name>/checkpoints/student_{dim}_{n}.pt` to retrain |
| Invalid latent_dim for AE | `ConvAutoencoder` requires `latent_dim % 16 == 0` for (4×4) bottleneck; all 6 dims (64,128,256,384,512,1024) satisfy this |
| Teacher checkpoint selection | Step 2 saves three checkpoints per dim (`best_fid`, `best_loss`, `latest`); step 4 defaults to `best_fid`. Override with `--teacher-ckpt best_loss` or `latest` if needed |
