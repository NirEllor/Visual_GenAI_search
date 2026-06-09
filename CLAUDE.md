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

**Goal:** Study how latent space dimensionality (64, 128, 256, 384, 512, 1024) and synthetic dataset size (250k, 500k, 1M, 2M) affect the FID score of distilled flow matching models.

**Distillation approach:** Dataset distillation — the teacher generates large synthetic latent datasets via Euler sampling; students are trained from scratch on this synthetic data using a combined loss (50% flow matching on the synthetic data + 50% KD imitation of the frozen teacher).

### Generative Model: Rectified Flow / Flow Matching

All generative models use **Flow Matching** (Rectified Flow), not DDPM. Implemented in `models/diffusion.py`:

- **Training**: straight-line interpolation `x_t = (1-t)*x_0 + t*x_1`, constant velocity target `v = x_1 - x_0`, loss `MSE(v_theta(x_t, t), v)`
- **Sampling**: Euler ODE integration from `t=1` (noise) → `t=0` (data), 50 steps
- Both teacher and student predict velocity `v_theta(x_t, t) → (B, D)`

### Autoencoder

Custom PyTorch `ConvAutoencoder` (`models/autoencoder.py`):
- 3-layer strided conv encoder: `(3,32,32)` → `(64,4,4)` → 1×1 conv → `(C,4,4)` → flatten → `latent_dim`
- `latent_channels = latent_dim // 16`; **`latent_dim` must be divisible by 16**
- 3-layer transposed conv decoder (mirror): unflatten → `(C,4,4)` → 1×1 conv → `(64,4,4)` → upsample
- **No FC bottleneck** — the information bottleneck is controlled entirely by channel width `C` at the 1×1 projection
- The `encoder_conv` (3→16→32→64) is a fixed feature extractor; `encoder_proj` (64→C) is the experimental variable
- This is by design: varying `latent_dim` varies `C`, which varies bottleneck capacity — not layer width
- Trained from scratch in Step 0 using LPIPS (VGG) loss

### File Structure

```
Guided_Research/
├── requirements.txt
├── checkpoints/                     # ae_{dim}.pt — trained PyTorch autoencoders
├── data/                            # Auto-downloaded CIFAR-10
├── latents/                         # latents_{dim}.npy, latents_{dim}_norm_stats.npy
├── synthetic/                       # synthetic/{dim}/synthetic_{dim}_{n}.npy + trajectories
├── models/
│   ├── autoencoder.py               # PyTorch ConvAutoencoder
│   ├── diffusion.py                 # FlowMatching class (forward + euler_sample)
│   └── denoiser.py                  # MLPDenoiser, TeacherDenoiser, StudentDenoiser
├── step0_train_autoencoder.py       # Train 6 ConvAutoencoders from scratch
├── step0b_eval_ae.py                # Evaluate AE reconstruction quality
├── step1_extract_latents.py         # Encode 50k CIFAR-10 images → latents
├── step2_train_teachers.py          # Train 6 flow matching teacher MLPs
├── step3a_generate.py               # Teacher generates synthetic latent datasets
├── step3b_distill.py                # Train 24 student MLPs on synthetic data
├── step4_evaluate.py                # Generate images, compute FID/IS, plot
└── results/trained_AE/              # Metrics, plots, generated images
```

### Pipeline Steps

**Step 0 — `step0_train_autoencoder.py`**
- Train a `ConvAutoencoder` from scratch on CIFAR-10 for each of 6 latent dims
- Loss: LPIPS (VGG backbone, frozen) — perceptual quality
- 1000 epochs, AdamW lr=2e-3, batch_size=128, cosine LR decay, grad_clip=5.0
- Save: `checkpoints/ae_{64,128,256,384,512,1024}.pt`

**Step 0b — `step0b_eval_ae.py`**
- Evaluate AE reconstruction quality (FID, visual samples)
- Results: `results/ae_eval/`

**Step 1 — `step1_extract_latents.py`**
- Load each trained `ConvAutoencoder` from `checkpoints/ae_{dim}.pt`
- Encode all 50k CIFAR-10 training images (batch_size=512) through the frozen encoder
- Save: `latents/latents_{64,128,256,384,512,1024}.npy` — shape `(50000, dim)`

**Step 2 — `step2_train_teachers.py`**
- Normalise latents to zero-mean unit-variance → saves `latents/latents_{dim}_norm_stats.npy`
- Train a flow matching teacher `TeacherDenoiser` (4 res blocks, hidden_dim=512) on normalised latents
- 1000 epochs, AdamW lr=3e-4, weight_decay=1e-4, batch_size=256, cosine LR decay, grad_clip=1.0
- EMA of model weights (decay=0.9999); checkpoint saved every 50 epochs
- Save: `models/teacher_{dim}.pt` (includes `latent_mean`, `latent_std` for denormalisation)

**Step 3a — `step3a_generate.py`**
- Load each teacher; run Euler sampling (50 steps, batch=2048) to produce synthetic latent datasets
- 4 sizes per dim: 250k, 500k, 1M, 2M samples (normalised space, float32 memmap)
- Also generates a trajectory dataset: 50k × 51 frames × dim (float16 memmap) for analysis
- Save: `synthetic/{dim}/synthetic_{dim}_{n}.npy`, `synthetic/{dim}/trajectories_{dim}.npy`

**Step 3b — `step3b_distill.py`**
- Train 24 students (6 dims × 4 sizes) — each `StudentDenoiser` (4 res blocks, hidden_channels scales with dim)
- Combined loss: 0.5 · flow_matching + 0.5 · KD imitation of frozen teacher
- x_1 and t sampled once per batch; shared by x_t, v_target, and the teacher call
- 1000 epochs, AdamW lr=1e-4, cosine LR decay, EMA (decay=0.9999), batch_size=256
- Save: `models/student_{dim}_{n_samples}.pt`

**Step 4 — `step4_evaluate.py`**
- Four independently restartable phases: `--generate`, `--decode`, `--metrics`, `--plot`
- `--generate`: Euler sampling (50 steps) → 10k latent samples → denormalise → `.npy`
- `--decode`: PyTorch AE decoder → PNG images; also reconstructs CIFAR-10 test set for AE-FID
- `--metrics`: FID (clean-fid vs CIFAR-10 train), IS (torch-fidelity), AE-FID
- `--plot`: produces `fid_vs_size.png` (FID vs dataset size, one line per dim) and `fid_vs_dim.png` (FID vs dim, one line per size), unified `metrics_all.json`
- Evaluates both teachers (`--teacher`) and students (`--size N`)

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

Steps 0, 2, 3a, 3b, and 4 support `--dim` (and `--size` for 3b/4) to pin a run to a specific GPU:

```bash
# Step 2 — train all 6 teachers in parallel (round-robin across available GPUs)
for dim in 64 128 256 384 512 1024; do
    python step2_train_teachers.py --dim $dim &
done
wait

# Step 3a — generate synthetic datasets
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

| Model | Blocks | Hidden dim | Params (dim=128) |
|-------|--------|-----------|-----------------|
| TeacherDenoiser | 4 ResBlocks | 512 | ~large |
| StudentDenoiser | 2 ResBlocks | 256 | ~4× fewer |

Both use: sinusoidal time embedding (dim=256) → 2-layer MLP → time projection into each ResBlock via `LayerNorm → Linear → GELU + time_proj → Linear + residual`.

### Pitfalls

| Risk | Mitigation |
|------|-----------|
| AE latent scale mismatch | Normalise latents before training; denormalise at eval |
| x_t / v_target inconsistency | Step 3b sanity-checks that x_1 and t are shared across x_t and v_target |
| Large synthetic datasets | 2M × 1024 × 4 bytes ≈ 8 GB; use memmap, optionally `--no-load-to-ram` |
| Stale student checkpoint | 3b skips if `student_{dim}_{size}.pt` exists — delete to retrain |
| Invalid latent_dim for AE | `ConvAutoencoder` requires `latent_dim % 16 == 0`; all 6 dims (64,128,256,384,512,1024) satisfy this |
